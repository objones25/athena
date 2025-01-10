package milvus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"container/heap"

	lru "github.com/hashicorp/golang-lru"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/objones25/athena/internal/embedding"
	"github.com/objones25/athena/internal/storage"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Config holds the configuration for the Milvus store
type Config struct {
	Host           string
	Port           int
	CollectionName string
	Dimension      int
	BatchSize      int
	MaxRetries     int
	PoolSize       int
	Quantization   QuantizationConfig
	Graph          GraphConfig
	// Add embedding service configuration
	EmbeddingService embedding.Service
}

const (
	defaultDimension      = 1536 // Increased for modern embedding models
	defaultTimeout        = 30 * time.Second
	batchSize             = 2000
	maxConnections        = 10
	maxPoolSize           = 30
	connTimeout           = 10 * time.Second
	flushInterval         = 5 * time.Second
	maxBufferSize         = 5000
	prefetchSize          = 100
	cacheWarmupSize       = 1000
	patternExpiryDuration = 24 * time.Hour

	// New optimization constants
	vectorChunkSize     = 512   // Size for parallel vector processing
	maxConcurrentChunks = 8     // Maximum concurrent vector chunks
	vectorCacheSize     = 10000 // Number of vectors to cache
	similarityThreshold = 0.95  // Threshold for similarity matching
)

// SearchStrategy defines different search approaches
type SearchStrategy int

const (
	// SearchStrategyStandard uses direct Milvus search
	SearchStrategyStandard SearchStrategy = iota
	// SearchStrategyParallel uses parallel processing for search
	SearchStrategyParallel
	// SearchStrategyLSH uses Locality Sensitive Hashing
	SearchStrategyLSH
	// SearchStrategyQuantization uses vector quantization
	SearchStrategyQuantization
)

var (
	ErrNotFound = errors.New("item not found")
	ErrTimeout  = errors.New("operation timed out")
)

// Ensure MilvusStore implements VectorStore
var _ storage.VectorStore = (*MilvusStore)(nil)

// WorkQueue manages concurrent operations
type WorkQueue struct {
	tasks    chan func() error
	workers  int
	maxQueue int
	wg       sync.WaitGroup
	closed   atomic.Bool
	logger   zerolog.Logger
}

func newWorkQueue(workers, maxQueue int) *WorkQueue {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if maxQueue <= 0 {
		maxQueue = workers * 100
	}

	wq := &WorkQueue{
		tasks:    make(chan func() error, maxQueue),
		workers:  workers,
		maxQueue: maxQueue,
		logger:   log.With().Str("component", "milvus").Logger(),
	}

	// Start worker goroutines
	wq.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go wq.worker()
	}

	return wq
}

func (wq *WorkQueue) worker() {
	defer wq.wg.Done()
	for task := range wq.tasks {
		if task == nil {
			continue
		}
		if err := task(); err != nil {
			wq.logger.Error().Err(err).Msg("Task error")
		}
	}
}

func (wq *WorkQueue) Submit(task func() error) error {
	if wq.closed.Load() {
		return fmt.Errorf("work queue is closed")
	}

	select {
	case wq.tasks <- task:
		return nil
	default:
		return fmt.Errorf("work queue is full")
	}
}

func (wq *WorkQueue) Close() {
	if !wq.closed.CompareAndSwap(false, true) {
		return // Already closed
	}
	close(wq.tasks)
	wq.wg.Wait() // Wait for all workers to finish
}

// ShardedCache implements a sharded LRU cache to reduce lock contention
type ShardedCache struct {
	shards    []*lru.Cache
	numShards int
}

// newShardedCache creates a new sharded cache with the specified total size
func newShardedCache(totalSize int) (*ShardedCache, error) {
	numShards := runtime.NumCPU() * 2 // Use 2x CPU cores for number of shards
	shardSize := totalSize / numShards

	sc := &ShardedCache{
		shards:    make([]*lru.Cache, numShards),
		numShards: numShards,
	}

	for i := 0; i < numShards; i++ {
		cache, err := lru.New(shardSize)
		if err != nil {
			return nil, fmt.Errorf("failed to create cache shard %d: %w", i, err)
		}
		sc.shards[i] = cache
	}

	return sc, nil
}

// getShard returns the appropriate shard for a given key
func (sc *ShardedCache) getShard(key string) *lru.Cache {
	// Use FNV hash for better distribution
	h := fnv.New32a()
	h.Write([]byte(key))
	return sc.shards[h.Sum32()%uint32(sc.numShards)]
}

func (sc *ShardedCache) Add(key string, value interface{}) bool {
	return sc.getShard(key).Add(key, value)
}

func (sc *ShardedCache) Get(key string) (interface{}, bool) {
	return sc.getShard(key).Get(key)
}

func (sc *ShardedCache) Remove(key string) {
	sc.getShard(key).Remove(key)
}

// MilvusStore implements the storage.Store interface using Milvus as the backend.
type MilvusStore struct {
	collectionName string
	dimension      int
	batchSize      int
	maxRetries     int
	poolSize       int
	pool           *connectionPool
	logger         zerolog.Logger
	host           string
	port           int
	collection     string
	timeout        time.Duration
	embedder       embedding.Service
	metrics        struct {
		failedOps       int64
		failedInserts   int64
		totalInsertTime int64
		totalInserts    int64
		cacheHits       int64
		cacheMisses     int64
		avgBatchSize    float64
	}
	workQueue    *WorkQueue
	insertBuffer struct {
		sync.Mutex
		items     []*storage.Item
		lastFlush time.Time
	}
	localCache     *ShardedCache
	accessPatterns struct {
		sync.RWMutex
		patterns      map[string]*AccessPattern
		relationships map[string][]string
	}
	vectorPool       *vectorBufferPool
	quantizer        *Quantizer
	graph            *SimilarityGraph
	vectorStore      *LockFreeVectorStore
	simdProc         *SimdProcessor
	batchProc        *BatchProcessor
	lshIndex         *LSHIndex
	hybridSearch     *HybridSearch
	indexUpdater     *ConcurrentIndexUpdater
	optimisticSearch *OptimisticSearch
	adaptiveIndex    *AdaptiveIndex
	indexStats       struct {
		sync.RWMutex
		searchLatency   []time.Duration
		searchHitRate   float64
		totalSearches   int64
		successSearches int64
	}
}

// AccessPattern tracks item access history
type AccessPattern struct {
	LastAccessed time.Time
	AccessCount  int
	// Track which items are accessed before/after this one
	BeforeItems map[string]int
	AfterItems  map[string]int
}

// connectionPool manages a pool of Milvus connections
type connectionPool struct {
	connections    chan *pooledConnection
	maxSize        int
	connectedConns int32 // Track actual number of connected clients
	metrics        struct {
		inUse      int64
		failed     int64
		reconnects int64
		waits      int64 // Track connection wait times
		waitTime   int64 // Total wait time in nanoseconds
	}
}

type pooledConnection struct {
	client    client.Client
	lastUsed  time.Time
	failures  int
	inUse     bool
	healthCtx context.CancelFunc
}

// vectorBufferPool manages pre-allocated vector buffers with size classes
type vectorBufferPool struct {
	pools     []sync.Pool
	sizes     []int
	dimension int
}

// Update newVectorBufferPool
func newVectorBufferPool(dimension int) *vectorBufferPool {
	// Reduce size classes and make them more targeted
	sizes := []int{32, 128, 512, 2048} // Reduced from 7 to 4 size classes
	pools := make([]sync.Pool, len(sizes))

	for i, size := range sizes {
		size := size // Capture for closure
		pools[i] = sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, size*dimension)
			},
		}
	}

	return &vectorBufferPool{
		pools:     pools,
		sizes:     sizes,
		dimension: dimension,
	}
}

// Update get method
func (p *vectorBufferPool) get(size int) []float32 {
	// Find the appropriate size class
	poolIndex := p.findSizeClass(size)
	if poolIndex >= 0 {
		buf := p.pools[poolIndex].Get().([]float32)
		return buf[:0] // Reset length but keep capacity
	}

	// If no suitable pool found, allocate new buffer
	return make([]float32, 0, size)
}

// Update put method
func (p *vectorBufferPool) put(buf []float32) {
	if cap(buf) == 0 {
		return
	}

	// Find the appropriate size class
	poolIndex := p.findSizeClass(cap(buf))
	if poolIndex >= 0 {
		p.pools[poolIndex].Put(buf)
	}
}

// Add findSizeClass method
func (p *vectorBufferPool) findSizeClass(size int) int {
	// Find the smallest size class that can accommodate the requested size
	for i, s := range p.sizes {
		if s*p.dimension >= size {
			return i
		}
	}
	return -1
}

// Update preallocate method to reduce initial allocations
func (p *vectorBufferPool) preallocate() {
	for i, size := range p.sizes {
		// Reduce pre-allocations from 4 to 2 per size class
		for j := 0; j < 2; j++ {
			p.pools[i].Put(make([]float32, 0, size*p.dimension))
		}
	}
}

// NewMilvusStore creates a new Milvus store instance with the provided configuration.
func NewMilvusStore(cfg Config) (*MilvusStore, error) {
	fmt.Printf("[MilvusStore:Init] Starting store initialization\n")
	if cfg.Dimension <= 0 {
		cfg.Dimension = defaultDimension
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 100
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
	}
	if cfg.PoolSize <= 0 {
		cfg.PoolSize = 5
	}

	if cfg.EmbeddingService == nil {
		return nil, fmt.Errorf("embedding service is required")
	}
	fmt.Printf("[MilvusStore:Init] Configuration validated\n")

	// Create sharded cache instead of simple LRU
	fmt.Printf("[MilvusStore:Init] Creating sharded cache\n")
	localCache, err := newShardedCache(10000) // Cache 10k items
	if err != nil {
		return nil, fmt.Errorf("failed to create local cache: %w", err)
	}
	fmt.Printf("[MilvusStore:Init] Sharded cache created\n")

	// Initialize connection pool
	fmt.Printf("[MilvusStore:Init] Creating connection pool\n")
	pool, err := newConnectionPool(cfg.PoolSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}
	fmt.Printf("[MilvusStore:Init] Connection pool created\n")

	// Initialize quantizer
	fmt.Printf("[MilvusStore:Init] Initializing quantizer\n")
	quantizer := NewQuantizer(cfg.Quantization, cfg.Dimension)
	fmt.Printf("[MilvusStore:Init] Quantizer initialized\n")

	// Initialize similarity graph
	fmt.Printf("[MilvusStore:Init] Initializing similarity graph\n")
	graph := NewSimilarityGraph(cfg.Graph, cfg.Dimension)
	fmt.Printf("[MilvusStore:Init] Similarity graph initialized\n")

	// Initialize optimized components
	fmt.Printf("[MilvusStore:Init] Initializing vector store and processors\n")
	vectorStore := NewLockFreeVectorStore(cfg.Dimension, 10000)
	simdProc := NewSimdProcessor(runtime.NumCPU())
	batchProc := NewBatchProcessor(runtime.NumCPU())
	fmt.Printf("[MilvusStore:Init] Vector store and processors initialized\n")

	// Initialize and preallocate vector pool
	fmt.Printf("[MilvusStore:Init] Initializing vector pool\n")
	vectorPool := newVectorBufferPool(cfg.Dimension)
	vectorPool.preallocate()
	fmt.Printf("[MilvusStore:Init] Vector pool initialized\n")

	// Initialize LSH index
	fmt.Printf("[MilvusStore:Init] Initializing LSH index\n")
	lshConfig := DefaultLSHConfig()
	lshIndex := NewLSHIndex(lshConfig, cfg.Dimension)
	fmt.Printf("[MilvusStore:Init] LSH index initialized\n")

	// Initialize hybrid search
	fmt.Printf("[MilvusStore:Init] Initializing hybrid search\n")
	hybridConfig := HybridSearchConfig{
		LSHTables:          cfg.Graph.BatchSize,
		LSHFunctions:       4,
		LSHThreshold:       similarityThreshold,
		HNSWMaxNeighbors:   cfg.Graph.MaxNeighbors,
		HNSWMaxSearchDepth: cfg.Graph.MaxSearchDepth,
		NumWorkers:         cfg.PoolSize,
		BatchSize:          cfg.BatchSize,
		SearchTimeout:      defaultTimeout,
		QualityThreshold:   similarityThreshold,
	}

	hybridSearch, err := NewHybridSearch(hybridConfig, cfg.Dimension)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize hybrid search: %w", err)
	}
	fmt.Printf("[MilvusStore:Init] Hybrid search initialized\n")

	// Initialize concurrent index updater
	fmt.Printf("[MilvusStore:Init] Initializing index updater\n")
	indexUpdater := NewConcurrentIndexUpdater(cfg.Dimension)
	fmt.Printf("[MilvusStore:Init] Index updater initialized\n")

	// Initialize optimistic search
	fmt.Printf("[MilvusStore:Init] Initializing optimistic search\n")
	optimisticSearch := NewOptimisticSearch(cfg.Dimension)
	fmt.Printf("[MilvusStore:Init] Optimistic search initialized\n")

	// Initialize adaptive index
	fmt.Printf("[MilvusStore:Init] Initializing adaptive index\n")
	adaptiveConfig := &AdaptiveConfig{
		DatasetSizeThreshold: 1000000, // 1M items
		LatencyThreshold:     100 * time.Millisecond,
		MemoryThreshold:      1 << 30, // 1GB
	}
	adaptiveIndex := NewAdaptiveIndex(adaptiveConfig)
	fmt.Printf("[MilvusStore:Init] Adaptive index initialized\n")

	store := &MilvusStore{
		collectionName: cfg.CollectionName,
		dimension:      cfg.Dimension,
		batchSize:      cfg.BatchSize,
		maxRetries:     cfg.MaxRetries,
		poolSize:       cfg.PoolSize,
		pool:           pool,
		logger:         log.With().Str("component", "milvus").Logger(),
		host:           cfg.Host,
		port:           cfg.Port,
		collection:     cfg.CollectionName,
		timeout:        defaultTimeout,
		embedder:       cfg.EmbeddingService,
		workQueue:      newWorkQueue(cfg.PoolSize, cfg.PoolSize*100),
		insertBuffer: struct {
			sync.Mutex
			items     []*storage.Item
			lastFlush time.Time
		}{
			items:     make([]*storage.Item, 0, maxBufferSize),
			lastFlush: time.Now(),
		},
		localCache:       localCache,
		vectorPool:       vectorPool,
		quantizer:        quantizer,
		graph:            graph,
		vectorStore:      vectorStore,
		simdProc:         simdProc,
		batchProc:        batchProc,
		lshIndex:         lshIndex,
		hybridSearch:     hybridSearch,
		indexUpdater:     indexUpdater,
		optimisticSearch: optimisticSearch,
		adaptiveIndex:    adaptiveIndex,
		indexStats: struct {
			sync.RWMutex
			searchLatency   []time.Duration
			searchHitRate   float64
			totalSearches   int64
			successSearches int64
		}{
			searchLatency: make([]time.Duration, 0, 1000),
		},
	}
	fmt.Printf("[MilvusStore:Init] Store instance created\n")

	// Initialize access patterns tracking
	store.accessPatterns.patterns = make(map[string]*AccessPattern)
	store.accessPatterns.relationships = make(map[string][]string)

	// Initialize pool connections
	fmt.Printf("[MilvusStore:Init] Initializing connection pool\n")
	if err := store.initializePool(cfg.PoolSize); err != nil {
		return nil, fmt.Errorf("failed to initialize connection pool: %w", err)
	}
	fmt.Printf("[MilvusStore:Init] Connection pool initialized\n")

	// Initialize collection
	fmt.Printf("[MilvusStore:Init] Ensuring collection exists\n")
	ctx := context.Background()
	if err := store.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	fmt.Printf("[MilvusStore:Init] Collection initialization complete\n")

	// Start background flush task
	fmt.Printf("[MilvusStore:Init] Starting background tasks\n")
	store.startBackgroundFlush(context.Background())

	// Start metrics logging
	go store.logPoolMetrics()

	// Warm up the cache with recent items
	if err := store.warmCache(ctx); err != nil {
		// Continue even if cache warmup fails
		store.logger.Warn().Err(err).Msg("Cache warmup failed")
	}

	// Start index performance monitoring
	go store.monitorIndexPerformance()
	fmt.Printf("[MilvusStore:Init] Background tasks started\n")

	fmt.Printf("[MilvusStore:Init] Store initialization completed successfully\n")
	return store, nil
}

// initializePool initializes the connection pool with the specified size
func (s *MilvusStore) initializePool(size int) error {
	fmt.Printf("[MilvusStore:Pool] Starting pool initialization with size %d\n", size)

	// Create initial connections
	for i := 0; i < size; i++ {
		fmt.Printf("[MilvusStore:Pool] Creating connection %d/%d\n", i+1, size)
		conn, err := s.createConnection()
		if err != nil {
			fmt.Printf("[MilvusStore:Pool] Failed to create connection %d: %v\n", i+1, err)
			return fmt.Errorf("failed to create initial connection %d: %w", i+1, err)
		}
		fmt.Printf("[MilvusStore:Pool] Successfully created connection %d\n", i+1)

		pc := &pooledConnection{
			client:   conn,
			lastUsed: time.Now(),
		}

		// Start health check goroutine for each connection
		fmt.Printf("[MilvusStore:Pool] Starting health check for connection %d\n", i+1)
		ctx, cancel := context.WithCancel(context.Background())
		pc.healthCtx = cancel
		go s.connectionHealthCheck(ctx, pc)

		fmt.Printf("[MilvusStore:Pool] Adding connection %d to pool\n", i+1)
		s.pool.connections <- pc
		fmt.Printf("[MilvusStore:Pool] Successfully added connection %d to pool\n", i+1)
	}

	fmt.Printf("[MilvusStore:Pool] All initial connections created, starting connection manager\n")
	// Start connection manager
	go s.manageConnections()
	fmt.Printf("[MilvusStore:Pool] Pool initialization completed\n")

	return nil
}

// createConnection creates a new connection to Milvus
func (s *MilvusStore) createConnection() (client.Client, error) {
	addr := fmt.Sprintf("%s:%d", s.host, s.port)
	fmt.Printf("[MilvusStore:Connection] Attempting to connect to %s\n", addr)
	conn, err := client.NewGrpcClient(context.Background(), addr)
	if err != nil {
		fmt.Printf("[MilvusStore:Connection] Failed to create connection: %v\n", err)
		return nil, err
	}
	fmt.Printf("[MilvusStore:Connection] Successfully created connection\n")
	return conn, nil
}

// connectionHealthCheck monitors the health of a connection
func (s *MilvusStore) connectionHealthCheck(ctx context.Context, pc *pooledConnection) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !pc.inUse {
				if err := s.checkConnection(ctx, pc.client); err != nil {
					pc.failures++
					if pc.failures > 3 {
						// Connection is unhealthy, replace it
						atomic.AddInt64(&s.pool.metrics.failed, 1)
						if newConn, err := s.createConnection(); err == nil {
							pc.client.Close()
							pc.client = newConn
							pc.failures = 0
							atomic.AddInt64(&s.pool.metrics.reconnects, 1)
						}
					}
				} else {
					pc.failures = 0
				}
			}
		}
	}
}

func (s *MilvusStore) manageConnections() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		inUse := atomic.LoadInt64(&s.pool.metrics.inUse)
		if inUse < int64(s.pool.maxSize/2) {
			// Pool is underutilized, we could scale down
			continue
		}

		// Pool is heavily utilized, we could scale up
		if inUse > int64(s.pool.maxSize*8/10) {
			s.scalePool(true)
		}
	}
}

func (s *MilvusStore) scalePool(up bool) {
	if up {
		// Increase pool size by 25% up to the maximum
		newSize := int(float64(s.pool.maxSize) * 1.25)
		if newSize > maxPoolSize {
			return
		}

		// Create new connections
		for i := s.pool.maxSize; i < newSize; i++ {
			if conn, err := s.createConnection(); err == nil {
				pc := &pooledConnection{
					client:   conn,
					lastUsed: time.Now(),
				}
				ctx, cancel := context.WithCancel(context.Background())
				pc.healthCtx = cancel
				go s.connectionHealthCheck(ctx, pc)

				// Try to add to pool with timeout
				select {
				case s.pool.connections <- pc:
					// Successfully added
				case <-time.After(time.Second):
					// Pool is full, close the connection
					pc.healthCtx()
					pc.client.Close()
					return
				}
			}
		}
		s.pool.maxSize = newSize
	} else {
		// Decrease pool size by 25% down to a minimum
		newSize := int(float64(s.pool.maxSize) * 0.75)
		if newSize < maxConnections {
			return
		}
		s.pool.maxSize = newSize

		// Close excess connections
		for i := 0; i < (s.pool.maxSize - newSize); i++ {
			select {
			case pc := <-s.pool.connections:
				if pc.healthCtx != nil {
					pc.healthCtx()
				}
				pc.client.Close()
			default:
				// No more connections to close
				return
			}
		}
	}
}

// getConnection gets a connection from the pool with improved handling
func (s *MilvusStore) getConnection() (client.Client, error) {
	start := time.Now()
	atomic.AddInt64(&s.pool.metrics.waits, 1)

	// Try to get connection with adaptive timeout
	timeout := time.Duration(float64(s.timeout) * 0.1) // Start with 10% of operation timeout
	maxTimeout := s.timeout / 2                        // Max 50% of operation timeout

	for {
		select {
		case pc := <-s.pool.connections:
			atomic.AddInt64(&s.pool.metrics.inUse, 1)
			pc.inUse = true
			pc.lastUsed = time.Now()

			waitTime := time.Since(start)
			atomic.AddInt64(&s.pool.metrics.waitTime, int64(waitTime))

			// Only check health if connection hasn't been used recently
			if time.Since(pc.lastUsed) > 30*time.Second {
				if err := s.checkConnection(context.Background(), pc.client); err != nil {
					pc.failures++
					if pc.failures > 3 {
						atomic.AddInt64(&s.pool.metrics.failed, 1)
						if newConn, err := s.createConnection(); err == nil {
							pc.client.Close()
							pc.client = newConn
							pc.failures = 0
							atomic.AddInt64(&s.pool.metrics.reconnects, 1)
						}
					}
				} else {
					pc.failures = 0
				}
			}

			return pc.client, nil

		case <-time.After(timeout):
			// Exponential backoff with max timeout
			timeout *= 2
			if timeout > maxTimeout {
				timeout = maxTimeout
			}

			// Check if we need to create new connections
			currentConns := atomic.LoadInt32(&s.pool.connectedConns)
			if currentConns < int32(s.pool.maxSize) {
				if atomic.CompareAndSwapInt32(&s.pool.connectedConns, currentConns, currentConns+1) {
					// Create new connection
					conn, err := s.createConnection()
					if err != nil {
						atomic.AddInt32(&s.pool.connectedConns, -1)
						continue
					}

					pc := &pooledConnection{
						client:   conn,
						lastUsed: time.Now(),
					}

					// Start health check goroutine
					ctx, cancel := context.WithCancel(context.Background())
					pc.healthCtx = cancel
					go s.connectionHealthCheck(ctx, pc)

					atomic.AddInt64(&s.pool.metrics.inUse, 1)
					return conn, nil
				}
			}
		}
	}
}

// releaseConnection returns a connection to the pool with improved handling
func (s *MilvusStore) releaseConnection(conn client.Client) {
	pc := &pooledConnection{
		client:   conn,
		lastUsed: time.Now(),
		inUse:    false,
	}

	// Try to return to pool with timeout
	select {
	case s.pool.connections <- pc:
		atomic.AddInt64(&s.pool.metrics.inUse, -1)
	case <-time.After(time.Second):
		// Pool is full, close the connection
		conn.Close()
		atomic.AddInt32(&s.pool.connectedConns, -1)
	}
}

func (s *MilvusStore) checkConnection(ctx context.Context, conn client.Client) error {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	_, err := conn.HasCollection(ctx, s.collection)
	return err
}

// withRetry executes an operation with retry logic
func (s *MilvusStore) withRetry(ctx context.Context, op func(context.Context) error) error {
	var lastErr error
	for attempt := 0; attempt < s.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during retry: %w", ctx.Err())
			case <-time.After(time.Second * time.Duration(attempt)):
				// Exponential backoff
			}
		}

		if err := op(ctx); err == nil {
			return nil
		} else {
			lastErr = err
			atomic.AddInt64(&s.metrics.failedOps, 1)
		}
	}
	return fmt.Errorf("operation failed after %d attempts: %w", s.maxRetries, lastErr)
}

func encodeMetadata(metadata map[string]interface{}) string {
	if metadata == nil {
		return "{}"
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		return "{}"
	}
	return string(data)
}

func decodeMetadata(data string) map[string]interface{} {
	var metadata map[string]interface{}
	if err := json.Unmarshal([]byte(data), &metadata); err != nil {
		return make(map[string]interface{})
	}

	// Convert float64 to int for integer values immediately
	converted := make(map[string]interface{}, len(metadata))
	for k, v := range metadata {
		if f, ok := v.(float64); ok && f == float64(int64(f)) {
			converted[k] = int(f)
		} else {
			converted[k] = v
		}
	}

	return converted
}

// verifyItem checks if an item was successfully inserted with retries
func (s *MilvusStore) verifyItem(ctx context.Context, id string, retries int) error {
	var lastErr error
	for i := 0; i < retries; i++ {
		// Exponential backoff with initial delay of 100ms
		if i > 0 {
			backoffDuration := time.Duration(100*(1<<uint(i-1))) * time.Millisecond
			select {
			case <-ctx.Done():
				return fmt.Errorf("context canceled during verification: %w", ctx.Err())
			case <-time.After(backoffDuration):
			}
		}

		conn, err := s.getConnection()
		if err != nil {
			lastErr = fmt.Errorf("failed to get connection for verification: %w", err)
			continue
		}

		expr := fmt.Sprintf("id == '%s'", id)
		result, err := conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
		s.releaseConnection(conn)

		if err != nil {
			lastErr = fmt.Errorf("failed to verify item %s: %w", id, err)
			continue
		}

		if len(result) > 0 {
			idCol, ok := result[0].(*entity.ColumnVarChar)
			if ok && len(idCol.Data()) > 0 {
				return nil // Item found
			}
		}

		lastErr = fmt.Errorf("item %s not found after insertion", id)
	}
	return lastErr
}

// processBatchEfficient processes a batch of items efficiently with verification
func (s *MilvusStore) processBatchEfficient(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	// Get a connection from the pool
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	s.logger.Debug().Int("batch_size", len(items)).Msg("Processing batch")

	// Pre-allocate slices with capacity
	batchSize := len(items)
	vectors := make([][]float32, batchSize)
	ids := make([]string, batchSize)
	contentTypes := make([]string, batchSize)
	contentData := make([]string, batchSize)
	metadata := make([]string, batchSize)
	createdAt := make([]int64, batchSize)
	expiresAt := make([]int64, batchSize)

	// Use vector buffer pool for vector data
	for i, item := range items {
		// Get buffer from pool
		vector := s.vectorPool.get(s.dimension)
		vector = append(vector, item.Vector...)
		vectors[i] = vector

		ids[i] = item.ID
		contentTypes[i] = string(item.Content.Type)
		contentData[i] = string(item.Content.Data)
		metadata[i] = encodeMetadata(item.Metadata)
		createdAt[i] = item.CreatedAt.UnixNano()
		if !item.ExpiresAt.IsZero() {
			expiresAt[i] = item.ExpiresAt.UnixNano()
		}
	}

	// Ensure vectors are returned to pool
	defer func() {
		for _, vec := range vectors {
			s.vectorPool.put(vec)
		}
	}()

	// Create column data
	columns := []entity.Column{
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnFloatVector("vector", s.dimension, vectors),
		entity.NewColumnVarChar("content_type", contentTypes),
		entity.NewColumnVarChar("content_data", contentData),
		entity.NewColumnVarChar("metadata", metadata),
		entity.NewColumnInt64("created_at", createdAt),
		entity.NewColumnInt64("expires_at", expiresAt),
	}

	// Insert data with retry and metric tracking
	insertStart := time.Now()
	err = s.withRetry(ctx, func(ctx context.Context) error {
		_, err := conn.Insert(ctx, s.collection, "", columns...)
		if err != nil {
			return fmt.Errorf("failed to insert data: %w", err)
		}

		// Flush to ensure data is persisted
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to flush data: %w", err)
		}

		// Load collection to ensure changes are reflected
		if err := conn.LoadCollection(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}

		// Wait for collection to be fully loaded
		timeout := time.After(10 * time.Second)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-timeout:
				return fmt.Errorf("timeout waiting for collection to be loaded")
			case <-ticker.C:
				loadState, err := conn.GetLoadState(ctx, s.collection, []string{})
				if err != nil {
					s.logger.Warn().Err(err).Msg("Failed to get load state")
					continue
				}
				if loadState == entity.LoadStateLoaded {
					s.logger.Debug().Msg("Collection loaded after insert")
					return nil
				}
			}
		}
	})

	if err != nil {
		atomic.AddInt64(&s.metrics.failedInserts, 1)
		return fmt.Errorf("failed to insert batch: %w", err)
	}

	insertDuration := time.Since(insertStart)
	atomic.AddInt64(&s.metrics.totalInsertTime, insertDuration.Nanoseconds())
	atomic.AddInt64(&s.metrics.totalInserts, int64(len(items)))

	// Verify a sample of items from the batch with increased retries and backoff
	verificationStart := time.Now()
	sampleSize := min(5, len(items)) // Verify up to 5 random items
	verifiedIDs := make(map[string]bool)

	for i := 0; i < sampleSize; i++ {
		idx := rand.Intn(len(items))
		id := items[idx].ID
		if verifiedIDs[id] {
			continue // Skip if already verified
		}
		verifiedIDs[id] = true

		if err := s.verifyItem(ctx, id, 5); err != nil { // Increased retries to 5
			s.logger.Error().
				Err(err).
				Str("item_id", id).
				Dur("verification_time", time.Since(verificationStart)).
				Msg("Batch verification failed")
			return fmt.Errorf("batch verification failed: %w", err)
		}
	}

	s.logger.Debug().
		Int("verified_items", len(verifiedIDs)).
		Dur("verification_time", time.Since(verificationStart)).
		Msg("Batch verification completed")

	return nil
}

// Background flush task
func (s *MilvusStore) startBackgroundFlush(ctx context.Context) {
	ticker := time.NewTicker(flushInterval)
	go func() {
		for {
			select {
			case <-ctx.Done():
				ticker.Stop()
				return
			case <-ticker.C:
				s.insertBuffer.Lock()
				if err := s.flushBuffer(ctx); err != nil {
					fmt.Printf("[Milvus:Flush] Background flush failed: %v\n", err)
				}
				s.insertBuffer.Unlock()
			}
		}
	}()
}

// GetByID retrieves an item by its ID
func (s *MilvusStore) GetByID(ctx context.Context, id string) (*storage.Item, error) {
	// Track access pattern (no lastAccessedID for first call)
	s.trackAccess(id, "")

	// Check local cache first
	if cached, ok := s.localCache.Get(id); ok {
		s.logger.Debug().Str("id", id).Msg("Cache hit from local cache")

		// Trigger prefetch of predicted next items in background
		s.prefetchPredicted(ctx, id)

		return cached.(*storage.Item), nil
	}

	s.logger.Debug().Str("id", id).Msg("Starting retrieval for ID")

	// Prefetch the item
	if err := s.prefetchItems(ctx, []string{id}); err != nil {
		return nil, fmt.Errorf("failed to prefetch item: %w", err)
	}

	// Get a connection from the pool
	conn, err := s.getConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Build the query
	expr := fmt.Sprintf("id == '%s'", id)
	outputFields := []string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"}

	// Execute the query
	s.logger.Debug().Str("expression", expr).Msg("Executing query")
	fmt.Printf("[Milvus:Get] Executing query with expression: %s\n", expr)
	queryResult, err := conn.Query(ctx, s.collection, []string{}, expr, outputFields)
	if err != nil {
		fmt.Printf("[Milvus:Get] Query failed: %v\n", err)
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	fmt.Printf("[Milvus:Get] Query returned %d columns\n", len(queryResult))

	// Process results
	item, err := s.processQueryResult(queryResult)
	if err != nil {
		fmt.Printf("[Milvus:Get] Failed to process query result: %v\n", err)
		return nil, fmt.Errorf("failed to process query result: %w", err)
	}
	if item == nil {
		fmt.Printf("[Milvus:Get] No item found for ID: %s\n", id)
	} else {
		fmt.Printf("[Milvus:Get] Successfully retrieved item with ID: %s\n", item.ID)
	}

	// If we found the item, cache it and trigger prefetch
	if item != nil {
		s.localCache.Add(id, item)
		s.prefetchPredicted(ctx, id)
	}

	return item, nil
}

// DeleteFromStore removes items by their IDs in batches
func (s *MilvusStore) DeleteFromStore(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	fmt.Printf("[Milvus:Delete] Starting deletion of %d items\n", len(ids))
	totalStart := time.Now()

	// Process deletions in parallel for large batches
	if len(ids) > s.batchSize {
		fmt.Printf("[Milvus:Delete] Processing in batches of %d\n", s.batchSize)
		chunks := (len(ids) + s.batchSize - 1) / s.batchSize
		errChan := make(chan error, chunks)

		for i := 0; i < len(ids); i += s.batchSize {
			end := i + s.batchSize
			if end > len(ids) {
				end = len(ids)
			}

			chunk := ids[i:end]
			batchNum := (i / s.batchSize) + 1
			fmt.Printf("[Milvus:Delete] Submitting batch %d/%d (%d items)\n",
				batchNum, chunks, len(chunk))

			go func(batch []string, batchNum int) {
				batchStart := time.Now()
				err := s.deleteChunk(ctx, batch)
				if err != nil {
					fmt.Printf("[Milvus:Delete] Batch %d failed after %v: %v\n",
						batchNum, time.Since(batchStart), err)
				} else {
					fmt.Printf("[Milvus:Delete] Batch %d completed in %v\n",
						batchNum, time.Since(batchStart))
				}
				errChan <- err
			}(chunk, batchNum)
		}

		// Wait for all deletions to complete
		var errs []error
		for i := 0; i < chunks; i++ {
			if err := <-errChan; err != nil {
				errs = append(errs, err)
			}
		}

		fmt.Printf("[Milvus:Delete] Total parallel deletion operation took %v\n",
			time.Since(totalStart))

		if len(errs) > 0 {
			return fmt.Errorf("batch deletion errors: %v", errs)
		}
		return nil
	}

	return s.deleteChunk(ctx, ids)
}

// deleteChunk processes a chunk of IDs for deletion
func (s *MilvusStore) deleteChunk(ctx context.Context, ids []string) error {
	return s.withRetry(ctx, func(ctx context.Context) error {
		// Get connection
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)

		// Build the expression for deletion
		idList := make([]string, len(ids))
		for i, id := range ids {
			idList[i] = fmt.Sprintf("'%s'", id)
		}
		expr := fmt.Sprintf("id in [%s]", strings.Join(idList, ","))

		// Verify items exist before deletion
		queryResult, err := conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
		if err != nil {
			return fmt.Errorf("pre-deletion verification query failed: %w", err)
		}

		// Delete the records with retry
		maxRetries := 3
		for i := 0; i < maxRetries; i++ {
			if err := conn.Delete(ctx, s.collection, "", expr); err != nil {
				if i == maxRetries-1 {
					return fmt.Errorf("failed to delete chunk after %d attempts: %w", maxRetries, err)
				}
				time.Sleep(100 * time.Millisecond)
				continue
			}
			break
		}

		// Flush to ensure deletion is persisted
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to flush after deletion: %w", err)
		}

		// Load collection to ensure changes are reflected in memory
		if err := conn.LoadCollection(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to load collection after deletion: %w", err)
		}

		// Wait for collection to be fully loaded with timeout
		timeout := time.After(10 * time.Second)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-timeout:
				return fmt.Errorf("timeout waiting for collection to be loaded")
			case <-ticker.C:
				loadState, err := conn.GetLoadState(ctx, s.collection, []string{})
				if err != nil {
					return fmt.Errorf("failed to get load state: %w", err)
				}
				if loadState == entity.LoadStateLoaded {
					goto verifyDeletion
				}
			}
		}

	verifyDeletion:
		// Verify items are deleted with retry
		maxVerifyRetries := 3
		for i := 0; i < maxVerifyRetries; i++ {
			queryResult, err = conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
			if err != nil {
				continue
			}
			if len(queryResult) == 0 || len(queryResult[0].(*entity.ColumnVarChar).Data()) == 0 {
				break
			}
			if i == maxVerifyRetries-1 {
				if idCol, ok := queryResult[0].(*entity.ColumnVarChar); ok {
					remainingIds := idCol.Data()
					return fmt.Errorf("deletion verification failed: %d items still present after %d retries", len(remainingIds), maxVerifyRetries)
				}
			}
			time.Sleep(100 * time.Millisecond)
		}

		// Remove from cache
		for _, id := range ids {
			s.localCache.Remove(id)
			s.localCache.Remove("search:" + id)
		}

		// Clear the entire cache to ensure no stale search results remain
		newCache, err := newShardedCache(s.poolSize * 100)
		if err != nil {
			s.logger.Error().Err(err).Msg("Failed to create new cache after deletion")
		} else {
			s.localCache = newCache
		}

		return nil
	})
}

// Update updates existing items
func (s *MilvusStore) Update(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	// Generate vectors for items that don't have them
	if err := s.generateVectors(ctx, items); err != nil {
		return fmt.Errorf("failed to generate vectors: %w", err)
	}

	// Validate vector dimensions
	for _, item := range items {
		if len(item.Vector) != s.dimension {
			return fmt.Errorf("invalid vector dimension: got %d, want %d", len(item.Vector), s.dimension)
		}
	}

	// Delete existing items
	ids := make([]string, len(items))
	for i, item := range items {
		ids[i] = item.ID
	}
	if err := s.DeleteFromStore(ctx, ids); err != nil {
		return fmt.Errorf("failed to delete existing items: %w", err)
	}

	// Insert updated items
	return s.Insert(ctx, items)
}

// Health checks the health of the Milvus connection
func (s *MilvusStore) Health(ctx context.Context) error {
	return s.withRetry(ctx, func(ctx context.Context) error {
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)

		exists, err := conn.HasCollection(ctx, s.collection)
		if err != nil {
			return fmt.Errorf("failed to check collection: %w", err)
		}
		if !exists {
			return fmt.Errorf("collection %s does not exist", s.collection)
		}
		return nil
	})
}

// ensureCollection ensures the collection exists with the correct schema
func (s *MilvusStore) ensureCollection(ctx context.Context) error {
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	s.logger.Debug().Msg("Checking if collection exists")
	exists, err := conn.HasCollection(ctx, s.collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if exists {
		s.logger.Debug().Msg("Collection exists, checking metric type")
		// Get index info to check metric type
		indexes, err := conn.DescribeIndex(ctx, s.collectionName, "vector")
		if err != nil {
			return fmt.Errorf("failed to describe index: %w", err)
		}

		// Check if we need to recreate the collection
		needsRecreation := true
		if len(indexes) > 0 {
			params := indexes[0].Params()
			if params != nil {
				if metricType, ok := params["metric_type"]; ok && metricType == "IP" {
					needsRecreation = false
				}
			}
		}

		if needsRecreation {
			s.logger.Info().Msg("Collection has wrong metric type, dropping and recreating")
			err = conn.DropCollection(ctx, s.collectionName)
			if err != nil {
				return fmt.Errorf("failed to drop collection: %w", err)
			}
			exists = false
		}
	}

	if !exists {
		s.logger.Debug().Msg("Creating collection")
		schema := &entity.Schema{
			CollectionName: s.collectionName,
			Description:    "Vector store collection",
			Fields: []*entity.Field{
				{
					Name:       "id",
					DataType:   entity.FieldTypeVarChar,
					PrimaryKey: true,
					TypeParams: map[string]string{
						"max_length": "100",
					},
				},
				{
					Name:     "vector",
					DataType: entity.FieldTypeFloatVector,
					TypeParams: map[string]string{
						"dim": strconv.Itoa(s.dimension),
					},
				},
				{
					Name:     "content_type",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "50",
					},
				},
				{
					Name:     "content_data",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "65535",
					},
				},
				{
					Name:     "metadata",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "65535",
					},
				},
				{
					Name:     "created_at",
					DataType: entity.FieldTypeInt64,
				},
				{
					Name:     "expires_at",
					DataType: entity.FieldTypeInt64,
				},
			},
		}

		err = conn.CreateCollection(ctx, schema, 2) // Use 2 shards for better performance
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		s.logger.Debug().Msg("Collection created successfully")

		// Create index with improved parameters for IP (Inner Product) metric
		s.logger.Debug().Msg("Creating index")
		idx, err := entity.NewIndexIvfFlat(entity.IP, 1024) // Changed to IP metric
		if err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}

		err = conn.CreateIndex(ctx, s.collectionName, "vector", idx, false)
		if err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
		s.logger.Debug().Msg("Index created successfully")

		// Load collection
		err = conn.LoadCollection(ctx, s.collectionName, false)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}

		// Wait for collection to be fully loaded
		timeout := time.After(30 * time.Second)
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-timeout:
				return fmt.Errorf("timeout waiting for collection to be loaded")
			case <-ticker.C:
				loadState, err := conn.GetLoadState(ctx, s.collectionName, []string{})
				if err != nil {
					s.logger.Warn().Err(err).Msg("Failed to get load state")
					continue
				}
				if loadState == entity.LoadStateLoaded {
					s.logger.Debug().Msg("Collection loaded successfully")
					return nil
				}
			}
		}
	}

	return nil
}

// searchResult represents a single search result with its similarity to the query vector
type searchResult struct {
	item       *storage.Item
	similarity float32 // Using similarity instead of distance
}

// resultHeap implements a max-heap of search results for efficient top-k retrieval
type resultHeap struct {
	results []searchResult
	limit   int
}

func (h *resultHeap) init(limit int) {
	h.limit = limit
	h.results = make([]searchResult, 0, limit)
	heap.Init(h)
}

func (h *resultHeap) Len() int { return len(h.results) }

func (h *resultHeap) Less(i, j int) bool {
	// Higher similarity is better
	return h.results[i].similarity < h.results[j].similarity
}

func (h *resultHeap) Swap(i, j int) {
	h.results[i], h.results[j] = h.results[j], h.results[i]
}

func (h *resultHeap) Push(x interface{}) {
	h.results = append(h.results, x.(searchResult))
}

func (h *resultHeap) Pop() interface{} {
	old := h.results
	n := len(old)
	x := old[n-1]
	h.results = old[0 : n-1]
	return x
}

func (h *resultHeap) worst() searchResult {
	if h.Len() == 0 {
		return searchResult{similarity: -1}
	}
	return h.results[0]
}

func (h *resultHeap) replace(result searchResult) {
	if result.similarity > h.results[0].similarity {
		h.results[0] = result
		heap.Fix(h, 0)
	}
}

// parallelSearch performs parallel vector search
func (s *MilvusStore) parallelSearch(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Split search space into segments for parallel processing
	nodes := s.graph.GetNodes()
	if len(nodes) == 0 {
		s.logger.Debug().Msg("No nodes available for parallel search")
		return nil, nil
	}

	s.logger.Debug().
		Int("total_nodes", len(nodes)).
		Int("cpu_count", runtime.NumCPU()).
		Msg("Starting parallel search")

	segmentSize := (len(nodes) + runtime.NumCPU() - 1) / runtime.NumCPU()
	segments := make([][]float32, 0, runtime.NumCPU())

	// Create segments of vectors
	vectorBuffer := s.getVectorBuffer(len(nodes))
	defer s.putVectorBuffer(vectorBuffer)

	offset := 0
	for i := 0; i < len(nodes); i += segmentSize {
		end := i + segmentSize
		if end > len(nodes) {
			end = len(nodes)
		}

		// Copy vectors to buffer
		segStart := offset
		for _, node := range nodes[i:end] {
			copy(vectorBuffer[offset:], node.Vector)
			offset += len(node.Vector)
		}
		segments = append(segments, vectorBuffer[segStart:offset])

		s.logger.Debug().
			Int("segment_index", len(segments)-1).
			Int("segment_size", end-i).
			Msg("Created search segment")
	}

	// Channel for collecting results with buffer to prevent blocking
	results := make(chan searchResult, limit*2)
	done := make(chan struct{})

	// Start parallel search
	var wg sync.WaitGroup
	for i, segment := range segments {
		wg.Add(1)
		go func(segmentID int, segmentVectors []float32) {
			defer wg.Done()

			searchStart := time.Now()
			// Process segment using inner product similarity
			similarities := make([]float32, len(segmentVectors)/s.dimension)
			for j := 0; j < len(segmentVectors); j += s.dimension {
				similarities[j/s.dimension] = computeInnerProduct(vector, segmentVectors[j:j+s.dimension])
			}

			s.logger.Debug().
				Int("segment_id", segmentID).
				Int("similarities_computed", len(similarities)).
				Dur("computation_time", time.Since(searchStart)).
				Msg("Processed search segment")

			// Send results through channel
			for j, sim := range similarities {
				select {
				case <-done:
					return // Early stopping
				case results <- searchResult{
					item:       &storage.Item{Vector: segmentVectors[j*s.dimension : (j+1)*s.dimension]},
					similarity: sim,
				}:
				}
			}
		}(i, segment)
	}

	// Start result collector
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect and merge results with early stopping
	resultHeap := &resultHeap{}
	resultHeap.init(limit)

	count := 0
	for searchResult := range results {
		if count < limit {
			heap.Push(resultHeap, searchResult)
			count++
			continue
		}

		// Check if this result is better than our worst result
		worst := resultHeap.worst()
		if searchResult.similarity > worst.similarity {
			resultHeap.replace(searchResult)

			// If we have enough good results, stop other goroutines
			if count >= limit*2 {
				close(done)
				break
			}
		}
		count++
	}

	s.logger.Debug().
		Int("total_results_processed", count).
		Int("final_results", resultHeap.Len()).
		Msg("Parallel search completed")

	// Convert heap to sorted slice
	items := make([]*storage.Item, 0, resultHeap.Len())
	for resultHeap.Len() > 0 {
		result := heap.Pop(resultHeap).(searchResult)
		items = append(items, result.item)
	}

	// Reverse slice to get descending order of similarity
	for i := 0; i < len(items)/2; i++ {
		j := len(items) - i - 1
		items[i], items[j] = items[j], items[i]
	}

	return items, nil
}

// Search performs a vector similarity search
func (s *MilvusStore) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	s.logger.Debug().
		Int("vector_dim", len(vector)).
		Int("limit", limit).
		Msg("Starting vector search")

	// Normalize the query vector
	queryVector := s.normalizeVector(vector)
	s.logger.Debug().
		Float32("query_vector_norm", computeNorm(queryVector)).
		Msg("Normalized query vector")

	// Select search strategy based on dataset size and result set size
	strategy := s.selectSearchStrategy(limit)
	s.logger.Debug().
		Str("strategy", string(strategy)).
		Int("limit", limit).
		Msg("Selected search strategy")

	var results []*storage.Item
	var err error

	switch strategy {
	case SearchStrategyStandard:
		results, err = s.standardSearch(ctx, queryVector, limit)
	case SearchStrategyParallel:
		results, err = s.parallelSearch(ctx, queryVector, limit)
	case SearchStrategyLSH:
		results, err = s.lshSearch(ctx, queryVector, limit)
	case SearchStrategyQuantization:
		results, err = s.quantizationSearch(ctx, queryVector, limit)
	default:
		results, err = s.standardSearch(ctx, queryVector, limit)
	}

	if err != nil {
		return nil, fmt.Errorf("search failed: %w", err)
	}

	s.logger.Debug().
		Str("strategy", string(strategy)).
		Int("results_found", len(results)).
		Msg("Search completed")

	return results, nil
}

// selectSearchStrategy selects the appropriate search strategy based on conditions
func (s *MilvusStore) selectSearchStrategy(limit int) SearchStrategy {
	// Use metrics to determine dataset size
	totalInserts := atomic.LoadInt64(&s.metrics.totalInserts)

	switch {
	case totalInserts > 1_000_000:
		// For very large datasets, use LSH
		return SearchStrategyLSH
	case totalInserts > 100_000:
		// For large datasets, use quantization
		return SearchStrategyQuantization
	case limit > 100:
		// For large result sets, use parallel search
		return SearchStrategyParallel
	default:
		// For small datasets and result sets, use standard search
		return SearchStrategyStandard
	}
}

// String returns the string representation of the search strategy
func (s SearchStrategy) String() string {
	switch s {
	case SearchStrategyParallel:
		return "parallel"
	case SearchStrategyLSH:
		return "lsh"
	case SearchStrategyQuantization:
		return "quantization"
	default:
		return "standard"
	}
}

// standardSearch is the fallback search method using Milvus directly
func (s *MilvusStore) standardSearch(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	searchStart := time.Now()

	// Normalize the query vector for cosine similarity
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		normalizedVector := make([]float32, len(vector))
		for i, v := range vector {
			normalizedVector[i] = v / norm
		}
		vector = normalizedVector
	}

	s.logger.Debug().
		Float64("query_vector_norm", float64(norm)).
		Msg("Normalized query vector")

	conn, err := s.getConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	s.logger.Debug().Msg("Starting standard search")

	// Ensure collection is loaded
	loadStart := time.Now()
	err = conn.LoadCollection(ctx, s.collection, false)
	if err != nil {
		s.logger.Error().
			Err(err).
			Msg("Failed to load collection")
		return nil, fmt.Errorf("failed to load collection: %w", err)
	}

	s.logger.Debug().
		Dur("load_time", time.Since(loadStart)).
		Msg("Collection loaded")

	// Wait for collection to be fully loaded
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return nil, fmt.Errorf("timeout waiting for collection to be loaded")
		case <-ticker.C:
			loadState, err := conn.GetLoadState(ctx, s.collection, []string{})
			if err != nil {
				s.logger.Warn().
					Err(err).
					Msg("Failed to get load state")
				continue
			}
			if loadState == entity.LoadStateLoaded {
				goto startSearch
			}
		}
	}

startSearch:
	s.logger.Debug().
		Dur("preparation_time", time.Since(searchStart)).
		Msg("Starting Milvus search")

	// Use optimized search parameters for better accuracy
	sp, err := entity.NewIndexIvfFlatSearchParam(64) // Increased nprobe for better recall
	if err != nil {
		s.logger.Error().
			Err(err).
			Msg("Failed to create search parameters")
		return nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	// Execute search with retry
	var searchResults []client.SearchResult
	err = s.withRetry(ctx, func(ctx context.Context) error {
		queryStart := time.Now()
		results, err := conn.Search(
			ctx,
			s.collection,
			[]string{}, // No partition
			"",         // No expression
			[]string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"},
			[]entity.Vector{entity.FloatVector(vector)},
			"vector",
			entity.IP, // Use Inner Product for cosine similarity with normalized vectors
			limit,
			sp,
		)
		if err != nil {
			s.logger.Error().
				Err(err).
				Dur("query_time", time.Since(queryStart)).
				Msg("Search operation failed")
			return fmt.Errorf("search operation failed: %w", err)
		}
		searchResults = results

		s.logger.Debug().
			Int("results_count", len(results)).
			Dur("query_time", time.Since(queryStart)).
			Msg("Search query completed")
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to execute search: %w", err)
	}

	// Process results
	items, err := s.processSearchResults(searchResults)
	if err != nil {
		s.logger.Error().
			Err(err).
			Msg("Failed to process search results")
		return nil, err
	}

	s.logger.Debug().
		Int("final_results", len(items)).
		Dur("total_time", time.Since(searchStart)).
		Msg("Standard search completed")

	return items, nil
}

// Add helper function to extract vectors from items
func vectorsFromItems(items []*storage.Item) [][]float32 {
	vectors := make([][]float32, len(items))
	for i, item := range items {
		vectors[i] = item.Vector
	}
	return vectors
}

// Add helper function to get item by ID
func (s *MilvusStore) getItemByID(ctx context.Context, id string) (*storage.Item, error) {
	// First check cache
	if cached, ok := s.localCache.Get(id); ok {
		if item, ok := cached.(*storage.Item); ok {
			return item, nil
		}
	}

	// Query Milvus for the item
	expr := fmt.Sprintf("id == '%s'", id)
	items, err := s.queryItems(ctx, expr, 1)
	if err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, nil
	}

	// Cache the result
	s.localCache.Add(id, items[0])
	return items[0], nil
}

// processSearchResults processes search results from Milvus
func (s *MilvusStore) processSearchResults(results []client.SearchResult) ([]*storage.Item, error) {
	if len(results) == 0 {
		return nil, nil
	}

	s.logger.Debug().
		Int("num_results", len(results)).
		Msg("Processing search results")

	// Get the first result (we only search with one vector)
	result := results[0]
	if result.ResultCount == 0 {
		return nil, nil
	}

	s.logger.Debug().
		Int("result_count", result.ResultCount).
		Int("num_fields", len(result.Fields)).
		Msg("Processing result fields")

	// Find fields by name
	var idCol *entity.ColumnVarChar
	var vectorCol *entity.ColumnFloatVector
	var contentTypeCol *entity.ColumnVarChar
	var contentDataCol *entity.ColumnVarChar
	var metadataCol *entity.ColumnVarChar
	var createdAtCol *entity.ColumnInt64
	var expiresAtCol *entity.ColumnInt64

	for _, field := range result.Fields {
		switch field.Name() {
		case "id":
			var ok bool
			idCol, ok = field.(*entity.ColumnVarChar)
			if !ok {
				return nil, fmt.Errorf("invalid id column type: %T", field)
			}
		case "vector":
			var ok bool
			vectorCol, ok = field.(*entity.ColumnFloatVector)
			if !ok {
				return nil, fmt.Errorf("invalid vector column type: %T", field)
			}
		case "content_type":
			var ok bool
			contentTypeCol, ok = field.(*entity.ColumnVarChar)
			if !ok {
				return nil, fmt.Errorf("invalid content_type column type: %T", field)
			}
		case "content_data":
			var ok bool
			contentDataCol, ok = field.(*entity.ColumnVarChar)
			if !ok {
				return nil, fmt.Errorf("invalid content_data column type: %T", field)
			}
		case "metadata":
			var ok bool
			metadataCol, ok = field.(*entity.ColumnVarChar)
			if !ok {
				return nil, fmt.Errorf("invalid metadata column type: %T", field)
			}
		case "created_at":
			var ok bool
			createdAtCol, ok = field.(*entity.ColumnInt64)
			if !ok {
				return nil, fmt.Errorf("invalid created_at column type: %T", field)
			}
		case "expires_at":
			var ok bool
			expiresAtCol, ok = field.(*entity.ColumnInt64)
			if !ok {
				return nil, fmt.Errorf("invalid expires_at column type: %T", field)
			}
		}
	}

	// Verify all required fields were found
	if idCol == nil {
		return nil, fmt.Errorf("id column not found in search results")
	}
	if vectorCol == nil {
		return nil, fmt.Errorf("vector column not found in search results")
	}
	if contentTypeCol == nil {
		return nil, fmt.Errorf("content_type column not found in search results")
	}
	if contentDataCol == nil {
		return nil, fmt.Errorf("content_data column not found in search results")
	}
	if metadataCol == nil {
		return nil, fmt.Errorf("metadata column not found in search results")
	}
	if createdAtCol == nil {
		return nil, fmt.Errorf("created_at column not found in search results")
	}
	if expiresAtCol == nil {
		return nil, fmt.Errorf("expires_at column not found in search results")
	}

	// Process each result
	items := make([]*storage.Item, 0, result.ResultCount)
	for i := 0; i < result.ResultCount; i++ {
		// Get vector data for current row
		vector := vectorCol.Data()[i]
		if len(vector) != s.dimension {
			s.logger.Warn().
				Int("expected_dim", s.dimension).
				Int("actual_dim", len(vector)).
				Msg("Invalid vector dimension")
			continue
		}

		// Parse metadata JSON
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataCol.Data()[i]), &metadata); err != nil {
			s.logger.Warn().
				Err(err).
				Str("item_id", idCol.Data()[i]).
				Msg("Failed to parse metadata")
			continue
		}

		// Create content
		content := storage.Content{
			Type: storage.ContentType(contentTypeCol.Data()[i]),
			Data: []byte(contentDataCol.Data()[i]),
		}

		item := &storage.Item{
			ID:        idCol.Data()[i],
			Vector:    vector,
			Content:   content,
			Metadata:  metadata,
			CreatedAt: time.Unix(0, createdAtCol.Data()[i]),
			ExpiresAt: time.Unix(0, expiresAtCol.Data()[i]),
		}
		items = append(items, item)

		s.logger.Debug().
			Str("item_id", item.ID).
			Int("vector_dim", len(item.Vector)).
			Msg("Processed search result item")
	}

	s.logger.Debug().
		Int("processed_items", len(items)).
		Msg("Completed processing search results")

	return items, nil
}

// isValidContentType validates the content type
func isValidContentType(ct storage.ContentType) bool {
	switch ct {
	case storage.ContentTypeText,
		storage.ContentTypeCode,
		storage.ContentTypeMath,
		storage.ContentTypeJSON,
		storage.ContentTypeMarkdown,
		"": // Allow empty content type for search results
		return true
	default:
		return false
	}
}

// processQueryResult converts Milvus query result fields to a storage item
func (s *MilvusStore) processQueryResult(queryResult []entity.Column) (*storage.Item, error) {
	if len(queryResult) == 0 {
		return nil, nil
	}

	// Check if any rows were returned by checking the length of any column
	hasRows := false
	for _, col := range queryResult {
		switch c := col.(type) {
		case *entity.ColumnVarChar:
			if len(c.Data()) > 0 {
				hasRows = true
				break
			}
		case *entity.ColumnFloatVector:
			if len(c.Data()) > 0 {
				hasRows = true
				break
			}
		case *entity.ColumnInt64:
			if len(c.Data()) > 0 {
				hasRows = true
				break
			}
		}
		if hasRows {
			break
		}
	}

	if !hasRows {
		return nil, nil
	}

	// Extract values from the first row
	var item storage.Item

	// Process each column
	for _, col := range queryResult {
		switch col.Name() {
		case "id":
			if idCol, ok := col.(*entity.ColumnVarChar); ok && len(idCol.Data()) > 0 {
				item.ID = idCol.Data()[0]
			} else {
				return nil, fmt.Errorf("invalid id column type or empty: %T", col)
			}
		case "vector":
			if vecCol, ok := col.(*entity.ColumnFloatVector); ok && len(vecCol.Data()) > 0 {
				item.Vector = vecCol.Data()[0]
			} else {
				return nil, fmt.Errorf("invalid vector column type or empty: %T", col)
			}
		case "content_type":
			if typeCol, ok := col.(*entity.ColumnVarChar); ok && len(typeCol.Data()) > 0 {
				item.Content.Type = storage.ContentType(typeCol.Data()[0])
			} else {
				return nil, fmt.Errorf("invalid content_type column type or empty: %T", col)
			}
		case "content_data":
			if dataCol, ok := col.(*entity.ColumnVarChar); ok && len(dataCol.Data()) > 0 {
				item.Content.Data = []byte(dataCol.Data()[0])
			} else {
				return nil, fmt.Errorf("invalid content_data column type or empty: %T", col)
			}
		case "metadata":
			if metaCol, ok := col.(*entity.ColumnVarChar); ok && len(metaCol.Data()) > 0 {
				if err := json.Unmarshal([]byte(metaCol.Data()[0]), &item.Metadata); err != nil {
					return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
				}
			} else {
				return nil, fmt.Errorf("invalid metadata column type or empty: %T", col)
			}
		case "created_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.CreatedAt = time.Unix(0, timeCol.Data()[0])
			} else {
				return nil, fmt.Errorf("invalid created_at column type or empty: %T", col)
			}
		case "expires_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.ExpiresAt = time.Unix(0, timeCol.Data()[0])
			} else {
				return nil, fmt.Errorf("invalid expires_at column type or empty: %T", col)
			}
		}
	}

	return &item, nil
}

// Close properly shuts down all storage connections
func (s *MilvusStore) Close() error {
	if s.simdProc != nil {
		s.simdProc.Close()
	}
	if s.graph != nil {
		s.graph.Close()
	}
	if s.quantizer != nil {
		s.quantizer.Close()
	}
	if s.optimisticSearch != nil {
		s.optimisticSearch.Close()
	}
	// Close work queue first
	s.workQueue.Close()

	// Close all connections in the pool
	closeTimeout := time.After(5 * time.Second)
	closed := make(chan struct{})

	go func() {
		for i := 0; i < s.pool.maxSize; i++ {
			select {
			case pc := <-s.pool.connections:
				if pc.healthCtx != nil {
					pc.healthCtx()
				}
				if err := pc.client.Close(); err != nil {
					fmt.Printf("failed to close connection: %v\n", err)
				}
			default:
				// Pool is empty
				continue
			}
		}
		close(s.pool.connections)
		close(closed)
	}()

	// Wait for cleanup with timeout
	select {
	case <-closed:
		return nil
	case <-closeTimeout:
		return fmt.Errorf("timeout while closing connections")
	}
}

// processBatch handles a single batch of items
func (s *MilvusStore) processBatch(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	err := s.withRetry(ctx, func(ctx context.Context) error {
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)

		// Get vector buffer from pool with appropriate size
		vectorBuffer := s.vectorPool.get(s.dimension)
		defer s.vectorPool.put(vectorBuffer)

		// Prepare column data
		ids := make([]string, len(items))
		vectors := make([][]float32, len(items))
		contentTypes := make([]string, len(items))
		contentData := make([]string, len(items))
		metadata := make([]string, len(items))
		createdAt := make([]int64, len(items))
		expiresAt := make([]int64, len(items))

		for i, item := range items {
			ids[i] = item.ID
			vectors[i] = item.Vector
			contentTypes[i] = string(item.Content.Type) // Fix: Set the content type
			contentData[i] = string(item.Content.Data)
			metadata[i] = encodeMetadata(item.Metadata)
			createdAt[i] = item.CreatedAt.UnixNano()
			expiresAt[i] = item.ExpiresAt.UnixNano()
		}

		// Create columns
		columns := []entity.Column{
			entity.NewColumnVarChar("id", ids),
			entity.NewColumnFloatVector("vector", s.dimension, vectors),
			entity.NewColumnVarChar("content_type", contentTypes),
			entity.NewColumnVarChar("content_data", contentData),
			entity.NewColumnVarChar("metadata", metadata),
			entity.NewColumnInt64("created_at", createdAt),
			entity.NewColumnInt64("expires_at", expiresAt),
		}

		// Insert data
		_, err = conn.Insert(ctx, s.collection, "", columns...)
		if err != nil {
			return fmt.Errorf("failed to insert data: %w", err)
		}

		// Update metrics
		atomic.AddInt64(&s.metrics.totalInserts, int64(len(items)))
		atomic.AddInt64(&s.metrics.totalInsertTime, time.Since(time.Now()).Nanoseconds())

		return nil
	})

	return err
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// trackAccess records an item access and updates patterns
func (s *MilvusStore) trackAccess(id string, lastAccessedID string) {
	s.accessPatterns.Lock()
	defer s.accessPatterns.Unlock()

	now := time.Now()

	// Create or update pattern for current item
	pattern, exists := s.accessPatterns.patterns[id]
	if !exists {
		pattern = &AccessPattern{
			BeforeItems: make(map[string]int),
			AfterItems:  make(map[string]int),
		}
		s.accessPatterns.patterns[id] = pattern
	}

	pattern.LastAccessed = now
	pattern.AccessCount++

	// Update relationships if we have a last accessed item
	if lastAccessedID != "" {
		// Current item comes after lastAccessedID
		if lastPattern, ok := s.accessPatterns.patterns[lastAccessedID]; ok {
			lastPattern.AfterItems[id]++
			pattern.BeforeItems[lastAccessedID]++
		}

		// Update relationships
		s.accessPatterns.relationships[lastAccessedID] = append(
			s.accessPatterns.relationships[lastAccessedID], id)

		// Cleanup old patterns (keep last 24 hours)
		if pattern.AccessCount%100 == 0 { // Don't clean up on every access
			s.cleanupOldPatterns()
		}
	}
}

// cleanupOldPatterns removes patterns older than 24 hours
func (s *MilvusStore) cleanupOldPatterns() {
	s.accessPatterns.Lock()
	defer s.accessPatterns.Unlock()

	now := time.Now()
	for id, pattern := range s.accessPatterns.patterns {
		if now.Sub(pattern.LastAccessed) > patternExpiryDuration {
			delete(s.accessPatterns.patterns, id)
			delete(s.accessPatterns.relationships, id)
		}
	}
}

// predictNextItems predicts which items are likely to be accessed next
func (s *MilvusStore) predictNextItems(id string, limit int) []string {
	s.accessPatterns.RLock()
	defer s.accessPatterns.RUnlock()

	scores := make(map[string]float64)

	// Score based on items that frequently follow the current item
	if pattern, ok := s.accessPatterns.patterns[id]; ok {
		for afterID, count := range pattern.AfterItems {
			scores[afterID] += float64(count) * 0.6 // Weight for direct following
		}

		// Also consider items that follow items that are related to current item
		for beforeID := range pattern.BeforeItems {
			if beforePattern, ok := s.accessPatterns.patterns[beforeID]; ok {
				for afterID, count := range beforePattern.AfterItems {
					scores[afterID] += float64(count) * 0.3 // Lower weight for indirect relationships
				}
			}
		}
	}

	// Convert scores to sorted slice
	type scoredItem struct {
		id    string
		score float64
	}
	var items []scoredItem
	for id, score := range scores {
		items = append(items, scoredItem{id, score})
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].score > items[j].score
	})

	// Return top N predicted items
	result := make([]string, 0, limit)
	for i := 0; i < len(items) && i < limit; i++ {
		result = append(result, items[i].id)
	}
	return result
}

// prefetchPredicted prefetches predicted items into cache
func (s *MilvusStore) prefetchPredicted(ctx context.Context, id string) {
	predictedIDs := s.predictNextItems(id, 5) // Prefetch top 5 predicted items
	if len(predictedIDs) == 0 {
		return
	}

	// Prefetch in background to not block current request
	go func() {
		// Create new context with timeout, derived from parent context
		prefetchCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()

		if err := s.prefetchItems(prefetchCtx, predictedIDs); err != nil {
			s.logger.Warn().
				Err(err).
				Str("trigger_id", id).
				Msg("Failed to prefetch predicted items")
		}
	}()
}

// generateVectors generates embeddings for the given items using the embedding service
func (s *MilvusStore) generateVectors(ctx context.Context, items []*storage.Item) error {
	// Group items by content type to use appropriate embedding models
	textItems := make(map[int]string)     // index -> text
	codeItems := make(map[int]string)     // index -> code
	markdownItems := make(map[int]string) // index -> markdown

	for i, item := range items {
		if item.Vector != nil && len(item.Vector) > 0 {
			continue // Skip if vector already exists
		}

		switch item.Content.Type {
		case storage.ContentTypeText, storage.ContentTypeJSON:
			textItems[i] = string(item.Content.Data)
		case storage.ContentTypeCode:
			codeItems[i] = string(item.Content.Data)
		case storage.ContentTypeMath:
			// For math content, we might want to use a specialized model or preprocess differently
			textItems[i] = string(item.Content.Data)
		case storage.ContentTypeMarkdown:
			markdownItems[i] = string(item.Content.Data)
		default:
			return fmt.Errorf("unsupported content type: %s", item.Content.Type)
		}
	}

	// Process text items
	if len(textItems) > 0 {
		texts := make([]string, 0, len(textItems))
		indices := make([]int, 0, len(textItems))
		for idx, text := range textItems {
			texts = append(texts, text)
			indices = append(indices, idx)
		}

		results, err := s.embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return fmt.Errorf("failed to generate text embeddings: %w", err)
		}

		for i, result := range results {
			items[indices[i]].Vector = result.Vector
		}
	}

	// Process code items (you might want to use a different model or preprocessing)
	if len(codeItems) > 0 {
		texts := make([]string, 0, len(codeItems))
		indices := make([]int, 0, len(codeItems))
		for idx, code := range codeItems {
			texts = append(texts, code)
			indices = append(indices, idx)
		}

		results, err := s.embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return fmt.Errorf("failed to generate code embeddings: %w", err)
		}

		for i, result := range results {
			items[indices[i]].Vector = result.Vector
		}
	}

	// Process markdown items
	if len(markdownItems) > 0 {
		texts := make([]string, 0, len(markdownItems))
		indices := make([]int, 0, len(markdownItems))
		for idx, md := range markdownItems {
			texts = append(texts, md)
			indices = append(indices, idx)
		}

		results, err := s.embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return fmt.Errorf("failed to generate markdown embeddings: %w", err)
		}

		for i, result := range results {
			items[indices[i]].Vector = result.Vector
		}
	}

	return nil
}

// normalizeVector normalizes a vector to unit length
func (s *MilvusStore) normalizeVector(vector []float32) []float32 {
	// Calculate the magnitude
	var sum float32
	for _, v := range vector {
		sum += v * v
	}
	magnitude := float32(math.Sqrt(float64(sum)))

	// Normalize the vector
	normalized := make([]float32, len(vector))
	if magnitude > 0 {
		for i, v := range vector {
			normalized[i] = v / magnitude
		}
	}
	return normalized
}

// Insert inserts items into the store
func (s *MilvusStore) Insert(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	// Generate vectors for items that don't have them
	if err := s.generateVectors(ctx, items); err != nil {
		return fmt.Errorf("failed to generate vectors: %w", err)
	}

	// Normalize vectors before insertion
	for _, item := range items {
		item.Vector = s.normalizeVector(item.Vector)
		s.logger.Debug().
			Str("item_id", item.ID).
			Float32("vector_norm", computeNorm(item.Vector)).
			Msg("Normalized vector for insertion")
	}

	// Process items in batches
	for i := 0; i < len(items); i += s.batchSize {
		end := i + s.batchSize
		if end > len(items) {
			end = len(items)
		}
		if err := s.processBatchEfficient(ctx, items[i:end]); err != nil {
			return fmt.Errorf("failed to process batch: %w", err)
		}
	}

	return nil
}

// flushBuffer processes all items in the buffer
func (s *MilvusStore) flushBuffer(ctx context.Context) error {
	if len(s.insertBuffer.items) == 0 {
		return nil
	}

	// Take all items from buffer
	items := s.insertBuffer.items
	s.insertBuffer.items = make([]*storage.Item, 0, maxBufferSize)
	s.insertBuffer.lastFlush = time.Now()

	// Process items in optimal batch sizes
	const optimalBatchSize = 2000
	for i := 0; i < len(items); i += optimalBatchSize {
		end := i + optimalBatchSize
		if end > len(items) {
			end = len(items)
		}
		batch := items[i:end]

		if err := s.processBatchEfficient(ctx, batch); err != nil {
			return fmt.Errorf("failed to process batch: %w", err)
		}
	}

	return nil
}

// Add connection pool metrics logging
func (s *MilvusStore) logPoolMetrics() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		waits := atomic.LoadInt64(&s.pool.metrics.waits)
		if waits > 0 {
			avgWaitTime := time.Duration(atomic.LoadInt64(&s.pool.metrics.waitTime) / waits)
			s.logger.Info().
				Int64("total_waits", waits).
				Dur("avg_wait_time", avgWaitTime).
				Int64("in_use", atomic.LoadInt64(&s.pool.metrics.inUse)).
				Int64("failed", atomic.LoadInt64(&s.pool.metrics.failed)).
				Int64("reconnects", atomic.LoadInt64(&s.pool.metrics.reconnects)).
				Int32("connected", atomic.LoadInt32(&s.pool.connectedConns)).
				Msg("Connection pool metrics")
		}
	}
}

// LockFreeVectorStore provides concurrent vector storage without locks
// using atomic operations and sharding for high-performance vector operations.
type LockFreeVectorStore struct {
	// dimension is the size of each vector
	dimension int
	// size tracks the number of vectors currently stored
	size atomic.Int64
	// capacity tracks the total storage capacity
	capacity atomic.Int64
	// chunks holds vector data in fixed-size chunks for better memory management
	// and concurrent access. Each chunk is protected by atomic operations.
	chunks []atomic.Value // Each chunk holds vectorChunkSize vectors
}

// NewLockFreeVectorStore creates a new lock-free vector store with the specified
// dimension and initial capacity. It uses atomic operations and chunking for
// thread-safe concurrent access without traditional locks.
func NewLockFreeVectorStore(dimension int, initialCapacity int) *LockFreeVectorStore {
	store := &LockFreeVectorStore{
		dimension: dimension,
		chunks:    make([]atomic.Value, (initialCapacity+vectorChunkSize-1)/vectorChunkSize),
	}
	store.capacity.Store(int64(initialCapacity))

	// Initialize chunks with zero-filled vectors
	for i := range store.chunks {
		chunk := make([]float32, vectorChunkSize*dimension)
		store.chunks[i].Store(chunk)
	}

	return store
}

// Add adds a vector to the store and returns its index.
// This operation is thread-safe and lock-free, using atomic operations
// to manage concurrent access.
func (s *LockFreeVectorStore) Add(vector []float32) uint64 {
	idx := s.size.Add(1) - 1
	chunkIdx := idx / int64(vectorChunkSize)
	offsetInChunk := (idx % int64(vectorChunkSize)) * int64(s.dimension)

	// Get current chunk using atomic operation
	chunk := s.chunks[chunkIdx].Load().([]float32)

	// Copy vector data
	copy(chunk[offsetInChunk:], vector)

	return uint64(idx)
}

// Get retrieves a vector by its index in a thread-safe manner.
// Returns the vector and a boolean indicating whether the vector was found.
func (s *LockFreeVectorStore) Get(idx uint64) ([]float32, bool) {
	if idx >= uint64(s.size.Load()) {
		return nil, false
	}

	chunkIdx := idx / uint64(vectorChunkSize)
	offsetInChunk := (idx % uint64(vectorChunkSize)) * uint64(s.dimension)

	// Get chunk using atomic operation
	chunk := s.chunks[chunkIdx].Load().([]float32)
	vector := make([]float32, s.dimension)
	copy(vector, chunk[offsetInChunk:offsetInChunk+uint64(s.dimension)])

	return vector, true
}

// Add vector pool helper methods
func (s *MilvusStore) getVectorBuffer(size int) []float32 {
	return s.vectorPool.get(size * s.dimension)
}

func (s *MilvusStore) putVectorBuffer(buf []float32) {
	s.vectorPool.put(buf)
}

// Add adaptive batch sizing methods
func (s *MilvusStore) getOptimalBatchSize() int {
	// Use metrics directly from MilvusStore
	totalInserts := atomic.LoadInt64(&s.metrics.totalInserts)
	if totalInserts == 0 {
		return s.batchSize // Use default if no data
	}

	failureRate := float64(atomic.LoadInt64(&s.metrics.failedInserts)) / float64(totalInserts)
	avgLatency := time.Duration(atomic.LoadInt64(&s.metrics.totalInsertTime) / totalInserts)

	// Adjust batch size based on metrics
	currentSize := float64(s.batchSize)
	switch {
	case failureRate > 0.1: // High failure rate
		currentSize *= 0.8
	case failureRate < 0.01 && avgLatency < 100*time.Millisecond:
		// Good performance
		currentSize *= 1.2
	}

	// Apply bounds
	if currentSize < 100 {
		currentSize = 100
	}
	if currentSize > float64(maxBufferSize) {
		currentSize = float64(maxBufferSize)
	}

	return int(currentSize)
}

// Add queryItems method
func (s *MilvusStore) queryItems(ctx context.Context, expr string, limit int) ([]*storage.Item, error) {
	// Get connection from pool
	conn, err := s.getConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Define output fields
	outputFields := "id,vector,content_type,content_data,metadata,created_at,expires_at"
	partitionNames := []string{}

	// Ensure collection is loaded
	if err := conn.LoadCollection(ctx, s.collection, false); err != nil {
		return nil, fmt.Errorf("failed to load collection: %w", err)
	}

	// Execute query
	queryStart := time.Now()
	results, err := conn.Query(
		ctx,
		s.collection,
		[]string{expr},
		outputFields,
		partitionNames,
		client.WithLimit(int64(limit)),
	)
	if err != nil {
		return nil, fmt.Errorf("query operation failed: %w", err)
	}
	s.logger.Debug().
		Dur("duration", time.Since(queryStart)).
		Msg("Query completed")

	// Process results
	return s.processQueryResults(results)
}

// processQueryResults converts Milvus query results into storage items
func (s *MilvusStore) processQueryResults(results client.ResultSet) ([]*storage.Item, error) {
	if results == nil {
		return nil, fmt.Errorf("no results found")
	}

	s.logger.Debug().Msg("Processing query results")

	// Get field values and validate types
	idCol, ok := results.GetColumn("id").(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("invalid id column type: %T", results.GetColumn("id"))
	}

	vectorCol, ok := results.GetColumn("vector").(*entity.ColumnFloatVector)
	if !ok {
		return nil, fmt.Errorf("invalid vector column type: %T", results.GetColumn("vector"))
	}

	contentTypeCol, ok := results.GetColumn("content_type").(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("invalid content_type column type: %T", results.GetColumn("content_type"))
	}

	contentDataCol, ok := results.GetColumn("content_data").(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("invalid content_data column type: %T", results.GetColumn("content_data"))
	}

	metadataCol, ok := results.GetColumn("metadata").(*entity.ColumnVarChar)
	if !ok {
		return nil, fmt.Errorf("invalid metadata column type: %T", results.GetColumn("metadata"))
	}

	createdAtCol, ok := results.GetColumn("created_at").(*entity.ColumnInt64)
	if !ok {
		return nil, fmt.Errorf("invalid created_at column type: %T", results.GetColumn("created_at"))
	}

	expiresAtCol, ok := results.GetColumn("expires_at").(*entity.ColumnInt64)
	if !ok {
		return nil, fmt.Errorf("invalid expires_at column type: %T", results.GetColumn("expires_at"))
	}

	idData := idCol.Data()
	if len(idData) == 0 {
		return nil, fmt.Errorf("no results found")
	}

	s.logger.Debug().
		Int("num_results", len(idData)).
		Msg("Processing items from query results")

	items := make([]*storage.Item, 0, len(idData))
	for i := 0; i < len(idData); i++ {
		// Get vector data for current row
		vector := vectorCol.Data()[i]
		if len(vector) != s.dimension {
			s.logger.Warn().
				Int("expected_dim", s.dimension).
				Int("actual_dim", len(vector)).
				Msg("Invalid vector dimension")
			continue
		}

		// Parse metadata JSON
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataCol.Data()[i]), &metadata); err != nil {
			s.logger.Warn().
				Err(err).
				Str("item_id", idData[i]).
				Msg("Failed to parse metadata")
			continue
		}

		// Create content
		content := storage.Content{
			Type: storage.ContentType(contentTypeCol.Data()[i]),
			Data: []byte(contentDataCol.Data()[i]),
		}

		item := &storage.Item{
			ID:        idData[i],
			Vector:    vector,
			Content:   content,
			Metadata:  metadata,
			CreatedAt: time.Unix(0, createdAtCol.Data()[i]),
			ExpiresAt: time.Unix(0, expiresAtCol.Data()[i]),
		}
		items = append(items, item)

		s.logger.Debug().
			Str("item_id", item.ID).
			Int("vector_dim", len(item.Vector)).
			Msg("Processed item")
	}

	s.logger.Debug().
		Int("processed_items", len(items)).
		Msg("Completed processing query results")

	return items, nil
}

// monitorIndexPerformance periodically evaluates index performance
func (s *MilvusStore) monitorIndexPerformance() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		s.indexStats.RLock()
		totalSearches := s.indexStats.totalSearches
		successSearches := s.indexStats.successSearches
		var avgLatency time.Duration
		if len(s.indexStats.searchLatency) > 0 {
			var total time.Duration
			for _, lat := range s.indexStats.searchLatency {
				total += lat
			}
			avgLatency = total / time.Duration(len(s.indexStats.searchLatency))
		}
		hitRate := float64(successSearches) / float64(totalSearches)
		s.indexStats.RUnlock()

		// Update adaptive index stats with current index type
		s.adaptiveIndex.UpdateStats(
			IndexTypeHNSW, // Default to HNSW as current index
			avgLatency,
			hitRate > 0.9,
			s.getCurrentMemoryUsage(),
		)

		// Clear old stats
		s.indexStats.Lock()
		s.indexStats.searchLatency = s.indexStats.searchLatency[:0]
		s.indexStats.Unlock()
	}
}

// getCurrentMemoryUsage returns the current memory usage of the index
func (s *MilvusStore) getCurrentMemoryUsage() int64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return int64(m.Alloc)
}

// lshSearch performs LSH-based vector search
func (s *MilvusStore) lshSearch(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	searchStart := time.Now()

	// Use LSH index for approximate search
	candidates := s.lshIndex.Query(vector, limit*2) // Get 2x candidates for better recall

	s.logger.Debug().
		Int("candidates_found", len(candidates)).
		Dur("lsh_query_time", time.Since(searchStart)).
		Msg("LSH search completed")

	items := make([]*storage.Item, 0, len(candidates))

	for i, candidate := range candidates {
		item, err := s.getItemByID(ctx, candidate.id)
		if err != nil {
			s.logger.Warn().
				Err(err).
				Str("item_id", candidate.id).
				Int("candidate_index", i).
				Msg("Failed to get item data")
			continue
		}
		if item != nil {
			items = append(items, item)
			s.logger.Debug().
				Str("item_id", item.ID).
				Msg("Retrieved LSH candidate")
		}
	}

	s.logger.Debug().
		Int("candidates_processed", len(candidates)).
		Int("items_retrieved", len(items)).
		Dur("total_time", time.Since(searchStart)).
		Msg("LSH search processing completed")

	return items, nil
}

// quantizationSearch performs quantization-based vector search
func (s *MilvusStore) quantizationSearch(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	searchStart := time.Now()

	// Quantize the query vector
	centroid, distances, err := s.quantizer.QuantizeVector(ctx, vector)
	if err != nil {
		s.logger.Error().
			Err(err).
			Msg("Failed to quantize query vector")
		return nil, fmt.Errorf("failed to quantize query vector: %w", err)
	}

	s.logger.Debug().
		Int("centroid", centroid).
		Float64("min_distance", float64(distances[0])).
		Dur("quantization_time", time.Since(searchStart)).
		Msg("Vector quantization completed")

	// Search using quantized vector and distances
	expr := fmt.Sprintf("centroid == '%v' AND distances <= %v", centroid, distances[0])
	items, err := s.queryItems(ctx, expr, limit*2)
	if err != nil {
		s.logger.Error().
			Err(err).
			Str("query_expr", expr).
			Msg("Failed to query items")
		return nil, err
	}

	s.logger.Debug().
		Int("items_found", len(items)).
		Str("query_expr", expr).
		Dur("query_time", time.Since(searchStart)).
		Msg("Initial quantization search completed")

	// Rerank results using inner product
	if len(items) > 0 {
		rerankStart := time.Now()
		sort.Slice(items, func(i, j int) bool {
			// Higher inner product means more similar for normalized vectors
			simI := computeInnerProduct(vector, items[i].Vector)
			simJ := computeInnerProduct(vector, items[j].Vector)
			return simI > simJ
		})

		if len(items) > limit {
			items = items[:limit]
		}

		s.logger.Debug().
			Int("reranked_items", len(items)).
			Dur("rerank_time", time.Since(rerankStart)).
			Msg("Results reranking completed")
	}

	s.logger.Debug().
		Int("final_results", len(items)).
		Dur("total_time", time.Since(searchStart)).
		Msg("Quantization search completed")

	return items, nil
}

// newConnectionPool creates a new connection pool
func newConnectionPool(size int) (*connectionPool, error) {
	fmt.Printf("[MilvusStore:Pool] Creating new connection pool with size %d\n", size)
	if size <= 0 {
		fmt.Printf("[MilvusStore:Pool] Invalid size %d, using default size %d\n", size, maxConnections)
		size = maxConnections
	}

	pool := &connectionPool{
		connections: make(chan *pooledConnection, size),
		maxSize:     size,
	}
	fmt.Printf("[MilvusStore:Pool] Connection pool created successfully\n")
	return pool, nil
}

// Add cache warming control
type warmingState struct {
	sync.Mutex
	isWarming  bool
	lastWarmed time.Time
}

var warmState = &warmingState{}

// warmCache performs intelligent cache warming
func (s *MilvusStore) warmCache(ctx context.Context) error {
	// Prevent concurrent warming
	warmState.Lock()
	if warmState.isWarming || time.Since(warmState.lastWarmed) < 10*time.Minute {
		warmState.Unlock()
		return nil
	}
	warmState.isWarming = true
	warmState.Unlock()

	defer func() {
		warmState.Lock()
		warmState.isWarming = false
		warmState.lastWarmed = time.Now()
		warmState.Unlock()
	}()

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for cache warmup: %w", err)
	}
	defer s.releaseConnection(conn)

	// Query most recently accessed and frequently accessed items
	expr := fmt.Sprintf("created_at >= %d", time.Now().Add(-24*time.Hour).UnixNano())
	outputFields := []string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"}

	result, err := conn.Query(
		ctx,
		s.collection,
		[]string{},
		expr,
		outputFields,
	)
	if err != nil {
		return fmt.Errorf("failed to query items for cache warmup: %w", err)
	}

	if len(result) == 0 {
		return nil
	}

	// Process results and add to cache
	items, err := s.processQueryResults(result)
	if err != nil {
		return fmt.Errorf("failed to process cache warmup results: %w", err)
	}

	// Add to cache
	for _, item := range items {
		s.localCache.Add(item.ID, item)
	}

	return nil
}

// prefetchItems performs batch prefetching of items
func (s *MilvusStore) prefetchItems(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	// Build ID list for query
	idList := make([]string, len(ids))
	for i, id := range ids {
		idList[i] = fmt.Sprintf("'%s'", id)
	}

	expr := fmt.Sprintf("id in [%s]", strings.Join(idList, ","))

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for prefetch: %w", err)
	}
	defer s.releaseConnection(conn)

	// Query items
	result, err := conn.Query(
		ctx,
		s.collection,
		[]string{},
		expr,
		[]string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"},
	)
	if err != nil {
		return fmt.Errorf("failed to query items for prefetch: %w", err)
	}

	if len(result) == 0 {
		return nil
	}

	// Process results and add to cache
	items, err := s.processQueryResults(result)
	if err != nil {
		return fmt.Errorf("failed to process prefetch results: %w", err)
	}

	// Add to cache
	for _, item := range items {
		s.localCache.Add(item.ID, item)
	}

	return nil
}

// BatchSet stores vectors with their associated data in batches
func (s *MilvusStore) BatchSet(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	// Validate items before processing
	for _, item := range items {
		// Validate content type
		if !isValidContentType(item.Content.Type) {
			return fmt.Errorf("invalid content type: %s", item.Content.Type)
		}

		// Validate vector dimension
		if len(item.Vector) != s.dimension {
			return fmt.Errorf("invalid vector dimension: got %d, want %d", len(item.Vector), s.dimension)
		}
	}

	// Process batches efficiently
	batchSize := s.getOptimalBatchSize()
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batchItems := items[i:end]

		err := s.processBatchEfficient(ctx, batchItems)
		if err != nil {
			return fmt.Errorf("batch processing error: %w", err)
		}
	}

	return nil
}

// computeNorm calculates the L2 norm of a vector
func computeNorm(vector []float32) float32 {
	var sum float32
	for _, v := range vector {
		sum += v * v
	}
	return float32(math.Sqrt(float64(sum)))
}

// computeInnerProduct calculates the inner product between two vectors
func computeInnerProduct(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}
