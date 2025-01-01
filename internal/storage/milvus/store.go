package milvus

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
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
			fmt.Printf("Task error: %v\n", err)
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
	metrics        struct {
		failedOps       int64
		failedInserts   int64
		totalInsertTime int64
		totalInserts    int64
		cacheHits       int64
		cacheMisses     int64
		avgBatchSize    float64
	}
	workQueue *WorkQueue
	// Add buffer for batching inserts
	insertBuffer struct {
		sync.Mutex
		items     []*storage.Item
		lastFlush time.Time
	}
	// Replace localCache with shardedCache
	localCache *ShardedCache
	// Add access pattern tracking
	accessPatterns struct {
		sync.RWMutex
		patterns map[string]*AccessPattern
		// Track relationships between items
		relationships map[string][]string
	}
	vectorPool *vectorBufferPool
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
	sync.Mutex
	buffers    map[int][][]float32
	dimension  int
	maxBuffers int
}

func newVectorBufferPool(dimension, maxBuffers int) *vectorBufferPool {
	return &vectorBufferPool{
		buffers:    make(map[int][][]float32),
		dimension:  dimension,
		maxBuffers: maxBuffers,
	}
}

func (p *vectorBufferPool) get(size int) []float32 {
	p.Lock()
	defer p.Unlock()

	// Round up to nearest power of 2 for better memory utilization
	size = nextPowerOfTwo(size)

	if buffers, ok := p.buffers[size]; ok && len(buffers) > 0 {
		buffer := buffers[len(buffers)-1]
		p.buffers[size] = buffers[:len(buffers)-1]
		return buffer[:0] // Reset length but keep capacity
	}

	return make([]float32, 0, size)
}

func (p *vectorBufferPool) put(buffer []float32) {
	p.Lock()
	defer p.Unlock()

	size := nextPowerOfTwo(cap(buffer))
	if len(p.buffers[size]) < p.maxBuffers {
		p.buffers[size] = append(p.buffers[size], buffer)
	}
}

// nextPowerOfTwo returns the next power of 2 >= n
func nextPowerOfTwo(n int) int {
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n |= n >> 32
	n++
	return n
}

// NewMilvusStore creates a new Milvus store instance with the provided configuration.
func NewMilvusStore(cfg Config) (*MilvusStore, error) {
	start := time.Now()
	fmt.Printf("[Milvus:Init] Starting initialization with config: %+v\n", cfg)

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

	// Create sharded cache instead of simple LRU
	localCache, err := newShardedCache(10000) // Cache 10k items
	if err != nil {
		return nil, fmt.Errorf("failed to create local cache: %w", err)
	}

	// Initialize connection pool
	poolStart := time.Now()
	fmt.Printf("[Milvus:Init] Creating connection pool...\n")
	pool, err := newConnectionPool(cfg.PoolSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}
	fmt.Printf("[Milvus:Init] Connection pool created in %v\n", time.Since(poolStart))

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
		timeout:        30 * time.Second,
		workQueue:      newWorkQueue(cfg.PoolSize, cfg.PoolSize*100),
		insertBuffer: struct {
			sync.Mutex
			items     []*storage.Item
			lastFlush time.Time
		}{
			items:     make([]*storage.Item, 0, maxBufferSize),
			lastFlush: time.Now(),
		},
		localCache: localCache,
		vectorPool: newVectorBufferPool(cfg.Dimension, 1000),
	}

	// Initialize access patterns tracking
	store.accessPatterns.patterns = make(map[string]*AccessPattern)
	store.accessPatterns.relationships = make(map[string][]string)

	// Initialize pool connections
	poolInitStart := time.Now()
	fmt.Printf("[Milvus:Init] Initializing connection pool with %d connections...\n", cfg.PoolSize)
	if err := store.initializePool(cfg.PoolSize); err != nil {
		return nil, fmt.Errorf("failed to initialize connection pool: %w", err)
	}
	fmt.Printf("[Milvus:Init] Pool initialization completed in %v\n", time.Since(poolInitStart))

	// Initialize collection
	collectionStart := time.Now()
	fmt.Printf("[Milvus:Init] Ensuring collection exists...\n")
	ctx := context.Background()
	if err := store.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize collection: %w", err)
	}
	fmt.Printf("[Milvus:Init] Collection initialization completed in %v\n", time.Since(collectionStart))

	// Start background flush task
	store.startBackgroundFlush(context.Background())
	fmt.Printf("[Milvus:Init] Background flush task started\n")

	// Start metrics logging
	go store.logPoolMetrics()
	fmt.Printf("[Milvus:Init] Pool metrics logging started\n")

	// Warm up the cache with recent items
	warmupStart := time.Now()
	fmt.Printf("[Milvus:Init] Starting cache warmup...\n")
	if err := store.warmCache(ctx); err != nil {
		fmt.Printf("[Milvus:Init] Cache warmup failed: %v\n", err)
		// Continue even if cache warmup fails
	} else {
		fmt.Printf("[Milvus:Init] Cache warmup completed in %v\n", time.Since(warmupStart))
	}

	fmt.Printf("[Milvus:Init] Total initialization completed in %v\n", time.Since(start))
	return store, nil
}

// initializePool initializes the connection pool with the specified size
func (s *MilvusStore) initializePool(size int) error {
	fmt.Printf("[Milvus:Pool] Starting pool initialization with size %d\n", size)

	// Create initial connections
	for i := 0; i < size; i++ {
		connStart := time.Now()
		fmt.Printf("[Milvus:Pool] Creating connection %d/%d\n", i+1, size)

		conn, err := s.createConnection()
		if err != nil {
			return fmt.Errorf("failed to create initial connection %d: %w", i+1, err)
		}
		fmt.Printf("[Milvus:Pool] Connection %d created in %v\n", i+1, time.Since(connStart))

		pc := &pooledConnection{
			client:   conn,
			lastUsed: time.Now(),
		}

		// Start health check goroutine for each connection
		ctx, cancel := context.WithCancel(context.Background())
		pc.healthCtx = cancel
		go s.connectionHealthCheck(ctx, pc)

		s.pool.connections <- pc
	}

	// Start connection manager
	go s.manageConnections()

	return nil
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

func (s *MilvusStore) createConnection() (client.Client, error) {
	addr := fmt.Sprintf("%s:%d", s.host, s.port)
	return client.NewGrpcClient(context.Background(), addr)
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
			time.Sleep(time.Duration(i*100) * time.Millisecond) // Exponential backoff
			continue
		}

		if len(result) > 0 && len(result[0].(*entity.ColumnVarChar).Data()) > 0 {
			return nil // Item found
		}

		lastErr = fmt.Errorf("item %s not found after insertion", id)
		time.Sleep(time.Duration(i*100) * time.Millisecond)
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
		expiresAt[i] = item.ExpiresAt.UnixNano()
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
		return err
	})

	if err != nil {
		atomic.AddInt64(&s.metrics.failedInserts, 1)
		return fmt.Errorf("failed to insert batch: %w", err)
	}

	insertDuration := time.Since(insertStart)
	atomic.AddInt64(&s.metrics.totalInsertTime, insertDuration.Nanoseconds())
	atomic.AddInt64(&s.metrics.totalInserts, int64(len(items)))

	// Verify a sample of items from the batch
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

		if err := s.verifyItem(ctx, id, 3); err != nil {
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

	// Update cache in background with rate limiting
	go func() {
		for _, item := range items {
			select {
			case <-time.After(time.Millisecond): // Rate limit cache updates
				s.localCache.Add(item.ID, item)
			case <-ctx.Done():
				return
			}
		}
	}()

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
	chunkStart := time.Now()
	fmt.Printf("[Milvus:DeleteChunk] Starting deletion of %d items\n", len(ids))
	fmt.Printf("[Milvus:DeleteChunk] Items to delete: %v\n", ids)

	return s.withRetry(ctx, func(ctx context.Context) error {
		// Get connection
		connStart := time.Now()
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)
		fmt.Printf("[Milvus:DeleteChunk] Got connection in %v\n", time.Since(connStart))

		// Build the expression for deletion
		exprStart := time.Now()
		idList := make([]string, len(ids))
		for i, id := range ids {
			idList[i] = fmt.Sprintf("'%s'", id)
		}
		expr := fmt.Sprintf("id in [%s]", strings.Join(idList, ","))
		fmt.Printf("[Milvus:DeleteChunk] Built expression in %v: %s\n", time.Since(exprStart), expr)

		// Verify items exist before deletion
		verifyStart := time.Now()
		queryResult, err := conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
		if err != nil {
			fmt.Printf("[Milvus:DeleteChunk] Pre-deletion verification query failed: %v\n", err)
		} else if len(queryResult) > 0 {
			if idCol, ok := queryResult[0].(*entity.ColumnVarChar); ok {
				fmt.Printf("[Milvus:DeleteChunk] Found %d items before deletion: %v\n",
					len(idCol.Data()), idCol.Data())
			}
		}
		fmt.Printf("[Milvus:DeleteChunk] Pre-deletion verification took %v\n", time.Since(verifyStart))

		// Delete the records with retry
		deleteStart := time.Now()
		maxRetries := 3
		for i := 0; i < maxRetries; i++ {
			if err := conn.Delete(ctx, s.collection, "", expr); err != nil {
				fmt.Printf("[Milvus:DeleteChunk] Deletion attempt %d failed after %v: %v\n",
					i+1, time.Since(deleteStart), err)
				if i == maxRetries-1 {
					return fmt.Errorf("failed to delete chunk after %d attempts: %w", maxRetries, err)
				}
				time.Sleep(100 * time.Millisecond)
				continue
			}
			fmt.Printf("[Milvus:DeleteChunk] Deletion executed in %v\n", time.Since(deleteStart))
			break
		}

		// Flush to ensure deletion is persisted
		flushStart := time.Now()
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			fmt.Printf("[Milvus:DeleteChunk] Flush failed after %v: %v\n",
				time.Since(flushStart), err)
			return fmt.Errorf("failed to flush after deletion: %w", err)
		}
		fmt.Printf("[Milvus:DeleteChunk] Flush completed in %v\n", time.Since(flushStart))

		// Load collection to ensure changes are reflected in memory
		loadStart := time.Now()
		if err := conn.LoadCollection(ctx, s.collection, false); err != nil {
			fmt.Printf("[Milvus:DeleteChunk] Load failed after %v: %v\n",
				time.Since(loadStart), err)
			return fmt.Errorf("failed to load collection after deletion: %w", err)
		}
		fmt.Printf("[Milvus:DeleteChunk] Collection loaded in %v\n", time.Since(loadStart))

		// Wait for collection to be fully loaded with timeout
		waitStart := time.Now()
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
					fmt.Printf("[Milvus:DeleteChunk] Failed to get load state after %v: %v\n",
						time.Since(waitStart), err)
					return fmt.Errorf("failed to get load state: %w", err)
				}
				fmt.Printf("[Milvus:DeleteChunk] Current load state: %d\n", loadState)
				if loadState == entity.LoadStateLoaded {
					fmt.Printf("[Milvus:DeleteChunk] Collection fully loaded in %v\n", time.Since(waitStart))
					goto verifyDeletion
				}
			}
		}

	verifyDeletion:
		// Verify items are deleted with retry
		verifyStart = time.Now()
		maxVerifyRetries := 3
		for i := 0; i < maxVerifyRetries; i++ {
			queryResult, err = conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
			if err != nil {
				fmt.Printf("[Milvus:DeleteChunk] Post-deletion verification query %d failed: %v\n", i+1, err)
				continue
			}
			if len(queryResult) == 0 || len(queryResult[0].(*entity.ColumnVarChar).Data()) == 0 {
				fmt.Printf("[Milvus:DeleteChunk] Deletion verified - no items remaining\n")
				break
			}
			if i == maxVerifyRetries-1 {
				if idCol, ok := queryResult[0].(*entity.ColumnVarChar); ok {
					remainingIds := idCol.Data()
					fmt.Printf("[Milvus:DeleteChunk] WARNING: Found %d items still present after deletion: %v\n",
						len(remainingIds), remainingIds)
					return fmt.Errorf("deletion verification failed: items still present after %d retries", maxVerifyRetries, maxVerifyRetries)
				}
			}
			time.Sleep(100 * time.Millisecond)
		}
		fmt.Printf("[Milvus:DeleteChunk] Post-deletion verification took %v\n", time.Since(verifyStart))

		// Remove from cache
		cacheStart := time.Now()
		for _, id := range ids {
			s.localCache.Remove(id)
			// Also remove any cached search results
			s.localCache.Remove("search:" + id)
		}
		fmt.Printf("[Milvus:DeleteChunk] Cache cleanup completed in %v\n", time.Since(cacheStart))

		// Clear the entire cache to ensure no stale search results remain
		cacheStart = time.Now()
		newCache, err := newShardedCache(s.poolSize * 100)
		if err != nil {
			s.logger.Error().Err(err).Msg("Failed to create new cache after deletion")
		} else {
			s.localCache = newCache
			fmt.Printf("[Milvus:DeleteChunk] Cache cleared in %v\n", time.Since(cacheStart))
		}

		fmt.Printf("[Milvus:DeleteChunk] Total chunk operation took %v\n", time.Since(chunkStart))
		return nil
	})
}

// Update updates existing items
func (s *MilvusStore) Update(ctx context.Context, items []*storage.Item) error {
	// In Milvus, we handle updates as delete + insert
	for _, item := range items {
		// Delete existing
		if err := s.DeleteFromStore(ctx, []string{item.ID}); err != nil {
			return err
		}

		// Insert new
		if err := s.Insert(ctx, []*storage.Item{item}); err != nil {
			return err
		}
	}
	return nil
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
	start := time.Now()
	fmt.Printf("[Milvus:Collection] Checking collection existence...\n")

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Check if collection exists
	exists, err := conn.HasCollection(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if exists {
		// Load collection to verify schema
		err = conn.LoadCollection(ctx, s.collection, false)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}

		// Get collection info
		info, err := conn.DescribeCollection(ctx, s.collection)
		if err != nil {
			return fmt.Errorf("failed to get collection info: %w", err)
		}

		// Check vector dimension
		for _, field := range info.Schema.Fields {
			if field.Name == "vector" {
				if dim, ok := field.TypeParams["dim"]; ok {
					if dim != fmt.Sprintf("%d", s.dimension) {
						// Drop collection if dimension mismatch
						err = conn.DropCollection(ctx, s.collection)
						if err != nil {
							return fmt.Errorf("failed to drop collection with wrong dimension: %w", err)
						}
						exists = false
						break
					}
				}
			}
		}
	}

	if !exists {
		createStart := time.Now()
		fmt.Printf("[Milvus:Collection] Creating collection with schema...\n")

		schema := &entity.Schema{
			CollectionName: s.collection,
			Description:    "Vector storage collection",
			Fields: []*entity.Field{
				{
					Name:       "id",
					DataType:   entity.FieldTypeVarChar,
					PrimaryKey: true,
					TypeParams: map[string]string{
						"max_length": "256",
					},
				},
				{
					Name:     "vector",
					DataType: entity.FieldTypeFloatVector,
					TypeParams: map[string]string{
						"dim": fmt.Sprintf("%d", s.dimension),
					},
				},
				{
					Name:     "content_type",
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						"max_length": "64",
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

		err = conn.CreateCollection(ctx, schema, 2)
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		fmt.Printf("[Milvus:Collection] Collection created in %v\n", time.Since(createStart))

		// Create index for vector field
		indexStart := time.Now()
		fmt.Printf("[Milvus:Collection] Creating index...\n")
		idx, err := entity.NewIndexIvfFlat(entity.L2, 1024)
		if err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}

		err = conn.CreateIndex(ctx, s.collection, "vector", idx, false)
		if err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
		fmt.Printf("[Milvus:Collection] Index created in %v\n", time.Since(indexStart))

		// Load collection
		loadStart := time.Now()
		fmt.Printf("[Milvus:Collection] Loading collection...\n")
		err = conn.LoadCollection(ctx, s.collection, false)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}
		fmt.Printf("[Milvus:Collection] Collection loaded in %v\n", time.Since(loadStart))
	}

	fmt.Printf("[Milvus:Collection] Collection initialization completed in %v\n", time.Since(start))
	return nil
}

// searchParams holds optimized search parameters
type searchParams struct {
	nprobe       int
	ef           int
	metric       entity.MetricType
	useParallel  bool
	numPartition int
}

// getOptimizedSearchParams returns optimized search parameters based on dataset size
func (s *MilvusStore) getOptimizedSearchParams(ctx context.Context) (*searchParams, error) {
	conn, err := s.getConnection()
	if err != nil {
		return nil, fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Get collection stats
	stats, err := conn.GetCollectionStatistics(ctx, s.collection)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection stats: %w", err)
	}

	var rowCount int64
	rowCountStr, ok := stats["row_count"]
	if ok {
		rowCount, _ = strconv.ParseInt(rowCountStr, 10, 64)
	}

	// Optimize parameters based on dataset size
	params := &searchParams{
		metric: entity.L2,
	}

	switch {
	case rowCount < 10000: // Small dataset
		params.nprobe = 8
		params.ef = 64
		params.useParallel = false
		params.numPartition = 1
	case rowCount < 100000: // Medium dataset
		params.nprobe = 16
		params.ef = 128
		params.useParallel = true
		params.numPartition = 2
	case rowCount < 1000000: // Large dataset
		params.nprobe = 32
		params.ef = 256
		params.useParallel = true
		params.numPartition = 4
	default: // Very large dataset
		params.nprobe = 64
		params.ef = 512
		params.useParallel = true
		params.numPartition = 8
	}

	return params, nil
}

// Search performs a similarity search using the provided vector
func (s *MilvusStore) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	start := time.Now()
	fmt.Printf("[Milvus:Search] Starting search with vector dimension %d, limit %d\n", len(vector), limit)

	// Check cache first
	cacheKey := s.vectorCacheKey(vector)
	if cached, ok := s.localCache.Get(cacheKey); ok {
		if items, ok := cached.([]*storage.Item); ok {
			fmt.Printf("[Milvus:Search] Cache hit, returning %d items\n", len(items))
			atomic.AddInt64(&s.metrics.cacheHits, 1)
			return items, nil
		}
	}
	atomic.AddInt64(&s.metrics.cacheMisses, 1)

	if len(vector) != s.dimension {
		return nil, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), s.dimension)
	}

	// Get optimized search parameters
	params, err := s.getOptimizedSearchParams(ctx)
	if err != nil {
		s.logger.Warn().Err(err).Msg("Failed to get optimized search params, using defaults")
		params = &searchParams{
			nprobe:       16,
			ef:           128,
			metric:       entity.L2,
			useParallel:  true,
			numPartition: 2,
		}
	}

	var searchResults []client.SearchResult
	err = s.withRetry(ctx, func(ctx context.Context) error {
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)

		// Create optimized search parameters
		sp, err := entity.NewIndexIvfFlatSearchParam(params.nprobe)
		if err != nil {
			return fmt.Errorf("failed to create search parameters: %w", err)
		}

		// Define output fields
		outputFields := []string{
			"id",
			"vector",
			"content_type",
			"content_data",
			"metadata",
			"created_at",
			"expires_at",
		}

		// Ensure collection is loaded
		err = conn.LoadCollection(ctx, s.collection, params.useParallel)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}

		// Perform search with optimized parameters
		searchStart := time.Now()
		fmt.Printf("[Milvus:Search] Executing search with dimension %d, nprobe %d\n",
			len(vector), params.nprobe)

		result, err := conn.Search(
			ctx,
			s.collection,
			[]string{},
			"",
			outputFields,
			[]entity.Vector{entity.FloatVector(vector)},
			"vector",
			params.metric,
			limit,
			sp,
		)
		if err != nil {
			return fmt.Errorf("search operation failed: %w", err)
		}
		fmt.Printf("[Milvus:Search] Search completed in %v\n", time.Since(searchStart))

		searchResults = result
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Process results
	items, err := s.processSearchResults(searchResults)
	if err != nil {
		return nil, fmt.Errorf("failed to process search results: %w", err)
	}

	// Cache results
	if len(items) > 0 {
		s.localCache.Add(cacheKey, items)
	}

	fmt.Printf("[Milvus:Search] Search completed in %v, found %d items\n",
		time.Since(start), len(items))
	return items, nil
}

// vectorCacheKey generates a cache key for vector search results
func (s *MilvusStore) vectorCacheKey(vector []float32) string {
	h := fnv.New64a()
	for _, v := range vector {
		binary.Write(h, binary.LittleEndian, v)
	}
	return fmt.Sprintf("vector:%x", h.Sum64())
}

// processSearchResults processes search results
func (s *MilvusStore) processSearchResults(results []client.SearchResult) ([]*storage.Item, error) {
	if len(results) == 0 {
		return nil, nil
	}

	items := make([]*storage.Item, 0)
	for i := 0; i < results[0].ResultCount; i++ {
		// Extract fields for this result
		var fields []entity.Column
		for _, col := range results[0].Fields {
			fields = append(fields, col)
		}

		item, err := s.processQueryResult(fields)
		if err != nil {
			return nil, fmt.Errorf("failed to process query result at index %d: %w", i, err)
		}

		if item != nil {
			items = append(items, item)
		}
	}

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

// processLargeBatch handles large batch insertions efficiently
func (s *MilvusStore) processLargeBatch(ctx context.Context, items []*storage.Item) error {
	start := time.Now()
	totalItems := len(items)
	logger := s.logger.With().Int("total_items", totalItems).Logger()
	logger.Debug().Msg("Starting large batch processing")

	// Calculate optimal batch size based on vector dimension
	optimalBatchSize := s.calculateOptimalBatchSize()
	batches := (totalItems + optimalBatchSize - 1) / optimalBatchSize

	// Create worker pool for parallel processing
	pool := newVectorWorkerPool(maxConcurrentChunks)
	pool.start()
	defer pool.stop()

	// Process batches in parallel
	for i := 0; i < totalItems; i += optimalBatchSize {
		end := i + optimalBatchSize
		if end > totalItems {
			end = totalItems
		}
		batch := items[i:end]
		batchNum := (i / optimalBatchSize) + 1

		// Submit batch processing task
		pool.tasks <- func() error {
			batchStart := time.Now()
			batchLogger := logger.With().
				Int("batch_number", batchNum).
				Int("total_batches", batches).
				Int("batch_size", len(batch)).
				Logger()

			// Process batch with optimized memory allocation
			if err := s.processBatchOptimized(ctx, batch, batchLogger); err != nil {
				batchLogger.Error().
					Err(err).
					Dur("duration", time.Since(batchStart)).
					Msg("Batch processing failed")
				return fmt.Errorf("batch %d failed: %v", batchNum, err)
			}

			batchLogger.Debug().
				Dur("duration", time.Since(batchStart)).
				Msg("Batch completed")
			return nil
		}
	}

	// Collect results
	var errs []error
	for i := 0; i < batches; i++ {
		if err := <-pool.results; err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("batch processing errors: %v", errs)
	}

	logger.Debug().
		Dur("duration", time.Since(start)).
		Msg("Large batch processing completed")
	return nil
}

// processBatchOptimized processes a batch with optimized memory allocation
func (s *MilvusStore) processBatchOptimized(ctx context.Context, batch []*storage.Item, logger zerolog.Logger) error {
	batchSize := len(batch)
	if batchSize == 0 {
		return nil
	}

	// Pre-allocate all slices with optimal size
	ids := make([]string, 0, batchSize)
	vectors := make([][]float32, 0, batchSize)
	contentTypes := make([]string, 0, batchSize)
	contentData := make([]string, 0, batchSize)
	metadata := make([]string, 0, batchSize)
	createdAt := make([]int64, 0, batchSize)
	expiresAt := make([]int64, 0, batchSize)

	// Get vector buffer from pool
	vectorBuffer := s.vectorPool.get(s.dimension * batchSize)
	defer s.vectorPool.put(vectorBuffer)

	// Process items with minimal allocations
	for _, item := range batch {
		ids = append(ids, item.ID)

		// Copy vector data to buffer
		start := len(vectorBuffer)
		vectorBuffer = append(vectorBuffer, item.Vector...)
		vectors = append(vectors, vectorBuffer[start:len(vectorBuffer)])

		contentTypes = append(contentTypes, string(item.Content.Type))
		contentData = append(contentData, string(item.Content.Data))
		metadata = append(metadata, encodeMetadata(item.Metadata))
		createdAt = append(createdAt, item.CreatedAt.UnixNano())
		expiresAt = append(expiresAt, item.ExpiresAt.UnixNano())
	}

	// Create columns efficiently
	columns := []entity.Column{
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnFloatVector("vector", s.dimension, vectors),
		entity.NewColumnVarChar("content_type", contentTypes),
		entity.NewColumnVarChar("content_data", contentData),
		entity.NewColumnVarChar("metadata", metadata),
		entity.NewColumnInt64("created_at", createdAt),
		entity.NewColumnInt64("expires_at", expiresAt),
	}

	// Get connection from pool
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Insert data with retry and metric tracking
	insertStart := time.Now()
	logger.Debug().Msg("Starting batch insert")

	err = s.withRetry(ctx, func(ctx context.Context) error {
		_, err := conn.Insert(ctx, s.collection, "", columns...)
		if err != nil {
			return fmt.Errorf("failed to insert batch: %w", err)
		}

		// Flush to ensure data is persisted
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to flush after insert: %w", err)
		}

		// Load collection to ensure data is indexed
		if err := conn.LoadCollection(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to load collection after insert: %w", err)
		}

		return nil
	})

	if err != nil {
		atomic.AddInt64(&s.metrics.failedInserts, 1)
		logger.Error().
			Err(err).
			Dur("duration", time.Since(insertStart)).
			Msg("Batch insert failed")
		return fmt.Errorf("failed to insert batch: %w", err)
	}

	insertDuration := time.Since(insertStart)
	atomic.AddInt64(&s.metrics.totalInsertTime, insertDuration.Nanoseconds())
	atomic.AddInt64(&s.metrics.totalInserts, int64(len(batch)))

	logger.Debug().
		Dur("duration", insertDuration).
		Msg("Batch inserted")

	// Update cache in background with rate limiting
	go func() {
		for _, item := range batch {
			select {
			case <-time.After(time.Millisecond): // Rate limit cache updates
				s.localCache.Add(item.ID, item)
			case <-ctx.Done():
				return
			}
		}
	}()

	return nil
}

// calculateOptimalBatchSize determines the optimal batch size based on performance metrics
func (s *MilvusStore) calculateOptimalBatchSize() int {
	// Calculate memory per vector
	memoryPerVector := s.dimension * 4 // 4 bytes per float32

	// Get system memory info
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Calculate memory-based size
	availableMemory := (m.Sys - m.HeapAlloc) / 10 // Use up to 10% of available memory
	maxVectorsInMemory := availableMemory / uint64(memoryPerVector)

	// Get performance metrics
	totalInserts := atomic.LoadInt64(&s.metrics.totalInserts)
	if totalInserts == 0 {
		// No metrics yet, use memory-based calculation
		if maxVectorsInMemory > uint64(batchSize) {
			maxVectorsInMemory = uint64(batchSize)
		}
		if maxVectorsInMemory < 100 {
			maxVectorsInMemory = 100
		}
		return int(maxVectorsInMemory)
	}

	// Calculate average insert time per item
	totalTime := time.Duration(atomic.LoadInt64(&s.metrics.totalInsertTime))
	avgTimePerItem := totalTime.Nanoseconds() / totalInserts

	// Adjust batch size based on performance
	failureRate := float64(atomic.LoadInt64(&s.metrics.failedInserts)) / float64(totalInserts)

	var optimalSize int
	switch {
	case failureRate > 0.1: // High failure rate, reduce batch size
		optimalSize = int(float64(s.batchSize) * 0.8)
	case failureRate < 0.01 && avgTimePerItem < 1000000: // Low failure rate and good performance
		optimalSize = int(float64(s.batchSize) * 1.2)
	default:
		optimalSize = s.batchSize
	}

	// Apply bounds
	if optimalSize > int(maxVectorsInMemory) {
		optimalSize = int(maxVectorsInMemory)
	}
	if optimalSize > batchSize {
		optimalSize = batchSize
	}
	if optimalSize < 100 {
		optimalSize = 100
	}

	// Update average batch size metric
	s.metrics.avgBatchSize = float64(optimalSize)

	return optimalSize
}

// newConnectionPool creates a new connection pool
func newConnectionPool(size int) (*connectionPool, error) {
	if size <= 0 {
		size = maxConnections
	}
	return &connectionPool{
		connections: make(chan *pooledConnection, size),
		maxSize:     size,
	}, nil
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

	fmt.Printf("[Milvus:Cache] Starting cache warmup\n")
	start := time.Now()

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for cache warmup: %w", err)
	}
	defer s.releaseConnection(conn)

	// Query most recently accessed and frequently accessed items
	expr := fmt.Sprintf("created_at >= %d", time.Now().Add(-24*time.Hour).UnixNano())
	outputFields := []string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"}

	fmt.Printf("[Milvus:Cache] Querying up to %d recent items for warmup\n", cacheWarmupSize)
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
		fmt.Printf("[Milvus:Cache] No results found\n")
		return nil
	}

	// Find the ID column to determine row count
	var totalRows int
	for _, col := range result {
		if col.Name() == "id" {
			if idCol, ok := col.(*entity.ColumnVarChar); ok {
				totalRows = len(idCol.Data())
				break
			}
		}
	}

	if totalRows == 0 {
		fmt.Printf("[Milvus:Cache] No valid rows found\n")
		return nil
	}

	fmt.Printf("[Milvus:Cache] Found %d items, processing up to %d for cache\n",
		totalRows, cacheWarmupSize)

	// Process only up to cacheWarmupSize items
	maxItems := totalRows
	if maxItems > cacheWarmupSize {
		maxItems = cacheWarmupSize
	}

	items := make([]*storage.Item, 0, maxItems)
	for i := 0; i < maxItems; i++ {
		item := &storage.Item{
			Content:  storage.Content{},
			Metadata: make(map[string]interface{}),
		}

		var conversionError bool
		for _, col := range result {
			if conversionError {
				break
			}

			switch col.Name() {
			case "id":
				if idCol, ok := col.(*entity.ColumnVarChar); ok && i < len(idCol.Data()) {
					item.ID = idCol.Data()[i]
				} else {
					conversionError = true
				}
			case "vector":
				if vecCol, ok := col.(*entity.ColumnFloatVector); ok && i < len(vecCol.Data()) {
					item.Vector = vecCol.Data()[i]
				} else {
					conversionError = true
				}
			case "content_type":
				if typeCol, ok := col.(*entity.ColumnVarChar); ok && i < len(typeCol.Data()) {
					item.Content.Type = storage.ContentType(typeCol.Data()[i])
				}
			case "content_data":
				if dataCol, ok := col.(*entity.ColumnVarChar); ok && i < len(dataCol.Data()) {
					item.Content.Data = []byte(dataCol.Data()[i])
				}
			case "metadata":
				if metaCol, ok := col.(*entity.ColumnVarChar); ok && i < len(metaCol.Data()) {
					item.Metadata = decodeMetadata(metaCol.Data()[i])
				}
			case "created_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && i < len(timeCol.Data()) {
					item.CreatedAt = time.Unix(0, timeCol.Data()[i])
				}
			case "expires_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && i < len(timeCol.Data()) {
					item.ExpiresAt = time.Unix(0, timeCol.Data()[i])
				}
			}
		}

		if !conversionError && item.ID != "" && item.Vector != nil {
			items = append(items, item)
		}
	}

	// Batch add items to cache
	for _, item := range items {
		s.localCache.Add(item.ID, item)
	}

	fmt.Printf("[Milvus:Cache] Successfully warmed up cache with %d items in %v (processing took %v)\n",
		len(items), time.Since(start), time.Since(start))
	return nil
}

// Add prefetching for batch operations
func (s *MilvusStore) prefetchItems(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	fmt.Printf("[Milvus:Prefetch] Starting prefetch for %d items\n", len(ids))
	start := time.Now()

	// Build ID list for query
	idList := make([]string, len(ids))
	listStart := time.Now()
	for i, id := range ids {
		idList[i] = fmt.Sprintf("'%s'", id)
	}
	fmt.Printf("[Milvus:Prefetch] Built ID list in %v\n", time.Since(listStart))

	expr := fmt.Sprintf("id in [%s]", strings.Join(idList, ","))
	fmt.Printf("[Milvus:Prefetch] Query expression: %s\n", expr)

	conn, err := s.getConnection()
	if err != nil {
		fmt.Printf("[Milvus:Prefetch] Failed to get connection: %v\n", err)
		return fmt.Errorf("failed to get connection for prefetch: %w", err)
	}
	defer s.releaseConnection(conn)

	// Query items
	queryStart := time.Now()
	result, err := conn.Query(
		ctx,
		s.collection,
		[]string{},
		expr,
		[]string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"},
	)
	if err != nil {
		fmt.Printf("[Milvus:Prefetch] Query failed after %v: %v\n",
			time.Since(queryStart), err)
		return fmt.Errorf("failed to query items for prefetch: %w", err)
	}
	fmt.Printf("[Milvus:Prefetch] Query completed in %v\n", time.Since(queryStart))

	if len(result) == 0 {
		fmt.Printf("[Milvus:Prefetch] No results found for IDs: %v\n", ids)
		return nil
	}

	// Process results and add to cache
	processStart := time.Now()
	rowCount := 0
	for _, col := range result {
		switch col.Name() {
		case "id":
			if idCol, ok := col.(*entity.ColumnVarChar); ok {
				rowCount = len(idCol.Data())
				fmt.Printf("[Milvus:Prefetch] Found %d rows\n", rowCount)
			}
		}
	}

	if rowCount == 0 {
		fmt.Printf("[Milvus:Prefetch] No valid rows in results\n")
		return nil
	}

	// Convert results to items
	items := make([]*storage.Item, 0, rowCount)
	for i := 0; i < rowCount; i++ {
		item := &storage.Item{
			Content:  storage.Content{},
			Metadata: make(map[string]interface{}),
		}

		var conversionError bool
		for _, col := range result {
			if conversionError {
				break
			}

			switch col.Name() {
			case "id":
				if idCol, ok := col.(*entity.ColumnVarChar); ok && i < len(idCol.Data()) {
					item.ID = idCol.Data()[i]
				} else {
					conversionError = true
				}
			case "vector":
				if vecCol, ok := col.(*entity.ColumnFloatVector); ok && i < len(vecCol.Data()) {
					item.Vector = vecCol.Data()[i]
				} else {
					conversionError = true
				}
			case "content_type":
				if typeCol, ok := col.(*entity.ColumnVarChar); ok && i < len(typeCol.Data()) {
					item.Content.Type = storage.ContentType(typeCol.Data()[i])
				}
			case "content_data":
				if dataCol, ok := col.(*entity.ColumnVarChar); ok && i < len(dataCol.Data()) {
					item.Content.Data = []byte(dataCol.Data()[i])
				}
			case "metadata":
				if metaCol, ok := col.(*entity.ColumnVarChar); ok && i < len(metaCol.Data()) {
					item.Metadata = decodeMetadata(metaCol.Data()[i])
				}
			case "created_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && i < len(timeCol.Data()) {
					item.CreatedAt = time.Unix(0, timeCol.Data()[i])
				}
			case "expires_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && i < len(timeCol.Data()) {
					item.ExpiresAt = time.Unix(0, timeCol.Data()[i])
				}
			}
		}

		if !conversionError && item.ID != "" && item.Vector != nil {
			items = append(items, item)
		}
	}

	fmt.Printf("[Milvus:Prefetch] Processed %d items in %v\n",
		len(items), time.Since(processStart))

	// Add to cache
	cacheStart := time.Now()
	for _, item := range items {
		s.localCache.Add(item.ID, item)
	}
	fmt.Printf("[Milvus:Prefetch] Added %d items to cache in %v\n",
		len(items), time.Since(cacheStart))

	fmt.Printf("[Milvus:Prefetch] Total prefetch operation took %v\n", time.Since(start))
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

	// Calculate optimal batch size based on vector dimension
	optimalBatchSize := s.calculateOptimalBatchSize()

	// Process batches
	for i := 0; i < len(items); i += optimalBatchSize {
		end := i + optimalBatchSize
		if end > len(items) {
			end = len(items)
		}
		batchItems := items[i:end]

		err := s.processBatch(ctx, batchItems)
		if err != nil {
			return fmt.Errorf("batch processing error: %w", err)
		}
	}

	return nil
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
			contentTypes[i] = string(item.Content.Type)
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

func max(a, b int) int {
	if a > b {
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

	s.logger.Debug().
		Str("trigger_id", id).
		Strs("predicted_ids", predictedIDs).
		Msg("Prefetching predicted items")

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

// Insert stores vectors with their associated data in batches
func (s *MilvusStore) Insert(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	logger := s.logger.With().Str("operation", "Insert").Int("item_count", len(items)).Logger()
	logger.Debug().Msg("Starting insertion of items")

	// For large batches, process directly
	if len(items) > maxBufferSize {
		return s.processLargeBatch(ctx, items)
	}

	// For smaller batches, use buffer
	s.insertBuffer.Lock()
	defer s.insertBuffer.Unlock()

	// Add items to buffer
	s.insertBuffer.items = append(s.insertBuffer.items, items...)

	// Check if we should flush
	shouldFlush := len(s.insertBuffer.items) >= maxBufferSize ||
		time.Since(s.insertBuffer.lastFlush) >= flushInterval

	if shouldFlush {
		return s.flushBuffer(ctx)
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

// vectorWorkerPool manages parallel vector processing
type vectorWorkerPool struct {
	workers int
	tasks   chan func() error
	results chan error
	wg      sync.WaitGroup
}

func newVectorWorkerPool(workers int) *vectorWorkerPool {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	return &vectorWorkerPool{
		workers: workers,
		tasks:   make(chan func() error, workers*2),
		results: make(chan error, workers*2),
	}
}

func (p *vectorWorkerPool) start() {
	p.wg.Add(p.workers)
	for i := 0; i < p.workers; i++ {
		go func() {
			defer p.wg.Done()
			for task := range p.tasks {
				p.results <- task()
			}
		}()
	}
}

func (p *vectorWorkerPool) stop() {
	close(p.tasks)
	p.wg.Wait()
	close(p.results)
}
