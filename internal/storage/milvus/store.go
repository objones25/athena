package milvus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"runtime"
	"sort"
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
	defaultDimension      = 128
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
		failedOps int64
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

// processBatchEfficient processes a batch of items efficiently
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

	// Build bulk insert data
	vectors := make([][]float32, len(items))
	ids := make([]string, len(items))
	contentTypes := make([]string, len(items))
	contentData := make([]string, len(items))
	metadata := make([]string, len(items))
	createdAt := make([]int64, len(items))
	expiresAt := make([]int64, len(items))

	for i, item := range items {
		vectors[i] = item.Vector
		ids[i] = item.ID
		contentTypes[i] = string(item.Content.Type)
		contentData[i] = string(item.Content.Data)
		metadata[i] = encodeMetadata(item.Metadata)
		createdAt[i] = item.CreatedAt.UnixNano()
		expiresAt[i] = item.ExpiresAt.UnixNano()
	}

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

	// Insert data with retry
	err = s.withRetry(ctx, func(ctx context.Context) error {
		_, err := conn.Insert(ctx, s.collectionName, "", columns...)
		return err
	})

	if err != nil {
		return fmt.Errorf("failed to insert batch: %w", err)
	}

	// Cache the items
	for _, item := range items {
		s.localCache.Add(item.ID, item)
	}

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

func (s *MilvusStore) verifyItem(ctx context.Context, id string) error {
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for verification: %w", err)
	}
	defer s.releaseConnection(conn)

	expr := fmt.Sprintf("id == '%s'", id)
	result, err := conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
	if err != nil {
		return fmt.Errorf("failed to verify item %s: %w", id, err)
	}
	if len(result) == 0 || len(result[0].(*entity.ColumnVarChar).Data()) == 0 {
		return fmt.Errorf("item %s not found after insertion", id)
	}
	return nil
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
	queryResult, err := conn.Query(ctx, s.collectionName, []string{}, expr, outputFields)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	// Process results
	item, err := s.processQueryResult(queryResult)
	if err != nil {
		return nil, fmt.Errorf("failed to process query result: %w", err)
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

func (s *MilvusStore) deleteChunk(ctx context.Context, ids []string) error {
	chunkStart := time.Now()
	fmt.Printf("[Milvus:DeleteChunk] Starting deletion of %d items\n", len(ids))

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
		fmt.Printf("[Milvus:DeleteChunk] Built expression in %v\n", time.Since(exprStart))

		// Delete the records
		deleteStart := time.Now()
		if err := conn.Delete(ctx, s.collection, "", expr); err != nil {
			fmt.Printf("[Milvus:DeleteChunk] Deletion failed after %v: %v\n",
				time.Since(deleteStart), err)
			return fmt.Errorf("failed to delete chunk: %w", err)
		}
		fmt.Printf("[Milvus:DeleteChunk] Deletion executed in %v\n", time.Since(deleteStart))

		// Flush to ensure deletion is persisted
		flushStart := time.Now()
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			fmt.Printf("[Milvus:DeleteChunk] Flush failed after %v: %v\n",
				time.Since(flushStart), err)
			return fmt.Errorf("failed to flush after deletion: %w", err)
		}
		fmt.Printf("[Milvus:DeleteChunk] Flush completed in %v\n", time.Since(flushStart))

		// Remove from cache
		cacheStart := time.Now()
		for _, id := range ids {
			s.localCache.Remove(id)
		}
		fmt.Printf("[Milvus:DeleteChunk] Cache cleanup completed in %v\n", time.Since(cacheStart))

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

// ensureCollection checks and initializes the collection if needed
func (s *MilvusStore) ensureCollection(ctx context.Context) error {
	start := time.Now()
	fmt.Printf("[Milvus:Collection] Checking collection existence...\n")

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	exists, err := conn.HasCollection(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}

	if !exists {
		createStart := time.Now()
		fmt.Printf("[Milvus:Collection] Collection does not exist, creating...\n")

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

		fmt.Printf("[Milvus:Collection] Creating collection with schema...\n")
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

		// Wait for index to be built
		waitStart := time.Now()
		fmt.Printf("[Milvus:Collection] Waiting for index to be built...\n")
		for {
			indexState, err := conn.DescribeIndex(ctx, s.collection, "vector")
			if err != nil {
				return fmt.Errorf("failed to get index state: %w", err)
			}
			if len(indexState) > 0 {
				break
			}
			time.Sleep(time.Second)
		}
		fmt.Printf("[Milvus:Collection] Index built in %v\n", time.Since(waitStart))

		// Load collection into memory
		loadStart := time.Now()
		fmt.Printf("[Milvus:Collection] Loading collection into memory...\n")
		err = conn.LoadCollection(ctx, s.collection, false)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}
		fmt.Printf("[Milvus:Collection] Collection loaded in %v\n", time.Since(loadStart))
	}

	fmt.Printf("[Milvus:Collection] Collection initialization completed in %v\n", time.Since(start))
	return nil
}

// Search performs a similarity search using the provided vector
func (s *MilvusStore) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	start := time.Now()
	fmt.Printf("[Milvus:Search] Starting search with vector dimension %d, limit %d\n", len(vector), limit)

	if len(vector) != s.dimension {
		return nil, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), s.dimension)
	}

	var results []client.SearchResult
	err := s.withRetry(ctx, func(ctx context.Context) error {
		connStart := time.Now()
		conn, err := s.getConnection()
		if err != nil {
			return fmt.Errorf("failed to get connection: %w", err)
		}
		defer s.releaseConnection(conn)
		fmt.Printf("[Milvus:Search] Got connection in %v\n", time.Since(connStart))

		// Use IVF_FLAT search parameters for better performance
		paramStart := time.Now()
		sp, err := entity.NewIndexIvfFlatSearchParam(10) // nprobe=10 for better recall
		if err != nil {
			return fmt.Errorf("failed to create search parameters: %w", err)
		}
		fmt.Printf("[Milvus:Search] Created search parameters in %v\n", time.Since(paramStart))

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

		// Perform search with optimized parameters
		searchStart := time.Now()
		fmt.Printf("[Milvus:Search] Starting Milvus search operation...\n")
		result, err := conn.Search(
			ctx,
			s.collection,
			[]string{}, // No partition
			"",         // No expression filter
			outputFields,
			[]entity.Vector{entity.FloatVector(vector)},
			"vector",
			entity.L2,
			limit,
			sp,
		)
		if err != nil {
			return fmt.Errorf("search operation failed: %w", err)
		}
		fmt.Printf("[Milvus:Search] Search operation completed in %v\n", time.Since(searchStart))

		results = result
		return nil
	})

	if err != nil {
		fmt.Printf("[Milvus:Search] Search failed: %v\n", err)
		return nil, err
	}

	if len(results) == 0 || len(results[0].Fields) == 0 {
		fmt.Printf("[Milvus:Search] No results found\n")
		return []*storage.Item{}, nil
	}

	// Process results efficiently
	processStart := time.Now()
	fmt.Printf("[Milvus:Search] Processing %d results\n", results[0].ResultCount)
	items := make([]*storage.Item, 0, results[0].ResultCount)

	for _, result := range results {
		for i := 0; i < result.ResultCount; i++ {
			item := &storage.Item{
				Content:  storage.Content{},
				Metadata: make(map[string]interface{}),
			}
			var convErr error

			// Extract fields from columns with safe type conversion and debug logging
			for _, col := range result.Fields {
				fmt.Printf("[Milvus:Search] Processing column %s (type: %T) for result %d\n",
					col.Name(), col, i)

				switch col.Name() {
				case "id":
					if idCol, ok := col.(*entity.ColumnVarChar); ok && len(idCol.Data()) > i {
						item.ID = idCol.Data()[i]
						fmt.Printf("[Milvus:Search] Successfully extracted ID: %s\n", item.ID)
					} else {
						convErr = fmt.Errorf("invalid id column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting ID: %v\n", convErr)
					}
				case "vector":
					if vecCol, ok := col.(*entity.ColumnFloatVector); ok && len(vecCol.Data()) > i {
						item.Vector = vecCol.Data()[i]
						fmt.Printf("[Milvus:Search] Successfully extracted vector of length %d\n",
							len(item.Vector))
					} else {
						convErr = fmt.Errorf("invalid vector column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting vector: %v\n", convErr)
					}
				case "content_type":
					if typeCol, ok := col.(*entity.ColumnVarChar); ok && len(typeCol.Data()) > i {
						item.Content.Type = storage.ContentType(typeCol.Data()[i])
						fmt.Printf("[Milvus:Search] Successfully extracted content type: %s\n",
							item.Content.Type)
					} else {
						convErr = fmt.Errorf("invalid content_type column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting content type: %v\n", convErr)
					}
				case "content_data":
					if dataCol, ok := col.(*entity.ColumnVarChar); ok && len(dataCol.Data()) > i {
						item.Content.Data = []byte(dataCol.Data()[i])
						fmt.Printf("[Milvus:Search] Successfully extracted content data of length %d\n",
							len(item.Content.Data))
					} else {
						convErr = fmt.Errorf("invalid content_data column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting content data: %v\n", convErr)
					}
				case "metadata":
					if metaCol, ok := col.(*entity.ColumnVarChar); ok && len(metaCol.Data()) > i {
						item.Metadata = decodeMetadata(metaCol.Data()[i])
						fmt.Printf("[Milvus:Search] Successfully extracted metadata with %d keys\n",
							len(item.Metadata))
					} else {
						convErr = fmt.Errorf("invalid metadata column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting metadata: %v\n", convErr)
					}
				case "created_at":
					if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > i {
						item.CreatedAt = time.Unix(0, timeCol.Data()[i])
						fmt.Printf("[Milvus:Search] Successfully extracted created_at: %v\n",
							item.CreatedAt)
					} else {
						convErr = fmt.Errorf("invalid created_at column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting created_at: %v\n", convErr)
					}
				case "expires_at":
					if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > i {
						item.ExpiresAt = time.Unix(0, timeCol.Data()[i])
						fmt.Printf("[Milvus:Search] Successfully extracted expires_at: %v\n",
							item.ExpiresAt)
					} else {
						convErr = fmt.Errorf("invalid expires_at column type or index: %T", col)
						fmt.Printf("[Milvus:Search] Error extracting expires_at: %v\n", convErr)
					}
				}

				if convErr != nil {
					return nil, fmt.Errorf("field conversion error at index %d: %w", i, convErr)
				}
			}

			// Validate required fields
			if item.ID == "" || item.Vector == nil {
				fmt.Printf("[Milvus:Search] Skipping invalid item (ID empty: %v, Vector nil: %v)\n",
					item.ID == "", item.Vector == nil)
				continue // Skip invalid items
			}

			items = append(items, item)
		}
	}

	fmt.Printf("[Milvus:Search] Results processed in %v\n", time.Since(processStart))
	fmt.Printf("[Milvus:Search] Total search operation took %v\n", time.Since(start))
	return items, nil
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
	fmt.Printf("[Milvus:LargeBatch] Starting large batch processing of %d items\n", totalItems)

	const optimalBatchSize = 2000
	batches := (totalItems + optimalBatchSize - 1) / optimalBatchSize

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Process all items in batches
	for i := 0; i < totalItems; i += optimalBatchSize {
		batchStart := time.Now()
		end := i + optimalBatchSize
		if end > totalItems {
			end = totalItems
		}
		batch := items[i:end]
		batchSize := len(batch)
		fmt.Printf("[Milvus:LargeBatch] Processing batch %d/%d (%d items)\n",
			(i/optimalBatchSize)+1, batches, batchSize)

		// Pre-allocate all slices for the batch
		ids := make([]string, batchSize)
		vectors := make([][]float32, batchSize)
		contentTypes := make([]string, batchSize)
		contentData := make([]string, batchSize)
		metadata := make([]string, batchSize)
		createdAt := make([]int64, batchSize)
		expiresAt := make([]int64, batchSize)

		// Process items in parallel
		processStart := time.Now()
		fmt.Printf("[Milvus:LargeBatch] Starting concurrent item processing for batch %d\n",
			(i/optimalBatchSize)+1)
		errChan := make(chan error, runtime.NumCPU())
		sem := make(chan struct{}, runtime.NumCPU())

		for j := range batch {
			sem <- struct{}{} // Acquire semaphore
			go func(j int) {
				defer func() { <-sem }() // Release semaphore

				if err := ctx.Err(); err != nil {
					errChan <- fmt.Errorf("context cancelled during processing: %w", err)
					return
				}

				ids[j] = batch[j].ID
				vectors[j] = batch[j].Vector
				contentTypes[j] = string(batch[j].Content.Type)
				contentData[j] = string(batch[j].Content.Data)
				metadata[j] = encodeMetadata(batch[j].Metadata)
				createdAt[j] = batch[j].CreatedAt.UnixNano()
				expiresAt[j] = batch[j].ExpiresAt.UnixNano()

				errChan <- nil
			}(j)
		}

		// Wait for all goroutines and check for errors
		for j := 0; j < len(batch); j++ {
			if err := <-errChan; err != nil {
				return err
			}
		}
		fmt.Printf("[Milvus:LargeBatch] Batch %d item processing completed in %v\n",
			(i/optimalBatchSize)+1, time.Since(processStart))

		// Create columns
		columnStart := time.Now()
		fmt.Printf("[Milvus:LargeBatch] Creating column data for batch %d\n",
			(i/optimalBatchSize)+1)
		idCol := entity.NewColumnVarChar("id", ids)
		vectorCol := entity.NewColumnFloatVector("vector", s.dimension, vectors)
		contentTypeCol := entity.NewColumnVarChar("content_type", contentTypes)
		contentDataCol := entity.NewColumnVarChar("content_data", contentData)
		metadataCol := entity.NewColumnVarChar("metadata", metadata)
		createdAtCol := entity.NewColumnInt64("created_at", createdAt)
		expiresAtCol := entity.NewColumnInt64("expires_at", expiresAt)
		fmt.Printf("[Milvus:LargeBatch] Column data created in %v\n", time.Since(columnStart))

		// Insert data
		insertStart := time.Now()
		fmt.Printf("[Milvus:LargeBatch] Starting insert for batch %d\n",
			(i/optimalBatchSize)+1)
		_, err = conn.Insert(ctx, s.collection, "", idCol, vectorCol, contentTypeCol,
			contentDataCol, metadataCol, createdAtCol, expiresAtCol)
		if err != nil {
			fmt.Printf("[Milvus:LargeBatch] Batch %d insert failed after %v: %v\n",
				(i/optimalBatchSize)+1, time.Since(insertStart), err)
			return fmt.Errorf("failed to insert batch: %w", err)
		}
		fmt.Printf("[Milvus:LargeBatch] Batch %d inserted in %v\n",
			(i/optimalBatchSize)+1, time.Since(insertStart))

		fmt.Printf("[Milvus:LargeBatch] Batch %d completed in %v\n",
			(i/optimalBatchSize)+1, time.Since(batchStart))
	}

	// Flush after all batches are inserted
	flushStart := time.Now()
	fmt.Printf("[Milvus:LargeBatch] Starting final flush\n")
	if err := conn.Flush(ctx, s.collection, false); err != nil {
		fmt.Printf("[Milvus:LargeBatch] Final flush failed after %v: %v\n",
			time.Since(flushStart), err)
		return fmt.Errorf("failed to flush after insertion: %w", err)
	}
	fmt.Printf("[Milvus:LargeBatch] Final flush completed in %v\n", time.Since(flushStart))

	// Verify a sample of items
	if totalItems > 10 {
		sampleSize := 5
		sampleIndices := make([]int, sampleSize)
		step := totalItems / sampleSize
		for i := range sampleIndices {
			sampleIndices[i] = i * step
		}

		verifyStart := time.Now()
		fmt.Printf("[Milvus:LargeBatch] Starting sample verification of %d items\n", sampleSize)
		for _, idx := range sampleIndices {
			if err := s.verifyItem(ctx, items[idx].ID); err != nil {
				fmt.Printf("[Milvus:LargeBatch] Sample verification failed after %v: %v\n",
					time.Since(verifyStart), err)
				return fmt.Errorf("sample verification failed: %w", err)
			}
		}
		fmt.Printf("[Milvus:LargeBatch] Sample verification completed in %v\n",
			time.Since(verifyStart))
	}

	fmt.Printf("[Milvus:LargeBatch] Total large batch processing took %v\n", time.Since(start))
	return nil
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

// BatchSet processes a batch of items efficiently
func (s *MilvusStore) BatchSet(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	fmt.Printf("[Milvus:BatchSet] Starting batch operation with %d items\n", len(items))
	start := time.Now()

	// Create columns for batch insert
	idColumn := make([]string, len(items))
	vectorColumn := make([][]float32, len(items))
	contentTypeColumn := make([]string, len(items))
	contentDataColumn := make([]string, len(items))
	metadataColumn := make([]string, len(items))
	createdAtColumn := make([]int64, len(items))
	expiresAtColumn := make([]int64, len(items))

	// Fill columns
	for i, item := range items {
		idColumn[i] = item.ID
		vectorColumn[i] = item.Vector
		contentTypeColumn[i] = string(item.Content.Type)
		contentDataColumn[i] = string(item.Content.Data)
		metadataColumn[i] = encodeMetadata(item.Metadata)
		createdAtColumn[i] = item.CreatedAt.UnixNano()
		expiresAtColumn[i] = item.ExpiresAt.UnixNano()
	}

	// Get connection from pool
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for batch insert: %w", err)
	}
	defer s.releaseConnection(conn)

	// Create insert columns
	insertColumns := []entity.Column{
		entity.NewColumnVarChar("id", idColumn),
		entity.NewColumnFloatVector("vector", s.dimension, vectorColumn),
		entity.NewColumnVarChar("content_type", contentTypeColumn),
		entity.NewColumnVarChar("content_data", contentDataColumn),
		entity.NewColumnVarChar("metadata", metadataColumn),
		entity.NewColumnInt64("created_at", createdAtColumn),
		entity.NewColumnInt64("expires_at", expiresAtColumn),
	}

	// Execute insert
	_, err = conn.Insert(ctx, s.collection, "", insertColumns...)
	if err != nil {
		return fmt.Errorf("failed to insert batch: %w", err)
	}

	fmt.Printf("[Milvus:BatchSet] Batch operation completed in %v\n", time.Since(start))
	return nil
}

// processQueryResult converts Milvus query results into a storage.Item
func (s *MilvusStore) processQueryResult(result []entity.Column) (*storage.Item, error) {
	if len(result) == 0 {
		return nil, nil
	}

	item := &storage.Item{
		Content:  storage.Content{},
		Metadata: make(map[string]interface{}),
	}
	foundData := false

	for _, col := range result {
		switch col.Name() {
		case "id":
			if idCol, ok := col.(*entity.ColumnVarChar); ok && len(idCol.Data()) > 0 {
				item.ID = idCol.Data()[0]
				foundData = true
			}
		case "vector":
			if vecCol, ok := col.(*entity.ColumnFloatVector); ok && len(vecCol.Data()) > 0 {
				item.Vector = vecCol.Data()[0]
			}
		case "content_type":
			if typeCol, ok := col.(*entity.ColumnVarChar); ok && len(typeCol.Data()) > 0 {
				item.Content.Type = storage.ContentType(typeCol.Data()[0])
			}
		case "content_data":
			if dataCol, ok := col.(*entity.ColumnVarChar); ok && len(dataCol.Data()) > 0 {
				item.Content.Data = []byte(dataCol.Data()[0])
			}
		case "metadata":
			if metaCol, ok := col.(*entity.ColumnVarChar); ok && len(metaCol.Data()) > 0 {
				item.Metadata = decodeMetadata(metaCol.Data()[0])
			}
		case "created_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.CreatedAt = time.Unix(0, timeCol.Data()[0])
			}
		case "expires_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.ExpiresAt = time.Unix(0, timeCol.Data()[0])
			}
		}
	}

	if !foundData || item.ID == "" {
		return nil, nil
	}

	return item, nil
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
