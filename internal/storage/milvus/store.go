package milvus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

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
	defaultDimension = 128
	defaultTimeout   = 30 * time.Second
	batchSize        = 2000
	maxConnections   = 10
	maxPoolSize      = 30
	connTimeout      = 10 * time.Second
	flushInterval    = 5 * time.Second
	maxBufferSize    = 5000
	prefetchSize     = 100
	cacheWarmupSize  = 1000
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
}

// connectionPool manages a pool of Milvus connections
type connectionPool struct {
	connections chan *pooledConnection
	maxSize     int
	metrics     struct {
		inUse      int64
		failed     int64
		reconnects int64
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
		cfg.BatchSize = batchSize
	}
	if cfg.PoolSize <= 0 {
		cfg.PoolSize = maxConnections
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = 3
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
		timeout:        defaultTimeout,
		workQueue:      newWorkQueue(cfg.PoolSize, cfg.PoolSize*100),
		insertBuffer: struct {
			sync.Mutex
			items     []*storage.Item
			lastFlush time.Time
		}{
			items:     make([]*storage.Item, 0, maxBufferSize),
			lastFlush: time.Now(),
		},
	}

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
	fmt.Printf("[Milvus:Pool] Attempting to get connection from pool (size: %d, in use: %d)\n",
		s.pool.maxSize, atomic.LoadInt64(&s.pool.metrics.inUse))

	// Try to get connection with exponential backoff
	for attempt := 0; attempt < 3; attempt++ {
		select {
		case pc := <-s.pool.connections:
			atomic.AddInt64(&s.pool.metrics.inUse, 1)
			pc.inUse = true
			pc.lastUsed = time.Now()

			// Only check health if connection hasn't been used recently
			if time.Since(pc.lastUsed) > 30*time.Second {
				if err := s.checkConnection(context.Background(), pc.client); err != nil {
					pc.failures++
					if pc.failures > 3 {
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

			fmt.Printf("[Milvus:Pool] Got connection in %v\n", time.Since(start))
			return pc.client, nil

		case <-time.After(time.Duration(attempt*50) * time.Millisecond):
			continue
		}
	}

	// Create new connection if pool is exhausted
	fmt.Printf("[Milvus:Pool] Pool exhausted, creating new connection\n")
	return s.createConnection()
}

// releaseConnection returns a connection to the pool with improved handling
func (s *MilvusStore) releaseConnection(conn client.Client) {
	start := time.Now()
	fmt.Printf("[Milvus:Pool] Attempting to release connection\n")

	select {
	case s.pool.connections <- &pooledConnection{
		client:   conn,
		lastUsed: time.Now(),
	}:
		atomic.AddInt64(&s.pool.metrics.inUse, -1)
		fmt.Printf("[Milvus:Pool] Connection released successfully in %v\n", time.Since(start))
	default:
		// Pool is full, close the connection
		fmt.Printf("[Milvus:Pool] Pool is full, closing connection\n")
		conn.Close()
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

// processBatchWithQueue processes a batch of items using the work queue
func (s *MilvusStore) processBatchWithQueue(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	// Process in smaller chunks for better efficiency
	chunkSize := s.batchSize
	if chunkSize > 100 {
		chunkSize = 100 // Cap chunk size for better parallelism
	}

	chunks := (len(items) + chunkSize - 1) / chunkSize
	errChan := make(chan error, chunks)

	// Submit chunks to work queue
	for i := 0; i < len(items); i += chunkSize {
		end := i + chunkSize
		if end > len(items) {
			end = len(items)
		}

		chunk := items[i:end]
		if err := s.workQueue.Submit(func() error {
			return s.processBatch(ctx, chunk)
		}); err != nil {
			return fmt.Errorf("failed to submit batch: %w", err)
		}
	}

	// Collect errors
	var errs []error
	for i := 0; i < chunks; i++ {
		if err := <-errChan; err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("batch processing errors: %v", errs)
	}
	return nil
}

// processBatch handles a batch of items efficiently
func (s *MilvusStore) processBatch(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	fmt.Printf("[Milvus:Batch] Starting efficient batch processing of %d items\n", len(items))
	start := time.Now()

	// Process items in parallel
	fmt.Printf("[Milvus:Batch] Starting parallel processing\n")
	parallelStart := time.Now()

	// Create columns for batch insert
	idColumn := make([]string, len(items))
	vectorColumn := make([][]float32, len(items))
	contentTypeColumn := make([]string, len(items))
	contentDataColumn := make([]string, len(items))
	metadataColumn := make([]string, len(items))
	createdAtColumn := make([]int64, len(items))
	expiresAtColumn := make([]int64, len(items))

	// Fill columns in parallel using worker pool
	var wg sync.WaitGroup
	workerCount := runtime.NumCPU()
	itemsPerWorker := (len(items) + workerCount - 1) / workerCount

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			start := workerID * itemsPerWorker
			end := start + itemsPerWorker
			if end > len(items) {
				end = len(items)
			}
			for j := start; j < end; j++ {
				item := items[j]
				idColumn[j] = item.ID
				vectorColumn[j] = item.Vector
				contentTypeColumn[j] = string(item.Content.Type)
				contentDataColumn[j] = string(item.Content.Data)
				metadataColumn[j] = encodeMetadata(item.Metadata)
				createdAtColumn[j] = item.CreatedAt.UnixNano()
				expiresAtColumn[j] = item.ExpiresAt.UnixNano()
			}
		}(i)
	}
	wg.Wait()

	fmt.Printf("[Milvus:Batch] Parallel processing completed in %v\n", time.Since(parallelStart))

	// Create insert columns
	columnStart := time.Now()
	insertColumns := []entity.Column{
		entity.NewColumnVarChar("id", idColumn),
		entity.NewColumnFloatVector("vector", s.dimension, vectorColumn),
		entity.NewColumnVarChar("content_type", contentTypeColumn),
		entity.NewColumnVarChar("content_data", contentDataColumn),
		entity.NewColumnVarChar("metadata", metadataColumn),
		entity.NewColumnInt64("created_at", createdAtColumn),
		entity.NewColumnInt64("expires_at", expiresAtColumn),
	}
	fmt.Printf("[Milvus:Batch] Columns created in %v\n", time.Since(columnStart))

	// Get connection from pool
	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for batch processing: %w", err)
	}
	defer s.releaseConnection(conn)

	// Execute insert
	fmt.Printf("[Milvus:Batch] Starting insert operation\n")
	insertStart := time.Now()
	_, err = conn.Insert(ctx, s.collection, "", insertColumns...)
	if err != nil {
		return fmt.Errorf("failed to insert batch: %w", err)
	}
	fmt.Printf("[Milvus:Batch] Insert completed in %v\n", time.Since(insertStart))

	fmt.Printf("[Milvus:Batch] Total batch processing took %v\n", time.Since(start))
	return nil
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

// processBatchEfficient is an optimized version of processBatch
func (s *MilvusStore) processBatchEfficient(ctx context.Context, items []*storage.Item) error {
	start := time.Now()
	fmt.Printf("[Milvus:Batch] Starting efficient batch processing of %d items\n", len(items))

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.releaseConnection(conn)

	// Pre-allocate slices with exact capacity
	batchSize := len(items)
	ids := make([]string, batchSize)
	vectors := make([][]float32, batchSize)
	contentTypes := make([]string, batchSize)
	contentData := make([]string, batchSize)
	metadata := make([]string, batchSize)
	createdAt := make([]int64, batchSize)
	expiresAt := make([]int64, batchSize)

	// Process items in parallel with worker pool
	processStart := time.Now()
	fmt.Printf("[Milvus:Batch] Starting parallel processing\n")

	workers := runtime.NumCPU()
	var wg sync.WaitGroup
	errChan := make(chan error, workers)
	itemChan := make(chan int, batchSize)

	// Start worker pool
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range itemChan {
				item := items[i]
				ids[i] = item.ID
				vectors[i] = item.Vector
				contentTypes[i] = string(item.Content.Type)
				contentData[i] = string(item.Content.Data)
				metadata[i] = encodeMetadata(item.Metadata)
				createdAt[i] = item.CreatedAt.UnixNano()
				expiresAt[i] = item.ExpiresAt.UnixNano()
			}
		}()
	}

	// Feed items to workers
	for i := range items {
		itemChan <- i
	}
	close(itemChan)

	// Wait for processing to complete
	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			return err
		}
	}
	fmt.Printf("[Milvus:Batch] Parallel processing completed in %v\n", time.Since(processStart))

	// Create columns efficiently
	columnStart := time.Now()
	columns := []entity.Column{
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnFloatVector("vector", s.dimension, vectors),
		entity.NewColumnVarChar("content_type", contentTypes),
		entity.NewColumnVarChar("content_data", contentData),
		entity.NewColumnVarChar("metadata", metadata),
		entity.NewColumnInt64("created_at", createdAt),
		entity.NewColumnInt64("expires_at", expiresAt),
	}
	fmt.Printf("[Milvus:Batch] Columns created in %v\n", time.Since(columnStart))

	// Insert with optimized retry logic
	insertStart := time.Now()
	fmt.Printf("[Milvus:Batch] Starting insert operation\n")
	err = s.withRetry(ctx, func(ctx context.Context) error {
		if _, err := conn.Insert(ctx, s.collection, "", columns...); err != nil {
			return fmt.Errorf("insert failed: %w", err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("batch insert failed: %w", err)
	}
	fmt.Printf("[Milvus:Batch] Insert completed in %v\n", time.Since(insertStart))

	// Only flush if enough time has passed since last flush
	if time.Since(s.insertBuffer.lastFlush) >= flushInterval {
		flushStart := time.Now()
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			return fmt.Errorf("flush failed: %w", err)
		}
		fmt.Printf("[Milvus:Batch] Flush completed in %v\n", time.Since(flushStart))
		s.insertBuffer.lastFlush = time.Now()
	}

	fmt.Printf("[Milvus:Batch] Total batch processing took %v\n", time.Since(start))
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
	start := time.Now()
	fmt.Printf("[Milvus:GetByID] Starting retrieval for ID: %s\n", id)

	// Try to prefetch nearby IDs
	if err := s.prefetchItems(ctx, []string{id}); err != nil {
		fmt.Printf("[Milvus:GetByID] Prefetch failed: %v\n", err)
	}

	conn, err := s.getConnection()
	if err != nil {
		fmt.Printf("[Milvus:GetByID] Failed to get connection: %v\n", err)
		return nil, err
	}
	defer s.releaseConnection(conn)

	// Prepare the query
	expr := fmt.Sprintf(`id == '%s'`, id)
	outputFields := []string{"id", "vector", "content_type", "content_data", "metadata", "created_at", "expires_at"}

	queryStart := time.Now()
	fmt.Printf("[Milvus:GetByID] Executing query with expression: %s\n", expr)

	// Execute query with timeout
	queryCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	result, err := conn.Query(
		queryCtx,
		s.collection,
		[]string{},
		expr,
		outputFields,
	)
	fmt.Printf("[Milvus:GetByID] Query execution took %v\n", time.Since(queryStart))

	if err != nil {
		fmt.Printf("[Milvus:GetByID] Query failed: %v\n", err)
		return nil, err
	}

	// No results found
	if len(result) == 0 {
		fmt.Printf("[Milvus:GetByID] No results found for ID: %s\n", id)
		return nil, nil
	}

	// Process the first result
	processStart := time.Now()
	fmt.Printf("[Milvus:GetByID] Processing result columns: %d\n", len(result))

	item := &storage.Item{
		Content:  storage.Content{},
		Metadata: make(map[string]interface{}),
	}
	foundData := false

	for _, col := range result {
		fmt.Printf("[Milvus:GetByID] Processing column %s (type: %T)\n", col.Name(), col)

		switch col.Name() {
		case "id":
			if idCol, ok := col.(*entity.ColumnVarChar); ok && len(idCol.Data()) > 0 {
				item.ID = idCol.Data()[0]
				foundData = true
				fmt.Printf("[Milvus:GetByID] Successfully extracted ID: %s\n", item.ID)
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract ID, type: %T, has data: %v\n",
					col, ok && len(idCol.Data()) > 0)
			}
		case "vector":
			if vecCol, ok := col.(*entity.ColumnFloatVector); ok && len(vecCol.Data()) > 0 {
				item.Vector = vecCol.Data()[0]
				fmt.Printf("[Milvus:GetByID] Successfully extracted vector of length %d\n",
					len(item.Vector))
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract vector, type: %T\n", col)
			}
		case "content_type":
			if typeCol, ok := col.(*entity.ColumnVarChar); ok && len(typeCol.Data()) > 0 {
				item.Content.Type = storage.ContentType(typeCol.Data()[0])
				fmt.Printf("[Milvus:GetByID] Successfully extracted content type: %s\n",
					item.Content.Type)
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract content type, type: %T\n", col)
			}
		case "content_data":
			if dataCol, ok := col.(*entity.ColumnVarChar); ok && len(dataCol.Data()) > 0 {
				item.Content.Data = []byte(dataCol.Data()[0])
				fmt.Printf("[Milvus:GetByID] Successfully extracted content data of length %d\n",
					len(item.Content.Data))
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract content data, type: %T\n", col)
			}
		case "metadata":
			if metaCol, ok := col.(*entity.ColumnVarChar); ok && len(metaCol.Data()) > 0 {
				item.Metadata = decodeMetadata(metaCol.Data()[0])
				fmt.Printf("[Milvus:GetByID] Successfully extracted metadata with %d keys\n",
					len(item.Metadata))
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract metadata, type: %T\n", col)
			}
		case "created_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.CreatedAt = time.Unix(0, timeCol.Data()[0])
				fmt.Printf("[Milvus:GetByID] Successfully extracted created_at: %v\n",
					item.CreatedAt)
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract created_at, type: %T\n", col)
			}
		case "expires_at":
			if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > 0 {
				item.ExpiresAt = time.Unix(0, timeCol.Data()[0])
				fmt.Printf("[Milvus:GetByID] Successfully extracted expires_at: %v\n",
					item.ExpiresAt)
			} else {
				fmt.Printf("[Milvus:GetByID] Failed to extract expires_at, type: %T\n", col)
			}
		}
	}

	fmt.Printf("[Milvus:GetByID] Result processing took %v\n", time.Since(processStart))
	fmt.Printf("[Milvus:GetByID] Total operation took %v\n", time.Since(start))

	// If no actual data was found, return nil
	if !foundData || item.ID == "" {
		fmt.Printf("[Milvus:GetByID] No valid data found for ID: %s\n", id)
		return nil, nil
	}

	return item, nil
}

// DeleteFromStore removes items by their IDs in batches
func (s *MilvusStore) DeleteFromStore(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	fmt.Printf("[Milvus:Delete] Starting deletion of %d items\n", len(ids))

	// Process deletions in parallel for large batches
	if len(ids) > s.batchSize {
		chunks := (len(ids) + s.batchSize - 1) / s.batchSize
		errChan := make(chan error, chunks)

		for i := 0; i < len(ids); i += s.batchSize {
			end := i + s.batchSize
			if end > len(ids) {
				end = len(ids)
			}

			chunk := ids[i:end]
			s.workQueue.Submit(func() error {
				return s.deleteChunk(ctx, chunk)
			})
		}

		// Wait for all deletions to complete
		var errs []error
		for i := 0; i < chunks; i++ {
			if err := <-errChan; err != nil {
				errs = append(errs, err)
			}
		}

		if len(errs) > 0 {
			return fmt.Errorf("batch deletion errors: %v", errs)
		}
		return nil
	}

	return s.deleteChunk(ctx, ids)
}

func (s *MilvusStore) deleteChunk(ctx context.Context, ids []string) error {
	return s.withRetry(ctx, func(ctx context.Context) error {
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

		// Delete the records
		if err := conn.Delete(ctx, s.collection, "", expr); err != nil {
			return fmt.Errorf("failed to delete chunk: %w", err)
		}

		// Flush to ensure deletion is persisted
		if err := conn.Flush(ctx, s.collection, false); err != nil {
			return fmt.Errorf("failed to flush after deletion: %w", err)
		}

		// Verify deletion
		if err := s.verifyDeletion(ctx, conn, expr); err != nil {
			return fmt.Errorf("deletion verification failed: %w", err)
		}

		return nil
	})
}

func (s *MilvusStore) verifyDeletion(ctx context.Context, conn client.Client, expr string) error {
	result, err := conn.Query(ctx, s.collection, []string{}, expr, []string{"id"})
	if err != nil {
		return fmt.Errorf("failed to verify deletion: %w", err)
	}

	if len(result) > 0 && len(result[0].(*entity.ColumnVarChar).Data()) > 0 {
		return fmt.Errorf("%d items still exist", len(result[0].(*entity.ColumnVarChar).Data()))
	}
	return nil
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

// Add cache warming functionality
func (s *MilvusStore) warmCache(ctx context.Context) error {
	fmt.Printf("[Milvus:Cache] Starting cache warmup\n")
	start := time.Now()

	conn, err := s.getConnection()
	if err != nil {
		return fmt.Errorf("failed to get connection for cache warmup: %w", err)
	}
	defer s.releaseConnection(conn)

	// Get most recently accessed items
	expr := ""
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

	// Process results and add to cache
	items := make([]*storage.Item, 0, len(result))
	for i := 0; i < len(result[0].(*entity.ColumnVarChar).Data()); i++ {
		item := &storage.Item{
			Content:  storage.Content{},
			Metadata: make(map[string]interface{}),
		}

		// Extract fields from columns
		for _, col := range result {
			switch col.Name() {
			case "id":
				if idCol, ok := col.(*entity.ColumnVarChar); ok {
					item.ID = idCol.Data()[i]
				}
			case "vector":
				if vecCol, ok := col.(*entity.ColumnFloatVector); ok {
					item.Vector = vecCol.Data()[i]
				}
			case "content_type":
				if typeCol, ok := col.(*entity.ColumnVarChar); ok {
					item.Content.Type = storage.ContentType(typeCol.Data()[i])
				}
			case "content_data":
				if dataCol, ok := col.(*entity.ColumnVarChar); ok {
					item.Content.Data = []byte(dataCol.Data()[i])
				}
			case "metadata":
				if metaCol, ok := col.(*entity.ColumnVarChar); ok {
					item.Metadata = decodeMetadata(metaCol.Data()[i])
				}
			case "created_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok {
					item.CreatedAt = time.Unix(0, timeCol.Data()[i])
				}
			case "expires_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok {
					item.ExpiresAt = time.Unix(0, timeCol.Data()[i])
				}
			}
		}

		if item.ID != "" && item.Vector != nil {
			items = append(items, item)
		}
	}

	fmt.Printf("[Milvus:Cache] Warmed up cache with %d items in %v\n", len(items), time.Since(start))
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
		fmt.Printf("[Milvus:Prefetch] No results found\n")
		return nil
	}

	// Process results and add to cache
	items := make([]*storage.Item, 0)
	rowCount := 0
	for _, col := range result {
		switch col.Name() {
		case "id":
			if idCol, ok := col.(*entity.ColumnVarChar); ok {
				rowCount = len(idCol.Data())
			}
		}
	}

	if rowCount == 0 {
		fmt.Printf("[Milvus:Prefetch] No rows found\n")
		return nil
	}

	for i := 0; i < rowCount; i++ {
		item := &storage.Item{
			Content:  storage.Content{},
			Metadata: make(map[string]interface{}),
		}

		// Extract fields from columns
		for _, col := range result {
			switch col.Name() {
			case "id":
				if idCol, ok := col.(*entity.ColumnVarChar); ok && len(idCol.Data()) > i {
					item.ID = idCol.Data()[i]
				}
			case "vector":
				if vecCol, ok := col.(*entity.ColumnFloatVector); ok && len(vecCol.Data()) > i {
					item.Vector = vecCol.Data()[i]
				}
			case "content_type":
				if typeCol, ok := col.(*entity.ColumnVarChar); ok && len(typeCol.Data()) > i {
					item.Content.Type = storage.ContentType(typeCol.Data()[i])
				}
			case "content_data":
				if dataCol, ok := col.(*entity.ColumnVarChar); ok && len(dataCol.Data()) > i {
					item.Content.Data = []byte(dataCol.Data()[i])
				}
			case "metadata":
				if metaCol, ok := col.(*entity.ColumnVarChar); ok && len(metaCol.Data()) > i {
					item.Metadata = decodeMetadata(metaCol.Data()[i])
				}
			case "created_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > i {
					item.CreatedAt = time.Unix(0, timeCol.Data()[i])
				}
			case "expires_at":
				if timeCol, ok := col.(*entity.ColumnInt64); ok && len(timeCol.Data()) > i {
					item.ExpiresAt = time.Unix(0, timeCol.Data()[i])
				}
			}
		}

		if item.ID != "" && item.Vector != nil {
			items = append(items, item)
		}
	}

	fmt.Printf("[Milvus:Prefetch] Prefetched %d items in %v\n", len(items), time.Since(start))
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
