package manager

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/cache"
	"github.com/objones25/athena/internal/storage/milvus"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Metrics
	cacheHits = promauto.NewCounter(prometheus.CounterOpts{
		Name: "storage_cache_hits_total",
		Help: "The total number of cache hits",
	})
	cacheMisses = promauto.NewCounter(prometheus.CounterOpts{
		Name: "storage_cache_misses_total",
		Help: "The total number of cache misses",
	})
	operationDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "storage_operation_duration_seconds",
		Help:    "Duration of storage operations",
		Buckets: prometheus.DefBuckets,
	}, []string{"operation"})
)

// Config holds configuration for the storage manager
type Config struct {
	Cache struct {
		Config cache.Config
		// Cache warming configuration
		WarmupInterval time.Duration
		WarmupQueries  []string
	}
	VectorStore struct {
		Config milvus.Config
	}
	// Circuit breaker configuration
	MaxRetries     int
	RetryInterval  time.Duration
	BreakDuration  time.Duration
	HealthInterval time.Duration
}

// Manager orchestrates interactions between the cache and vector store
type Manager struct {
	cache       *cache.RedisCache
	vectorStore *milvus.MilvusStore
	config      Config
	mu          sync.RWMutex
	metrics     struct {
		lastError    time.Time
		errorCount   int
		circuitOpen  bool
		healthStatus bool
		lastHealthy  time.Time
	}
	stopChan   chan struct{}
	warmupChan chan struct{}
}

// NewManager creates a new storage manager with the provided configuration
func NewManager(cfg Config) (*Manager, error) {
	fmt.Printf("Initializing storage manager with config: %+v\n", cfg)

	fmt.Printf("Creating Redis cache...\n")
	cache, err := cache.NewRedisCache(cfg.Cache.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}
	fmt.Printf("Redis cache created successfully\n")

	fmt.Printf("Creating Milvus vector store...\n")
	vectorStore, err := milvus.NewMilvusStore(cfg.VectorStore.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}
	fmt.Printf("Milvus vector store created successfully\n")

	m := &Manager{
		cache:       cache,
		vectorStore: vectorStore,
		config:      cfg,
		stopChan:    make(chan struct{}),
		warmupChan:  make(chan struct{}, 1),
	}

	fmt.Printf("Starting background tasks...\n")
	// Start background tasks
	go m.healthCheck()
	go m.cacheWarmer()
	fmt.Printf("Background tasks started\n")

	fmt.Printf("Storage manager initialized successfully\n")
	return m, nil
}

// BatchSet stores multiple items efficiently
func (m *Manager) BatchSet(ctx context.Context, items []*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	start := time.Now()
	fmt.Printf("[BatchSet] Starting batch operation with %d items\n", len(items))

	// Store in vector store first
	timer := prometheus.NewTimer(operationDuration.WithLabelValues("batch_set"))
	defer timer.ObserveDuration()

	// Vector store operation
	vectorChan := make(chan error, 1)
	go func() {
		vectorStart := time.Now()
		err := m.vectorStore.Insert(ctx, items)
		fmt.Printf("[BatchSet] Vector store insertion took %v\n", time.Since(vectorStart))
		vectorChan <- err
	}()

	// Cache operations in parallel
	cacheChan := make(chan error, 1)
	go func() {
		cacheStart := time.Now()
		// Convert items to map for batch operation
		itemMap := make(map[string]*storage.Item, len(items))
		for _, item := range items {
			itemMap[item.ID] = item
		}
		err := m.cache.BatchSet(ctx, itemMap)
		fmt.Printf("[BatchSet] Cache operations took %v\n", time.Since(cacheStart))
		cacheChan <- err
	}()

	// Wait for both operations
	select {
	case err := <-vectorChan:
		if err != nil {
			fmt.Printf("[BatchSet] Vector store insertion failed: %v\n", err)
			return fmt.Errorf("failed to store items in vector store: %w", err)
		}
	case <-ctx.Done():
		return ctx.Err()
	}

	// Cache errors are non-fatal
	select {
	case err := <-cacheChan:
		if err != nil {
			fmt.Printf("[BatchSet] Cache batch operation failed: %v\n", err)
		}
	case <-ctx.Done():
		return ctx.Err()
	}

	fmt.Printf("[BatchSet] Total operation took %v\n", time.Since(start))
	return nil
}

// Set stores an item in both the cache and vector store
func (m *Manager) Set(ctx context.Context, key string, item *storage.Item) error {
	return m.BatchSet(ctx, []*storage.Item{item})
}

// Get retrieves an item by its key
func (m *Manager) Get(ctx context.Context, key string) (*storage.Item, error) {
	start := time.Now()
	fmt.Printf("[Get] Starting retrieval for key: %s\n", key)

	timer := prometheus.NewTimer(operationDuration.WithLabelValues("get"))
	defer timer.ObserveDuration()

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Try cache first
	cacheStart := time.Now()
	item, err := m.cache.Get(ctx, key)
	cacheDuration := time.Since(cacheStart)

	if err == nil && item != nil {
		cacheHits.Inc()
		fmt.Printf("[Get] Cache hit for key %s, took %v\n", key, cacheDuration)
		// Convert metadata types
		if item.Metadata != nil {
			for k, v := range item.Metadata {
				if f, ok := v.(float64); ok && f == float64(int64(f)) {
					item.Metadata[k] = int(f)
				}
			}
		}
		return item, nil
	}
	cacheMisses.Inc()
	fmt.Printf("[Get] Cache miss for key %s, took %v\n", key, cacheDuration)

	// Cache miss, try vector store
	vectorStart := time.Now()
	item, err = m.vectorStore.GetByID(ctx, key)
	fmt.Printf("[Get] Vector store lookup took %v\n", time.Since(vectorStart))

	if err != nil {
		fmt.Printf("[Get] Vector store error: %v\n", err)
		return nil, fmt.Errorf("failed to get item from vector store: %w", err)
	}
	if item == nil {
		fmt.Printf("[Get] Item not found in vector store\n")
		return nil, nil
	}

	// Convert metadata types before caching
	if item.Metadata != nil {
		for k, v := range item.Metadata {
			if f, ok := v.(float64); ok && f == float64(int64(f)) {
				item.Metadata[k] = int(f)
			}
		}
	}

	// Cache the item for future requests
	cacheStart = time.Now()
	if err := m.cache.Set(ctx, key, item); err != nil {
		fmt.Printf("[Get] Failed to cache item %s: %v\n", key, err)
	}
	fmt.Printf("[Get] Cache set took %v\n", time.Since(cacheStart))
	fmt.Printf("[Get] Total operation took %v\n", time.Since(start))

	return item, nil
}

// Search performs a similarity search
func (m *Manager) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	timer := prometheus.NewTimer(operationDuration.WithLabelValues("search"))
	defer timer.ObserveDuration()

	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.metrics.circuitOpen {
		return nil, fmt.Errorf("circuit breaker is open")
	}

	// Generate cache key for this search
	cacheKey := fmt.Sprintf("search:%x:%d", vector, limit)

	// Try cache first
	if cached, err := m.cache.Get(ctx, cacheKey); err == nil && cached != nil {
		cacheHits.Inc()
		// For searches, we store the results as a special type
		if results, ok := cached.Metadata["search_results"].([]*storage.Item); ok {
			return results, nil
		}
	}

	// Perform search
	items, err := m.vectorStore.Search(ctx, vector, limit)
	if err != nil {
		m.recordError()
		return nil, fmt.Errorf("vector store search failed: %w", err)
	}

	// Cache results
	cacheItem := &storage.Item{
		ID: cacheKey,
		Metadata: map[string]interface{}{
			"search_results": items,
		},
		ExpiresAt: time.Now().Add(m.config.Cache.Config.DefaultTTL),
	}
	if err := m.cache.Set(ctx, cacheKey, cacheItem); err != nil {
		fmt.Printf("warning: failed to cache search results: %v\n", err)
	}

	return items, nil
}

// DeleteFromStore removes items from both the vector store and cache
func (m *Manager) DeleteFromStore(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	start := time.Now()
	fmt.Printf("[Delete] Starting deletion of %d items\n", len(ids))

	timer := prometheus.NewTimer(operationDuration.WithLabelValues("delete"))
	defer timer.ObserveDuration()

	// Delete from cache and vector store concurrently
	var wg sync.WaitGroup
	errChan := make(chan error, 2)

	// Cache deletion
	wg.Add(1)
	go func() {
		defer wg.Done()
		cacheStart := time.Now()
		if err := m.cache.BatchDelete(ctx, ids); err != nil {
			fmt.Printf("[Delete] Cache batch deletion failed: %v\n", err)
			errChan <- fmt.Errorf("cache deletion failed: %w", err)
		}
		fmt.Printf("[Delete] Cache deletion took %v\n", time.Since(cacheStart))
	}()

	// Vector store deletion
	wg.Add(1)
	go func() {
		defer wg.Done()
		vectorStart := time.Now()
		if err := m.vectorStore.DeleteFromStore(ctx, ids); err != nil {
			fmt.Printf("[Delete] Vector store deletion failed: %v\n", err)
			errChan <- fmt.Errorf("vector store deletion failed: %w", err)
		}
		fmt.Printf("[Delete] Vector store deletion took %v\n", time.Since(vectorStart))
	}()

	// Wait for all operations to complete
	go func() {
		wg.Wait()
		close(errChan)
	}()

	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	fmt.Printf("[Delete] Total operation took %v\n", time.Since(start))
	return nil
}

// Health checks the health of both storage systems
func (m *Manager) Health(ctx context.Context) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if err := m.cache.Health(ctx); err != nil {
		return fmt.Errorf("cache health check failed: %w", err)
	}

	if err := m.vectorStore.Health(ctx); err != nil {
		return fmt.Errorf("vector store health check failed: %w", err)
	}

	return nil
}

// Close properly shuts down all storage connections
func (m *Manager) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Signal background tasks to stop
	close(m.stopChan)

	var errs []error

	if err := m.vectorStore.Close(); err != nil {
		errs = append(errs, fmt.Errorf("vector store close failed: %w", err))
	}

	if err := m.cache.Close(); err != nil {
		errs = append(errs, fmt.Errorf("cache close failed: %w", err))
	}

	if len(errs) > 0 {
		return fmt.Errorf("multiple close errors: %v", errs)
	}
	return nil
}

// Background tasks

func (m *Manager) healthCheck() {
	ticker := time.NewTicker(m.config.HealthInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			if err := m.Health(ctx); err != nil {
				m.metrics.healthStatus = false
				fmt.Printf("health check failed: %v\n", err)
			} else {
				m.metrics.healthStatus = true
				m.metrics.lastHealthy = time.Now()
			}
			cancel()
		}
	}
}

func (m *Manager) cacheWarmer() {
	ticker := time.NewTicker(m.config.Cache.WarmupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.warmCache()
		case <-m.warmupChan:
			m.warmCache()
		}
	}
}

func (m *Manager) warmCache() {
	for _, query := range m.config.Cache.WarmupQueries {
		// Execute the warmup query
		// This is a placeholder - implement based on your specific needs
		fmt.Printf("Warming cache with query: %s\n", query)
	}
}

// Circuit breaker methods

func (m *Manager) recordError() {
	m.metrics.lastError = time.Now()
	m.metrics.errorCount++

	if m.metrics.errorCount >= m.config.MaxRetries {
		m.metrics.circuitOpen = true
		go m.resetCircuitBreaker()
	}
}

func (m *Manager) resetCircuitBreaker() {
	time.Sleep(m.config.BreakDuration)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metrics.circuitOpen = false
	m.metrics.errorCount = 0
}

// TriggerCacheWarmup manually triggers cache warming
func (m *Manager) TriggerCacheWarmup() {
	select {
	case m.warmupChan <- struct{}{}:
	default:
		// Channel is full, warmup already pending
	}
}
