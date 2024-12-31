package manager

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/cache"
	"github.com/objones25/athena/internal/storage/milvus"
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

// BatchSet stores multiple items
func (m *Manager) BatchSet(ctx context.Context, items []*storage.Item) error {
	start := time.Now()
	fmt.Printf("[BatchSet] Starting batch operation with %d items\n", len(items))

	// Store in vector store first
	vectorStart := time.Now()
	fmt.Printf("[BatchSet] Starting vector store insertion\n")
	if err := m.vectorStore.BatchSet(ctx, items); err != nil {
		fmt.Printf("[BatchSet] Vector store insertion failed after %v: %v\n", time.Since(vectorStart), err)
		return fmt.Errorf("failed to store in vector store: %w", err)
	}
	fmt.Printf("[BatchSet] Vector store insertion completed in %v\n", time.Since(vectorStart))

	// Then cache all items
	cacheStart := time.Now()
	fmt.Printf("[BatchSet] Starting cache operations\n")
	for _, item := range items {
		if err := m.cache.Set(ctx, item.ID, item); err != nil {
			fmt.Printf("[BatchSet] Cache operation failed for item %s after %v: %v\n",
				item.ID, time.Since(cacheStart), err)
			return fmt.Errorf("failed to cache item %s: %w", item.ID, err)
		}
	}
	fmt.Printf("[BatchSet] Cache operations took %v\n", time.Since(cacheStart))
	fmt.Printf("[BatchSet] Total operation took %v\n", time.Since(start))
	return nil
}

// Set stores an item in both the cache and vector store
func (m *Manager) Set(ctx context.Context, key string, item *storage.Item) error {
	return m.BatchSet(ctx, []*storage.Item{item})
}

// Get retrieves an item by ID
func (m *Manager) Get(ctx context.Context, id string) (*storage.Item, error) {
	start := time.Now()
	fmt.Printf("[Get] Starting retrieval for ID: %s\n", id)

	// Try cache first
	cacheStart := time.Now()
	item, err := m.cache.Get(ctx, id)
	cacheDuration := time.Since(cacheStart)

	if err == nil && item != nil {
		fmt.Printf("[Get] Cache hit for %s in %v\n", id, cacheDuration)
		return item, nil
	}
	fmt.Printf("[Get] Cache miss for %s in %v\n", id, cacheDuration)

	// Fallback to vector store
	vectorStart := time.Now()
	fmt.Printf("[Get] Falling back to vector store for %s\n", id)
	item, err = m.vectorStore.GetByID(ctx, id)
	if err != nil {
		fmt.Printf("[Get] Vector store retrieval failed after %v: %v\n", time.Since(vectorStart), err)
		return nil, fmt.Errorf("failed to get from vector store: %w", err)
	}
	fmt.Printf("[Get] Vector store retrieval took %v\n", time.Since(vectorStart))

	if item != nil {
		// Update cache with found item
		cacheUpdateStart := time.Now()
		if err := m.cache.Set(ctx, id, item); err != nil {
			fmt.Printf("[Get] Cache update failed after %v: %v\n", time.Since(cacheUpdateStart), err)
			// Log but don't fail on cache update error
		} else {
			fmt.Printf("[Get] Cache updated in %v\n", time.Since(cacheUpdateStart))
		}
	}

	fmt.Printf("[Get] Total operation took %v\n", time.Since(start))
	return item, nil
}

// Search performs a vector similarity search
func (m *Manager) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	start := time.Now()
	fmt.Printf("[Search] Starting search with vector dimension %d, limit %d\n", len(vector), limit)

	// Perform search
	searchStart := time.Now()
	fmt.Printf("[Search] Executing vector store search\n")
	results, err := m.vectorStore.Search(ctx, vector, limit)
	if err != nil {
		fmt.Printf("[Search] Vector store search failed after %v: %v\n", time.Since(searchStart), err)
		return nil, fmt.Errorf("failed to perform vector search: %w", err)
	}
	fmt.Printf("[Search] Vector store search completed in %v\n", time.Since(searchStart))

	// Update cache with results
	if len(results) > 0 {
		cacheStart := time.Now()
		fmt.Printf("[Search] Updating cache with %d results\n", len(results))
		for _, item := range results {
			if err := m.cache.Set(ctx, item.ID, item); err != nil {
				fmt.Printf("[Search] Cache update failed for item %s after %v: %v\n",
					item.ID, time.Since(cacheStart), err)
				// Log but don't fail on cache update error
			}
		}
		fmt.Printf("[Search] Cache updates completed in %v\n", time.Since(cacheStart))
	}

	fmt.Printf("[Search] Total operation took %v\n", time.Since(start))
	return results, nil
}

// DeleteFromStore removes items from both stores
func (m *Manager) DeleteFromStore(ctx context.Context, ids []string) error {
	start := time.Now()
	fmt.Printf("[Delete] Starting deletion of %d items: %v\n", len(ids), ids)

	// Verify items exist before deletion
	verifyStart := time.Now()
	fmt.Printf("[Delete] Verifying items before deletion\n")
	for _, id := range ids {
		item, err := m.Get(ctx, id)
		if err != nil {
			fmt.Printf("[Delete] Error verifying item %s: %v\n", id, err)
		} else if item != nil {
			fmt.Printf("[Delete] Found item %s before deletion\n", id)
		} else {
			fmt.Printf("[Delete] Item %s not found before deletion\n", id)
		}
	}
	fmt.Printf("[Delete] Pre-deletion verification took %v\n", time.Since(verifyStart))

	// Delete from vector store first
	vectorStart := time.Now()
	fmt.Printf("[Delete] Starting vector store deletion\n")
	if err := m.vectorStore.DeleteFromStore(ctx, ids); err != nil {
		fmt.Printf("[Delete] Vector store deletion failed after %v: %v\n", time.Since(vectorStart), err)
		return fmt.Errorf("failed to delete from vector store: %w", err)
	}
	fmt.Printf("[Delete] Vector store deletion completed in %v\n", time.Since(vectorStart))

	// Then remove from cache
	cacheStart := time.Now()
	fmt.Printf("[Delete] Starting cache deletion\n")
	for _, id := range ids {
		if err := m.cache.DeleteFromCache(ctx, id); err != nil {
			fmt.Printf("[Delete] Cache deletion failed for item %s after %v: %v\n",
				id, time.Since(cacheStart), err)
			return fmt.Errorf("failed to delete from cache: %w", err)
		}
	}
	fmt.Printf("[Delete] Cache deletion completed in %v\n", time.Since(cacheStart))

	// Verify items are deleted
	verifyStart = time.Now()
	fmt.Printf("[Delete] Verifying items after deletion\n")
	for _, id := range ids {
		item, err := m.Get(ctx, id)
		if err != nil {
			fmt.Printf("[Delete] Error verifying deletion of item %s: %v\n", id, err)
		} else if item != nil {
			fmt.Printf("[Delete] WARNING: Item %s still exists after deletion: %+v\n", id, item)
		} else {
			fmt.Printf("[Delete] Confirmed deletion of item %s\n", id)
		}
	}
	fmt.Printf("[Delete] Post-deletion verification took %v\n", time.Since(verifyStart))

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

// TriggerCacheWarmup manually triggers cache warming
func (m *Manager) TriggerCacheWarmup() {
	select {
	case m.warmupChan <- struct{}{}:
	default:
		// Channel is full, warmup already pending
	}
}
