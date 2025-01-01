package manager

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/cache"
	"github.com/objones25/athena/internal/storage/milvus"
	"github.com/rs/zerolog/log"
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
	logger := log.With().Str("component", "storage_manager").Logger()
	logger.Debug().Interface("config", cfg).Msg("Initializing storage manager")

	logger.Debug().Msg("Creating Redis cache")
	cache, err := cache.NewRedisCache(cfg.Cache.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}
	logger.Debug().Msg("Redis cache created successfully")

	logger.Debug().Msg("Creating Milvus vector store")
	vectorStore, err := milvus.NewMilvusStore(cfg.VectorStore.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}
	logger.Debug().Msg("Milvus vector store created successfully")

	m := &Manager{
		cache:       cache,
		vectorStore: vectorStore,
		config:      cfg,
		stopChan:    make(chan struct{}),
		warmupChan:  make(chan struct{}, 1),
	}

	logger.Debug().Msg("Starting background tasks")
	// Start background tasks
	go m.healthCheck()
	go m.cacheWarmer()
	logger.Debug().Msg("Background tasks started")

	logger.Info().Msg("Storage manager initialized successfully")
	return m, nil
}

// BatchSet stores multiple items
func (m *Manager) BatchSet(ctx context.Context, items []*storage.Item) error {
	start := time.Now()
	logger := log.With().
		Str("component", "storage_manager").
		Str("operation", "batch_set").
		Int("item_count", len(items)).
		Logger()

	logger.Debug().Msg("Starting batch operation")

	// Store in vector store first
	vectorStart := time.Now()
	logger.Debug().Msg("Starting vector store insertion")
	if err := m.vectorStore.BatchSet(ctx, items); err != nil {
		logger.Error().
			Err(err).
			Dur("duration", time.Since(vectorStart)).
			Msg("Vector store insertion failed")
		return fmt.Errorf("failed to store in vector store: %w", err)
	}
	logger.Debug().
		Dur("duration", time.Since(vectorStart)).
		Msg("Vector store insertion completed")

	// Then cache all items
	cacheStart := time.Now()
	logger.Debug().Msg("Starting cache operations")
	for _, item := range items {
		if err := m.cache.Set(ctx, item.ID, item); err != nil {
			logger.Error().
				Err(err).
				Str("item_id", item.ID).
				Dur("duration", time.Since(cacheStart)).
				Msg("Cache operation failed")
			return fmt.Errorf("failed to cache item %s: %w", item.ID, err)
		}
	}
	logger.Debug().
		Dur("duration", time.Since(cacheStart)).
		Msg("Cache operations completed")

	logger.Debug().
		Dur("duration", time.Since(start)).
		Msg("Total batch operation completed")
	return nil
}

// Set stores an item in both the cache and vector store
func (m *Manager) Set(ctx context.Context, key string, item *storage.Item) error {
	return m.BatchSet(ctx, []*storage.Item{item})
}

// Get retrieves an item by ID
func (m *Manager) Get(ctx context.Context, id string) (*storage.Item, error) {
	start := time.Now()
	logger := log.With().
		Str("component", "storage_manager").
		Str("operation", "get").
		Str("id", id).
		Logger()

	logger.Debug().Msg("Starting retrieval")

	// Try cache first
	cacheStart := time.Now()
	item, err := m.cache.Get(ctx, id)
	cacheDuration := time.Since(cacheStart)

	if err == nil && item != nil {
		logger.Debug().
			Dur("duration", cacheDuration).
			Msg("Cache hit")
		return item, nil
	}
	logger.Debug().
		Dur("duration", cacheDuration).
		Msg("Cache miss")

	// Fallback to vector store
	vectorStart := time.Now()
	logger.Debug().Msg("Falling back to vector store")
	item, err = m.vectorStore.GetByID(ctx, id)
	if err != nil {
		logger.Error().
			Err(err).
			Dur("duration", time.Since(vectorStart)).
			Msg("Vector store retrieval failed")
		return nil, fmt.Errorf("failed to get from vector store: %w", err)
	}
	logger.Debug().
		Dur("duration", time.Since(vectorStart)).
		Msg("Vector store retrieval completed")

	if item != nil {
		// Update cache with found item
		cacheUpdateStart := time.Now()
		if err := m.cache.Set(ctx, id, item); err != nil {
			logger.Warn().
				Err(err).
				Dur("duration", time.Since(cacheUpdateStart)).
				Msg("Cache update failed")
			// Log but don't fail on cache update error
		} else {
			logger.Debug().
				Dur("duration", time.Since(cacheUpdateStart)).
				Msg("Cache updated")
		}
	}

	logger.Debug().
		Dur("duration", time.Since(start)).
		Msg("Total operation completed")
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
	logger := log.With().
		Str("component", "storage_manager").
		Str("operation", "delete").
		Interface("ids", ids).
		Logger()

	logger.Debug().
		Int("count", len(ids)).
		Msg("Starting deletion")

	// Verify items exist before deletion
	verifyStart := time.Now()
	logger.Debug().Msg("Verifying items before deletion")
	for _, id := range ids {
		item, err := m.Get(ctx, id)
		if err != nil {
			logger.Warn().
				Err(err).
				Str("id", id).
				Msg("Error verifying item")
		} else if item != nil {
			logger.Debug().
				Str("id", id).
				Msg("Found item before deletion")
		} else {
			logger.Debug().
				Str("id", id).
				Msg("Item not found before deletion")
		}
	}
	logger.Debug().
		Dur("duration", time.Since(verifyStart)).
		Msg("Pre-deletion verification completed")

	// Delete from vector store first
	vectorStart := time.Now()
	logger.Debug().Msg("Starting vector store deletion")
	if err := m.vectorStore.DeleteFromStore(ctx, ids); err != nil {
		logger.Error().
			Err(err).
			Dur("duration", time.Since(vectorStart)).
			Msg("Vector store deletion failed")
		return fmt.Errorf("failed to delete from vector store: %w", err)
	}
	logger.Debug().
		Dur("duration", time.Since(vectorStart)).
		Msg("Vector store deletion completed")

	// Then remove from cache
	cacheStart := time.Now()
	logger.Debug().Msg("Starting cache deletion")
	for _, id := range ids {
		if err := m.cache.DeleteFromCache(ctx, id); err != nil {
			logger.Error().
				Err(err).
				Str("id", id).
				Dur("duration", time.Since(cacheStart)).
				Msg("Cache deletion failed")
			return fmt.Errorf("failed to delete from cache: %w", err)
		}
	}
	logger.Debug().
		Dur("duration", time.Since(cacheStart)).
		Msg("Cache deletion completed")

	// Verify items are deleted
	verifyStart = time.Now()
	logger.Debug().Msg("Verifying items after deletion")
	for _, id := range ids {
		item, err := m.Get(ctx, id)
		if err != nil {
			logger.Warn().
				Err(err).
				Str("id", id).
				Msg("Error verifying deletion")
		} else if item != nil {
			logger.Warn().
				Str("id", id).
				Interface("item", item).
				Msg("Item still exists after deletion")
		} else {
			logger.Debug().
				Str("id", id).
				Msg("Confirmed deletion")
		}
	}
	logger.Debug().
		Dur("duration", time.Since(verifyStart)).
		Msg("Post-deletion verification completed")

	logger.Debug().
		Dur("duration", time.Since(start)).
		Msg("Total operation completed")
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
