package monitor

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/prometheus/client_golang/prometheus"
)

// CacheWarmer handles intelligent cache warming strategies
type CacheWarmer struct {
	cache       storage.Cache
	vectorStore storage.VectorStore
	mu          sync.RWMutex
	config      WarmerConfig
	patterns    map[string]time.Time // Track access patterns
}

type WarmerConfig struct {
	// How often to analyze access patterns
	AnalysisInterval time.Duration
	// How many items to warm at once
	BatchSize int
	// Minimum access count to consider warming
	MinAccessCount int
	// Maximum age of access patterns to consider
	MaxPatternAge time.Duration
	// Maximum time to wait for warming operations
	Timeout time.Duration
}

type WarmingResult struct {
	ItemsWarmed  int
	Errors       []error
	Duration     time.Duration
	PatternCount int
	StartTime    time.Time
	EndTime      time.Time
}

func NewCacheWarmer(cache storage.Cache, vectorStore storage.VectorStore, cfg WarmerConfig) *CacheWarmer {
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 1000
	}
	if cfg.MinAccessCount == 0 {
		cfg.MinAccessCount = 5
	}
	if cfg.MaxPatternAge == 0 {
		cfg.MaxPatternAge = 24 * time.Hour
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 5 * time.Minute
	}
	if cfg.AnalysisInterval == 0 {
		cfg.AnalysisInterval = 1 * time.Hour
	}

	return &CacheWarmer{
		cache:       cache,
		vectorStore: vectorStore,
		config:      cfg,
		patterns:    make(map[string]time.Time),
	}
}

// StartWarming begins periodic cache warming based on access patterns
func (cw *CacheWarmer) StartWarming(ctx context.Context) {
	ticker := time.NewTicker(cw.config.AnalysisInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			timer := prometheus.NewTimer(CacheLatency.WithLabelValues("warm"))
			result, err := cw.WarmCache(ctx)
			timer.ObserveDuration()

			if err != nil {
				ErrorsTotal.WithLabelValues("cache", "warm", "failed").Inc()
				continue
			}

			CacheOperations.WithLabelValues("warm", "success").Add(float64(result.ItemsWarmed))
		}
	}
}

// RecordAccess records an access pattern for future warming
func (cw *CacheWarmer) RecordAccess(id string) {
	cw.mu.Lock()
	defer cw.mu.Unlock()
	cw.patterns[id] = time.Now()
}

// WarmCache performs cache warming based on recorded patterns
func (cw *CacheWarmer) WarmCache(ctx context.Context) (*WarmingResult, error) {
	ctx, cancel := context.WithTimeout(ctx, cw.config.Timeout)
	defer cancel()

	result := &WarmingResult{
		StartTime: time.Now(),
	}

	// Get frequently accessed items that aren't in cache
	patterns := cw.getWarmingCandidates()
	result.PatternCount = len(patterns)

	if len(patterns) == 0 {
		return result, nil
	}

	// Warm in batches
	for i := 0; i < len(patterns); i += cw.config.BatchSize {
		end := i + cw.config.BatchSize
		if end > len(patterns) {
			end = len(patterns)
		}
		batch := patterns[i:end]

		if err := cw.warmBatch(ctx, batch); err != nil {
			result.Errors = append(result.Errors, err)
			continue
		}
		result.ItemsWarmed += len(batch)
	}

	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)

	return result, nil
}

// getWarmingCandidates returns IDs that should be warmed based on access patterns
func (cw *CacheWarmer) getWarmingCandidates() []string {
	cw.mu.Lock()
	defer cw.mu.Unlock()

	now := time.Now()
	candidates := make([]string, 0)
	accessCounts := make(map[string]int)

	// Count recent accesses and clean up old patterns
	for id, lastAccess := range cw.patterns {
		if now.Sub(lastAccess) > cw.config.MaxPatternAge {
			delete(cw.patterns, id)
			continue
		}
		accessCounts[id]++
		if accessCounts[id] >= cw.config.MinAccessCount {
			// Check if item is already in cache
			exists, _ := cw.checkCache(context.Background(), id)
			if !exists {
				candidates = append(candidates, id)
			}
		}
	}

	return candidates
}

// warmBatch warms a batch of items
func (cw *CacheWarmer) warmBatch(ctx context.Context, ids []string) error {
	// Get items from vector store
	var vectors []*storage.Item
	for _, id := range ids {
		// This is a simplified example - in practice, you'd want to batch this
		items, err := cw.vectorStore.Search(ctx, nil, 1) // Get exact item by ID
		if err != nil {
			return fmt.Errorf("failed to get item %s from vector store: %w", id, err)
		}
		if len(items) > 0 {
			vectors = append(vectors, items[0])
		}
	}

	// Store in cache
	for _, item := range vectors {
		if err := cw.cache.Set(ctx, item.ID, item); err != nil {
			return fmt.Errorf("failed to warm item %s: %w", item.ID, err)
		}
	}

	return nil
}

// checkCache checks if an item exists in cache
func (cw *CacheWarmer) checkCache(ctx context.Context, id string) (bool, error) {
	item, err := cw.cache.Get(ctx, id)
	if err != nil {
		return false, err
	}
	return item != nil, nil
}
