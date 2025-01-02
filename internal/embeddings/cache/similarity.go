package cache

import (
	"context"
	"fmt"
	"sync"

	"github.com/objones25/athena/internal/embeddings/similarity"
)

// SimilarityCache extends the base Cache with similarity-specific operations
type SimilarityCache struct {
	cache Cache
	sim   *similarity.Context
	mu    sync.RWMutex
}

// NewSimilarityCache creates a new similarity cache wrapper
func NewSimilarityCache(cache Cache, simCtx *similarity.Context) *SimilarityCache {
	if simCtx == nil {
		simCtx = &similarity.Context{
			TopicalWeight:   0.4,
			SemanticWeight:  0.3,
			SyntacticWeight: 0.2,
			LanguageWeight:  0.1,
			BatchSize:       1000,
			UseConcurrency:  true,
			CacheResults:    true,
		}
	}

	return &SimilarityCache{
		cache: cache,
		sim:   simCtx,
	}
}

// CompareCached calculates similarity metrics between two cached embeddings
func (sc *SimilarityCache) CompareCached(ctx context.Context, key1, key2 string) (similarity.Metrics, error) {
	// Get embeddings from cache
	embed1, err := sc.cache.Get(ctx, key1)
	if err != nil {
		return similarity.Metrics{}, fmt.Errorf("failed to get first embedding: %w", err)
	}

	embed2, err := sc.cache.Get(ctx, key2)
	if err != nil {
		return similarity.Metrics{}, fmt.Errorf("failed to get second embedding: %w", err)
	}

	// Calculate similarity metrics
	return similarity.Calculate(embed1, embed2, *sc.sim)
}

// BatchCompareCached calculates similarity metrics for multiple pairs of cached embeddings
func (sc *SimilarityCache) BatchCompareCached(ctx context.Context, keyPairs [][2]string) ([]similarity.Metrics, error) {
	// Get all unique keys
	uniqueKeys := make(map[string]struct{})
	for _, pair := range keyPairs {
		uniqueKeys[pair[0]] = struct{}{}
		uniqueKeys[pair[1]] = struct{}{}
	}

	// Convert to slice for MGet
	keys := make([]string, 0, len(uniqueKeys))
	for k := range uniqueKeys {
		keys = append(keys, k)
	}

	// Fetch all embeddings in one call
	embeddings, err := sc.cache.MGet(ctx, keys)
	if err != nil {
		return nil, fmt.Errorf("failed to get embeddings: %w", err)
	}

	// Create embedding lookup map
	embedMap := make(map[string][]float32)
	for i, key := range keys {
		embedMap[key] = embeddings[i]
	}

	// Prepare input for batch calculation
	vectors1 := make([][]float32, len(keyPairs))
	vectors2 := make([][]float32, len(keyPairs))
	for i, pair := range keyPairs {
		embed1, ok := embedMap[pair[0]]
		if !ok {
			return nil, fmt.Errorf("embedding not found for key: %s", pair[0])
		}
		embed2, ok := embedMap[pair[1]]
		if !ok {
			return nil, fmt.Errorf("embedding not found for key: %s", pair[1])
		}
		vectors1[i] = embed1
		vectors2[i] = embed2
	}

	// Calculate similarities in batch
	return similarity.BatchCalculate(vectors1, vectors2, *sc.sim)
}

// FindSimilar finds similar embeddings in cache based on a query embedding
func (sc *SimilarityCache) FindSimilar(ctx context.Context, queryKey string, threshold float64, limit int) ([]SimilarityMatch, error) {
	// Get all keys (this is inefficient and should be replaced with proper indexing)
	allKeys := []string{} // TODO: Implement way to get all keys from cache

	// Prepare batch comparison
	var keyPairs [][2]string
	for _, key := range allKeys {
		if key == queryKey {
			continue
		}
		keyPairs = append(keyPairs, [2]string{queryKey, key})
	}

	// Batch calculate similarities
	metrics, err := sc.BatchCompareCached(ctx, keyPairs)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate similarities: %w", err)
	}

	// Filter and sort matches
	matches := make([]SimilarityMatch, 0, limit)
	for i, m := range metrics {
		if m.Contextual >= threshold {
			matches = append(matches, SimilarityMatch{
				Key:     keyPairs[i][1],
				Metrics: m,
			})
		}
		if len(matches) >= limit {
			break
		}
	}

	return matches, nil
}

// SimilarityMatch represents a matching embedding with its similarity score
type SimilarityMatch struct {
	Key     string
	Metrics similarity.Metrics
}

// UpdateContext updates the similarity context configuration
func (sc *SimilarityCache) UpdateContext(ctx *similarity.Context) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	sc.sim = ctx
}

// GetContext returns the current similarity context configuration
func (sc *SimilarityCache) GetContext() *similarity.Context {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.sim
}
