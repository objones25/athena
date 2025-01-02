package integration

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/search/cache"
	"github.com/objones25/athena/internal/search/index"
	"github.com/objones25/athena/internal/search/similarity"
)

// mockStore implements search.Store interface for testing
type mockStore struct {
	vectors  map[string][]float32
	metadata map[string]map[string]interface{}
}

func newMockStore() *mockStore {
	return &mockStore{
		vectors:  make(map[string][]float32),
		metadata: make(map[string]map[string]interface{}),
	}
}

func (s *mockStore) Get(ctx context.Context, key string) ([]float32, error) {
	if vec, ok := s.vectors[key]; ok {
		return vec, nil
	}
	return nil, fmt.Errorf("key not found: %s", key)
}

func (s *mockStore) MGet(ctx context.Context, keys []string) ([][]float32, error) {
	vectors := make([][]float32, len(keys))
	for i, key := range keys {
		if vec, ok := s.vectors[key]; ok {
			vectors[i] = vec
		} else {
			return nil, fmt.Errorf("key not found: %s", key)
		}
	}
	return vectors, nil
}

func (s *mockStore) Keys(ctx context.Context) ([]string, error) {
	keys := make([]string, 0, len(s.vectors))
	for k := range s.vectors {
		keys = append(keys, k)
	}
	return keys, nil
}

func (s *mockStore) GetMetadata(ctx context.Context, key string) (map[string]interface{}, error) {
	if meta, ok := s.metadata[key]; ok {
		return meta, nil
	}
	return nil, fmt.Errorf("metadata not found: %s", key)
}

// TestSearchPipeline tests the entire search pipeline
func TestSearchPipeline(t *testing.T) {
	dimensions := 128
	numVectors := 1000
	k := 10

	// Initialize components
	store := newMockStore()
	idx := index.New(dimensions, index.DefaultConfig())
	cache := cache.New(dimensions, cache.DefaultConfig())
	defer cache.Close()

	ctx := context.Background()

	// Generate test vectors with known clusters
	clusters := []struct {
		center []float32
		radius float32
		count  int
	}{
		{generateRandomVector(dimensions), 0.1, numVectors / 4},
		{generateRandomVector(dimensions), 0.1, numVectors / 4},
		{generateRandomVector(dimensions), 0.1, numVectors / 4},
		{generateRandomVector(dimensions), 0.1, numVectors / 4},
	}

	// Add vectors to store and index
	t.Run("Adding Vectors", func(t *testing.T) {
		for i, cluster := range clusters {
			for j := 0; j < cluster.count; j++ {
				key := fmt.Sprintf("cluster_%d_vec_%d", i, j)
				vector := perturbVector(cluster.center, cluster.radius)
				metadata := map[string]interface{}{
					"cluster": i,
					"index":   j,
				}

				// Add to store
				store.vectors[key] = vector
				store.metadata[key] = metadata

				// Add to index
				err := idx.Add(ctx, key, vector)
				if err != nil {
					t.Errorf("Failed to add vector: %v", err)
				}

				// Add to cache
				err = cache.Set(ctx, key, vector, metadata)
				if err != nil {
					t.Errorf("Failed to cache vector: %v", err)
				}
			}
		}
	})

	// Build search index
	t.Run("Building Index", func(t *testing.T) {
		err := idx.BuildGraph(ctx, index.DefaultConfig())
		if err != nil {
			t.Errorf("Failed to build index: %v", err)
		}
	})

	// Test search functionality
	t.Run("Search Functionality", func(t *testing.T) {
		for i, cluster := range clusters {
			// Search near cluster center
			results, err := idx.SearchNearest(ctx, cluster.center, k, index.DefaultConfig())
			if err != nil {
				t.Errorf("Search failed: %v", err)
				continue
			}

			// Verify results
			if len(results) != k {
				t.Errorf("Expected %d results, got %d", k, len(results))
			}

			// Check cluster membership
			clusterHits := 0
			for _, result := range results {
				meta := store.metadata[result.Key]
				if meta["cluster"] == i {
					clusterHits++
				}
			}

			// Most results should be from the same cluster
			minExpectedHits := k * 7 / 10 // At least 70% should be from the same cluster
			if clusterHits < minExpectedHits {
				t.Errorf("Expected at least %d hits from cluster %d, got %d", minExpectedHits, i, clusterHits)
			}
		}
	})

	// Test cache functionality
	t.Run("Cache Operations", func(t *testing.T) {
		// Test cache hit
		key := "cluster_0_vec_0"
		vector, err := cache.Get(ctx, key)
		if err != nil {
			t.Errorf("Cache miss for existing key: %v", err)
		}

		// Verify vector
		expectedVector := store.vectors[key]
		if !vectorEqual(vector, expectedVector) {
			t.Error("Cached vector doesn't match original")
		}

		// Test cache eviction
		cfg := cache.DefaultConfig()
		cfg.MaxSize = 10
		cfg.TTL = 100 * time.Millisecond
		smallCache := cache.New(dimensions, cfg)
		defer smallCache.Close()

		// Fill cache beyond capacity
		for i := 0; i < cfg.MaxSize+5; i++ {
			key := fmt.Sprintf("test_vec_%d", i)
			vector := generateRandomVector(dimensions)
			err := smallCache.Set(ctx, key, vector, nil)
			if err != nil {
				t.Errorf("Failed to set cache entry: %v", err)
			}
		}

		// Wait for TTL
		time.Sleep(cfg.TTL + 10*time.Millisecond)

		// Verify eviction
		_, err = smallCache.Get(ctx, "test_vec_0")
		if err == nil {
			t.Error("Expected cache miss after eviction")
		}
	})

	// Test similarity metrics
	t.Run("Similarity Metrics", func(t *testing.T) {
		simCtx := similarity.DefaultContext()
		vec1 := clusters[0].center
		vec2 := clusters[1].center
		vec3 := perturbVector(vec1, 0.1)

		// Compare similar vectors
		metrics1, err := similarity.Calculate(vec1, vec3, simCtx)
		if err != nil {
			t.Errorf("Similarity calculation failed: %v", err)
		}

		// Compare different vectors
		metrics2, err := similarity.Calculate(vec1, vec2, simCtx)
		if err != nil {
			t.Errorf("Similarity calculation failed: %v", err)
		}

		// Similar vectors should have higher scores
		if metrics1.Contextual <= metrics2.Contextual {
			t.Error("Similar vectors scored lower than different vectors")
		}
	})

	// Test concurrent search operations
	t.Run("Concurrent Search", func(t *testing.T) {
		var wg sync.WaitGroup
		concurrency := 10
		queries := make([][]float32, concurrency)
		for i := range queries {
			queries[i] = generateRandomVector(dimensions)
		}

		// Perform concurrent searches
		wg.Add(concurrency)
		for i := 0; i < concurrency; i++ {
			go func(i int) {
				defer wg.Done()
				results, err := idx.SearchNearest(ctx, queries[i], k, index.DefaultConfig())
				if err != nil {
					t.Errorf("Concurrent search failed: %v", err)
				}
				if len(results) != k {
					t.Errorf("Expected %d results, got %d", k, len(results))
				}
			}(i)
		}
		wg.Wait()
	})
}

// Helper functions

func generateRandomVector(dimensions int) []float32 {
	vec := make([]float32, dimensions)
	for i := range vec {
		vec[i] = rand.Float32()
	}
	normalizeVector(vec)
	return vec
}

func perturbVector(base []float32, radius float32) []float32 {
	vec := make([]float32, len(base))
	for i := range base {
		noise := (rand.Float32()*2 - 1) * radius
		vec[i] = base[i] + noise
	}
	normalizeVector(vec)
	return vec
}

func normalizeVector(vec []float32) {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range vec {
		vec[i] /= norm
	}
}

func vectorEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > 1e-6 {
			return false
		}
	}
	return true
}
