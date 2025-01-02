package cache

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/search/similarity"
)

func TestCache(t *testing.T) {
	dimensions := 4
	cfg := DefaultConfig()
	cfg.MaxSize = 10
	cfg.TTL = 100 * time.Millisecond
	cfg.CleanupInterval = 50 * time.Millisecond

	cache := New(dimensions, cfg)
	defer cache.Close()

	ctx := context.Background()

	// Test Set and Get
	t.Run("Set and Get", func(t *testing.T) {
		vector := []float32{1, 2, 3, 4}
		metadata := map[string]interface{}{
			"type": "test",
			"id":   1,
		}

		err := cache.Set(ctx, "key1", vector, metadata)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		got, err := cache.Get(ctx, "key1")
		if err != nil {
			t.Errorf("Get() error = %v", err)
			return
		}

		if !vectorEqual(got, vector) {
			t.Errorf("Get() = %v, want %v", got, vector)
		}

		// Test GetWithMetadata
		gotVec, gotMeta, err := cache.GetWithMetadata(ctx, "key1")
		if err != nil {
			t.Errorf("GetWithMetadata() error = %v", err)
			return
		}

		if !vectorEqual(gotVec, vector) {
			t.Errorf("GetWithMetadata() vector = %v, want %v", gotVec, vector)
		}

		if gotMeta["type"] != metadata["type"] || gotMeta["id"] != metadata["id"] {
			t.Errorf("GetWithMetadata() metadata = %v, want %v", gotMeta, metadata)
		}
	})

	// Test SetBatch and GetBatch
	t.Run("Batch Operations", func(t *testing.T) {
		keys := []string{"batch1", "batch2"}
		vectors := [][]float32{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		}
		metadata := []map[string]interface{}{
			{"id": 1},
			{"id": 2},
		}

		err := cache.SetBatch(ctx, keys, vectors, metadata)
		if err != nil {
			t.Errorf("SetBatch() error = %v", err)
			return
		}

		// Verify each entry
		for i, key := range keys {
			got, err := cache.Get(ctx, key)
			if err != nil {
				t.Errorf("Get() error = %v", err)
				continue
			}

			if !vectorEqual(got, vectors[i]) {
				t.Errorf("Get() = %v, want %v", got, vectors[i])
			}
		}
	})

	// Test LRU eviction
	t.Run("LRU Eviction", func(t *testing.T) {
		// Fill cache beyond capacity
		for i := 0; i < cfg.MaxSize+5; i++ {
			key := fmt.Sprintf("lru%d", i)
			vector := []float32{float32(i), float32(i), float32(i), float32(i)}
			err := cache.Set(ctx, key, vector, nil)
			if err != nil {
				t.Errorf("Set() error = %v", err)
				return
			}
		}

		// First few entries should be evicted
		for i := 0; i < 5; i++ {
			key := fmt.Sprintf("lru%d", i)
			_, err := cache.Get(ctx, key)
			if err == nil {
				t.Errorf("Get() should return error for evicted key %s", key)
			}
		}
	})

	// Test TTL expiration
	t.Run("TTL Expiration", func(t *testing.T) {
		key := "ttl_test"
		vector := []float32{1, 2, 3, 4}
		err := cache.Set(ctx, key, vector, nil)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		// Wait for TTL to expire
		time.Sleep(cfg.TTL + 10*time.Millisecond)

		// Wait for cleanup
		time.Sleep(cfg.CleanupInterval + 10*time.Millisecond)

		_, err = cache.Get(ctx, key)
		if err == nil {
			t.Error("Get() should return error for expired key")
		}
	})

	// Test similarity score caching
	t.Run("Similarity Score Cache", func(t *testing.T) {
		key1 := "sim1"
		key2 := "sim2"
		vec1 := []float32{1, 0, 0, 0}
		vec2 := []float32{0, 1, 0, 0}

		err := cache.Set(ctx, key1, vec1, nil)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		err = cache.Set(ctx, key2, vec2, nil)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		// Calculate and cache similarity
		metrics := similarity.Metrics{
			Cosine:     0.0,
			Angular:    90.0,
			Euclidean:  1.414,
			Contextual: 0.5,
		}
		cache.SetSimilarityScore(key1, key2, metrics)

		// Retrieve cached score
		got, ok := cache.GetSimilarityScore(key1, key2)
		if !ok {
			t.Error("GetSimilarityScore() should find cached score")
			return
		}

		if got.Cosine != metrics.Cosine || got.Angular != metrics.Angular {
			t.Errorf("GetSimilarityScore() = %v, want %v", got, metrics)
		}

		// Score should be symmetric
		got, ok = cache.GetSimilarityScore(key2, key1)
		if !ok {
			t.Error("GetSimilarityScore() should find cached score (symmetric)")
			return
		}

		if got.Cosine != metrics.Cosine || got.Angular != metrics.Angular {
			t.Errorf("GetSimilarityScore() symmetric = %v, want %v", got, metrics)
		}
	})

	// Test Remove
	t.Run("Remove", func(t *testing.T) {
		key := "remove_test"
		vector := []float32{1, 2, 3, 4}
		err := cache.Set(ctx, key, vector, nil)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		cache.Remove(ctx, key)

		_, err = cache.Get(ctx, key)
		if err == nil {
			t.Error("Get() should return error for removed key")
		}
	})

	// Test Clear
	t.Run("Clear", func(t *testing.T) {
		key := "clear_test"
		vector := []float32{1, 2, 3, 4}
		err := cache.Set(ctx, key, vector, nil)
		if err != nil {
			t.Errorf("Set() error = %v", err)
			return
		}

		cache.Clear(ctx)

		_, err = cache.Get(ctx, key)
		if err == nil {
			t.Error("Get() should return error after clear")
		}
	})
}

func TestCacheConcurrency(t *testing.T) {
	dimensions := 4
	cfg := DefaultConfig()
	cache := New(dimensions, cfg)
	defer cache.Close()

	ctx := context.Background()
	concurrency := 10
	iterations := 100

	// Concurrent Set and Get
	t.Run("Concurrent Set/Get", func(t *testing.T) {
		var wg sync.WaitGroup
		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					key := fmt.Sprintf("conc%d_%d", id, j)
					vector := []float32{float32(id), float32(j), 0, 0}

					err := cache.Set(ctx, key, vector, nil)
					if err != nil {
						t.Errorf("Set() error = %v", err)
						return
					}

					got, err := cache.Get(ctx, key)
					if err != nil {
						t.Errorf("Get() error = %v", err)
						return
					}

					if !vectorEqual(got, vector) {
						t.Errorf("Get() = %v, want %v", got, vector)
					}
				}
			}(i)
		}
		wg.Wait()
	})

	// Concurrent similarity score operations
	t.Run("Concurrent Similarity Scores", func(t *testing.T) {
		var wg sync.WaitGroup
		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					key1 := fmt.Sprintf("sim%d_%d_1", id, j)
					key2 := fmt.Sprintf("sim%d_%d_2", id, j)
					metrics := similarity.Metrics{
						Cosine: float64(id) / float64(concurrency),
					}

					cache.SetSimilarityScore(key1, key2, metrics)
					got, _ := cache.GetSimilarityScore(key1, key2)
					if got.Cosine != metrics.Cosine {
						t.Errorf("GetSimilarityScore() = %v, want %v", got, metrics)
					}
				}
			}(i)
		}
		wg.Wait()
	})
}

func vectorEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkCache(b *testing.B) {
	dimensions := 4
	cfg := DefaultConfig()
	cache := New(dimensions, cfg)
	defer cache.Close()

	ctx := context.Background()
	vector := []float32{1, 2, 3, 4}
	metadata := map[string]interface{}{"id": 1}

	b.Run("Set", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			key := fmt.Sprintf("bench%d", i)
			_ = cache.Set(ctx, key, vector, metadata)
		}
	})

	b.Run("Get", func(b *testing.B) {
		key := "bench_get"
		_ = cache.Set(ctx, key, vector, metadata)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = cache.Get(ctx, key)
		}
	})

	b.Run("GetWithMetadata", func(b *testing.B) {
		key := "bench_get_meta"
		_ = cache.Set(ctx, key, vector, metadata)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, _ = cache.GetWithMetadata(ctx, key)
		}
	})
}
