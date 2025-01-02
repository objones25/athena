package index_test

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/objones25/athena/internal/search/index"
)

func TestVectorIndex(t *testing.T) {
	dimensions := 4
	cfg := index.DefaultConfig()
	cfg.InitialCapacity = 100
	cfg.MaxConnections = 5
	cfg.BuildBatchSize = 10
	cfg.ScoreThreshold = 0.5

	idx := index.New(dimensions, cfg)
	ctx := context.Background()

	// Test Add and Search
	t.Run("Add and Search", func(t *testing.T) {
		vectors := []struct {
			key    string
			vector []float32
		}{
			{"vec1", []float32{1, 0, 0, 0}},
			{"vec2", []float32{0, 1, 0, 0}},
			{"vec3", []float32{0, 0, 1, 0}},
			{"vec4", []float32{0.7, 0.7, 0, 0}},
		}

		// Add vectors
		for _, v := range vectors {
			err := idx.Add(ctx, v.key, v.vector)
			if err != nil {
				t.Errorf("Add() error = %v", err)
				return
			}
		}

		// Build graph
		err := idx.BuildGraph(ctx, cfg)
		if err != nil {
			t.Errorf("BuildGraph() error = %v", err)
			return
		}

		// Search nearest to vec1
		results, err := idx.SearchNearest(ctx, vectors[0].vector, 2, cfg)
		if err != nil {
			t.Errorf("SearchNearest() error = %v", err)
			return
		}

		if len(results) != 2 {
			t.Errorf("SearchNearest() returned %d results, want 2", len(results))
			return
		}

		// vec4 should be closer to vec1 than vec2
		if results[0].Key != "vec1" || results[1].Key != "vec4" {
			t.Errorf("SearchNearest() wrong order: got %v, %v", results[0].Key, results[1].Key)
		}
	})

	// Test batch operations
	t.Run("Batch Operations", func(t *testing.T) {
		keys := []string{"batch1", "batch2", "batch3"}
		vectors := [][]float32{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
		}

		err := idx.AddBatch(ctx, keys, vectors)
		if err != nil {
			t.Errorf("AddBatch() error = %v", err)
			return
		}

		// Build graph
		err = idx.BuildGraph(ctx, cfg)
		if err != nil {
			t.Errorf("BuildGraph() error = %v", err)
			return
		}

		// Search nearest to batch1
		results, err := idx.SearchNearest(ctx, vectors[0], 3, cfg)
		if err != nil {
			t.Errorf("SearchNearest() error = %v", err)
			return
		}

		if len(results) != 3 {
			t.Errorf("SearchNearest() returned %d results, want 3", len(results))
		}
	})

	// Test dimension validation
	t.Run("Dimension Validation", func(t *testing.T) {
		err := idx.Add(ctx, "invalid", []float32{1, 2, 3}) // Wrong dimensions
		if err == nil {
			t.Error("Add() should return error for wrong dimensions")
		}

		_, err = idx.SearchNearest(ctx, []float32{1, 2, 3}, 1, cfg)
		if err == nil {
			t.Error("SearchNearest() should return error for wrong dimensions")
		}
	})

	// Test remove
	t.Run("Remove", func(t *testing.T) {
		key := "remove_test"
		vector := []float32{1, 1, 1, 1}

		err := idx.Add(ctx, key, vector)
		if err != nil {
			t.Errorf("Add() error = %v", err)
			return
		}

		err = idx.BuildGraph(ctx, cfg)
		if err != nil {
			t.Errorf("BuildGraph() error = %v", err)
			return
		}

		idx.Remove(ctx, key)

		// Search should not return removed vector
		results, err := idx.SearchNearest(ctx, vector, 1, cfg)
		if err != nil {
			t.Errorf("SearchNearest() error = %v", err)
			return
		}

		for _, result := range results {
			if result.Key == key {
				t.Error("SearchNearest() returned removed vector")
			}
		}
	})
}

func TestGraphBuilding(t *testing.T) {
	dimensions := 4
	cfg := index.DefaultConfig()
	cfg.MaxConnections = 3
	cfg.ScoreThreshold = 0.5

	idx := index.New(dimensions, cfg)
	ctx := context.Background()

	// Create a grid of vectors
	gridSize := 5
	for i := 0; i < gridSize; i++ {
		for j := 0; j < gridSize; j++ {
			key := fmt.Sprintf("grid_%d_%d", i, j)
			vector := []float32{float32(i) / float32(gridSize-1), float32(j) / float32(gridSize-1), 0, 0}
			err := idx.Add(ctx, key, vector)
			if err != nil {
				t.Errorf("Add() error = %v", err)
				return
			}
		}
	}

	// Build graph
	err := idx.BuildGraph(ctx, cfg)
	if err != nil {
		t.Errorf("BuildGraph() error = %v", err)
		return
	}

	// Verify graph properties
	t.Run("Graph Properties", func(t *testing.T) {
		// Each node should have at most MaxConnections neighbors
		for _, key := range idx.GetVectorKeys() {
			neighbors := idx.GetNeighbors(key)
			if len(neighbors) > cfg.MaxConnections {
				t.Errorf("Node %s has %d neighbors, want <= %d", key, len(neighbors), cfg.MaxConnections)
			}

			// Verify neighbor scores
			for _, neighbor := range neighbors {
				score := idx.GetScore(key, neighbor)
				if score < cfg.ScoreThreshold {
					t.Errorf("Edge score %.2f below threshold %.2f", score, cfg.ScoreThreshold)
				}
			}
		}
	})

	// Test search with graph
	t.Run("Graph Search", func(t *testing.T) {
		// Search for nearest neighbors of a point
		queryVec := []float32{0.5, 0.5, 0, 0} // Center point
		k := 4
		results, err := idx.SearchNearest(ctx, queryVec, k, cfg)
		if err != nil {
			t.Errorf("SearchNearest() error = %v", err)
			return
		}

		if len(results) != k {
			t.Errorf("SearchNearest() returned %d results, want %d", len(results), k)
		}

		// Results should be sorted by score
		for i := 1; i < len(results); i++ {
			if results[i-1].Score < results[i].Score {
				t.Error("Results not properly sorted")
			}
		}

		// Verify that closest points are found
		centerKey := "grid_2_2" // Middle of the grid
		found := false
		for _, result := range results {
			if result.Key == centerKey {
				found = true
				break
			}
		}
		if !found {
			t.Error("Center point not found in nearest neighbors")
		}
	})
}

func TestConcurrentAccess(t *testing.T) {
	dimensions := 4
	cfg := index.DefaultConfig()
	idx := index.New(dimensions, cfg)
	ctx := context.Background()

	// Concurrent Add and Search
	t.Run("Concurrent Operations", func(t *testing.T) {
		var wg sync.WaitGroup
		concurrency := 10
		vectorsPerWorker := 100

		// Add vectors concurrently
		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < vectorsPerWorker; j++ {
					key := fmt.Sprintf("worker_%d_vec_%d", workerID, j)
					vector := make([]float32, dimensions)
					for k := range vector {
						vector[k] = rand.Float32()
					}
					err := idx.Add(ctx, key, vector)
					if err != nil {
						t.Errorf("Add() error = %v", err)
					}
				}
			}(i)
		}
		wg.Wait()

		// Build graph
		err := idx.BuildGraph(ctx, cfg)
		if err != nil {
			t.Errorf("BuildGraph() error = %v", err)
			return
		}

		// Search concurrently
		wg.Add(concurrency)
		for i := 0; i < concurrency; i++ {
			go func() {
				defer wg.Done()
				query := make([]float32, dimensions)
				for k := range query {
					query[k] = rand.Float32()
				}
				_, err := idx.SearchNearest(ctx, query, 10, cfg)
				if err != nil {
					t.Errorf("SearchNearest() error = %v", err)
				}
			}()
		}
		wg.Wait()
	})
}

func BenchmarkVectorIndex(b *testing.B) {
	dimensions := 128
	numVectors := 10000
	cfg := index.DefaultConfig()
	idx := index.New(dimensions, cfg)
	ctx := context.Background()

	// Generate random vectors
	vectors := make([][]float32, numVectors)
	keys := make([]string, numVectors)
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
		keys[i] = fmt.Sprintf("vec_%d", i)
	}

	// Benchmark Add
	b.Run("Add", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			idx.Add(ctx, fmt.Sprintf("bench_%d", i), vectors[i%numVectors])
		}
	})

	// Add vectors for search benchmark
	err := idx.AddBatch(ctx, keys, vectors)
	if err != nil {
		b.Fatalf("AddBatch() error = %v", err)
	}

	err = idx.BuildGraph(ctx, cfg)
	if err != nil {
		b.Fatalf("BuildGraph() error = %v", err)
	}

	// Benchmark Search
	b.Run("Search", func(b *testing.B) {
		query := make([]float32, dimensions)
		for i := range query {
			query[i] = rand.Float32()
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = idx.SearchNearest(ctx, query, 10, cfg)
		}
	})
}
