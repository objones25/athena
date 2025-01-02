package index_test

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/objones25/athena/internal/search/index"
)

func TestParallelSearch(t *testing.T) {
	// Create test data
	dimensions := 128
	numVectors := 10000
	vectors := make([][]float32, numVectors)
	rng := rand.New(rand.NewSource(42))

	// Generate random vectors
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		for j := range vectors[i] {
			vectors[i][j] = float32(rng.Float64())
		}
	}

	// Create query vector
	query := make([]float32, dimensions)
	for i := range query {
		query[i] = float32(rng.Float64())
	}

	tests := []struct {
		name        string
		cfg         index.Config
		k           int
		wantResults int
		wantTimeout bool
	}{
		{
			name: "single worker",
			cfg: index.Config{
				InitialCapacity: 10000,
				MaxConnections:  32,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
			},
			k:           10,
			wantResults: 10,
		},
		{
			name: "multiple connections",
			cfg: index.Config{
				InitialCapacity: 10000,
				MaxConnections:  64,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
			},
			k:           10,
			wantResults: 10,
		},
		{
			name: "large batch",
			cfg: index.Config{
				InitialCapacity: 10000,
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
			},
			k:           10,
			wantResults: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			// Create index
			idx := index.New(dimensions, tt.cfg)

			// Add vectors
			keys := make([]string, len(vectors))
			for i := range vectors {
				keys[i] = fmt.Sprintf("vec_%d", i)
			}
			err := idx.AddBatch(ctx, keys, vectors)
			if err != nil {
				t.Fatalf("AddBatch() error = %v", err)
			}

			// Build graph
			err = idx.BuildGraph(ctx, tt.cfg)
			if err != nil {
				t.Fatalf("BuildGraph() error = %v", err)
			}

			// Perform search
			results, err := idx.SearchNearest(ctx, query, tt.k, tt.cfg)
			if err != nil {
				t.Errorf("SearchNearest() error = %v", err)
				return
			}

			if len(results) != tt.wantResults {
				t.Errorf("SearchNearest() got %v results, want %v", len(results), tt.wantResults)
			}

			// Verify results are sorted by score
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Errorf("Results not sorted: score[%d]=%v > score[%d]=%v",
						i, results[i].Score, i-1, results[i-1].Score)
				}
			}

			// Verify all results meet minimum score threshold
			for i, result := range results {
				if result.Score < tt.cfg.ScoreThreshold {
					t.Errorf("Result %d has score %v below threshold %v",
						i, result.Score, tt.cfg.ScoreThreshold)
				}
			}
		})
	}
}

func TestConcurrentSearches(t *testing.T) {
	dimensions := 128
	numVectors := 10000
	vectors := make([][]float32, numVectors)
	rng := rand.New(rand.NewSource(42))

	// Generate random vectors
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		for j := range vectors[i] {
			vectors[i][j] = float32(rng.Float64())
		}
	}

	// Create index
	cfg := index.Config{
		InitialCapacity: 10000,
		MaxConnections:  32,
		BuildBatchSize:  100,
		ScoreThreshold:  0.5,
	}
	idx := index.New(dimensions, cfg)
	ctx := context.Background()

	// Add vectors
	keys := make([]string, len(vectors))
	for i := range vectors {
		keys[i] = fmt.Sprintf("vec_%d", i)
	}
	err := idx.AddBatch(ctx, keys, vectors)
	if err != nil {
		t.Fatalf("AddBatch() error = %v", err)
	}

	// Build graph
	err = idx.BuildGraph(ctx, cfg)
	if err != nil {
		t.Fatalf("BuildGraph() error = %v", err)
	}

	// Run concurrent searches
	numQueries := 10
	var wg sync.WaitGroup
	wg.Add(numQueries)
	errors := make(chan error, numQueries)

	for i := 0; i < numQueries; i++ {
		go func() {
			defer wg.Done()

			// Create random query vector
			query := make([]float32, dimensions)
			for j := range query {
				query[j] = float32(rng.Float64())
			}

			// Perform search
			_, err := idx.SearchNearest(ctx, query, 10, cfg)
			if err != nil {
				errors <- err
			}
		}()
	}

	// Wait for all searches to complete
	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		t.Errorf("Concurrent search error: %v", err)
	}
}

func BenchmarkSearch(b *testing.B) {
	dimensions := 128
	numVectors := 100000
	vectors := make([][]float32, numVectors)
	rng := rand.New(rand.NewSource(42))

	// Generate random vectors
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		for j := range vectors[i] {
			vectors[i][j] = float32(rng.Float64())
		}
	}

	// Create query vector
	query := make([]float32, dimensions)
	for i := range query {
		query[i] = float32(rng.Float64())
	}

	benchmarks := []struct {
		name    string
		config  index.Config
		k       int
		vectors [][]float32
	}{
		{
			name: "small_sequential",
			config: index.Config{
				InitialCapacity: 1000,
				MaxConnections:  32,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
			},
			k:       10,
			vectors: vectors[:1000],
		},
		{
			name: "small_parallel",
			config: index.Config{
				InitialCapacity: 1000,
				MaxConnections:  64,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
			},
			k:       10,
			vectors: vectors[:1000],
		},
		{
			name: "large_sequential",
			config: index.Config{
				InitialCapacity: 100000,
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
			},
			k:       10,
			vectors: vectors,
		},
		{
			name: "large_parallel",
			config: index.Config{
				InitialCapacity: 100000,
				MaxConnections:  64,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
			},
			k:       10,
			vectors: vectors,
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			ctx := context.Background()

			// Create index
			idx := index.New(dimensions, bm.config)

			// Add vectors
			keys := make([]string, len(bm.vectors))
			for i := range bm.vectors {
				keys[i] = fmt.Sprintf("vec_%d", i)
			}
			err := idx.AddBatch(ctx, keys, bm.vectors)
			if err != nil {
				b.Fatalf("AddBatch() error = %v", err)
			}

			// Build graph
			err = idx.BuildGraph(ctx, bm.config)
			if err != nil {
				b.Fatalf("BuildGraph() error = %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := idx.SearchNearest(ctx, query, bm.k, bm.config)
				if err != nil {
					b.Fatalf("SearchNearest() error = %v", err)
				}
			}
		})
	}
}
