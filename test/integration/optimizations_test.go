package integration

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/objones25/athena/internal/search/index"
	"github.com/objones25/athena/internal/search/optimizations"
)

func TestOptimizedSearch(t *testing.T) {
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
		name string
		cfg  optimizations.Config
	}{
		{
			name: "pca_only",
			cfg: optimizations.Config{
				TargetDimension: 32,
			},
		},
		{
			name: "quantization_only",
			cfg: optimizations.Config{
				NumCentroids:   100,
				MaxIterations:  100,
				ConvergenceEps: 1e-6,
			},
		},
		{
			name: "pca_and_quantization",
			cfg: optimizations.Config{
				TargetDimension: 32,
				NumCentroids:    100,
				MaxIterations:   100,
				ConvergenceEps:  1e-6,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply optimizations
			var optimizedVectors [][]float32
			var projection [][]float32
			var err error

			if tt.cfg.TargetDimension > 0 {
				// Apply PCA
				optimizedVectors, projection, err = optimizations.PCA(vectors, tt.cfg)
				if err != nil {
					t.Fatalf("PCA() error = %v", err)
				}

				// Project query
				query, err = optimizations.Project(query, projection)
				if err != nil {
					t.Fatalf("Project() error = %v", err)
				}
			} else {
				optimizedVectors = vectors
			}

			if tt.cfg.NumCentroids > 0 {
				// Apply quantization
				centroids, assignments, err := optimizations.Quantize(optimizedVectors, tt.cfg)
				if err != nil {
					t.Fatalf("Quantize() error = %v", err)
				}

				// Use centroids as optimized vectors
				optimizedVectors = make([][]float32, len(optimizedVectors))
				for i, cluster := range assignments {
					optimizedVectors[i] = centroids[cluster]
				}

				// Quantize query
				query, err = optimizations.Compress(query, centroids)
				if err != nil {
					t.Fatalf("Compress() error = %v", err)
				}
			}

			// Create index with optimized vectors
			idxCfg := index.Config{
				InitialCapacity: len(optimizedVectors),
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
			}
			idx := index.New(len(optimizedVectors[0]), idxCfg)
			ctx := context.Background()

			// Add vectors
			keys := make([]string, len(optimizedVectors))
			for i := range optimizedVectors {
				keys[i] = fmt.Sprintf("vec_%d", i)
			}
			err = idx.AddBatch(ctx, keys, optimizedVectors)
			if err != nil {
				t.Fatalf("AddBatch() error = %v", err)
			}

			// Build graph
			err = idx.BuildGraph(ctx, idxCfg)
			if err != nil {
				t.Fatalf("BuildGraph() error = %v", err)
			}

			// Search with optimized query
			k := 10
			results, err := idx.SearchNearest(ctx, query, k, idxCfg)
			if err != nil {
				t.Errorf("SearchNearest() error = %v", err)
				return
			}

			// Verify results
			if len(results) != k {
				t.Errorf("SearchNearest() returned %d results, want %d", len(results), k)
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
				if result.Score < idxCfg.ScoreThreshold {
					t.Errorf("Result %d has score %v below threshold %v",
						i, result.Score, idxCfg.ScoreThreshold)
				}
			}
		})
	}
}

func BenchmarkOptimizedSearch(b *testing.B) {
	// Create test data
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
		name string
		cfg  optimizations.Config
	}{
		{
			name: "no_optimization",
			cfg:  optimizations.Config{},
		},
		{
			name: "pca_only",
			cfg: optimizations.Config{
				TargetDimension: 32,
			},
		},
		{
			name: "quantization_only",
			cfg: optimizations.Config{
				NumCentroids:   100,
				MaxIterations:  100,
				ConvergenceEps: 1e-6,
			},
		},
		{
			name: "pca_and_quantization",
			cfg: optimizations.Config{
				TargetDimension: 32,
				NumCentroids:    100,
				MaxIterations:   100,
				ConvergenceEps:  1e-6,
			},
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Apply optimizations
			var optimizedVectors [][]float32
			var projection [][]float32
			var err error

			if bm.cfg.TargetDimension > 0 {
				// Apply PCA
				optimizedVectors, projection, err = optimizations.PCA(vectors, bm.cfg)
				if err != nil {
					b.Fatalf("PCA() error = %v", err)
				}

				// Project query
				query, err = optimizations.Project(query, projection)
				if err != nil {
					b.Fatalf("Project() error = %v", err)
				}
			} else {
				optimizedVectors = vectors
			}

			if bm.cfg.NumCentroids > 0 {
				// Apply quantization
				centroids, assignments, err := optimizations.Quantize(optimizedVectors, bm.cfg)
				if err != nil {
					b.Fatalf("Quantize() error = %v", err)
				}

				// Use centroids as optimized vectors
				optimizedVectors = make([][]float32, len(optimizedVectors))
				for i, cluster := range assignments {
					optimizedVectors[i] = centroids[cluster]
				}

				// Quantize query
				query, err = optimizations.Compress(query, centroids)
				if err != nil {
					b.Fatalf("Compress() error = %v", err)
				}
			}

			// Create index with optimized vectors
			idxCfg := index.Config{
				InitialCapacity: len(optimizedVectors),
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
			}
			idx := index.New(len(optimizedVectors[0]), idxCfg)
			ctx := context.Background()

			// Add vectors
			keys := make([]string, len(optimizedVectors))
			for i := range optimizedVectors {
				keys[i] = fmt.Sprintf("vec_%d", i)
			}
			err = idx.AddBatch(ctx, keys, optimizedVectors)
			if err != nil {
				b.Fatalf("AddBatch() error = %v", err)
			}

			// Build graph
			err = idx.BuildGraph(ctx, idxCfg)
			if err != nil {
				b.Fatalf("BuildGraph() error = %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := idx.SearchNearest(ctx, query, 10, idxCfg)
				if err != nil {
					b.Fatalf("SearchNearest() error = %v", err)
				}
			}
		})
	}
}
