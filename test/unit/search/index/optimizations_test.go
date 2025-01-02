package index_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/objones25/athena/internal/search/index"
)

func TestDimensionalityReduction(t *testing.T) {
	// Create test data with known structure
	dimensions := 10
	numVectors := 1000
	vectors := make([][]float32, numVectors)
	rng := rand.New(rand.NewSource(42))

	// Generate vectors with most variance in first two dimensions
	for i := range vectors {
		vectors[i] = make([]float32, dimensions)
		vectors[i][0] = float32(math.Sin(float64(i) * 0.1)) // Primary direction
		vectors[i][1] = float32(math.Cos(float64(i) * 0.1)) // Secondary direction
		// Add noise to other dimensions
		for j := 2; j < dimensions; j++ {
			vectors[i][j] = float32(0.01 * (rng.Float64() - 0.5))
		}
	}

	tests := []struct {
		name          string
		targetDim     int
		minVariance   float64
		wantDimension int
	}{
		{
			name:          "fixed dimensions",
			targetDim:     2,
			wantDimension: 2,
		},
		{
			name:          "variance based",
			minVariance:   0.95,
			wantDimension: 2, // Should capture most variance in 2 dimensions
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := index.Config{
				InitialCapacity: numVectors,
				MaxConnections:  32,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
				TargetDimension: tt.targetDim,
				MinVariance:     tt.minVariance,
			}

			// Create index with dimensionality reduction
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

			// Create query vector
			query := make([]float32, dimensions)
			for i := range query {
				query[i] = float32(rng.Float64())
			}

			// Search in reduced space
			results, err := idx.SearchNearest(ctx, query, 10, cfg)
			if err != nil {
				t.Errorf("SearchNearest() error = %v", err)
				return
			}

			// Verify results
			if len(results) == 0 {
				t.Error("SearchNearest() returned no results")
			}

			// Verify results are sorted by score
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Errorf("Results not sorted: score[%d]=%v > score[%d]=%v",
						i, results[i].Score, i-1, results[i-1].Score)
				}
			}
		})
	}
}

func TestQuantization(t *testing.T) {
	// Create test data with known clusters
	dimensions := 4
	numVectors := 1000
	numClusters := 4
	vectors := make([][]float32, numVectors)
	rng := rand.New(rand.NewSource(42))

	// Generate vectors around known centroids
	centroids := make([][]float32, numClusters)
	for i := range centroids {
		centroids[i] = make([]float32, dimensions)
		for j := range centroids[i] {
			centroids[i][j] = float32(rng.Float64())
		}
	}

	for i := range vectors {
		cluster := i % numClusters
		vectors[i] = make([]float32, dimensions)
		// Add noise around centroid
		for j := range vectors[i] {
			noise := float32(0.1 * (rng.Float64() - 0.5))
			vectors[i][j] = centroids[cluster][j] + noise
		}
	}

	tests := []struct {
		name         string
		numCentroids int
		maxIter      int
		eps          float64
	}{
		{
			name:         "basic quantization",
			numCentroids: numClusters,
			maxIter:      100,
			eps:          1e-6,
		},
		{
			name:         "more centroids",
			numCentroids: numClusters * 2,
			maxIter:      100,
			eps:          1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := index.Config{
				InitialCapacity: numVectors,
				MaxConnections:  32,
				BuildBatchSize:  100,
				ScoreThreshold:  0.5,
				NumCentroids:    tt.numCentroids,
				MaxIterations:   tt.maxIter,
				ConvergenceEps:  tt.eps,
			}

			// Create index with quantization
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

			// Create query vector near a known centroid
			query := make([]float32, dimensions)
			centroidIdx := rng.Intn(numClusters)
			for j := range query {
				noise := float32(0.05 * (rng.Float64() - 0.5))
				query[j] = centroids[centroidIdx][j] + noise
			}

			// Search with quantization
			results, err := idx.SearchNearest(ctx, query, 10, cfg)
			if err != nil {
				t.Errorf("SearchNearest() error = %v", err)
				return
			}

			// Verify results
			if len(results) == 0 {
				t.Error("SearchNearest() returned no results")
			}

			// Verify results are sorted by score
			for i := 1; i < len(results); i++ {
				if results[i].Score > results[i-1].Score {
					t.Errorf("Results not sorted: score[%d]=%v > score[%d]=%v",
						i, results[i].Score, i-1, results[i-1].Score)
				}
			}

			// Verify that some vectors from the same cluster are found
			found := false
			for _, result := range results {
				resultIdx, _ := strconv.Atoi(strings.TrimPrefix(result.Key, "vec_"))
				if resultIdx%numClusters == centroidIdx {
					found = true
					break
				}
			}
			if !found {
				t.Error("No vectors from the same cluster found")
			}
		})
	}
}

func BenchmarkOptimizations(b *testing.B) {
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
			name: "no_optimization",
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
			name: "with_pca",
			config: index.Config{
				InitialCapacity: 100000,
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
				TargetDimension: 32,
			},
			k:       10,
			vectors: vectors,
		},
		{
			name: "with_quantization",
			config: index.Config{
				InitialCapacity: 100000,
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
				NumCentroids:    1000,
				MaxIterations:   100,
				ConvergenceEps:  1e-6,
			},
			k:       10,
			vectors: vectors,
		},
		{
			name: "pca_and_quantization",
			config: index.Config{
				InitialCapacity: 100000,
				MaxConnections:  32,
				BuildBatchSize:  1000,
				ScoreThreshold:  0.5,
				TargetDimension: 32,
				NumCentroids:    1000,
				MaxIterations:   100,
				ConvergenceEps:  1e-6,
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
