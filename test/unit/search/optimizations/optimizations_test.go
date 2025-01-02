package optimizations_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/objones25/athena/internal/search/optimizations"
)

func TestPCA(t *testing.T) {
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
			cfg := optimizations.Config{
				TargetDimension: tt.targetDim,
				MinVariance:     tt.minVariance,
			}

			reduced, projection, err := optimizations.PCA(vectors, cfg)
			if err != nil {
				t.Errorf("PCA() error = %v", err)
				return
			}

			// Check dimensions
			if len(reduced) != len(vectors) {
				t.Errorf("PCA() wrong number of vectors = %v, want %v", len(reduced), len(vectors))
			}
			if len(reduced[0]) != tt.wantDimension {
				t.Errorf("PCA() wrong output dimensions = %v, want %v", len(reduced[0]), tt.wantDimension)
			}
			if len(projection) != tt.wantDimension {
				t.Errorf("PCA() wrong projection dimensions = %v, want %v", len(projection), tt.wantDimension)
			}

			// Verify that projection preserves distances approximately
			for i := 0; i < len(vectors); i++ {
				for j := i + 1; j < len(vectors); j++ {
					origDist := squaredDistance(vectors[i], vectors[j])
					redDist := squaredDistance(reduced[i], reduced[j])
					relError := math.Abs(origDist-redDist) / origDist
					if relError > 0.2 { // Allow 20% error due to dimensionality reduction
						t.Errorf("PCA() distance not preserved: original = %v, reduced = %v", origDist, redDist)
					}
				}
			}

			// Test projection and reconstruction
			for i := 0; i < 10; i++ {
				// Create random test vector
				test := make([]float32, dimensions)
				for j := range test {
					test[j] = float32(rng.Float64())
				}

				// Project and reconstruct
				projected, err := optimizations.Project(test, projection)
				if err != nil {
					t.Errorf("Project() error = %v", err)
					continue
				}

				reconstructed, err := optimizations.Reconstruct(projected, projection)
				if err != nil {
					t.Errorf("Reconstruct() error = %v", err)
					continue
				}

				// Check reconstruction error
				var mse float64
				for j := range test {
					diff := float64(test[j] - reconstructed[j])
					mse += diff * diff
				}
				mse /= float64(dimensions)

				if mse > 0.1 { // Allow reasonable reconstruction error
					t.Errorf("Reconstruction error too high: MSE = %v", mse)
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
			cfg := optimizations.Config{
				NumCentroids:   tt.numCentroids,
				MaxIterations:  tt.maxIter,
				ConvergenceEps: tt.eps,
			}

			foundCentroids, assignments, err := optimizations.Quantize(vectors, cfg)
			if err != nil {
				t.Errorf("Quantize() error = %v", err)
				return
			}

			// Check number of centroids
			if len(foundCentroids) != tt.numCentroids {
				t.Errorf("Quantize() wrong number of centroids = %v, want %v", len(foundCentroids), tt.numCentroids)
			}

			// Check assignments
			if len(assignments) != len(vectors) {
				t.Errorf("Quantize() wrong number of assignments = %v, want %v", len(assignments), len(vectors))
			}

			// Verify that vectors are assigned to nearest centroid
			for i, vec := range vectors {
				cluster := assignments[i]
				dist := squaredDistance(vec, foundCentroids[cluster])

				// Check if any other centroid is closer
				for j, centroid := range foundCentroids {
					if j != cluster {
						otherDist := squaredDistance(vec, centroid)
						if otherDist < dist {
							t.Errorf("Vector assigned to non-nearest centroid: dist = %v, better = %v", dist, otherDist)
						}
					}
				}
			}

			// Verify cluster sizes are reasonably balanced
			counts := make([]int, tt.numCentroids)
			for _, cluster := range assignments {
				counts[cluster]++
			}

			expectedSize := len(vectors) / tt.numCentroids
			for i, count := range counts {
				// Allow 50% deviation from expected size
				if count < expectedSize/2 || count > expectedSize*3/2 {
					t.Errorf("Cluster %d has unbalanced size: %d (expected around %d)", i, count, expectedSize)
				}
			}

			// Test compression
			for i := 0; i < 10; i++ {
				// Create random test vector
				test := make([]float32, dimensions)
				for j := range test {
					test[j] = float32(rng.Float64())
				}

				// Compress vector
				compressed, err := optimizations.Compress(test, foundCentroids)
				if err != nil {
					t.Errorf("Compress() error = %v", err)
					continue
				}

				// Verify compressed vector is one of the centroids
				found := false
				for _, centroid := range foundCentroids {
					if vectorEqual(compressed, centroid) {
						found = true
						break
					}
				}
				if !found {
					t.Error("Compressed vector is not a centroid")
				}
			}
		})
	}
}

func BenchmarkOptimizations(b *testing.B) {
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

	b.Run("PCA", func(b *testing.B) {
		cfg := optimizations.Config{
			TargetDimension: 32,
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, err := optimizations.PCA(vectors, cfg)
			if err != nil {
				b.Fatalf("PCA() error = %v", err)
			}
		}
	})

	b.Run("Quantization", func(b *testing.B) {
		cfg := optimizations.Config{
			NumCentroids:   100,
			MaxIterations:  100,
			ConvergenceEps: 1e-6,
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, err := optimizations.Quantize(vectors, cfg)
			if err != nil {
				b.Fatalf("Quantize() error = %v", err)
			}
		}
	})
}

// Helper functions

func squaredDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
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
