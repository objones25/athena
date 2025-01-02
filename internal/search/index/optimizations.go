package index

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// PCAConfig holds configuration for PCA dimensionality reduction
type PCAConfig struct {
	TargetDimensions int     // Target number of dimensions after reduction
	MinVariance      float64 // Minimum variance to preserve (0.0-1.0)
}

// QuantizationConfig holds configuration for vector quantization
type QuantizationConfig struct {
	NumCentroids   int     // Number of centroids for quantization
	MaxIterations  int     // Maximum iterations for k-means
	ConvergenceEps float64 // Convergence threshold
	SampleSize     int     // Number of vectors to sample for training
}

// PCA performs principal component analysis for dimensionality reduction
func PCA(vectors [][]float32, cfg PCAConfig) ([][]float32, [][]float32, error) {
	if len(vectors) == 0 {
		return nil, nil, fmt.Errorf("empty vector set")
	}

	// Convert to float64 for better numerical stability
	n := len(vectors)
	d := len(vectors[0])
	data := make([][]float64, n)
	for i := range data {
		data[i] = make([]float64, d)
		for j := range data[i] {
			data[i][j] = float64(vectors[i][j])
		}
	}

	// Center the data
	mean := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			mean[j] += data[i][j]
		}
	}
	for j := 0; j < d; j++ {
		mean[j] /= float64(n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			data[i][j] -= mean[j]
		}
	}

	// Compute covariance matrix
	cov := make([][]float64, d)
	for i := range cov {
		cov[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			for k := 0; k < n; k++ {
				cov[i][j] += data[k][i] * data[k][j]
			}
			cov[i][j] /= float64(n - 1)
		}
	}

	// Compute eigenvalues and eigenvectors using power iteration
	eigenvalues := make([]float64, d)
	eigenvectors := make([][]float64, d)
	for i := range eigenvectors {
		eigenvectors[i] = make([]float64, d)
		// Initialize with random vector
		for j := range eigenvectors[i] {
			eigenvectors[i][j] = rand.Float64()
		}
		normalize(eigenvectors[i])
	}

	// Power iteration
	maxIter := 100
	eps := 1e-10
	for i := 0; i < d; i++ {
		v := eigenvectors[i]
		var prevLambda float64
		for iter := 0; iter < maxIter; iter++ {
			// Multiply by covariance matrix
			u := make([]float64, d)
			for j := 0; j < d; j++ {
				for k := 0; k < d; k++ {
					u[j] += cov[j][k] * v[k]
				}
			}

			// Normalize and update eigenvalue
			lambda := vectorNorm(u)
			if lambda < eps {
				break
			}
			for j := range u {
				v[j] = u[j] / lambda
			}

			// Check convergence
			if math.Abs(lambda-prevLambda) < eps {
				break
			}
			prevLambda = lambda
			eigenvalues[i] = lambda
		}

		// Deflate covariance matrix
		for j := 0; j < d; j++ {
			for k := 0; k < d; k++ {
				cov[j][k] -= eigenvalues[i] * v[j] * v[k]
			}
		}
	}

	// Sort eigenvalues and eigenvectors
	indices := make([]int, d)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return eigenvalues[indices[i]] > eigenvalues[indices[j]]
	})

	// Determine number of components to keep
	totalVar := 0.0
	for _, val := range eigenvalues {
		totalVar += val
	}

	var numComponents int
	if cfg.TargetDimensions > 0 {
		numComponents = cfg.TargetDimensions
	} else {
		cumVar := 0.0
		for i, idx := range indices {
			cumVar += eigenvalues[idx] / totalVar
			if cumVar >= cfg.MinVariance {
				numComponents = i + 1
				break
			}
		}
	}

	// Create projection matrix
	projection := make([][]float32, numComponents)
	for i := range projection {
		projection[i] = make([]float32, d)
		for j := range projection[i] {
			projection[i][j] = float32(eigenvectors[indices[i]][j])
		}
	}

	// Project data
	reduced := make([][]float32, n)
	for i := range reduced {
		reduced[i] = make([]float32, numComponents)
		for j := 0; j < numComponents; j++ {
			for k := 0; k < d; k++ {
				reduced[i][j] += float32(data[i][k]) * projection[j][k]
			}
		}
	}

	return reduced, projection, nil
}

// Quantize performs vector quantization using k-means clustering
func Quantize(vectors [][]float32, cfg QuantizationConfig) ([][]float32, []int, error) {
	if len(vectors) == 0 {
		return nil, nil, fmt.Errorf("empty vector set")
	}

	n := len(vectors)
	d := len(vectors[0])

	// Sample vectors for training if specified
	trainVectors := vectors
	if cfg.SampleSize > 0 && cfg.SampleSize < n {
		indices := rand.Perm(n)[:cfg.SampleSize]
		trainVectors = make([][]float32, cfg.SampleSize)
		for i, idx := range indices {
			trainVectors[i] = vectors[idx]
		}
	}

	// Initialize centroids randomly
	centroids := make([][]float32, cfg.NumCentroids)
	for i := range centroids {
		centroids[i] = make([]float32, d)
		srcIdx := rand.IntN(len(trainVectors))
		copy(centroids[i], trainVectors[srcIdx])
	}

	// K-means clustering
	assignments := make([]int, n)
	for iter := 0; iter < cfg.MaxIterations; iter++ {
		// Assign vectors to nearest centroids
		changed := false
		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)
			bestCentroid := 0
			for j, centroid := range centroids {
				dist := euclideanDistanceSquared(vec, centroid)
				if dist < minDist {
					minDist = dist
					bestCentroid = j
				}
			}
			if assignments[i] != bestCentroid {
				assignments[i] = bestCentroid
				changed = true
			}
		}

		if !changed {
			break
		}

		// Update centroids
		counts := make([]int, cfg.NumCentroids)
		newCentroids := make([][]float32, cfg.NumCentroids)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, d)
		}

		for i, vec := range vectors {
			c := assignments[i]
			counts[c]++
			for j := range vec {
				newCentroids[c][j] += vec[j]
			}
		}

		// Calculate mean
		maxMove := float32(0)
		for i := range centroids {
			if counts[i] > 0 {
				for j := range centroids[i] {
					newCentroids[i][j] /= float32(counts[i])
					move := math.Abs(float64(newCentroids[i][j] - centroids[i][j]))
					if float32(move) > maxMove {
						maxMove = float32(move)
					}
					centroids[i][j] = newCentroids[i][j]
				}
			}
		}

		// Check convergence
		if maxMove < float32(cfg.ConvergenceEps) {
			break
		}
	}

	return centroids, assignments, nil
}

// Helper functions

func normalize(v []float64) {
	norm := vectorNorm(v)
	if norm > 0 {
		for i := range v {
			v[i] /= norm
		}
	}
}

func vectorNorm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func euclideanDistanceSquared(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
