package optimizations

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// Quantize performs vector quantization using k-means clustering
func Quantize(vectors [][]float32, cfg Config) ([][]float32, []int, error) {
	if len(vectors) == 0 {
		return nil, nil, fmt.Errorf("empty input vectors")
	}

	dimensions := len(vectors[0])
	k := cfg.NumCentroids
	if k > len(vectors) {
		k = len(vectors)
	}

	// Initialize centroids using k-means++
	centroids := make([][]float32, k)
	centroids[0] = make([]float32, dimensions)
	copy(centroids[0], vectors[rand.Intn(len(vectors))])

	for i := 1; i < k; i++ {
		// Compute distances to nearest centroid for each point
		distances := make([]float64, len(vectors))
		var totalDist float64
		var wg sync.WaitGroup
		var mu sync.Mutex
		numWorkers := runtime.NumCPU()
		chunkSize := (len(vectors) + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				var localTotal float64
				for j := start; j < end && j < len(vectors); j++ {
					minDist := math.MaxFloat64
					for c := 0; c < i; c++ {
						dist := squaredDistance(vectors[j], centroids[c])
						if dist < minDist {
							minDist = dist
						}
					}
					distances[j] = minDist
					localTotal += minDist
				}
				mu.Lock()
				totalDist += localTotal
				mu.Unlock()
			}(w*chunkSize, (w+1)*chunkSize)
		}
		wg.Wait()

		// Choose next centroid with probability proportional to squared distance
		target := rand.Float64() * totalDist
		var sum float64
		var chosen int
		for j, dist := range distances {
			sum += dist
			if sum >= target {
				chosen = j
				break
			}
		}

		centroids[i] = make([]float32, dimensions)
		copy(centroids[i], vectors[chosen])
	}

	// K-means iterations
	assignments := make([]int, len(vectors))
	newAssignments := make([]int, len(vectors))
	var changed bool
	iteration := 0

	for iteration < cfg.MaxIterations {
		changed = false

		// Assign points to nearest centroids
		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU()
		chunkSize := (len(vectors) + numWorkers - 1) / numWorkers

		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end && i < len(vectors); i++ {
					minDist := math.MaxFloat64
					bestCentroid := 0
					for j := range centroids {
						dist := squaredDistance(vectors[i], centroids[j])
						if dist < minDist {
							minDist = dist
							bestCentroid = j
						}
					}
					newAssignments[i] = bestCentroid
				}
			}(w*chunkSize, (w+1)*chunkSize)
		}
		wg.Wait()

		// Check for changes in assignments
		for i := range assignments {
			if assignments[i] != newAssignments[i] {
				changed = true
				assignments[i] = newAssignments[i]
			}
		}

		if !changed {
			break
		}

		// Update centroids
		counts := make([]int, k)
		newCentroids := make([][]float32, k)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, dimensions)
		}

		for i, cluster := range assignments {
			counts[cluster]++
			for j := range vectors[i] {
				newCentroids[cluster][j] += vectors[i][j]
			}
		}

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

		if maxMove < float32(cfg.ConvergenceEps) {
			break
		}

		iteration++
	}

	return centroids, assignments, nil
}

// squaredDistance computes the squared Euclidean distance between two vectors
func squaredDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return sum
}

// FindNearest finds the nearest centroid to a vector
func FindNearest(vector []float32, centroids [][]float32) (int, error) {
	if len(vector) != len(centroids[0]) {
		return 0, fmt.Errorf("vector dimension mismatch: got %d, want %d",
			len(vector), len(centroids[0]))
	}

	minDist := math.MaxFloat64
	nearest := 0
	for i, centroid := range centroids {
		dist := squaredDistance(vector, centroid)
		if dist < minDist {
			minDist = dist
			nearest = i
		}
	}

	return nearest, nil
}

// Compress compresses a vector by quantizing it to the nearest centroid
func Compress(vector []float32, centroids [][]float32) ([]float32, error) {
	nearest, err := FindNearest(vector, centroids)
	if err != nil {
		return nil, err
	}
	return centroids[nearest], nil
}
