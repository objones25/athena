package milvus

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/rs/zerolog/log"
)

// QuantizationType represents different quantization algorithms
type QuantizationType int

const (
	StandardQuantization QuantizationType = iota
	ProductQuantization
	OptimisticQuantization
	ResidualQuantization
)

// QuantizationConfig holds configuration for vector quantization
type QuantizationConfig struct {
	NumCentroids     int              // Number of centroids for quantization
	MaxIterations    int              // Maximum k-means iterations
	ConvergenceEps   float64          // Convergence threshold
	SampleSize       int              // Number of vectors to sample for training
	BatchSize        int              // Size of batches for processing
	UpdateInterval   time.Duration    // How often to update centroids
	NumWorkers       int              // Number of worker goroutines
	QuantizationType QuantizationType // Type of quantization to use
	NumSubspaces     int              // Number of subspaces for PQ
	NumBitsPerIdx    int              // Number of bits per subquantizer index
	OptimisticProbe  int              // Number of centroids to check in optimistic search
}

// DefaultQuantizationConfig returns the default configuration
func DefaultQuantizationConfig() QuantizationConfig {
	return QuantizationConfig{
		NumCentroids:     256,
		MaxIterations:    100,
		ConvergenceEps:   1e-6,
		SampleSize:       10000,
		BatchSize:        1000,
		UpdateInterval:   1 * time.Hour,
		NumWorkers:       runtime.NumCPU(),
		QuantizationType: StandardQuantization,
		NumSubspaces:     8,
		NumBitsPerIdx:    8,
		OptimisticProbe:  8,
	}
}

// Quantizer handles vector quantization
type Quantizer struct {
	config     QuantizationConfig
	centroids  [][]float32
	dimension  int
	version    uint64
	lastUpdate time.Time
	mu         sync.RWMutex

	// Worker pools and channels
	vectorChan   chan []float32
	resultChan   chan quantizeResult
	workerPool   chan struct{}
	updateTicker *time.Ticker
	done         chan struct{}

	// Add fields for Product Quantization
	subQuantizers   []*Quantizer                  // Subquantizers for PQ
	optimisticCache *lru.Cache[string, []float32] // Cache for optimistic search
}

type quantizeResult struct {
	vector    []float32
	centroid  int
	distances []float32
}

// NewQuantizer creates a new vector quantizer
func NewQuantizer(config QuantizationConfig, dimension int) *Quantizer {
	// Validate and set defaults
	if config.NumCentroids <= 0 {
		config.NumCentroids = 256
	}
	if config.MaxIterations <= 0 {
		config.MaxIterations = 100
	}
	if config.ConvergenceEps <= 0 {
		config.ConvergenceEps = 0.001
	}
	if config.SampleSize <= 0 {
		config.SampleSize = 10000
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 1000
	}
	if config.UpdateInterval <= 0 {
		config.UpdateInterval = time.Hour // Default to 1 hour if not set
	}
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}

	// Initialize cache for optimistic quantization
	cache, _ := lru.New[string, []float32](1000) // Cache size of 1000 vectors

	q := &Quantizer{
		config:          config,
		dimension:       dimension,
		vectorChan:      make(chan []float32, config.BatchSize),
		resultChan:      make(chan quantizeResult, config.BatchSize),
		workerPool:      make(chan struct{}, config.NumWorkers),
		done:            make(chan struct{}),
		optimisticCache: cache,
	}

	// Start worker goroutines
	for i := 0; i < config.NumWorkers; i++ {
		go q.worker()
	}

	// Start update monitor with validated interval
	q.updateTicker = time.NewTicker(config.UpdateInterval)
	go q.monitorUpdates()

	return q
}

// Close shuts down the quantizer
func (q *Quantizer) Close() {
	close(q.done)
	q.updateTicker.Stop()
}

// QuantizeVector quantizes a single vector
func (q *Quantizer) QuantizeVector(ctx context.Context, vector []float32) (int, []float32, error) {
	if len(vector) != q.dimension {
		return 0, nil, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), q.dimension)
	}

	switch q.config.QuantizationType {
	case ProductQuantization:
		indices, distances, err := q.quantizeProductVector(vector)
		if err != nil {
			return 0, nil, err
		}
		// Convert multiple indices to single index
		index := 0
		for i, idx := range indices {
			index |= idx << (uint(i) * uint(q.config.NumBitsPerIdx))
		}
		return index, distances, nil

	case OptimisticQuantization:
		return q.quantizeOptimistic(vector)

	case ResidualQuantization:
		indices, distances, err := q.quantizeResidual(vector)
		if err != nil {
			return 0, nil, err
		}
		// Use first index as main index
		if len(indices) == 0 {
			return 0, distances, nil
		}
		return indices[0], distances, nil

	default:
		// Use standard quantization
		return q.standardQuantizeVector(ctx, vector)
	}
}

// QuantizeBatch quantizes multiple vectors in parallel
func (q *Quantizer) QuantizeBatch(ctx context.Context, vectors [][]float32) ([]int, [][]float32, error) {
	if len(vectors) == 0 {
		return nil, nil, nil
	}

	centroids := make([]int, len(vectors))
	distances := make([][]float32, len(vectors))
	var wg sync.WaitGroup
	errChan := make(chan error, len(vectors))

	// Process vectors in batches
	for i := 0; i < len(vectors); i += q.config.BatchSize {
		end := i + q.config.BatchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				centroid, dist, err := q.QuantizeVector(ctx, vectors[j])
				if err != nil {
					errChan <- err
					return
				}
				centroids[j] = centroid
				distances[j] = dist
			}
		}(i, end)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	if err := <-errChan; err != nil {
		return nil, nil, err
	}

	return centroids, distances, nil
}

// UpdateCentroids updates the centroids using k-means clustering
func (q *Quantizer) UpdateCentroids(ctx context.Context, vectors [][]float32) error {
	if len(vectors) == 0 {
		return fmt.Errorf("no vectors provided for centroid update")
	}

	// Sample vectors if needed
	if len(vectors) > q.config.SampleSize {
		sampled := make([][]float32, q.config.SampleSize)
		for i := range sampled {
			idx := rand.Intn(len(vectors))
			sampled[i] = vectors[idx]
		}
		vectors = sampled
	}

	// Initialize centroids randomly
	centroids := make([][]float32, q.config.NumCentroids)
	for i := range centroids {
		centroids[i] = vectors[rand.Intn(len(vectors))]
	}

	// K-means iterations
	assignments := make([]int, len(vectors))
	for iter := 0; iter < q.config.MaxIterations; iter++ {
		// Assign vectors to nearest centroids
		var maxChange float64
		for i, vec := range vectors {
			minDist := math.MaxFloat64
			bestCentroid := 0
			for j, centroid := range centroids {
				dist := euclideanDistance(vec, centroid)
				if dist < minDist {
					minDist = dist
					bestCentroid = j
				}
			}
			if assignments[i] != bestCentroid {
				maxChange++
				assignments[i] = bestCentroid
			}
		}

		// Check convergence
		changeRate := maxChange / float64(len(vectors))
		if changeRate < q.config.ConvergenceEps {
			break
		}

		// Update centroids
		counts := make([]int, len(centroids))
		newCentroids := make([][]float32, len(centroids))
		for i := range newCentroids {
			newCentroids[i] = make([]float32, q.dimension)
		}

		for i, vec := range vectors {
			centroid := assignments[i]
			counts[centroid]++
			for j := range vec {
				newCentroids[centroid][j] += vec[j]
			}
		}

		for i := range newCentroids {
			if counts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float32(counts[i])
				}
			} else {
				// Reinitialize empty centroids
				newCentroids[i] = vectors[rand.Intn(len(vectors))]
			}
		}

		centroids = newCentroids
	}

	// Update centroids atomically
	q.mu.Lock()
	q.centroids = centroids
	atomic.AddUint64(&q.version, 1)
	q.lastUpdate = time.Now()
	q.mu.Unlock()

	log.Info().
		Int("num_centroids", len(centroids)).
		Int("num_vectors", len(vectors)).
		Time("update_time", q.lastUpdate).
		Msg("Updated quantization centroids")

	return nil
}

func (q *Quantizer) worker() {
	for {
		select {
		case <-q.done:
			return
		case vector := <-q.vectorChan:
			q.mu.RLock()
			centroids := q.centroids
			q.mu.RUnlock()

			if centroids == nil {
				q.resultChan <- quantizeResult{vector: vector}
				continue
			}

			// Find nearest centroid
			minDist := math.MaxFloat64
			nearest := 0
			distances := make([]float32, len(centroids))

			for i, centroid := range centroids {
				dist := euclideanDistance(vector, centroid)
				distances[i] = float32(dist)
				if dist < minDist {
					minDist = dist
					nearest = i
				}
			}

			q.resultChan <- quantizeResult{
				vector:    vector,
				centroid:  nearest,
				distances: distances,
			}
		}
	}
}

func (q *Quantizer) monitorUpdates() {
	for {
		select {
		case <-q.done:
			return
		case <-q.updateTicker.C:
			if time.Since(q.lastUpdate) >= q.config.UpdateInterval {
				log.Debug().Msg("Triggering centroid update")
				// Actual update will be triggered by the store
			}
		}
	}
}

func euclideanDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Add method for Product Quantization
func (q *Quantizer) quantizeProductVector(vector []float32) ([]int, []float32, error) {
	if len(vector) != q.dimension {
		return nil, nil, fmt.Errorf("invalid vector dimension")
	}

	subDim := q.dimension / q.config.NumSubspaces
	indices := make([]int, q.config.NumSubspaces)
	distances := make([]float32, q.config.NumSubspaces)

	// Process each subspace in parallel
	var wg sync.WaitGroup
	wg.Add(q.config.NumSubspaces)
	errChan := make(chan error, q.config.NumSubspaces)

	for i := 0; i < q.config.NumSubspaces; i++ {
		go func(subIdx int) {
			defer wg.Done()

			start := subIdx * subDim
			end := start + subDim
			subvec := vector[start:end]

			// Quantize subvector
			centroid, dist, err := q.subQuantizers[subIdx].QuantizeVector(context.Background(), subvec)
			if err != nil {
				errChan <- err
				return
			}

			indices[subIdx] = centroid
			distances[subIdx] = dist[0]
		}(i)
	}

	wg.Wait()
	close(errChan)

	if err := <-errChan; err != nil {
		return nil, nil, err
	}

	return indices, distances, nil
}

// quantizeOptimistic uses optimistic quantization with caching
func (q *Quantizer) quantizeOptimistic(vector []float32) (int, []float32, error) {
	// Generate cache key
	key := vectorKey(vector)

	// Check cache
	if cached, ok := q.optimisticCache.Get(key); ok {
		// Use cached result
		minDist := math.MaxFloat64
		nearest := 0
		distances := make([]float32, len(q.centroids))

		for i, centroid := range q.centroids {
			dist := euclideanDistance(cached, centroid)
			distances[i] = float32(dist)
			if dist < minDist {
				minDist = dist
				nearest = i
			}
		}
		return nearest, distances, nil
	}

	// Not in cache, perform standard quantization
	centroid, distances, err := q.standardQuantizeVector(context.Background(), vector)
	if err != nil {
		return 0, nil, err
	}

	// Cache the result
	q.optimisticCache.Add(key, vector)

	return centroid, distances, nil
}

// Add method for Residual Quantization
func (q *Quantizer) quantizeResidual(vector []float32) ([]int, []float32, error) {
	residual := make([]float32, len(vector))
	copy(residual, vector)

	var indices []int
	var totalDistances []float32

	// Iteratively quantize residuals
	for i := 0; i < len(q.subQuantizers); i++ {
		// Quantize current residual
		centroid, distances, err := q.subQuantizers[i].QuantizeVector(context.Background(), residual)
		if err != nil {
			return nil, nil, err
		}

		indices = append(indices, centroid)
		totalDistances = append(totalDistances, distances...)

		// Compute new residual
		centroidVec := q.subQuantizers[i].centroids[centroid]
		for j := range residual {
			residual[j] -= centroidVec[j]
		}

		// Stop if residual is small enough
		if vectorNorm(residual) < float32(q.config.ConvergenceEps) {
			break
		}
	}

	return indices, totalDistances, nil
}

// Helper function to compute vector norm
func vectorNorm(v []float32) float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// Helper function to generate vector key for cache
func vectorKey(v []float32) string {
	h := fnv.New64a()
	for _, x := range v {
		binary.Write(h, binary.LittleEndian, x)
	}
	return fmt.Sprintf("%x", h.Sum64())
}

// Rename original QuantizeVector to standardQuantizeVector
func (q *Quantizer) standardQuantizeVector(ctx context.Context, vector []float32) (int, []float32, error) {
	if len(vector) != q.dimension {
		return 0, nil, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), q.dimension)
	}

	q.mu.RLock()
	if q.centroids == nil {
		q.mu.RUnlock()
		return 0, vector, nil // Return original if not initialized
	}
	currentVersion := atomic.LoadUint64(&q.version)
	q.mu.RUnlock()

	// Send vector for processing
	select {
	case q.vectorChan <- vector:
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	}

	// Wait for result
	select {
	case result := <-q.resultChan:
		// Check if centroids were updated during processing
		if currentVersion != atomic.LoadUint64(&q.version) {
			// Retry once if versions don't match
			select {
			case q.vectorChan <- vector:
			case <-ctx.Done():
				return 0, nil, ctx.Err()
			}
			select {
			case result = <-q.resultChan:
			case <-ctx.Done():
				return 0, nil, ctx.Err()
			}
		}
		return result.centroid, result.distances, nil
	case <-ctx.Done():
		return 0, nil, ctx.Err()
	}
}
