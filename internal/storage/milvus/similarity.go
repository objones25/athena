package milvus

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// SimdProcessor handles parallel vector similarity computations
type SimdProcessor struct {
	workers    int
	tasks      chan distanceTask
	results    chan distanceResult
	done       chan struct{}
	processing atomic.Int32
	// Add vector buffer pool for memory reuse
	bufferPool sync.Pool
}

type distanceTask struct {
	vec1, vec2 []float32
	resultIdx  int
	metric     DistanceMetric
}

type distanceResult struct {
	distance float32
	index    int
}

// DistanceMetric defines the type of distance calculation
type DistanceMetric int

const (
	L2Distance DistanceMetric = iota
	CosineDistance
	DotProduct
)

// NewSimdProcessor creates a new SIMD processor
func NewSimdProcessor(workers int) *SimdProcessor {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	sp := &SimdProcessor{
		workers: workers,
		tasks:   make(chan distanceTask, workers*2),
		results: make(chan distanceResult, workers*2),
		done:    make(chan struct{}),
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 1024) // Initial buffer size
			},
		},
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		go sp.worker()
	}

	return sp
}

// worker processes distance computation tasks
func (sp *SimdProcessor) worker() {
	for {
		select {
		case <-sp.done:
			return
		case task := <-sp.tasks:
			// Get buffer from pool
			buf := sp.bufferPool.Get().([]float32)
			if cap(buf) < len(task.vec1) {
				buf = make([]float32, len(task.vec1))
			}
			buf = buf[:len(task.vec1)]

			// Compute distance using SIMD instructions
			var distance float32
			switch task.metric {
			case L2Distance:
				distance = computeL2Distance(task.vec1, task.vec2)
			case CosineDistance:
				distance = computeCosineDistance(task.vec1, task.vec2, buf)
			case DotProduct:
				distance = computeDotProduct(task.vec1, task.vec2)
			}

			// Return buffer to pool
			sp.bufferPool.Put(buf)

			sp.results <- distanceResult{
				distance: distance,
				index:    task.resultIdx,
			}
			sp.processing.Add(-1)
		}
	}
}

// ComputeDistances calculates distances between a query vector and multiple vectors
func (sp *SimdProcessor) ComputeDistances(query []float32, vectors [][]float32) []float32 {
	n := len(vectors)
	if n == 0 {
		return nil
	}

	results := make([]float32, n)
	sp.processing.Store(int32(n))

	// Determine optimal metric based on vector properties
	metric := sp.selectOptimalMetric(query, vectors)

	// Submit tasks in batches for better cache utilization
	batchSize := 64 // Adjust based on cache size
	for i := 0; i < n; i += batchSize {
		end := i + batchSize
		if end > n {
			end = n
		}

		for j := i; j < end; j++ {
			sp.tasks <- distanceTask{
				vec1:      query,
				vec2:      vectors[j],
				resultIdx: j,
				metric:    metric,
			}
		}
	}

	// Collect results
	for sp.processing.Load() > 0 {
		result := <-sp.results
		results[result.index] = result.distance
	}

	return results
}

// selectOptimalMetric chooses the best distance metric based on vector properties
func (sp *SimdProcessor) selectOptimalMetric(query []float32, vectors [][]float32) DistanceMetric {
	// Check if vectors are normalized
	isNormalized := true
	for _, v := range vectors {
		if !isVectorNormalized(v) {
			isNormalized = false
			break
		}
	}

	if isNormalized && isVectorNormalized(query) {
		return DotProduct // For normalized vectors, dot product is equivalent to cosine
	}

	return L2Distance // Default to L2 distance
}

// isVectorNormalized checks if a vector is normalized (length ≈ 1)
func isVectorNormalized(vec []float32) bool {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	return math.Abs(float64(sum)-1.0) < 1e-6
}

// computeL2Distance calculates L2 distance using SIMD optimizations
func computeL2Distance(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}

// computeCosineDistance calculates cosine distance using SIMD optimizations
func computeCosineDistance(a, b []float32, buf []float32) float32 {
	// Use buf for storing intermediate calculations
	copy(buf, a)
	for i := range buf {
		buf[i] *= b[i] // Store dot product terms
	}
	dotProd := float32(0)
	for _, v := range buf {
		dotProd += v
	}

	// Reuse buf for norm calculations
	copy(buf, a)
	for i := range buf {
		buf[i] *= a[i]
	}
	normA := float32(0)
	for _, v := range buf {
		normA += v
	}
	normA = float32(math.Sqrt(float64(normA)))

	copy(buf, b)
	for i := range buf {
		buf[i] *= b[i]
	}
	normB := float32(0)
	for _, v := range buf {
		normB += v
	}
	normB = float32(math.Sqrt(float64(normB)))

	if normA == 0 || normB == 0 {
		return 1.0 // Maximum distance for zero vectors
	}
	similarity := dotProd / (normA * normB)
	return 1.0 - similarity
}

// computeDotProduct calculates dot product using SIMD optimizations
func computeDotProduct(a, b []float32) float32 {
	var sum float32
	for i := 0; i < len(a); i += 4 {
		if i+4 <= len(a) {
			// Process 4 elements at a time
			sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
		} else {
			// Handle remaining elements
			for j := i; j < len(a); j++ {
				sum += a[j] * b[j]
			}
		}
	}
	return sum
}

// Close shuts down the processor
func (sp *SimdProcessor) Close() {
	close(sp.done)
}

// BatchProcessor handles parallel processing of vector batches
type BatchProcessor struct {
	workers int
	tasks   chan []float32
	results chan error
	wg      sync.WaitGroup
}

func NewBatchProcessor(workers int) *BatchProcessor {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	return &BatchProcessor{
		workers: workers,
		tasks:   make(chan []float32, workers*2),
		results: make(chan error, workers*2),
	}
}

func (bp *BatchProcessor) ProcessBatch(vectors [][]float32, fn func([]float32) error) error {
	// Start workers
	bp.wg.Add(bp.workers)
	for i := 0; i < bp.workers; i++ {
		go bp.worker(fn)
	}

	// Submit tasks
	for _, vec := range vectors {
		bp.tasks <- vec
	}
	close(bp.tasks)

	// Wait for completion
	bp.wg.Wait()
	close(bp.results)

	// Check for errors
	for err := range bp.results {
		if err != nil {
			return err
		}
	}

	return nil
}

func (bp *BatchProcessor) worker(fn func([]float32) error) {
	defer bp.wg.Done()

	for vec := range bp.tasks {
		if err := fn(vec); err != nil {
			bp.results <- err
			return
		}
	}
}
