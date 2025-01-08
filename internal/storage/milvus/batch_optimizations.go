package milvus

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
)

// SIMDBatchProcessor handles parallel vector batch processing with SIMD optimizations
type SIMDBatchProcessor struct {
	dimension  int
	numWorkers int
	tasks      chan batchTask
	results    chan batchResult
	bufferPool sync.Pool
	done       chan struct{}
	processing atomic.Int32
}

type batchTask struct {
	vectors   [][]float32
	query     []float32
	resultIdx int
	distFunc  func([]float32, []float32) float32
}

type batchResult struct {
	distances []float32
	index     int
}

// NewSIMDBatchProcessor creates a new batch processor
func NewSIMDBatchProcessor(dimension int) *SIMDBatchProcessor {
	numWorkers := runtime.NumCPU()
	bp := &SIMDBatchProcessor{
		dimension:  dimension,
		numWorkers: numWorkers,
		tasks:      make(chan batchTask, numWorkers*2),
		results:    make(chan batchResult, numWorkers*2),
		done:       make(chan struct{}),
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}

	// Start worker goroutines
	for i := 0; i < numWorkers; i++ {
		go bp.worker()
	}

	return bp
}

// worker processes batch tasks
func (bp *SIMDBatchProcessor) worker() {
	for {
		select {
		case <-bp.done:
			return
		case task := <-bp.tasks:
			// Process batch with SIMD
			distances := make([]float32, len(task.vectors))
			for i, vec := range task.vectors {
				distances[i] = task.distFunc(task.query, vec)
			}

			bp.results <- batchResult{
				distances: distances,
				index:     task.resultIdx,
			}
			bp.processing.Add(-1)
		}
	}
}

// ProcessBatch processes a batch of vectors in parallel
func (bp *SIMDBatchProcessor) ProcessBatch(query []float32, vectors [][]float32, distFunc func([]float32, []float32) float32) []float32 {
	if len(vectors) == 0 {
		return nil
	}

	// Calculate optimal batch size
	batchSize := (len(vectors) + bp.numWorkers - 1) / bp.numWorkers
	numBatches := (len(vectors) + batchSize - 1) / batchSize

	// Submit tasks
	bp.processing.Store(int32(numBatches))
	for i := 0; i < len(vectors); i += batchSize {
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		bp.tasks <- batchTask{
			vectors:   vectors[i:end],
			query:     query,
			resultIdx: i,
			distFunc:  distFunc,
		}
	}

	// Collect results
	distances := make([]float32, len(vectors))
	for bp.processing.Load() > 0 {
		result := <-bp.results
		copy(distances[result.index:], result.distances)
	}

	return distances
}

// Close shuts down the processor
func (bp *SIMDBatchProcessor) Close() {
	close(bp.done)
}

// ResidualQuantizer implements residual vector quantization
type ResidualQuantizer struct {
	dimension    int
	numLevels    int
	numCentroids int
	codebooks    [][][]float32
	mu           sync.RWMutex
	bufferPool   sync.Pool
}

// NewResidualQuantizer creates a new residual quantizer
func NewResidualQuantizer(dimension, numLevels, numCentroids int) *ResidualQuantizer {
	return &ResidualQuantizer{
		dimension:    dimension,
		numLevels:    numLevels,
		numCentroids: numCentroids,
		codebooks:    make([][][]float32, numLevels),
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
}

// Train trains the residual quantizer on a set of vectors
func (rq *ResidualQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return nil
	}

	rq.mu.Lock()
	defer rq.mu.Unlock()

	// Get buffer from pool
	residuals := make([][]float32, len(vectors))
	for i := range residuals {
		residuals[i] = make([]float32, rq.dimension)
		copy(residuals[i], vectors[i])
	}

	// Train each level
	for level := 0; level < rq.numLevels; level++ {
		// Run k-means clustering on residuals
		centroids := rq.kmeans(residuals, rq.numCentroids)
		rq.codebooks[level] = centroids

		// Compute residuals for next level
		for _, vec := range residuals {
			// Find nearest centroid
			minDist := float32(1e10)
			var nearest []float32
			for _, centroid := range centroids {
				dist := computeL2Distance(vec, centroid)
				if dist < minDist {
					minDist = dist
					nearest = centroid
				}
			}

			// Subtract centroid from vector
			for j := range vec {
				vec[j] -= nearest[j]
			}
		}
	}

	return nil
}

// kmeans performs k-means clustering
func (rq *ResidualQuantizer) kmeans(vectors [][]float32, k int) [][]float32 {
	if len(vectors) == 0 {
		return nil
	}

	dim := len(vectors[0])
	centroids := make([][]float32, k)

	// Initialize centroids randomly
	for i := 0; i < k; i++ {
		idx := rand.Intn(len(vectors))
		centroids[i] = make([]float32, dim)
		copy(centroids[i], vectors[idx])
	}

	maxIter := 100
	for iter := 0; iter < maxIter; iter++ {
		// Assign vectors to nearest centroids
		assignments := make([]int, len(vectors))
		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)
			for j, centroid := range centroids {
				dist := computeL2Distance(vec, centroid)
				if dist < minDist {
					minDist = dist
					assignments[i] = j
				}
			}
		}

		// Update centroids
		newCentroids := make([][]float32, k)
		counts := make([]int, k)
		for i := range newCentroids {
			newCentroids[i] = make([]float32, dim)
		}

		for i, vec := range vectors {
			cluster := assignments[i]
			counts[cluster]++
			for j := range vec {
				newCentroids[cluster][j] += vec[j]
			}
		}

		// Average centroids
		for i := range newCentroids {
			if counts[i] > 0 {
				for j := range newCentroids[i] {
					newCentroids[i][j] /= float32(counts[i])
				}
			} else {
				// Reinitialize empty clusters
				idx := rand.Intn(len(vectors))
				copy(newCentroids[i], vectors[idx])
			}
		}

		// Check convergence
		converged := true
		for i := range centroids {
			if computeL2Distance(centroids[i], newCentroids[i]) > 1e-6 {
				converged = false
				break
			}
		}

		centroids = newCentroids
		if converged {
			break
		}
	}

	return centroids
}

// Quantize performs residual quantization on a vector
func (rq *ResidualQuantizer) Quantize(vector []float32) []uint8 {
	rq.mu.RLock()
	defer rq.mu.RUnlock()

	// Get buffer from pool
	residual := rq.bufferPool.Get().([]float32)
	defer rq.bufferPool.Put(residual)
	copy(residual, vector)

	codes := make([]uint8, rq.numLevels)

	// Quantize each level
	for level := 0; level < rq.numLevels; level++ {
		// Find nearest centroid
		minDist := float32(1e10)
		var minIdx int
		for i, centroid := range rq.codebooks[level] {
			dist := computeL2Distance(residual, centroid)
			if dist < minDist {
				minDist = dist
				minIdx = i
			}
		}
		codes[level] = uint8(minIdx)

		// Subtract centroid from residual
		for i := range residual {
			residual[i] -= rq.codebooks[level][minIdx][i]
		}
	}

	return codes
}

// Reconstruct reconstructs a vector from its codes
func (rq *ResidualQuantizer) Reconstruct(codes []uint8) []float32 {
	rq.mu.RLock()
	defer rq.mu.RUnlock()

	// Get buffer from pool
	vector := rq.bufferPool.Get().([]float32)
	defer rq.bufferPool.Put(vector)

	// Add centroids from each level
	for level, code := range codes {
		centroid := rq.codebooks[level][code]
		for i := range vector {
			vector[i] += centroid[i]
		}
	}

	// Return copy of result
	result := make([]float32, len(vector))
	copy(result, vector)
	return result
}

// ConcurrentIndexUpdater handles concurrent index updates
type ConcurrentIndexUpdater struct {
	graph     *HNSWGraph
	lsh       *MultiProbeLSH
	quantizer *ResidualQuantizer
	tasks     chan updateTask
	done      chan struct{}
	wg        sync.WaitGroup
}

type updateTask struct {
	id     string
	vector []float32
}

// NewConcurrentIndexUpdater creates a new concurrent index updater
func NewConcurrentIndexUpdater(dimension int) *ConcurrentIndexUpdater {
	updater := &ConcurrentIndexUpdater{
		graph:     NewHNSWGraph(dimension, 32),
		lsh:       NewMultiProbeLSH(dimension, 8, 8),
		quantizer: NewResidualQuantizer(dimension, 4, 256),
		tasks:     make(chan updateTask, runtime.NumCPU()*2),
		done:      make(chan struct{}),
	}

	// Start worker goroutines
	for i := 0; i < runtime.NumCPU(); i++ {
		updater.wg.Add(1)
		go updater.worker()
	}

	return updater
}

// worker processes index update tasks
func (cu *ConcurrentIndexUpdater) worker() {
	defer cu.wg.Done()

	for {
		select {
		case <-cu.done:
			return
		case task := <-cu.tasks:
			// Update all indexes concurrently
			var wg sync.WaitGroup
			wg.Add(3)

			// Update HNSW graph
			go func() {
				defer wg.Done()
				cu.graph.Insert(task.id, task.vector)
			}()

			// Update LSH index
			go func() {
				defer wg.Done()
				cu.lsh.Insert(task.id, task.vector)
			}()

			// Update quantizer
			go func() {
				defer wg.Done()
				cu.quantizer.Quantize(task.vector)
			}()

			wg.Wait()
		}
	}
}

// Insert adds a vector to all indexes concurrently
func (cu *ConcurrentIndexUpdater) Insert(id string, vector []float32) error {
	// Make a copy of the vector to ensure thread safety
	vectorCopy := make([]float32, len(vector))
	copy(vectorCopy, vector)

	// Insert into graph using slice
	if err := cu.graph.Insert(id, vectorCopy); err != nil {
		return err
	}

	// Update LSH index with the copy
	cu.lsh.Insert(id, vectorCopy)

	// Update quantizer with the copy
	cu.quantizer.Quantize(vectorCopy)

	return nil
}

// Close shuts down the updater
func (cu *ConcurrentIndexUpdater) Close() {
	close(cu.done)
	cu.wg.Wait()
	cu.graph.Close()
	cu.lsh.Close()
}

// OptimisticDistanceComputer implements optimistic distance computation
type OptimisticDistanceComputer struct {
	dimension  int
	bufferPool sync.Pool
}

// NewOptimisticDistanceComputer creates a new optimistic distance computer
func NewOptimisticDistanceComputer(dimension int) *OptimisticDistanceComputer {
	return &OptimisticDistanceComputer{
		dimension: dimension,
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
}

// ComputeDistances computes distances with early termination
func (odc *OptimisticDistanceComputer) ComputeDistances(query []float32, vectors [][]float32, threshold float32) []float32 {
	// Get buffer from pool
	buf := odc.bufferPool.Get().([]float32)
	defer odc.bufferPool.Put(buf)

	distances := make([]float32, len(vectors))
	for i, vec := range vectors {
		distances[i] = odc.computeDistanceOptimistic(query, vec, threshold)
	}
	return distances
}

// computeDistanceOptimistic computes L2 distance with early termination
func (odc *OptimisticDistanceComputer) computeDistanceOptimistic(a, b []float32, threshold float32) float32 {
	var sum float32
	thresholdSq := threshold * threshold

	// Process vectors in blocks for better cache utilization
	const blockSize = 16
	for i := 0; i < len(a); i += blockSize {
		end := i + blockSize
		if end > len(a) {
			end = len(a)
		}

		// Compute partial sum for this block
		var blockSum float32
		for j := i; j < end; j++ {
			d := a[j] - b[j]
			blockSum += d * d
		}
		sum += blockSum

		// Early termination if we exceed threshold
		if sum > thresholdSq {
			return float32(math.Sqrt(float64(sum)))
		}
	}

	return float32(math.Sqrt(float64(sum)))
}

// ComputeDistancesParallel computes distances in parallel with early termination
func (odc *OptimisticDistanceComputer) ComputeDistancesParallel(query []float32, vectors [][]float32, threshold float32) []float32 {
	numWorkers := runtime.NumCPU()
	batchSize := (len(vectors) + numWorkers - 1) / numWorkers

	// Create channels for results
	results := make(chan struct {
		distances []float32
		start     int
	}, numWorkers)

	// Process batches in parallel
	var wg sync.WaitGroup
	for i := 0; i < len(vectors); i += batchSize {
		wg.Add(1)
		start := i
		end := i + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}

		go func(start, end int) {
			defer wg.Done()

			// Process batch
			distances := make([]float32, end-start)
			for i, vec := range vectors[start:end] {
				distances[i] = odc.computeDistanceOptimistic(query, vec, threshold)
			}

			// Send results
			results <- struct {
				distances []float32
				start     int
			}{distances, start}
		}(start, end)
	}

	// Close results channel when all workers are done
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	distances := make([]float32, len(vectors))
	for result := range results {
		copy(distances[result.start:], result.distances)
	}

	return distances
}

// Close releases resources
func (odc *OptimisticDistanceComputer) Close() {
	// Nothing to do here since sync.Pool handles cleanup
}
