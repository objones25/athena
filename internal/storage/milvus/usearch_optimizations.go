package milvus

import (
	"container/heap"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
)

// HNSWGraph implements a Hierarchical Navigable Small World graph
type HNSWGraph struct {
	maxLevel     int
	levelMult    float64
	efConstruct  int
	maxNeighbors int
	entryPoint   atomic.Pointer[hnswNode]
	nodes        sync.Map
	dimension    int
	distFunc     func(a, b []float32) float32
	nodePool     sync.Pool
	bufferPool   sync.Pool
}

type hnswNode struct {
	id        string
	vector    []float32
	neighbors [][]atomic.Pointer[hnswNode]
	level     int
}

// NewHNSWGraph creates a new HNSW graph
func NewHNSWGraph(dimension int, maxNeighbors int) *HNSWGraph {
	return &HNSWGraph{
		dimension:    dimension,
		maxNeighbors: maxNeighbors,
		maxLevel:     32,
		levelMult:    1.0 / math.Log(2),
		efConstruct:  64,
		distFunc:     computeL2Distance,
		nodePool: sync.Pool{
			New: func() interface{} {
				return &hnswNode{}
			},
		},
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
}

// Insert adds a new vector to the HNSW graph
func (g *HNSWGraph) Insert(id string, vector []float32) error {
	// Get buffer from pool
	buf := g.bufferPool.Get().([]float32)
	defer g.bufferPool.Put(buf)

	// Generate random level with exponential distribution
	level := g.generateLevel()

	// Create new node
	node := g.nodePool.Get().(*hnswNode)
	node.id = id
	node.vector = make([]float32, len(vector))
	copy(node.vector, vector)
	node.level = level
	node.neighbors = make([][]atomic.Pointer[hnswNode], level+1)

	for i := range node.neighbors {
		node.neighbors[i] = make([]atomic.Pointer[hnswNode], g.maxNeighbors)
	}

	// Get entry point
	entryPoint := g.entryPoint.Load()
	if entryPoint == nil {
		g.entryPoint.Store(node)
		g.nodes.Store(id, node)
		return nil
	}

	// Find entry points for all levels
	currObj := entryPoint
	currDist := g.computeDistance(vector, currObj.vector)

	// For each level, find closest neighbors
	maxLevel := min(level, currObj.level)
	for l := maxLevel; l >= 0; l-- {
		changed := true
		for changed {
			changed = false
			// Check neighbors at current level
			for i := 0; i < g.maxNeighbors; i++ {
				if currObj.neighbors[l][i].Load() == nil {
					continue
				}
				neighbor := currObj.neighbors[l][i].Load()
				dist := g.computeDistance(vector, neighbor.vector)
				if dist < currDist {
					currDist = dist
					currObj = neighbor
					changed = true
				}
			}
		}

		// Find neighbors for current level using beam search
		neighbors := g.searchLayer(vector, currObj, g.efConstruct, l)

		// Update bidirectional connections
		for i := 0; i < len(neighbors) && i < g.maxNeighbors; i++ {
			neighbor := neighbors[i]
			node.neighbors[l][i].Store(neighbor)

			// Add backlink using direct store
			for j := 0; j < g.maxNeighbors; j++ {
				if neighbor.neighbors[l][j].Load() == nil {
					neighbor.neighbors[l][j].Store(node)
					break
				}
			}
		}
	}

	g.nodes.Store(id, node)
	return nil
}

// Search performs approximate nearest neighbor search
func (g *HNSWGraph) Search(query []float32, k int) []string {
	currObj := g.entryPoint.Load()
	if currObj == nil {
		return nil
	}

	currDist := g.computeDistance(query, currObj.vector)

	// Search through levels
	for l := currObj.level; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			// Check neighbors at current level
			for i := 0; i < g.maxNeighbors; i++ {
				if currObj.neighbors[l][i].Load() == nil {
					continue
				}
				neighbor := currObj.neighbors[l][i].Load()
				dist := g.computeDistance(query, neighbor.vector)
				if dist < currDist {
					currDist = dist
					currObj = neighbor
					changed = true
				}
			}
		}
	}

	// Perform beam search at bottom layer
	neighbors := g.searchLayer(query, currObj, k*2, 0)

	// Extract top k results
	results := make([]string, 0, k)
	for i := 0; i < len(neighbors) && i < k; i++ {
		results = append(results, neighbors[i].id)
	}
	return results
}

// searchLayer performs beam search at a specific layer
func (g *HNSWGraph) searchLayer(query []float32, entryPoint *hnswNode, ef int, level int) []*hnswNode {
	candidates := &distanceHeap{}
	results := &distanceHeap{}
	visited := make(map[string]bool)

	heap.Init(candidates)
	heap.Init(results)

	// Add entry point
	heap.Push(candidates, &distanceItem{
		node:     entryPoint,
		distance: g.computeDistance(query, entryPoint.vector),
	})
	heap.Push(results, &distanceItem{
		node:     entryPoint,
		distance: g.computeDistance(query, entryPoint.vector),
	})
	visited[entryPoint.id] = true

	// Process candidates
	for candidates.Len() > 0 {
		curr := heap.Pop(candidates).(*distanceItem)
		furthestDist := (*results)[0].distance

		if curr.distance > furthestDist {
			break
		}

		// Check neighbors
		for i := 0; i < g.maxNeighbors; i++ {
			if curr.node.neighbors[level][i].Load() == nil {
				continue
			}
			neighbor := curr.node.neighbors[level][i].Load()
			if visited[neighbor.id] {
				continue
			}
			visited[neighbor.id] = true

			dist := g.computeDistance(query, neighbor.vector)
			if results.Len() < ef || dist < furthestDist {
				heap.Push(candidates, &distanceItem{
					node:     neighbor,
					distance: dist,
				})
				heap.Push(results, &distanceItem{
					node:     neighbor,
					distance: dist,
				})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}

	// Convert heap to slice
	resultNodes := make([]*hnswNode, results.Len())
	for i := results.Len() - 1; i >= 0; i-- {
		resultNodes[i] = heap.Pop(results).(*distanceItem).node
	}
	return resultNodes
}

// generateLevel generates a random level with exponential distribution
func (g *HNSWGraph) generateLevel() int {
	level := int(math.Floor(-math.Log(rand.Float64()) * g.levelMult))
	if level < 0 {
		level = 0
	}
	if level > g.maxLevel {
		level = g.maxLevel
	}
	return level
}

// Helper types for priority queue
type distanceItem struct {
	node     *hnswNode
	distance float32
}

type distanceHeap []*distanceItem

func (h distanceHeap) Len() int           { return len(h) }
func (h distanceHeap) Less(i, j int) bool { return h[i].distance < h[j].distance }
func (h distanceHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *distanceHeap) Push(x interface{}) {
	*h = append(*h, x.(*distanceItem))
}

func (h *distanceHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h *distanceHeap) Top() *distanceItem {
	if len(*h) == 0 {
		return nil
	}
	return (*h)[0]
}

// ProductQuantizer implements Product Quantization for vector compression
type ProductQuantizer struct {
	dimension    int
	numSubspaces int
	numCentroids int
	centroids    [][][]float32
	mu           sync.RWMutex
}

// NewProductQuantizer creates a new product quantizer
func NewProductQuantizer(dimension, numSubspaces, numCentroids int) *ProductQuantizer {
	if numSubspaces <= 0 {
		numSubspaces = 8
	}
	if numCentroids <= 0 {
		numCentroids = 256
	}

	return &ProductQuantizer{
		dimension:    dimension,
		numSubspaces: numSubspaces,
		numCentroids: numCentroids,
		centroids:    make([][][]float32, numSubspaces),
	}
}

// Train trains the product quantizer on a set of vectors
func (pq *ProductQuantizer) Train(vectors [][]float32) error {
	if len(vectors) == 0 {
		return nil
	}

	subDim := pq.dimension / pq.numSubspaces
	pq.mu.Lock()
	defer pq.mu.Unlock()

	// Initialize centroids for each subspace
	for i := 0; i < pq.numSubspaces; i++ {
		start := i * subDim
		end := start + subDim

		// Extract subvectors
		subvectors := make([][]float32, len(vectors))
		for j, vec := range vectors {
			subvectors[j] = vec[start:end]
		}

		// Run k-means clustering
		centroids := pq.kmeans(subvectors, pq.numCentroids)
		pq.centroids[i] = centroids
	}

	return nil
}

// Quantize compresses a vector using product quantization
func (pq *ProductQuantizer) Quantize(vector []float32) []uint8 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	subDim := pq.dimension / pq.numSubspaces
	codes := make([]uint8, pq.numSubspaces)

	// Quantize each subvector
	for i := 0; i < pq.numSubspaces; i++ {
		start := i * subDim
		end := start + subDim
		subvec := vector[start:end]

		// Find nearest centroid
		minDist := float32(math.MaxFloat32)
		var minIdx int
		for j, centroid := range pq.centroids[i] {
			dist := computeL2Distance(subvec, centroid)
			if dist < minDist {
				minDist = dist
				minIdx = j
			}
		}
		codes[i] = uint8(minIdx)
	}

	return codes
}

// Reconstruct decompresses a quantized vector
func (pq *ProductQuantizer) Reconstruct(codes []uint8) []float32 {
	pq.mu.RLock()
	defer pq.mu.RUnlock()

	subDim := pq.dimension / pq.numSubspaces
	vector := make([]float32, pq.dimension)

	// Reconstruct each subvector
	for i := 0; i < pq.numSubspaces; i++ {
		start := i * subDim
		end := start + subDim
		copy(vector[start:end], pq.centroids[i][codes[i]])
	}

	return vector
}

// kmeans performs k-means clustering
func (pq *ProductQuantizer) kmeans(vectors [][]float32, k int) [][]float32 {
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

// MultiProbeLSH implements Multi-probe Locality Sensitive Hashing
type MultiProbeLSH struct {
	numTables  int
	numHashes  int
	dimension  int
	hashTables []map[uint64][]string
	hashFuncs  [][]float32
	mu         sync.RWMutex
	bufferPool sync.Pool
}

// NewMultiProbeLSH creates a new Multi-probe LSH index
func NewMultiProbeLSH(dimension, numTables, numHashes int) *MultiProbeLSH {
	if numTables <= 0 {
		numTables = 8
	}
	if numHashes <= 0 {
		numHashes = 8
	}

	lsh := &MultiProbeLSH{
		numTables:  numTables,
		numHashes:  numHashes,
		dimension:  dimension,
		hashTables: make([]map[uint64][]string, numTables),
		hashFuncs:  make([][]float32, numTables*numHashes),
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}

	// Initialize hash tables
	for i := range lsh.hashTables {
		lsh.hashTables[i] = make(map[uint64][]string)
	}

	// Initialize random hash functions
	for i := range lsh.hashFuncs {
		lsh.hashFuncs[i] = make([]float32, dimension)
		for j := range lsh.hashFuncs[i] {
			lsh.hashFuncs[i][j] = rand.Float32()*2 - 1 // Random values in [-1, 1]
		}
	}

	return lsh
}

// Insert adds a vector to the LSH index
func (lsh *MultiProbeLSH) Insert(id string, vector []float32) {
	// Insert into each hash table
	for i := 0; i < lsh.numTables; i++ {
		hashValue := lsh.computeHash(vector, i)
		lsh.hashTables[i][hashValue] = append(lsh.hashTables[i][hashValue], id)
	}
}

// Search performs multi-probe LSH search
func (lsh *MultiProbeLSH) Search(query []float32, numProbes int) []string {
	// Get buffer from pool
	buf := lsh.bufferPool.Get().([]float32)
	defer lsh.bufferPool.Put(buf)

	lsh.mu.RLock()
	defer lsh.mu.RUnlock()

	// Track unique results
	results := make(map[string]struct{})

	// Search in each hash table
	for i := 0; i < lsh.numTables; i++ {
		// Get base hash value
		baseHash := lsh.computeHash(query, i)

		// Generate probe sequence
		probes := lsh.generateProbes(baseHash, numProbes)

		// Check each probe
		for _, probe := range probes {
			if ids, ok := lsh.hashTables[i][probe]; ok {
				for _, id := range ids {
					results[id] = struct{}{}
				}
			}
		}
	}

	// Convert results to slice
	ids := make([]string, 0, len(results))
	for id := range results {
		ids = append(ids, id)
	}

	return ids
}

// computeHash generates a hash value for a vector
func (lsh *MultiProbeLSH) computeHash(vector []float32, tableIdx int) uint64 {
	var hash uint64

	// Apply each hash function in the table
	for i := 0; i < lsh.numHashes; i++ {
		// Compute dot product with hash function
		hashFunc := lsh.hashFuncs[tableIdx*lsh.numHashes+i]
		var sum float32
		for j := range vector {
			sum += vector[j] * hashFunc[j]
		}

		// Add bit to hash based on sign
		if sum > 0 {
			hash |= 1 << i
		}
	}

	return hash
}

// generateProbes generates a sequence of hash values to probe
func (lsh *MultiProbeLSH) generateProbes(baseHash uint64, numProbes int) []uint64 {
	probes := make([]uint64, 0, numProbes)
	probes = append(probes, baseHash)

	// Generate perturbation vectors
	for i := 1; i < numProbes; i++ {
		// Flip i bits in the base hash
		for j := uint(0); j < uint(lsh.numHashes); j++ {
			probe := baseHash ^ (1 << j)
			probes = append(probes, probe)
			if len(probes) >= numProbes {
				return probes
			}
		}
	}

	return probes
}

// Close releases resources
func (lsh *MultiProbeLSH) Close() {
	lsh.mu.Lock()
	defer lsh.mu.Unlock()

	// Clear hash tables
	for i := range lsh.hashTables {
		lsh.hashTables[i] = nil
	}
	lsh.hashTables = nil
	lsh.hashFuncs = nil
}

// OptimisticSearch implements optimistic vector search with early termination
type OptimisticSearch struct {
	graph      *HNSWGraph
	lsh        *MultiProbeLSH
	quantizer  *ProductQuantizer
	bufferPool sync.Pool
}

// NewOptimisticSearch creates a new optimistic search instance
func NewOptimisticSearch(dimension int) *OptimisticSearch {
	return &OptimisticSearch{
		graph:     NewHNSWGraph(dimension, 32),
		lsh:       NewMultiProbeLSH(dimension, 8, 8),
		quantizer: NewProductQuantizer(dimension, 8, 256),
		bufferPool: sync.Pool{
			New: func() interface{} {
				return make([]float32, dimension)
			},
		},
	}
}

// Insert adds a vector to all indexes
func (os *OptimisticSearch) Insert(id string, vector []float32) error {
	// Insert into HNSW graph
	if err := os.graph.Insert(id, vector); err != nil {
		return err
	}

	// Insert into LSH index
	os.lsh.Insert(id, vector)

	return nil
}

// Search performs optimistic search with early termination
func (os *OptimisticSearch) Search(query []float32, k int) []string {
	// Get buffer from pool
	buf := os.bufferPool.Get().([]float32)
	defer os.bufferPool.Put(buf)

	// First try LSH for fast approximate results
	candidates := os.lsh.Search(query, 10)
	if len(candidates) >= k {
		return candidates[:k]
	}

	// If LSH doesn't find enough results, use HNSW
	results := os.graph.Search(query, k)

	// Combine and deduplicate results
	seen := make(map[string]bool)
	combined := make([]string, 0, k)

	// Add LSH results first
	for _, id := range candidates {
		if !seen[id] {
			seen[id] = true
			combined = append(combined, id)
		}
	}

	// Add HNSW results
	for _, id := range results {
		if !seen[id] && len(combined) < k {
			seen[id] = true
			combined = append(combined, id)
		}
	}

	return combined
}

// Close releases resources
func (os *OptimisticSearch) Close() {
	os.graph.Close()
	os.lsh.Close()
}

// Add Close method to HNSWGraph
func (g *HNSWGraph) Close() {
	// Clear nodes
	g.nodes.Range(func(key, value interface{}) bool {
		g.nodes.Delete(key)
		return true
	})

	// Clear entry point
	g.entryPoint.Store(nil)
}

// computeDistance calculates distance between vectors
func (g *HNSWGraph) computeDistance(a, b []float32) float32 {
	return g.distFunc(a, b)
}
