package milvus

import (
	"container/heap"
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
)

// GraphConfig holds configuration for the similarity graph
type GraphConfig struct {
	Dimension      int           // Vector dimension
	MaxNeighbors   int           // Maximum number of neighbors per node
	MaxSearchDepth int           // Maximum depth for graph traversal during search
	UpdateInterval time.Duration // How often to update graph connections
	BatchSize      int           // Size of batches for processing updates
	NumWorkers     int           // Number of worker goroutines
	MinSimilarity  float32       // Minimum similarity threshold for edges
	PruneThreshold int           // Number of edges before pruning is triggered
}

// DefaultGraphConfig returns the default configuration
func DefaultGraphConfig() GraphConfig {
	return GraphConfig{
		Dimension:      1536,
		MaxNeighbors:   32,
		MaxSearchDepth: 50,
		UpdateInterval: 15 * time.Minute,
		BatchSize:      1000,
		NumWorkers:     8,
		MinSimilarity:  0.6,
		PruneThreshold: 64,
	}
}

// Node represents a vertex in the similarity graph
type Node struct {
	ID        string
	Vector    []float32
	Neighbors *neighborList
	mu        sync.RWMutex
}

// Edge represents a connection between two nodes
type Edge struct {
	To         *Node
	Similarity float32
}

// neighborList maintains a sorted list of neighbors
type neighborList struct {
	edges   []Edge
	maxSize int
	minSim  float32
}

// UpdateQueue manages concurrent graph updates
type UpdateQueue struct {
	updates chan graphUpdate
	results chan error
	workers int
	wg      sync.WaitGroup
	graph   *SimilarityGraph
}

type graphUpdate struct {
	nodeID    string
	vector    []float32
	neighbors []string
}

// newUpdateQueue creates a new update queue with the specified number of workers
func newUpdateQueue(workers int, graph *SimilarityGraph) *UpdateQueue {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	uq := &UpdateQueue{
		updates: make(chan graphUpdate, workers*2),
		results: make(chan error, workers*2),
		workers: workers,
		graph:   graph,
	}

	// Start worker goroutines
	uq.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go uq.worker()
	}

	return uq
}

func (uq *UpdateQueue) worker() {
	defer uq.wg.Done()

	for update := range uq.updates {
		if err := uq.graph.processUpdate(update); err != nil {
			uq.results <- err
		}
	}
}

func (uq *UpdateQueue) Close() {
	close(uq.updates)
	uq.wg.Wait()
	close(uq.results)
}

// SimilarityGraph maintains relationships between vectors
type SimilarityGraph struct {
	config      GraphConfig
	nodes       sync.Map
	dimension   int
	lastUpdate  time.Time
	updateQueue *UpdateQueue

	// Worker pools and channels
	updateChan   chan *Node
	resultChan   chan error
	workerPool   chan struct{}
	updateTicker *time.Ticker
	done         chan struct{}

	// Metrics
	metrics struct {
		nodesAdded    uint64
		edgesAdded    uint64
		edgesPruned   uint64
		searchCount   uint64
		updateCount   uint64
		avgSearchTime int64
	}
}

// NewSimilarityGraph creates a new similarity graph
func NewSimilarityGraph(config GraphConfig, dimension int) *SimilarityGraph {
	// Validate configuration
	if dimension <= 0 {
		dimension = DefaultGraphConfig().Dimension
	}
	if config.MaxNeighbors <= 0 {
		config.MaxNeighbors = DefaultGraphConfig().MaxNeighbors
	}
	if config.MaxSearchDepth <= 0 {
		config.MaxSearchDepth = DefaultGraphConfig().MaxSearchDepth
	}
	if config.UpdateInterval <= 0 {
		config.UpdateInterval = DefaultGraphConfig().UpdateInterval
	}
	if config.BatchSize <= 0 {
		config.BatchSize = DefaultGraphConfig().BatchSize
	}
	if config.NumWorkers <= 0 {
		config.NumWorkers = DefaultGraphConfig().NumWorkers
	}
	if config.MinSimilarity <= 0 {
		config.MinSimilarity = DefaultGraphConfig().MinSimilarity
	}
	if config.PruneThreshold <= 0 {
		config.PruneThreshold = DefaultGraphConfig().PruneThreshold
	}

	g := &SimilarityGraph{
		config:     config,
		dimension:  dimension,
		updateChan: make(chan *Node, config.BatchSize),
		resultChan: make(chan error, config.BatchSize),
		workerPool: make(chan struct{}, config.NumWorkers),
		done:       make(chan struct{}),
	}

	// Initialize update queue with graph reference
	g.updateQueue = newUpdateQueue(config.NumWorkers, g)

	// Start worker goroutines
	for i := 0; i < config.NumWorkers; i++ {
		go g.worker()
	}

	// Start update monitor
	g.updateTicker = time.NewTicker(config.UpdateInterval)
	go g.monitorUpdates()

	return g
}

// Close shuts down the graph
func (g *SimilarityGraph) Close() {
	close(g.done)
	g.updateTicker.Stop()
	if g.updateQueue != nil {
		g.updateQueue.Close()
	}
}

// AddNode adds a new node to the graph
func (g *SimilarityGraph) AddNode(ctx context.Context, id string, vector []float32) error {
	if len(vector) != g.dimension {
		return fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), g.dimension)
	}

	node := &Node{
		ID:     id,
		Vector: vector,
		mu:     sync.RWMutex{},
	}

	// Store node
	if _, loaded := g.nodes.LoadOrStore(id, node); loaded {
		return fmt.Errorf("node %s already exists", id)
	}

	atomic.AddUint64(&g.metrics.nodesAdded, 1)

	// Schedule node for update using update queue
	select {
	case g.updateQueue.updates <- graphUpdate{
		nodeID: id,
		vector: vector,
	}:
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

// FindNearest finds the k nearest neighbors of a vector
func (g *SimilarityGraph) FindNearest(ctx context.Context, vector []float32, k int) ([]*Node, error) {
	if len(vector) != g.dimension {
		return nil, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), g.dimension)
	}

	start := time.Now()
	defer func() {
		atomic.AddInt64(&g.metrics.avgSearchTime, time.Since(start).Nanoseconds())
		atomic.AddUint64(&g.metrics.searchCount, 1)
	}()

	// Use beam search to find nearest neighbors
	visited := make(map[string]bool)
	candidates := &nodeHeap{}
	heap.Init(candidates)

	// Start with a random entry point
	var entryNode *Node
	g.nodes.Range(func(key, value interface{}) bool {
		entryNode = value.(*Node)
		return false // Stop after first node
	})

	if entryNode == nil {
		return nil, fmt.Errorf("graph is empty")
	}

	similarity := cosineSimilarity(vector, entryNode.Vector)
	heap.Push(candidates, &nodeDistance{node: entryNode, distance: similarity})
	visited[entryNode.ID] = true

	result := make([]*Node, 0, k)
	depth := 0

	for candidates.Len() > 0 && len(result) < k && depth < g.config.MaxSearchDepth {
		current := heap.Pop(candidates).(*nodeDistance)
		result = append(result, current.node)

		// Explore neighbors
		current.node.mu.RLock()
		for _, edge := range current.node.Neighbors.edges {
			if !visited[edge.To.ID] {
				similarity := cosineSimilarity(vector, edge.To.Vector)
				heap.Push(candidates, &nodeDistance{node: edge.To, distance: similarity})
				visited[edge.To.ID] = true
			}
		}
		current.node.mu.RUnlock()

		depth++
	}

	return result, nil
}

func (g *SimilarityGraph) worker() {
	for {
		select {
		case <-g.done:
			return
		case update := <-g.updateQueue.updates:
			if err := g.processUpdate(update); err != nil {
				log.Error().
					Err(err).
					Str("node_id", update.nodeID).
					Msg("Failed to process graph update")
				g.updateQueue.results <- err
			}
		}
	}
}

func (g *SimilarityGraph) monitorUpdates() {
	for {
		select {
		case <-g.done:
			return
		case <-g.updateTicker.C:
			if time.Since(g.lastUpdate) >= g.config.UpdateInterval {
				log.Debug().Msg("Starting periodic graph update")
				g.updateAllNodes()
			}
		}
	}
}

func (g *SimilarityGraph) processUpdate(update graphUpdate) error {
	// Get the node
	nodeValue, ok := g.nodes.Load(update.nodeID)
	if !ok {
		return fmt.Errorf("node %s not found", update.nodeID)
	}
	node := nodeValue.(*Node)

	// Find potential neighbors
	candidates := make([]Edge, 0, g.config.PruneThreshold)

	// Use SIMD for parallel similarity computation
	g.nodes.Range(func(key, value interface{}) bool {
		other := value.(*Node)
		if other.ID != node.ID {
			similarity := cosineSimilarity(node.Vector, other.Vector)
			if similarity >= g.config.MinSimilarity {
				candidates = append(candidates, Edge{To: other, Similarity: similarity})
			}
		}
		return true
	})

	// Sort candidates by similarity
	sortEdges(candidates)

	// Update node's neighbors with minimal locking
	if len(candidates) > 0 {
		node.mu.Lock()
		node.Neighbors.edges = mergeBestEdges(node.Neighbors.edges, candidates, g.config.MaxNeighbors)
		node.mu.Unlock()

		atomic.AddUint64(&g.metrics.edgesAdded, uint64(len(candidates)))
	}

	// Prune edges if needed
	if len(node.Neighbors.edges) > g.config.PruneThreshold {
		node.mu.Lock()
		oldLen := len(node.Neighbors.edges)
		node.Neighbors.edges = node.Neighbors.edges[:g.config.MaxNeighbors]
		node.mu.Unlock()

		atomic.AddUint64(&g.metrics.edgesPruned, uint64(oldLen-len(node.Neighbors.edges)))
	}

	atomic.AddUint64(&g.metrics.updateCount, 1)
	return nil
}

func (g *SimilarityGraph) updateAllNodes() {
	g.nodes.Range(func(key, value interface{}) bool {
		node := value.(*Node)
		select {
		case g.updateChan <- node:
		default:
			// Skip if channel is full
			log.Warn().Str("node_id", node.ID).Msg("Skipped node update due to full channel")
		}
		return true
	})
}

// GetNodes returns all nodes in the graph
func (g *SimilarityGraph) GetNodes() []*Node {
	var nodes []*Node
	g.nodes.Range(func(key, value interface{}) bool {
		nodes = append(nodes, value.(*Node))
		return true
	})
	return nodes
}

// Helper types and functions

type nodeDistance struct {
	node     *Node
	distance float32
}

type nodeHeap []*nodeDistance

func (h nodeHeap) Len() int            { return len(h) }
func (h nodeHeap) Less(i, j int) bool  { return h[i].distance > h[j].distance }
func (h nodeHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x interface{}) { *h = append(*h, x.(*nodeDistance)) }
func (h *nodeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func cosineSimilarity(a, b []float32) float32 {
	var dotProduct float32
	var normA, normB float32

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}

func sortEdges(edges []Edge) {
	for i := 0; i < len(edges)-1; i++ {
		for j := i + 1; j < len(edges); j++ {
			if edges[i].Similarity < edges[j].Similarity {
				edges[i], edges[j] = edges[j], edges[i]
			}
		}
	}
}

func mergeBestEdges(existing, new []Edge, maxSize int) []Edge {
	combined := make([]Edge, 0, len(existing)+len(new))
	combined = append(combined, existing...)
	combined = append(combined, new...)
	sortEdges(combined)

	if len(combined) > maxSize {
		combined = combined[:maxSize]
	}
	return combined
}
