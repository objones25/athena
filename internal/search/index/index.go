package index

import (
	"container/heap"
	"context"
	"fmt"
	"runtime"
	"sync"

	"github.com/objones25/athena/internal/search/similarity"
)

// VectorIndex provides efficient nearest neighbor search for embeddings
type VectorIndex struct {
	vectors    map[string][]float32          // Key -> Vector mapping
	graph      map[string][]string           // Adjacency list for approximate graph
	scores     map[string]map[string]float64 // Cached similarity scores
	dimensions int
	mu         sync.RWMutex
}

// Config holds index configuration
type Config struct {
	InitialCapacity int
	MaxConnections  int     // Maximum connections per node in the graph
	BuildBatchSize  int     // Batch size for parallel graph building
	ScoreThreshold  float64 // Minimum score to create an edge
}

// DefaultConfig returns default index configuration
func DefaultConfig() Config {
	return Config{
		InitialCapacity: 10000,
		MaxConnections:  32,
		BuildBatchSize:  1000,
		ScoreThreshold:  0.7,
	}
}

// New creates a new vector index
func New(dimensions int, cfg Config) *VectorIndex {
	return &VectorIndex{
		vectors:    make(map[string][]float32, cfg.InitialCapacity),
		graph:      make(map[string][]string, cfg.InitialCapacity),
		scores:     make(map[string]map[string]float64, cfg.InitialCapacity),
		dimensions: dimensions,
	}
}

// Add adds a vector to the index
func (idx *VectorIndex) Add(ctx context.Context, key string, vector []float32) error {
	if len(vector) != idx.dimensions {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", idx.dimensions, len(vector))
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store vector
	idx.vectors[key] = vector
	idx.graph[key] = make([]string, 0)
	idx.scores[key] = make(map[string]float64)

	return nil
}

// AddBatch adds multiple vectors in batch
func (idx *VectorIndex) AddBatch(ctx context.Context, keys []string, vectors [][]float32) error {
	if len(keys) != len(vectors) {
		return fmt.Errorf("keys and vectors length mismatch")
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	for i, key := range keys {
		if len(vectors[i]) != idx.dimensions {
			return fmt.Errorf("vector dimension mismatch for key %s: expected %d, got %d",
				key, idx.dimensions, len(vectors[i]))
		}
		idx.vectors[key] = vectors[i]
		idx.graph[key] = make([]string, 0)
		idx.scores[key] = make(map[string]float64)
	}

	return nil
}

// Remove removes a vector from the index
func (idx *VectorIndex) Remove(ctx context.Context, key string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Remove from vectors and scores
	delete(idx.vectors, key)
	delete(idx.scores, key)

	// Remove from graph
	delete(idx.graph, key)
	for _, neighbors := range idx.graph {
		for i := 0; i < len(neighbors); i++ {
			if neighbors[i] == key {
				neighbors = append(neighbors[:i], neighbors[i+1:]...)
				i--
			}
		}
	}
}

// SearchNearest finds nearest neighbors for a query vector
func (idx *VectorIndex) SearchNearest(ctx context.Context, queryVec []float32, k int, cfg Config) ([]SearchResult, error) {
	if len(queryVec) != idx.dimensions {
		return nil, fmt.Errorf("query vector dimension mismatch: expected %d, got %d",
			idx.dimensions, len(queryVec))
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Use priority queue for top-k
	pq := make(priorityQueue, 0, k)
	heap.Init(&pq)

	// Start with random entry points
	visited := make(map[string]bool)
	candidates := make(map[string]bool)

	// Add initial candidates
	for key := range idx.vectors {
		candidates[key] = true
		if len(candidates) >= cfg.MaxConnections {
			break
		}
	}

	simCtx := similarity.DefaultContext()
	for len(candidates) > 0 && len(visited) < len(idx.vectors) {
		// Get best candidate
		var bestKey string
		var bestScore float64
		for key := range candidates {
			if visited[key] {
				continue
			}
			metrics, err := similarity.Calculate(queryVec, idx.vectors[key], simCtx)
			if err != nil {
				continue
			}
			score := metrics.Contextual
			if score > bestScore {
				bestScore = score
				bestKey = key
			}
		}

		if bestKey == "" {
			break
		}

		// Mark as visited
		visited[bestKey] = true
		delete(candidates, bestKey)

		// Add to results if good enough
		if bestScore >= cfg.ScoreThreshold {
			heap.Push(&pq, &SearchResult{
				Key:   bestKey,
				Score: bestScore,
			})
			if pq.Len() > k {
				heap.Pop(&pq)
			}
		}

		// Add neighbors as candidates
		for _, neighbor := range idx.graph[bestKey] {
			if !visited[neighbor] {
				candidates[neighbor] = true
			}
		}
	}

	// Convert heap to sorted slice
	results := make([]SearchResult, pq.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = *(heap.Pop(&pq).(*SearchResult))
	}

	return results, nil
}

// BuildGraph builds the approximate nearest neighbor graph
func (idx *VectorIndex) BuildGraph(ctx context.Context, cfg Config) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Reset graph
	for key := range idx.graph {
		idx.graph[key] = make([]string, 0, cfg.MaxConnections)
	}

	// Process in batches
	keys := make([]string, 0, len(idx.vectors))
	for key := range idx.vectors {
		keys = append(keys, key)
	}

	var wg sync.WaitGroup
	errCh := make(chan error, 1)
	semaphore := make(chan struct{}, runtime.NumCPU())

	for i := 0; i < len(keys); i += cfg.BuildBatchSize {
		end := min(i+cfg.BuildBatchSize, len(keys))
		batch := keys[i:end]

		wg.Add(1)
		go func(batchKeys []string) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			simCtx := similarity.DefaultContext()
			for _, key1 := range batchKeys {
				vec1 := idx.vectors[key1]
				scores := make(map[string]float64)

				// Compare with all other vectors
				for key2, vec2 := range idx.vectors {
					if key1 == key2 {
						continue
					}

					metrics, err := similarity.Calculate(vec1, vec2, simCtx)
					if err != nil {
						select {
						case errCh <- fmt.Errorf("similarity calculation failed: %w", err):
						default:
						}
						return
					}

					score := metrics.Contextual
					if score >= cfg.ScoreThreshold {
						scores[key2] = score
					}
				}

				// Select top-k neighbors
				neighbors := make([]string, 0, cfg.MaxConnections)
				for key2, score := range scores {
					if len(neighbors) < cfg.MaxConnections {
						neighbors = append(neighbors, key2)
						continue
					}
					// Replace lowest scoring neighbor if current score is higher
					minIdx := 0
					minScore := scores[neighbors[0]]
					for j, n := range neighbors {
						if scores[n] < minScore {
							minIdx = j
							minScore = scores[n]
						}
					}
					if score > minScore {
						neighbors[minIdx] = key2
					}
				}

				idx.graph[key1] = neighbors
				idx.scores[key1] = scores
			}
		}(batch)
	}

	// Wait for all batches and check for errors
	go func() {
		wg.Wait()
		close(errCh)
	}()

	if err := <-errCh; err != nil {
		return err
	}

	return nil
}

// SearchResult represents a search result with score
type SearchResult struct {
	Key   string
	Score float64
}

// Priority queue implementation for top-k
type priorityQueue []*SearchResult

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].Score < pq[j].Score // Min heap
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *priorityQueue) Push(x interface{}) {
	item := x.(*SearchResult)
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
