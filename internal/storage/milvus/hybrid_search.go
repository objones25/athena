package milvus

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/rs/zerolog/log"
)

// HybridSearchConfig holds configuration for the hybrid search
type HybridSearchConfig struct {
	// LSH configuration
	LSHTables    int
	LSHFunctions int
	LSHThreshold float32

	// HNSW configuration
	HNSWMaxNeighbors   int
	HNSWMaxSearchDepth int
	HNSWBeamWidth      int

	// General configuration
	NumWorkers       int
	BatchSize        int
	SearchTimeout    time.Duration
	QualityThreshold float32
}

// DefaultHybridSearchConfig returns default configuration
func DefaultHybridSearchConfig() HybridSearchConfig {
	return HybridSearchConfig{
		LSHTables:          8,
		LSHFunctions:       4,
		LSHThreshold:       0.8,
		HNSWMaxNeighbors:   32,
		HNSWMaxSearchDepth: 50,
		HNSWBeamWidth:      100,
		NumWorkers:         runtime.NumCPU(),
		BatchSize:          1000,
		SearchTimeout:      time.Second * 2,
		QualityThreshold:   0.95,
	}
}

// HybridSearch combines LSH and HNSW for efficient vector search
type HybridSearch struct {
	config HybridSearchConfig
	lsh    *LSHIndex
	hnsw   *SimilarityGraph
	stats  struct {
		lshHits      atomic.Int64
		hnswHits     atomic.Int64
		totalQueries atomic.Int64
		avgLatency   atomic.Int64
	}
	workerPool chan struct{}
	mu         sync.RWMutex
}

// NewHybridSearch creates a new hybrid search instance
func NewHybridSearch(config HybridSearchConfig, dimension int) (*HybridSearch, error) {
	if config.NumWorkers <= 0 {
		config.NumWorkers = runtime.NumCPU()
	}

	// Initialize LSH
	lshConfig := LSHConfig{
		NumHashTables:    config.LSHTables,
		NumHashFunctions: config.LSHFunctions,
		Threshold:        config.LSHThreshold,
	}
	lsh := NewLSHIndex(lshConfig, dimension)

	// Initialize HNSW
	hnswConfig := GraphConfig{
		Dimension:      dimension,
		MaxNeighbors:   config.HNSWMaxNeighbors,
		MaxSearchDepth: config.HNSWMaxSearchDepth,
		NumWorkers:     config.NumWorkers,
	}
	hnsw := NewSimilarityGraph(hnswConfig, dimension)

	return &HybridSearch{
		config:     config,
		lsh:        lsh,
		hnsw:       hnsw,
		workerPool: make(chan struct{}, config.NumWorkers),
	}, nil
}

// Search performs hybrid vector search
func (hs *HybridSearch) Search(ctx context.Context, query []float32, k int) ([]*SearchResult, error) {
	start := time.Now()
	defer func() {
		hs.updateStats(time.Since(start))
	}()

	// Create context with timeout
	ctx, cancel := context.WithTimeout(ctx, hs.config.SearchTimeout)
	defer cancel()

	// First try LSH for fast approximate results
	lshResults := make(chan []*SearchResult, 1)
	go func() {
		candidates := hs.lsh.Query(query, k*2) // Get 2x candidates for better recall
		results := make([]*SearchResult, 0, len(candidates))
		for _, c := range candidates {
			results = append(results, &SearchResult{
				ID:       c.id,
				Vector:   c.vector,
				Distance: computeL2Distance(query, c.vector),
			})
		}
		lshResults <- results
	}()

	// Start HNSW search in parallel
	hnswResults := make(chan []*SearchResult, 1)
	go func() {
		// Convert graph search results to SearchResult format
		graphResults, err := hs.hnsw.FindNearest(ctx, query, k)
		if err != nil {
			log.Error().Err(err).Msg("HNSW search failed")
			hnswResults <- nil
			return
		}
		results := make([]*SearchResult, 0, len(graphResults))
		for _, node := range graphResults {
			results = append(results, &SearchResult{
				ID:       node.ID,
				Vector:   node.Vector,
				Distance: computeL2Distance(query, node.Vector),
			})
		}
		hnswResults <- results
	}()

	// Wait for either LSH or HNSW to complete
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("search timeout")
	case results := <-lshResults:
		hs.stats.lshHits.Add(1)
		if len(results) >= k && hs.validateResults(results[:k]) {
			return results[:k], nil
		}
		// Wait for HNSW results if LSH quality is insufficient
		select {
		case <-ctx.Done():
			return results[:k], nil // Return LSH results if timeout
		case hnswResults := <-hnswResults:
			hs.stats.hnswHits.Add(1)
			return hs.mergeResults(results, hnswResults, k), nil
		}
	case results := <-hnswResults:
		hs.stats.hnswHits.Add(1)
		return results[:k], nil
	}
}

// validateResults checks if the results meet the quality threshold
func (hs *HybridSearch) validateResults(results []*SearchResult) bool {
	if len(results) == 0 {
		return false
	}

	// Check if distances are within acceptable range
	maxDist := results[0].Distance * (1 + hs.config.QualityThreshold)
	for _, r := range results {
		if r.Distance > maxDist {
			return false
		}
	}
	return true
}

// mergeResults combines and deduplicates results from LSH and HNSW
func (hs *HybridSearch) mergeResults(lshResults, hnswResults []*SearchResult, k int) []*SearchResult {
	seen := make(map[string]bool)
	merged := make([]*SearchResult, 0, k)

	// Helper function to add unique results
	addUnique := func(result *SearchResult) {
		if !seen[result.ID] && len(merged) < k {
			seen[result.ID] = true
			merged = append(merged, result)
		}
	}

	// Merge results, prioritizing lower distances
	i, j := 0, 0
	for len(merged) < k && (i < len(lshResults) || j < len(hnswResults)) {
		if i < len(lshResults) && (j >= len(hnswResults) || lshResults[i].Distance <= hnswResults[j].Distance) {
			addUnique(lshResults[i])
			i++
		} else if j < len(hnswResults) {
			addUnique(hnswResults[j])
			j++
		}
	}

	return merged
}

// updateStats updates search statistics
func (hs *HybridSearch) updateStats(duration time.Duration) {
	hs.stats.totalQueries.Add(1)
	// Update moving average of latency
	current := hs.stats.avgLatency.Load()
	newAvg := (current*9 + duration.Nanoseconds()) / 10 // Simple moving average
	hs.stats.avgLatency.Store(newAvg)
}

// SearchResult represents a single search result
type SearchResult struct {
	ID       string
	Vector   []float32
	Distance float32
}
