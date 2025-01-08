package milvus

import (
	"context"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// IndexType represents different types of vector indices
type IndexType int

const (
	IndexTypeHNSW IndexType = iota
	IndexTypeLSH
	IndexTypeQuantization
)

// String returns the string representation of IndexType
func (i IndexType) String() string {
	switch i {
	case IndexTypeHNSW:
		return "hnsw"
	case IndexTypeLSH:
		return "lsh"
	case IndexTypeQuantization:
		return "quantization"
	default:
		return "unknown"
	}
}

// IndexStats tracks performance metrics for each index type
type IndexStats struct {
	AvgLatency     float64
	HitRate        float64
	MemoryUsage    int64
	LastUpdated    time.Time
	TotalQueries   int64
	SuccessQueries int64
	CurrentIndex   IndexType
	mu             sync.RWMutex
}

// AdaptiveIndex manages multiple index types and automatically selects the best one
type AdaptiveIndex struct {
	currentIndex IndexType
	stats        map[IndexType]*IndexStats
	config       *AdaptiveConfig
	mu           sync.RWMutex
}

type AdaptiveConfig struct {
	// Thresholds for switching index types
	DatasetSizeThreshold int64         // Switch to LSH/Quantization above this size
	LatencyThreshold     time.Duration // Switch if latency exceeds this
	MemoryThreshold      int64         // Switch if memory usage exceeds this
	MinQueryCount        int64         // Minimum queries before considering a switch
	EvaluationInterval   time.Duration // How often to evaluate index performance
}

func NewAdaptiveIndex(config *AdaptiveConfig) *AdaptiveIndex {
	if config == nil {
		config = &AdaptiveConfig{
			DatasetSizeThreshold: 1000000, // 1M vectors
			LatencyThreshold:     time.Millisecond * 100,
			MemoryThreshold:      1 << 30, // 1GB
			MinQueryCount:        1000,
			EvaluationInterval:   time.Minute * 5,
		}
	}

	return &AdaptiveIndex{
		currentIndex: IndexTypeHNSW, // Start with HNSW as default
		stats:        make(map[IndexType]*IndexStats),
		config:       config,
	}
}

// UpdateStats records performance metrics for the current index
func (ai *AdaptiveIndex) UpdateStats(indexType IndexType, latency time.Duration, success bool, memoryUsage int64) {
	ai.mu.Lock()
	defer ai.mu.Unlock()

	stats, ok := ai.stats[indexType]
	if !ok {
		stats = &IndexStats{}
		ai.stats[indexType] = stats
	}

	stats.mu.Lock()
	defer stats.mu.Unlock()

	stats.TotalQueries++
	if success {
		stats.SuccessQueries++
	}

	// Update moving average for latency
	if stats.TotalQueries == 1 {
		stats.AvgLatency = float64(latency.Nanoseconds())
	} else {
		alpha := 0.1 // Smoothing factor
		stats.AvgLatency = stats.AvgLatency*(1-alpha) + float64(latency.Nanoseconds())*alpha
	}

	stats.HitRate = float64(stats.SuccessQueries) / float64(stats.TotalQueries)
	stats.MemoryUsage = memoryUsage
	stats.LastUpdated = time.Now()
	stats.CurrentIndex = indexType
}

// SelectBestIndex determines the optimal index type based on current statistics
func (ai *AdaptiveIndex) SelectBestIndex(ctx context.Context, datasetSize int64) IndexType {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	// Check if we have enough data to make a decision
	for _, stats := range ai.stats {
		stats.mu.RLock()
		if stats.TotalQueries < ai.config.MinQueryCount {
			stats.mu.RUnlock()
			return ai.currentIndex
		}
		stats.mu.RUnlock()
	}

	// Large dataset optimization
	if datasetSize > ai.config.DatasetSizeThreshold {
		currentStats := ai.stats[ai.currentIndex]
		currentStats.mu.RLock()
		defer currentStats.mu.RUnlock()

		// Check if current index is struggling
		if time.Duration(currentStats.AvgLatency) > ai.config.LatencyThreshold ||
			currentStats.MemoryUsage > ai.config.MemoryThreshold {

			// Switch to LSH for very large datasets with memory pressure
			if currentStats.MemoryUsage > ai.config.MemoryThreshold {
				log.Info().
					Int64("dataset_size", datasetSize).
					Int64("memory_usage", currentStats.MemoryUsage).
					Msg("Switching to LSH index due to memory pressure")
				return IndexTypeLSH
			}

			// Switch to quantization for large datasets with high latency
			log.Info().
				Int64("dataset_size", datasetSize).
				Float64("avg_latency_ms", currentStats.AvgLatency/1e6).
				Msg("Switching to quantization index due to high latency")
			return IndexTypeQuantization
		}
	}

	// For smaller datasets or good performance, stick with HNSW
	return IndexTypeHNSW
}

// GetCurrentStats returns statistics for the current index
func (ai *AdaptiveIndex) GetCurrentStats() *IndexStats {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	stats := ai.stats[ai.currentIndex]
	if stats == nil {
		return &IndexStats{CurrentIndex: ai.currentIndex}
	}

	stats.mu.RLock()
	defer stats.mu.RUnlock()

	// Return a copy to avoid concurrent access issues
	return &IndexStats{
		AvgLatency:     stats.AvgLatency,
		HitRate:        stats.HitRate,
		MemoryUsage:    stats.MemoryUsage,
		LastUpdated:    stats.LastUpdated,
		TotalQueries:   stats.TotalQueries,
		SuccessQueries: stats.SuccessQueries,
		CurrentIndex:   stats.CurrentIndex,
	}
}

// ShouldEvaluate checks if it's time to reevaluate the index type
func (ai *AdaptiveIndex) ShouldEvaluate() bool {
	ai.mu.RLock()
	defer ai.mu.RUnlock()

	stats := ai.stats[ai.currentIndex]
	if stats == nil {
		return true
	}

	stats.mu.RLock()
	defer stats.mu.RUnlock()

	return time.Since(stats.LastUpdated) >= ai.config.EvaluationInterval
}
