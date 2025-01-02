package similarity

import (
	"fmt"
	"math"
	"sync"

	"golang.org/x/sync/errgroup"
)

// MetricType represents different types of similarity/distance metrics
type MetricType int

const (
	Cosine MetricType = iota
	Angular
	Euclidean
	Manhattan
	Contextual
)

// Metrics holds different similarity/distance measurements
type Metrics struct {
	Cosine     float64
	Angular    float64 // in degrees
	Euclidean  float64
	Manhattan  float64
	Contextual float64
}

// Context defines weights for different aspects of similarity
type Context struct {
	TopicalWeight   float64
	SemanticWeight  float64
	SyntacticWeight float64
	LanguageWeight  float64
	BatchSize       int  // for batch processing
	UseConcurrency  bool // enable/disable concurrent processing
	CacheResults    bool // enable result caching
}

// DefaultContext returns a balanced context configuration
func DefaultContext() Context {
	return Context{
		TopicalWeight:   0.4,
		SemanticWeight:  0.3,
		SyntacticWeight: 0.2,
		LanguageWeight:  0.1,
		BatchSize:       1000,
		UseConcurrency:  true,
		CacheResults:    true,
	}
}

// Cache for storing computed similarities
type similarityCache struct {
	sync.RWMutex
	cache map[string]Metrics
}

var (
	cache = &similarityCache{
		cache: make(map[string]Metrics),
	}
)

// Calculate computes all similarity metrics concurrently
func Calculate(vec1, vec2 []float32, ctx Context) (Metrics, error) {
	if ctx.CacheResults {
		if metrics, ok := getCachedMetrics(vec1, vec2); ok {
			return metrics, nil
		}
	}

	var metrics Metrics
	var g errgroup.Group
	var mu sync.Mutex

	if ctx.UseConcurrency {
		// Concurrent calculation of basic metrics
		g.Go(func() error {
			cosine := calculateCosine(vec1, vec2)
			mu.Lock()
			metrics.Cosine = cosine
			metrics.Angular = math.Acos(cosine) * 180 / math.Pi
			mu.Unlock()
			return nil
		})

		g.Go(func() error {
			euclidean := calculateEuclidean(vec1, vec2)
			mu.Lock()
			metrics.Euclidean = euclidean
			mu.Unlock()
			return nil
		})

		g.Go(func() error {
			manhattan := calculateManhattan(vec1, vec2)
			mu.Lock()
			metrics.Manhattan = manhattan
			mu.Unlock()
			return nil
		})

		if err := g.Wait(); err != nil {
			return Metrics{}, err
		}
	} else {
		// Sequential calculation
		metrics.Cosine = calculateCosine(vec1, vec2)
		metrics.Angular = math.Acos(metrics.Cosine) * 180 / math.Pi
		metrics.Euclidean = calculateEuclidean(vec1, vec2)
		metrics.Manhattan = calculateManhattan(vec1, vec2)
	}

	// Calculate contextual score using all metrics
	metrics.Contextual = calculateContextual(metrics, ctx)

	if ctx.CacheResults {
		cacheMetrics(vec1, vec2, metrics)
	}

	return metrics, nil
}

// BatchCalculate processes multiple vector pairs efficiently
func BatchCalculate(vectors1, vectors2 [][]float32, ctx Context) ([]Metrics, error) {
	if len(vectors1) != len(vectors2) {
		return nil, ErrMismatchedVectors
	}

	results := make([]Metrics, len(vectors1))

	if !ctx.UseConcurrency {
		// Sequential processing
		for i := range vectors1 {
			metrics, err := Calculate(vectors1[i], vectors2[i], ctx)
			if err != nil {
				return nil, err
			}
			results[i] = metrics
		}
		return results, nil
	}

	// Concurrent batch processing
	var g errgroup.Group
	var mu sync.Mutex

	// Process in batches
	for batchStart := 0; batchStart < len(vectors1); batchStart += ctx.BatchSize {
		batchEnd := min(batchStart+ctx.BatchSize, len(vectors1))
		batchIndex := batchStart // Capture for closure

		g.Go(func() error {
			batchResults := make([]Metrics, batchEnd-batchIndex)
			for i := 0; i < len(batchResults); i++ {
				metrics, err := Calculate(
					vectors1[batchIndex+i],
					vectors2[batchIndex+i],
					ctx,
				)
				if err != nil {
					return err
				}
				batchResults[i] = metrics
			}

			mu.Lock()
			copy(results[batchIndex:], batchResults)
			mu.Unlock()
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return results, nil
}

// Helper functions for metric calculations
func calculateCosine(a, b []float32) float64 {
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func calculateEuclidean(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func calculateManhattan(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += math.Abs(float64(a[i] - b[i]))
	}
	return sum
}

func calculateContextual(m Metrics, ctx Context) float64 {
	// Normalize distances to [0,1] range
	normalizedEuclidean := 1 / (1 + m.Euclidean)
	normalizedManhattan := 1 / (1 + m.Manhattan)

	// Combine metrics with weights
	score := m.Cosine*ctx.SemanticWeight +
		((180-m.Angular)/180)*ctx.SyntacticWeight +
		normalizedEuclidean*ctx.TopicalWeight +
		normalizedManhattan*ctx.LanguageWeight

	totalWeight := ctx.SemanticWeight + ctx.SyntacticWeight +
		ctx.TopicalWeight + ctx.LanguageWeight

	return score / totalWeight
}

// Cache management
func getCachedMetrics(vec1, vec2 []float32) (Metrics, bool) {
	key := getCacheKey(vec1, vec2)
	cache.RLock()
	metrics, ok := cache.cache[key]
	cache.RUnlock()
	return metrics, ok
}

func cacheMetrics(vec1, vec2 []float32, metrics Metrics) {
	key := getCacheKey(vec1, vec2)
	cache.Lock()
	cache.cache[key] = metrics
	cache.Unlock()
}

func getCacheKey(vec1, vec2 []float32) string {
	// Implement a fast hash function for vectors
	// This is a placeholder - implement actual hashing
	return fmt.Sprintf("%v-%v", vec1[:5], vec2[:5])
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
