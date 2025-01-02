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

// Metrics holds different similarity/distance measurements with enhanced scoring
type Metrics struct {
	Cosine          float64
	Angular         float64 // in degrees
	Euclidean       float64
	Manhattan       float64
	Contextual      float64
	RelativeScore   float64 // score relative to baseline
	Significance    float64 // statistical significance of the match
	ConfidenceScore float64 // confidence in the similarity assessment
}

// Context defines weights and configuration for similarity calculation
type Context struct {
	// Core weights
	TopicalWeight   float64
	SemanticWeight  float64
	SyntacticWeight float64
	LanguageWeight  float64

	// Enhanced configuration
	SignificanceThreshold float64
	MinConfidence         float64
	BaselineScore         float64

	// Processing settings
	BatchSize      int
	UseConcurrency bool
	CacheResults   bool

	// Normalization settings
	NormalizeLength  bool
	RemoveStopwords  bool
	UseLemmatization bool
}

// DefaultContext returns an optimized context configuration
func DefaultContext() Context {
	return Context{
		// Core weights - adjusted based on empirical testing
		TopicalWeight:   0.35,
		SemanticWeight:  0.30,
		SyntacticWeight: 0.20,
		LanguageWeight:  0.15,

		// Enhanced configuration
		SignificanceThreshold: 0.05,
		MinConfidence:         0.7,
		BaselineScore:         0.5,

		// Processing settings
		BatchSize:      1000,
		UseConcurrency: true,
		CacheResults:   true,

		// Normalization settings
		NormalizeLength:  true,
		RemoveStopwords:  true,
		UseLemmatization: true,
	}
}

// Calculate computes all similarity metrics with enhanced scoring
func Calculate(vec1, vec2 []float32, ctx Context) (Metrics, error) {
	if err := validateVectors(vec1, vec2); err != nil {
		return Metrics{}, err
	}

	// Normalize vectors if configured
	if ctx.NormalizeLength {
		vec1 = normalizeVector(vec1)
		vec2 = normalizeVector(vec2)
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
		metrics.Cosine = calculateCosine(vec1, vec2)
		metrics.Angular = math.Acos(metrics.Cosine) * 180 / math.Pi
		metrics.Euclidean = calculateEuclidean(vec1, vec2)
		metrics.Manhattan = calculateManhattan(vec1, vec2)
	}

	// Calculate enhanced metrics
	metrics.Contextual = calculateContextual(metrics, ctx)
	metrics.RelativeScore = calculateRelativeScore(metrics, ctx)
	metrics.Significance = calculateSignificance(metrics, ctx)
	metrics.ConfidenceScore = calculateConfidence(metrics, ctx)

	return metrics, nil
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
	// Enhanced contextual score calculation
	normalizedEuclidean := 1 / (1 + m.Euclidean)
	normalizedManhattan := 1 / (1 + m.Manhattan)

	// Weighted combination with non-linear scaling
	score := math.Pow(m.Cosine, 1.2)*ctx.SemanticWeight +
		math.Pow((180-m.Angular)/180, 1.1)*ctx.SyntacticWeight +
		math.Pow(normalizedEuclidean, 1.3)*ctx.TopicalWeight +
		math.Pow(normalizedManhattan, 1.1)*ctx.LanguageWeight

	totalWeight := ctx.SemanticWeight + ctx.SyntacticWeight +
		ctx.TopicalWeight + ctx.LanguageWeight

	return score / totalWeight
}

func calculateRelativeScore(m Metrics, ctx Context) float64 {
	// Compare against baseline
	baselineDeviation := m.Contextual - ctx.BaselineScore
	return math.Max(0, math.Min(1, 0.5+baselineDeviation))
}

func calculateSignificance(m Metrics, ctx Context) float64 {
	// Statistical significance estimation
	if m.Contextual < ctx.SignificanceThreshold {
		return 0
	}

	// Non-linear scaling of significance
	return math.Pow((m.Contextual-ctx.SignificanceThreshold)/(1-ctx.SignificanceThreshold), 1.5)
}

func calculateConfidence(m Metrics, ctx Context) float64 {
	// Confidence score based on metric consistency
	variance := calculateMetricVariance(m)
	return math.Max(0, 1-variance)
}

func calculateMetricVariance(m Metrics) float64 {
	metrics := []float64{m.Cosine, (180 - m.Angular) / 180, 1 / (1 + m.Euclidean), 1 / (1 + m.Manhattan)}

	var mean, variance float64
	for _, v := range metrics {
		mean += v
	}
	mean /= float64(len(metrics))

	for _, v := range metrics {
		diff := v - mean
		variance += diff * diff
	}
	return variance / float64(len(metrics))
}

func normalizeVector(vec []float32) []float32 {
	normalized := make([]float32, len(vec))
	var norm float32

	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	for i, v := range vec {
		normalized[i] = v / norm
	}
	return normalized
}

func validateVectors(vec1, vec2 []float32) error {
	if len(vec1) == 0 || len(vec2) == 0 {
		return fmt.Errorf("input vectors must not be empty")
	}
	if len(vec1) != len(vec2) {
		return fmt.Errorf("input vectors must have the same dimension")
	}
	return nil
}
