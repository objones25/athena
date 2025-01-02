package similarity_test

import (
	"math"
	"testing"

	"github.com/objones25/athena/internal/search/similarity"
)

func TestCalculate(t *testing.T) {
	tests := []struct {
		name    string
		vec1    []float32
		vec2    []float32
		ctx     similarity.Context
		want    float64
		wantErr bool
	}{
		{
			name: "identical vectors",
			vec1: []float32{1, 0, 0, 0},
			vec2: []float32{1, 0, 0, 0},
			ctx:  similarity.DefaultContext(),
			want: 1.0,
		},
		{
			name: "orthogonal vectors",
			vec1: []float32{1, 0, 0, 0},
			vec2: []float32{0, 1, 0, 0},
			ctx:  similarity.DefaultContext(),
			want: 0.0,
		},
		{
			name: "similar vectors",
			vec1: []float32{0.8, 0.1, 0.1, 0},
			vec2: []float32{0.7, 0.2, 0.1, 0},
			ctx:  similarity.DefaultContext(),
			want: 0.9,
		},
		{
			name:    "different dimensions",
			vec1:    []float32{1, 0},
			vec2:    []float32{1, 0, 0},
			ctx:     similarity.DefaultContext(),
			wantErr: true,
		},
		{
			name:    "empty vectors",
			vec1:    []float32{},
			vec2:    []float32{},
			ctx:     similarity.DefaultContext(),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics, err := similarity.Calculate(tt.vec1, tt.vec2, tt.ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("Calculate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if err != nil {
				return
			}

			// Check if contextual score is within acceptable range
			if math.Abs(metrics.Contextual-tt.want) > 0.1 {
				t.Errorf("Calculate() contextual score = %v, want %v", metrics.Contextual, tt.want)
			}

			// Verify metric properties
			if metrics.Cosine < -1 || metrics.Cosine > 1 {
				t.Errorf("Invalid cosine similarity: %v", metrics.Cosine)
			}
			if metrics.Angular < 0 || metrics.Angular > 180 {
				t.Errorf("Invalid angular distance: %v", metrics.Angular)
			}
			if metrics.Euclidean < 0 {
				t.Errorf("Invalid Euclidean distance: %v", metrics.Euclidean)
			}
			if metrics.Manhattan < 0 {
				t.Errorf("Invalid Manhattan distance: %v", metrics.Manhattan)
			}
		})
	}
}

func TestCalculateWithNormalization(t *testing.T) {
	tests := []struct {
		name string
		vec1 []float32
		vec2 []float32
		ctx  similarity.Context
		want float64
	}{
		{
			name: "different magnitudes",
			vec1: []float32{2, 0, 0, 0},   // Magnitude 2
			vec2: []float32{0.5, 0, 0, 0}, // Magnitude 0.5
			ctx: similarity.Context{
				TopicalWeight:   0.35,
				SemanticWeight:  0.30,
				SyntacticWeight: 0.20,
				LanguageWeight:  0.15,
				NormalizeLength: true,
			},
			want: 1.0, // Should be 1.0 after normalization
		},
		{
			name: "without normalization",
			vec1: []float32{2, 0, 0, 0},
			vec2: []float32{0.5, 0, 0, 0},
			ctx: similarity.Context{
				TopicalWeight:   0.35,
				SemanticWeight:  0.30,
				SyntacticWeight: 0.20,
				LanguageWeight:  0.15,
				NormalizeLength: false,
			},
			want: 0.25, // Should reflect magnitude difference
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics, err := similarity.Calculate(tt.vec1, tt.vec2, tt.ctx)
			if err != nil {
				t.Errorf("Calculate() error = %v", err)
				return
			}

			if math.Abs(metrics.Cosine-tt.want) > 0.1 {
				t.Errorf("Calculate() cosine = %v, want %v", metrics.Cosine, tt.want)
			}
		})
	}
}

func TestMetricsConsistency(t *testing.T) {
	tests := []struct {
		name    string
		vec1    []float32
		vec2    []float32
		ctx     similarity.Context
		wantErr bool
	}{
		{
			name: "consistent metrics",
			vec1: []float32{1, 0, 0, 0},
			vec2: []float32{0.9, 0.1, 0, 0},
			ctx:  similarity.DefaultContext(),
		},
		{
			name: "orthogonal vectors",
			vec1: []float32{1, 0, 0, 0},
			vec2: []float32{0, 1, 0, 0},
			ctx:  similarity.DefaultContext(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metrics, err := similarity.Calculate(tt.vec1, tt.vec2, tt.ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("Calculate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Verify consistency between metrics
			cosineAngle := math.Acos(metrics.Cosine) * 180 / math.Pi
			if math.Abs(cosineAngle-metrics.Angular) > 1e-6 {
				t.Errorf("Inconsistent angle: cosine angle = %v, angular = %v", cosineAngle, metrics.Angular)
			}

			// Verify that confidence score is reasonable
			if metrics.ConfidenceScore < 0 || metrics.ConfidenceScore > 1 {
				t.Errorf("Invalid confidence score: %v", metrics.ConfidenceScore)
			}

			// Verify that relative score is reasonable
			if metrics.RelativeScore < 0 || metrics.RelativeScore > 1 {
				t.Errorf("Invalid relative score: %v", metrics.RelativeScore)
			}

			// Verify that significance is reasonable
			if metrics.Significance < 0 || metrics.Significance > 1 {
				t.Errorf("Invalid significance: %v", metrics.Significance)
			}
		})
	}
}

func TestConcurrentCalculation(t *testing.T) {
	vec1 := make([]float32, 1000)
	vec2 := make([]float32, 1000)
	for i := range vec1 {
		vec1[i] = float32(i)
		vec2[i] = float32(i)
	}

	ctx := similarity.DefaultContext()
	ctx.UseConcurrency = true

	metrics, err := similarity.Calculate(vec1, vec2, ctx)
	if err != nil {
		t.Errorf("Calculate() error = %v", err)
		return
	}

	if metrics.Cosine != 1.0 {
		t.Errorf("Calculate() cosine = %v, want 1.0", metrics.Cosine)
	}
}

func BenchmarkCalculate(b *testing.B) {
	vec1 := make([]float32, 1000)
	vec2 := make([]float32, 1000)
	for i := range vec1 {
		vec1[i] = float32(i)
		vec2[i] = float32(i)
	}

	ctx := similarity.DefaultContext()

	b.Run("sequential", func(b *testing.B) {
		ctx.UseConcurrency = false
		for i := 0; i < b.N; i++ {
			_, _ = similarity.Calculate(vec1, vec2, ctx)
		}
	})

	b.Run("concurrent", func(b *testing.B) {
		ctx.UseConcurrency = true
		for i := 0; i < b.N; i++ {
			_, _ = similarity.Calculate(vec1, vec2, ctx)
		}
	})
}
