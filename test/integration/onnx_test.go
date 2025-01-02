package integration

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/objones25/athena/internal/embeddings"
	"github.com/objones25/athena/internal/embeddings/similarity"
)

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct float64
	var normA float64
	var normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// validateEmbedding checks if an embedding vector has valid properties
func validateEmbedding(t *testing.T, embedding []float32, dimension int) {
	t.Helper()

	// Check dimension
	require.Equal(t, dimension, len(embedding), "Embedding dimension mismatch")

	// Check if vector is all zeros
	allZeros := true
	for _, v := range embedding {
		if v != 0 {
			allZeros = false
			break
		}
	}
	require.False(t, allZeros, "Embedding is a zero vector")

	// Calculate and log vector statistics
	var sum, sumSquares float64
	var min, max float32 = embedding[0], embedding[0]

	for _, v := range embedding {
		sum += float64(v)
		sumSquares += float64(v * v)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	mean := sum / float64(len(embedding))
	variance := (sumSquares / float64(len(embedding))) - (mean * mean)
	stdDev := math.Sqrt(variance)
	l2Norm := math.Sqrt(sumSquares)

	t.Logf("Vector stats: mean=%.6f, stdDev=%.6f, min=%.6f, max=%.6f, l2Norm=%.6f",
		mean, stdDev, min, max, l2Norm)

	// Check if vector has reasonable values
	require.NotZero(t, l2Norm, "Vector L2 norm should not be zero")
	require.True(t, stdDev > 0, "Vector should have non-zero standard deviation")
}

func TestONNXClient(t *testing.T) {
	// Reset ONNX runtime state before starting tests
	embeddings.ResetRuntime()

	modelsDir := os.Getenv("TEST_MODELS_DIR")
	if modelsDir == "" {
		t.Skip("TEST_MODELS_DIR not set")
	}

	// Test configurations for different models
	configs := []embeddings.ModelConfig{
		{
			ModelPath:           filepath.Join(modelsDir, "all-MiniLM-L6-v2.onnx"),
			TokenizerConfigPath: filepath.Join(modelsDir, "all-MiniLM-L6-v2_tokenizer"),
			Dimension:           384,
			MaxLength:           512,
			UseGPU:              false,
		},
		{
			ModelPath:           filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2.onnx"),
			TokenizerConfigPath: filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2_tokenizer"),
			Dimension:           384,
			MaxLength:           512,
			UseGPU:              false,
		},
	}

	for _, cfg := range configs {
		t.Run(filepath.Base(cfg.ModelPath), func(t *testing.T) {
			// Check if model exists
			if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
				t.Skipf("Model not found at %s, skipping test", cfg.ModelPath)
			}

			// Check if tokenizer exists
			if _, err := os.Stat(cfg.TokenizerConfigPath); os.IsNotExist(err) {
				t.Skipf("Tokenizer not found at %s, skipping test", cfg.TokenizerConfigPath)
			}

			// Create client
			client, err := embeddings.NewONNXClient(cfg)
			require.NoError(t, err)
			defer client.Close()

			// Initialize client
			err = client.Initialize(context.Background())
			require.NoError(t, err)

			// Test semantic similarity
			t.Run("Semantic_Similarity", func(t *testing.T) {
				// Create similarity context with test-specific settings
				simCtx := similarity.Context{
					TopicalWeight:   0.4, // Weight for topic differentiation
					SemanticWeight:  0.3, // Weight for semantic meaning
					SyntacticWeight: 0.2, // Weight for structural similarity
					LanguageWeight:  0.1, // Weight for cross-lingual cases
					UseConcurrency:  true,
					CacheResults:    false, // Disable caching for tests
				}

				testCases := []struct {
					name     string
					text1    string
					text2    string
					similar  bool // Whether the texts should be semantically similar
					minScore float64
					maxScore float64
					weights  *similarity.Context // Optional case-specific weights
				}{
					{
						name:     "Same Meaning Different Words",
						text1:    "The quick brown fox jumps over the lazy dog",
						text2:    "A fast auburn canine leaps across a sleepy hound",
						similar:  true,
						minScore: 0.7,
						maxScore: 1.0,
						weights: &similarity.Context{ // Emphasize semantic similarity
							TopicalWeight:   0.2,
							SemanticWeight:  0.5,
							SyntacticWeight: 0.2,
							LanguageWeight:  0.1,
						},
					},
					{
						name:     "Different Topics",
						text1:    "Quantum physics deals with subatomic particles",
						text2:    "The Renaissance was an important cultural movement",
						similar:  false,
						minScore: 0.0,
						maxScore: 0.5,
						weights: &similarity.Context{ // Emphasize topic differentiation
							TopicalWeight:   0.6,
							SemanticWeight:  0.2,
							SyntacticWeight: 0.1,
							LanguageWeight:  0.1,
						},
					},
					{
						name:     "Multilingual Similarity",
						text1:    "Hello, how are you?",
						text2:    "¡Hola, cómo estás?",
						similar:  true,
						minScore: 0.6,
						maxScore: 1.0,
						weights: &similarity.Context{ // Emphasize cross-lingual understanding
							TopicalWeight:   0.2,
							SemanticWeight:  0.3,
							SyntacticWeight: 0.1,
							LanguageWeight:  0.4,
						},
					},
					{
						name:     "Opposite Meanings",
						text1:    "The stock market is going up today",
						text2:    "The stock market is crashing today",
						similar:  false,
						minScore: 0.0,
						maxScore: 0.6,
						weights: &similarity.Context{ // Emphasize semantic differences
							TopicalWeight:   0.2,
							SemanticWeight:  0.5,
							SyntacticWeight: 0.2,
							LanguageWeight:  0.1,
						},
					},
				}

				for _, tc := range testCases {
					t.Run(tc.name, func(t *testing.T) {
						// Generate embeddings
						embedding1, err := client.GenerateEmbedding(context.Background(), tc.text1)
						require.NoError(t, err)
						t.Log("\nEmbedding 1 stats:")
						validateEmbedding(t, embedding1, cfg.Dimension)

						embedding2, err := client.GenerateEmbedding(context.Background(), tc.text2)
						require.NoError(t, err)
						t.Log("\nEmbedding 2 stats:")
						validateEmbedding(t, embedding2, cfg.Dimension)

						// Use test-specific weights if provided, otherwise use default
						ctx := simCtx
						if tc.weights != nil {
							ctx = *tc.weights
							ctx.UseConcurrency = simCtx.UseConcurrency
							ctx.CacheResults = simCtx.CacheResults
						}

						// Calculate similarity metrics
						metrics, err := similarity.Calculate(embedding1, embedding2, ctx)
						require.NoError(t, err)

						// Log detailed similarity analysis
						t.Logf("\nSimilarity Analysis:")
						t.Logf("Text 1: %q", tc.text1)
						t.Logf("Text 2: %q", tc.text2)
						t.Logf("Metrics:")
						t.Logf("  - Cosine: %.6f", metrics.Cosine)
						t.Logf("  - Angular: %.2f degrees", metrics.Angular)
						t.Logf("  - Euclidean: %.6f", metrics.Euclidean)
						t.Logf("  - Manhattan: %.6f", metrics.Manhattan)
						t.Logf("  - Contextual: %.6f", metrics.Contextual)
						t.Logf("Expected Range: %.6f to %.6f", tc.minScore, tc.maxScore)

						// Validate results
						if tc.similar {
							require.GreaterOrEqual(t, metrics.Contextual, tc.minScore,
								"Expected texts to be similar with score >= %f, got %f (angle: %.2f°)",
								tc.minScore, metrics.Contextual, metrics.Angular)
						} else {
							require.LessOrEqual(t, metrics.Contextual, tc.maxScore,
								"Expected texts to be different with score <= %f, got %f (angle: %.2f°)",
								tc.maxScore, metrics.Contextual, metrics.Angular)
						}
					})
				}
			})

			// Test performance
			t.Run("Performance", func(t *testing.T) {
				// Test single embedding generation performance
				t.Run("Single_Embedding", func(t *testing.T) {
					start := time.Now()
					embedding, err := client.GenerateEmbedding(context.Background(), "This is a test sentence for performance measurement.")
					duration := time.Since(start)
					require.NoError(t, err)
					validateEmbedding(t, embedding, cfg.Dimension)

					t.Logf("Single embedding generation took: %v", duration)
					require.Less(t, duration.Seconds(), 0.5, "Single embedding generation took too long: %v", duration)
				})

				// Test batch embedding generation performance
				t.Run("Batch_Embedding", func(t *testing.T) {
					texts := []string{
						"First test sentence for batch processing.",
						"Second test sentence with different content.",
						"Third sentence to ensure proper batch handling.",
						"Fourth sentence with more diverse vocabulary.",
						"Fifth sentence to complete the batch test.",
					}

					start := time.Now()
					embeddings, err := client.BatchGenerateEmbeddings(context.Background(), texts)
					duration := time.Since(start)
					require.NoError(t, err)

					for _, embedding := range embeddings {
						validateEmbedding(t, embedding, cfg.Dimension)
					}

					t.Logf("Batch embedding generation took: %v", duration)
					require.Less(t, duration.Seconds(), 3.0, "Batch embedding generation took too long: %v", duration)
				})
			})
		})
	}
}

func BenchmarkONNXClient(b *testing.B) {
	modelsDir := os.Getenv("TEST_MODELS_DIR")
	if modelsDir == "" {
		b.Skip("TEST_MODELS_DIR not set")
	}

	cfg := embeddings.ModelConfig{
		ModelPath:           filepath.Join(modelsDir, "all-MiniLM-L6-v2.onnx"),
		TokenizerConfigPath: filepath.Join(modelsDir, "all-MiniLM-L6-v2_tokenizer"),
		Dimension:           384,
		MaxLength:           512,
		UseGPU:              false,
	}

	client, err := embeddings.NewONNXClient(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer client.Close()

	if err := client.Initialize(context.Background()); err != nil {
		b.Fatal(err)
	}

	texts := []string{
		"Short text",
		"Medium length text for testing embedding generation",
		"A longer text that contains multiple sentences. This should test how the model handles longer sequences. It includes punctuation and various lengths.",
	}

	for _, text := range texts {
		name := fmt.Sprintf("Len_%d", len(text))
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := client.GenerateEmbedding(context.Background(), text)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}

	// Benchmark batch processing
	batchSizes := []int{1, 10, 50, 100}
	for _, size := range batchSizes {
		name := fmt.Sprintf("Batch_%d", size)
		b.Run(name, func(b *testing.B) {
			batch := make([]string, size)
			for i := range batch {
				batch[i] = fmt.Sprintf("Test text for batch processing %d", i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := client.BatchGenerateEmbeddings(context.Background(), batch)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func sqrt(x float64) float64 {
	return math.Sqrt(x)
}

func abs(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// Helper function to calculate vector norm
func vectorNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	return math.Sqrt(sum)
}
