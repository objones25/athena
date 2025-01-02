package unit

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/objones25/athena/internal/embeddings"
)

func TestMockService(t *testing.T) {
	mockConfig := embeddings.ModelConfig{
		ModelPath:           "models/test_model.onnx",
		TokenizerConfigPath: "models/test_tokenizer.json",
		Dimension:           384,
		MaxLength:           512,
		UseGPU:              false,
	}

	t.Run("Generate", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		req := &embeddings.EmbeddingRequest{
			Content:     "test content",
			ContentType: embeddings.ContentTypeText,
			Model:       embeddings.ModelMultilingualLM,
		}

		// Test successful generation
		result, err := svc.Generate(context.Background(), req)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, mockConfig.Dimension, len(result.Vector))
		assert.Equal(t, embeddings.ModelMultilingualLM, result.Model)

		// Test with simulated failure
		svc.SetFailureRate(1.0) // 100% failure rate
		_, err = svc.Generate(context.Background(), req)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "simulated failure")
	})

	t.Run("BatchGenerate", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		reqs := []*embeddings.EmbeddingRequest{
			{
				Content:     "test content 1",
				ContentType: embeddings.ContentTypeText,
				Model:       embeddings.ModelMultilingualLM,
			},
			{
				Content:     "test content 2",
				ContentType: embeddings.ContentTypeText,
				Model:       embeddings.ModelMultilingualLM,
			},
		}

		// Test successful batch generation
		results, err := svc.BatchGenerate(context.Background(), reqs)
		assert.NoError(t, err)
		assert.Len(t, results, len(reqs))
		for _, result := range results {
			assert.Equal(t, mockConfig.Dimension, len(result.Vector))
			assert.Equal(t, embeddings.ModelMultilingualLM, result.Model)
		}

		// Test with simulated failure
		svc.SetFailureRate(1.0) // 100% failure rate
		_, err = svc.BatchGenerate(context.Background(), reqs)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "simulated failure")
	})

	t.Run("Latency", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		req := &embeddings.EmbeddingRequest{
			Content:     "test content",
			ContentType: embeddings.ContentTypeText,
			Model:       embeddings.ModelMultilingualLM,
		}

		// Set artificial latency
		expectedLatency := 100 * time.Millisecond
		svc.SetLatency(expectedLatency)

		start := time.Now()
		_, err := svc.Generate(context.Background(), req)
		duration := time.Since(start)

		assert.NoError(t, err)
		assert.GreaterOrEqual(t, duration, expectedLatency)
	})

	t.Run("Health", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)

		// Test healthy state
		assert.NoError(t, svc.Health(context.Background()))

		// Test unhealthy state
		svc.SetHealthStatus(false)
		assert.Error(t, svc.Health(context.Background()))
	})

	t.Run("GetModelInfo", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		info, err := svc.GetModelInfo(embeddings.ModelMultilingualLM)
		assert.NoError(t, err)
		assert.Equal(t, mockConfig, *info)

		// Test invalid model type
		_, err = svc.GetModelInfo("invalid-model")
		assert.Error(t, err)
	})

	t.Run("CallCount", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		req := &embeddings.EmbeddingRequest{
			Content:     "test content",
			ContentType: embeddings.ContentTypeText,
			Model:       embeddings.ModelMultilingualLM,
		}

		// Generate a few embeddings
		for i := 0; i < 3; i++ {
			_, err := svc.Generate(context.Background(), req)
			assert.NoError(t, err)
		}

		assert.Equal(t, 3, svc.GetCallCount("generate"))
	})

	t.Run("Close", func(t *testing.T) {
		svc := embeddings.NewMockService(mockConfig)
		assert.NoError(t, svc.Close())

		// Test with simulated failure
		svc.SetError("close", assert.AnError)
		assert.Error(t, svc.Close())
	})
}
