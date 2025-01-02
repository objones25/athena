package cache

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/objones25/athena/internal/embeddings/similarity"
)

// mockCache implements Cache interface for testing
type mockCache struct {
	data map[string][]float32
}

func newMockCache() *mockCache {
	return &mockCache{
		data: make(map[string][]float32),
	}
}

func (m *mockCache) Get(_ context.Context, key string) ([]float32, error) {
	if vec, ok := m.data[key]; ok {
		return vec, nil
	}
	return nil, ErrNotFound
}

func (m *mockCache) Set(_ context.Context, key string, embedding []float32) error {
	m.data[key] = embedding
	return nil
}

func (m *mockCache) MGet(_ context.Context, keys []string) ([][]float32, error) {
	result := make([][]float32, len(keys))
	for i, key := range keys {
		if vec, ok := m.data[key]; ok {
			result[i] = vec
		} else {
			return nil, ErrNotFound
		}
	}
	return result, nil
}

func (m *mockCache) MSet(_ context.Context, items map[string][]float32) error {
	for k, v := range items {
		m.data[k] = v
	}
	return nil
}

func (m *mockCache) Delete(_ context.Context, keys []string) error {
	for _, key := range keys {
		delete(m.data, key)
	}
	return nil
}

func (m *mockCache) Clear(_ context.Context) error {
	m.data = make(map[string][]float32)
	return nil
}

func (m *mockCache) Close() error {
	return nil
}

func (m *mockCache) Health(_ context.Context) error {
	return nil
}

func TestSimilarityCache(t *testing.T) {
	ctx := context.Background()
	cache := newMockCache()
	simCache := NewSimilarityCache(cache, nil)

	// Test vectors
	vec1 := []float32{1.0, 0.0, 0.0} // pointing along x-axis
	vec2 := []float32{0.0, 1.0, 0.0} // pointing along y-axis
	vec3 := []float32{0.5, 0.5, 0.0} // 45 degrees between x and y

	// Store test vectors
	require.NoError(t, cache.Set(ctx, "vec1", vec1))
	require.NoError(t, cache.Set(ctx, "vec2", vec2))
	require.NoError(t, cache.Set(ctx, "vec3", vec3))

	t.Run("CompareCached", func(t *testing.T) {
		metrics, err := simCache.CompareCached(ctx, "vec1", "vec2")
		require.NoError(t, err)

		// vec1 and vec2 are perpendicular, so cosine should be 0
		assert.InDelta(t, 0.0, metrics.Cosine, 0.001)
		assert.InDelta(t, 90.0, metrics.Angular, 0.001)
	})

	t.Run("BatchCompareCached", func(t *testing.T) {
		keyPairs := [][2]string{
			{"vec1", "vec2"},
			{"vec1", "vec3"},
			{"vec2", "vec3"},
		}

		metrics, err := simCache.BatchCompareCached(ctx, keyPairs)
		require.NoError(t, err)
		require.Len(t, metrics, 3)

		// Check vec1-vec2 comparison (perpendicular)
		assert.InDelta(t, 0.0, metrics[0].Cosine, 0.001)
		assert.InDelta(t, 90.0, metrics[0].Angular, 0.001)

		// Check vec1-vec3 comparison (45 degrees)
		assert.InDelta(t, 0.707, metrics[1].Cosine, 0.001) // cos(45°) ≈ 0.707
		assert.InDelta(t, 45.0, metrics[1].Angular, 0.001)

		// Check vec2-vec3 comparison (45 degrees)
		assert.InDelta(t, 0.707, metrics[2].Cosine, 0.001)
		assert.InDelta(t, 45.0, metrics[2].Angular, 0.001)
	})

	t.Run("UpdateContext", func(t *testing.T) {
		newCtx := &similarity.Context{
			TopicalWeight:   0.5,
			SemanticWeight:  0.2,
			SyntacticWeight: 0.2,
			LanguageWeight:  0.1,
			BatchSize:       500,
			UseConcurrency:  false,
			CacheResults:    false,
		}

		simCache.UpdateContext(newCtx)
		currentCtx := simCache.GetContext()

		assert.Equal(t, newCtx.TopicalWeight, currentCtx.TopicalWeight)
		assert.Equal(t, newCtx.SemanticWeight, currentCtx.SemanticWeight)
		assert.Equal(t, newCtx.BatchSize, currentCtx.BatchSize)
		assert.Equal(t, newCtx.UseConcurrency, currentCtx.UseConcurrency)
	})
}
