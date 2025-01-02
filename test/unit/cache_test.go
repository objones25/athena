package unit

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/objones25/athena/internal/embeddings/cache"
)

func TestRedisCache(t *testing.T) {
	// Start miniredis server
	s, err := miniredis.Run()
	require.NoError(t, err)
	defer s.Close()

	// Create cache client
	cfg := cache.Config{
		Addr: s.Addr(),
		TTL:  time.Hour,
	}
	c, err := cache.NewRedisCache(cfg)
	require.NoError(t, err)
	defer c.Close()

	ctx := context.Background()

	t.Run("Single Operations", func(t *testing.T) {
		// Test Set and Get
		key := "test:key"
		embedding := []float32{1.0, 2.0, 3.0}

		err := c.Set(ctx, key, embedding)
		require.NoError(t, err)

		result, err := c.Get(ctx, key)
		require.NoError(t, err)
		assert.Equal(t, embedding, result)

		// Test non-existent key
		result, err = c.Get(ctx, "nonexistent")
		require.NoError(t, err)
		assert.Nil(t, result)

		// Test Delete
		err = c.Delete(ctx, []string{key})
		require.NoError(t, err)

		result, err = c.Get(ctx, key)
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("Batch Operations", func(t *testing.T) {
		// Test MSet and MGet
		items := map[string][]float32{
			"key1": {1.0, 2.0, 3.0},
			"key2": {4.0, 5.0, 6.0},
			"key3": {7.0, 8.0, 9.0},
		}

		err := c.MSet(ctx, items)
		require.NoError(t, err)

		keys := []string{"key1", "key2", "nonexistent", "key3"}
		results, err := c.MGet(ctx, keys)
		require.NoError(t, err)
		assert.Len(t, results, 4)
		assert.Equal(t, items["key1"], results[0])
		assert.Equal(t, items["key2"], results[1])
		assert.Nil(t, results[2]) // nonexistent key
		assert.Equal(t, items["key3"], results[3])

		// Test Clear
		err = c.Clear(ctx)
		require.NoError(t, err)

		results, err = c.MGet(ctx, keys)
		require.NoError(t, err)
		for _, result := range results {
			assert.Nil(t, result)
		}
	})

	t.Run("TTL", func(t *testing.T) {
		key := "ttl:test"
		embedding := []float32{1.0, 2.0, 3.0}

		err := c.Set(ctx, key, embedding)
		require.NoError(t, err)

		// Fast forward time
		s.FastForward(2 * time.Hour)

		result, err := c.Get(ctx, key)
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("Health", func(t *testing.T) {
		err := c.Health(ctx)
		require.NoError(t, err)

		// Test with closed connection
		s.Close()
		err = c.Health(ctx)
		assert.Error(t, err)
	})
}

func TestDefaultKeyGenerator(t *testing.T) {
	g := cache.NewDefaultKeyGenerator("")

	t.Run("Single Key Generation", func(t *testing.T) {
		content := "test content"
		model := "BERT"

		key := g.GenerateKey(content, model)
		assert.NotEmpty(t, key)
		assert.Contains(t, key, "emb")  // default prefix
		assert.Contains(t, key, "bert") // lowercase model name
	})

	t.Run("Multiple Keys Generation", func(t *testing.T) {
		contents := []string{
			"first content",
			"second content",
			"third content",
		}
		model := "CodeBERT"

		keys := g.GenerateKeys(contents, model)
		assert.Len(t, keys, len(contents))
		for _, key := range keys {
			assert.NotEmpty(t, key)
			assert.Contains(t, key, "emb")      // default prefix
			assert.Contains(t, key, "codebert") // lowercase model name
		}

		// Keys should be unique
		uniqueKeys := make(map[string]bool)
		for _, key := range keys {
			uniqueKeys[key] = true
		}
		assert.Len(t, uniqueKeys, len(contents))
	})

	t.Run("Custom Prefix", func(t *testing.T) {
		customPrefix := "custom"
		g := cache.NewDefaultKeyGenerator(customPrefix)

		key := g.GenerateKey("test", "BERT")
		assert.Contains(t, key, customPrefix)
	})

	t.Run("Deterministic", func(t *testing.T) {
		content := "test content"
		model := "BERT"

		key1 := g.GenerateKey(content, model)
		key2 := g.GenerateKey(content, model)
		assert.Equal(t, key1, key2)
	})
}
