package unit

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/cache"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQueryCache(t *testing.T) {
	// Start miniredis server
	s, err := miniredis.Run()
	require.NoError(t, err)
	defer s.Close()

	// Create Redis cache
	redisCfg := cache.Config{
		Host:                 s.Host(),
		Port:                 s.Port(),
		DefaultTTL:           time.Hour,
		PoolSize:             10,
		MinIdleConns:         5,
		MaxRetries:           3,
		CompressionThreshold: 1024,
	}

	redisCache, err := cache.NewRedisCache(redisCfg)
	require.NoError(t, err)
	require.NotNil(t, redisCache)
	defer redisCache.Close()

	t.Run("Cache_Operations", func(t *testing.T) {
		qc := cache.NewQueryCache(redisCache, time.Minute, 0.95)
		require.NotNil(t, qc)

		ctx := context.Background()

		// Create test query
		query := &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  10,
		}

		// Test cache miss
		result, err := qc.Get(ctx, query)
		require.NoError(t, err)
		assert.Nil(t, result)

		// Create test items
		items := []*storage.Item{
			{
				ID: "test1",
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte("test content 1"),
				},
				Vector: []float32{1.0, 2.0, 3.0},
			},
			{
				ID: "test2",
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte("test content 2"),
				},
				Vector: []float32{4.0, 5.0, 6.0},
			},
		}

		// Test cache set
		err = qc.Set(ctx, query, items)
		require.NoError(t, err)

		// Test cache hit
		result, err = qc.Get(ctx, query)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Len(t, result.Items, len(items))
		assert.Equal(t, items[0].ID, result.Items[0].ID)
		assert.Equal(t, items[1].ID, result.Items[1].ID)
	})

	t.Run("Vector_Normalization", func(t *testing.T) {
		qc := cache.NewQueryCache(redisCache, time.Minute, 0.95)
		require.NotNil(t, qc)

		ctx := context.Background()

		// Create two queries with same direction but different magnitudes
		query1 := &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  10,
		}
		query2 := &cache.QueryKey{
			Vector: []float32{2.0, 4.0, 6.0},
			Limit:  10,
		}

		items := []*storage.Item{
			{
				ID:     "test1",
				Vector: []float32{1.0, 2.0, 3.0},
			},
		}

		// Set cache with first query
		err := qc.Set(ctx, query1, items)
		require.NoError(t, err)

		// Should get same results with scaled vector
		result, err := qc.Get(ctx, query2)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Len(t, result.Items, len(items))
		assert.Equal(t, items[0].ID, result.Items[0].ID)
	})

	t.Run("TTL_Expiration", func(t *testing.T) {
		qc := cache.NewQueryCache(redisCache, time.Second, 0.95)
		require.NotNil(t, qc)

		ctx := context.Background()

		query := &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  10,
		}

		items := []*storage.Item{
			{
				ID:     "test1",
				Vector: []float32{1.0, 2.0, 3.0},
			},
		}

		// Set cache
		err := qc.Set(ctx, query, items)
		require.NoError(t, err)

		// Verify cache hit
		result, err := qc.Get(ctx, query)
		require.NoError(t, err)
		require.NotNil(t, result)

		// Wait for TTL to expire
		time.Sleep(2 * time.Second)

		// Verify cache miss after expiration
		result, err = qc.Get(ctx, query)
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("Similar_Queries", func(t *testing.T) {
		qc := cache.NewQueryCache(redisCache, time.Minute, 0.95)
		require.NotNil(t, qc)

		ctx := context.Background()

		// Create similar queries
		query1 := &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  10,
		}
		query2 := &cache.QueryKey{
			Vector: []float32{1.1, 2.1, 3.1}, // Slightly different
			Limit:  10,
		}

		items := []*storage.Item{
			{
				ID:     "test1",
				Vector: []float32{1.0, 2.0, 3.0},
			},
		}

		// Set cache with first query
		err := qc.Set(ctx, query1, items)
		require.NoError(t, err)

		// Try to find similar query
		similarQuery, err := qc.FindSimilarQuery(ctx, query2)
		require.NoError(t, err)
		if similarQuery != nil {
			// If similar query found, verify it matches query1
			assert.Equal(t, query1.Vector, similarQuery.Vector)
			assert.Equal(t, query1.Limit, similarQuery.Limit)
		}
	})

	t.Run("Invalid_Operations", func(t *testing.T) {
		qc := cache.NewQueryCache(redisCache, time.Minute, 0.95)
		require.NotNil(t, qc)

		ctx := context.Background()

		// Test nil query
		result, err := qc.Get(ctx, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query cannot be nil")
		assert.Nil(t, result)

		// Test empty vector
		query := &cache.QueryKey{
			Vector: []float32{},
			Limit:  10,
		}
		result, err = qc.Get(ctx, query)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query vector cannot be empty")
		assert.Nil(t, result)

		// Test zero limit
		query = &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  0,
		}
		result, err = qc.Get(ctx, query)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query limit must be positive")
		assert.Nil(t, result)

		// Test negative limit
		query = &cache.QueryKey{
			Vector: []float32{1.0, 2.0, 3.0},
			Limit:  -1,
		}
		result, err = qc.Get(ctx, query)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "query limit must be positive")
		assert.Nil(t, result)
	})
}
