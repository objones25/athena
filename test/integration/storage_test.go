package integration

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/cache"
	"github.com/objones25/athena/internal/storage/manager"
	"github.com/objones25/athena/internal/storage/milvus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStorageIntegration(t *testing.T) {
	ctx := context.Background()

	// Create storage manager
	config := manager.Config{
		Cache: struct {
			Config         cache.Config
			WarmupInterval time.Duration
			WarmupQueries  []string
		}{
			Config: cache.Config{
				Host: "localhost",
				Port: "6379",
			},
			WarmupInterval: 5 * time.Minute,
			WarmupQueries:  []string{},
		},
		VectorStore: struct {
			Config milvus.Config
		}{
			Config: milvus.Config{
				Host:           "localhost",
				Port:           19530,
				CollectionName: "test_collection",
				Dimension:      1536,
				BatchSize:      1000,
				MaxRetries:     3,
				PoolSize:       14,
			},
		},
		MaxRetries:     3,
		RetryInterval:  time.Second,
		BreakDuration:  5 * time.Second,
		HealthInterval: 30 * time.Second,
	}

	t.Logf("Initializing storage manager with config: %+v", config)
	mgr, err := manager.NewManager(config)
	require.NoError(t, err)
	defer mgr.Close()

	// Clear existing data
	t.Log("Clearing existing data...")
	err = mgr.DeleteFromStore(ctx, []string{"*"})
	require.NoError(t, err)

	t.Run("Basic_Operations", func(t *testing.T) {
		// Create test item with high-dimensional vector
		vector := make([]float32, config.VectorStore.Config.Dimension)
		for i := range vector {
			vector[i] = rand.Float32()
		}

		item := &storage.Item{
			ID: "test_high_dim",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test content"),
			},
			Vector:    vector,
			Metadata:  map[string]interface{}{"test": "metadata"},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(24 * time.Hour),
		}

		// Test Set operation
		err := mgr.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Test Get operation
		retrieved, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		require.NotNil(t, retrieved)

		// Verify retrieved item
		assert.Equal(t, item.ID, retrieved.ID)
		assert.Equal(t, item.Content.Type, retrieved.Content.Type)
		assert.Equal(t, item.Content.Data, retrieved.Content.Data)
		assert.Equal(t, len(item.Vector), len(retrieved.Vector))
		assert.Equal(t, item.Metadata, retrieved.Metadata)

		// Test Search operation
		results, err := mgr.Search(ctx, vector, 1)
		require.NoError(t, err)
		require.Len(t, results, 1)
		assert.Equal(t, item.ID, results[0].ID)

		// Test Delete operation
		err = mgr.DeleteFromStore(ctx, []string{item.ID})
		require.NoError(t, err)

		// Verify deletion
		deleted, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.Nil(t, deleted)
	})

	t.Run("Type_Safety", func(t *testing.T) {
		// Test invalid content type
		invalidTypeItem := &storage.Item{
			ID: "invalid_type",
			Content: storage.Content{
				Type: "invalid_type",
				Data: []byte("test content"),
			},
			Vector:    make([]float32, config.VectorStore.Config.Dimension),
			Metadata:  map[string]interface{}{"test": "metadata"},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(24 * time.Hour),
		}

		// Initialize vector with random values
		for i := range invalidTypeItem.Vector {
			invalidTypeItem.Vector[i] = rand.Float32()
		}

		err := mgr.Set(ctx, invalidTypeItem.ID, invalidTypeItem)
		if err == nil {
			t.Error("Expected error for invalid content type, got nil")
		} else {
			assert.Contains(t, err.Error(), "invalid content type")
		}

		// Test invalid vector dimension
		invalidDimItem := &storage.Item{
			ID: "invalid_dim",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test content"),
			},
			Vector:    make([]float32, 512), // Wrong dimension
			Metadata:  map[string]interface{}{"test": "metadata"},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(24 * time.Hour),
		}

		// Initialize vector with random values
		for i := range invalidDimItem.Vector {
			invalidDimItem.Vector[i] = rand.Float32()
		}

		err = mgr.Set(ctx, invalidDimItem.ID, invalidDimItem)
		if err == nil {
			t.Error("Expected error for invalid vector dimension, got nil")
		} else {
			assert.Contains(t, err.Error(), "invalid vector dimension")
		}
	})

	t.Run("Performance", func(t *testing.T) {
		// Create multiple test items
		items := make([]*storage.Item, 100)
		for i := range items {
			vector := make([]float32, config.VectorStore.Config.Dimension)
			for j := range vector {
				vector[j] = rand.Float32()
			}
			items[i] = &storage.Item{
				ID: fmt.Sprintf("perf_test_%d", i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test content %d", i)),
				},
				Vector:    vector,
				Metadata:  map[string]interface{}{"test": fmt.Sprintf("metadata_%d", i)},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(24 * time.Hour),
			}
		}

		// Test batch insert performance
		start := time.Now()
		err := mgr.BatchSet(ctx, items)
		require.NoError(t, err)
		insertDuration := time.Since(start)
		t.Logf("Batch insert of %d items took %v", len(items), insertDuration)

		// Test search performance
		start = time.Now()
		results, err := mgr.Search(ctx, items[0].Vector, 10)
		require.NoError(t, err)
		searchDuration := time.Since(start)
		t.Logf("Search operation took %v", searchDuration)
		assert.NotEmpty(t, results)

		// Test batch retrieval performance
		start = time.Now()
		var wg sync.WaitGroup
		for _, item := range items[:10] {
			wg.Add(1)
			go func(id string) {
				defer wg.Done()
				retrieved, err := mgr.Get(ctx, id)
				require.NoError(t, err)
				require.NotNil(t, retrieved)
			}(item.ID)
		}
		wg.Wait()
		retrievalDuration := time.Since(start)
		t.Logf("Parallel retrieval of 10 items took %v", retrievalDuration)

		// Clean up
		var ids []string
		for _, item := range items {
			ids = append(ids, item.ID)
		}
		err = mgr.DeleteFromStore(ctx, ids)
		require.NoError(t, err)
	})
}
