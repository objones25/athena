package integration

import (
	"context"
	"fmt"
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
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Initialize components
	redisConfig := cache.Config{
		Host:                 "localhost",
		Port:                 "6379",
		DefaultTTL:           time.Hour,
		PoolSize:             10,
		MinIdleConns:         2,
		MaxRetries:           3,
		CompressionThreshold: 1024,
	}

	milvusConfig := milvus.Config{
		Host:           "localhost",
		Port:           19530,
		CollectionName: "test_collection",
		Dimension:      128,
		BatchSize:      100,
		MaxRetries:     3,
		PoolSize:       5,
	}

	managerConfig := manager.Config{
		Cache: struct {
			Config         cache.Config
			WarmupInterval time.Duration
			WarmupQueries  []string
		}{
			Config:         redisConfig,
			WarmupInterval: time.Minute,
			WarmupQueries:  []string{"test_query"},
		},
		VectorStore: struct {
			Config milvus.Config
		}{
			Config: milvusConfig,
		},
		MaxRetries:     3,
		RetryInterval:  time.Second,
		BreakDuration:  time.Minute,
		HealthInterval: time.Minute,
	}

	mgr, err := manager.NewManager(managerConfig)
	require.NoError(t, err)
	defer mgr.Close()

	ctx := context.Background()

	t.Run("End-to-End Flow", func(t *testing.T) {
		// Create test items
		items := make([]*storage.Item, 10)
		for i := 0; i < 10; i++ {
			items[i] = &storage.Item{
				ID: fmt.Sprintf("test_item_%d", i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test content %d", i)),
					Format: map[string]string{
						"encoding": "utf-8",
					},
				},
				Vector:    make([]float32, 128), // Fill with test data
				Metadata:  map[string]interface{}{"test": i},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(time.Hour),
			}
			// Fill vector with some test data
			for j := range items[i].Vector {
				items[i].Vector[j] = float32(i) * 0.1
			}
		}

		// Test insertion
		err = mgr.BatchSet(ctx, items)
		require.NoError(t, err)

		// Test retrieval
		for _, item := range items {
			retrieved, err := mgr.Get(ctx, item.ID)
			require.NoError(t, err)
			assert.NotNil(t, retrieved)
			assert.Equal(t, item.ID, retrieved.ID)
			assert.Equal(t, item.Content.Type, retrieved.Content.Type)
			assert.Equal(t, item.Content.Data, retrieved.Content.Data)
			assert.Equal(t, item.Vector, retrieved.Vector)
			assert.Equal(t, item.Metadata["test"], retrieved.Metadata["test"])
		}

		// Test vector search
		results, err := mgr.Search(ctx, items[0].Vector, 5)
		require.NoError(t, err)
		assert.NotEmpty(t, results)
		assert.LessOrEqual(t, len(results), 5)

		// Test deletion in batch
		idsToDelete := make([]string, 5)
		for i := 0; i < 5; i++ {
			idsToDelete[i] = items[i].ID
		}
		err = mgr.DeleteFromStore(ctx, idsToDelete)
		require.NoError(t, err)

		// Verify deletions
		for i := 0; i < 5; i++ {
			retrieved, err := mgr.Get(ctx, items[i].ID)
			require.NoError(t, err)
			assert.Nil(t, retrieved)
		}

		// Verify remaining items
		for i := 5; i < 10; i++ {
			retrieved, err := mgr.Get(ctx, items[i].ID)
			require.NoError(t, err)
			assert.NotNil(t, retrieved)
		}
	})

	t.Run("Consistency Check", func(t *testing.T) {
		// Create test item
		item := &storage.Item{
			ID: "consistency_test",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("consistency test"),
			},
			Vector:    make([]float32, 128),
			Metadata:  map[string]interface{}{"test": "consistency"},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(time.Hour),
		}

		// Store item
		err := mgr.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Verify in both stores
		cached, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.NotNil(t, cached)

		results, err := mgr.Search(ctx, item.Vector, 1)
		require.NoError(t, err)
		assert.NotEmpty(t, results)

		// Verify consistency
		assert.Equal(t, cached.ID, results[0].ID)
		assert.Equal(t, cached.Content.Data, results[0].Content.Data)
		assert.Equal(t, cached.Vector, results[0].Vector)
	})

	t.Run("Error Recovery", func(t *testing.T) {
		// Create test item
		item := &storage.Item{
			ID: "error_recovery_test",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("error recovery test"),
			},
			Vector:    make([]float32, 128),
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(time.Hour),
		}

		// Test health check
		err := mgr.Health(ctx)
		require.NoError(t, err)

		// Store item
		err = mgr.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Verify storage
		retrieved, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.NotNil(t, retrieved)
		assert.Equal(t, item.ID, retrieved.ID)
	})

	t.Run("Performance", func(t *testing.T) {
		const (
			itemCount  = 1000
			batchSize  = 100
			searchTopK = 10
		)

		// Test different scenarios
		scenarios := []struct {
			name     string
			itemFunc func() []*storage.Item
		}{
			{
				name: "Single Item Operations",
				itemFunc: func() []*storage.Item {
					items := make([]*storage.Item, 1)
					items[0] = &storage.Item{
						ID: "single_test",
						Content: storage.Content{
							Type: storage.ContentTypeText,
							Data: []byte("single item test"),
						},
						Vector:    make([]float32, 128),
						CreatedAt: time.Now(),
						ExpiresAt: time.Now().Add(time.Hour),
					}
					return items
				},
			},
			{
				name: "Small Batch Operations",
				itemFunc: func() []*storage.Item {
					items := make([]*storage.Item, 10)
					for i := range items {
						items[i] = &storage.Item{
							ID: fmt.Sprintf("small_batch_%d", i),
							Content: storage.Content{
								Type: storage.ContentTypeText,
								Data: []byte(fmt.Sprintf("small batch test %d", i)),
							},
							Vector:    make([]float32, 128),
							CreatedAt: time.Now(),
							ExpiresAt: time.Now().Add(time.Hour),
						}
					}
					return items
				},
			},
			{
				name: "Large Batch Operations",
				itemFunc: func() []*storage.Item {
					items := make([]*storage.Item, itemCount)
					for i := range items {
						items[i] = &storage.Item{
							ID: fmt.Sprintf("large_batch_%d", i),
							Content: storage.Content{
								Type: storage.ContentTypeText,
								Data: []byte(fmt.Sprintf("large batch test %d", i)),
							},
							Vector:    make([]float32, 128),
							CreatedAt: time.Now(),
							ExpiresAt: time.Now().Add(time.Hour),
						}
					}
					return items
				},
			},
		}

		for _, sc := range scenarios {
			t.Run(sc.name, func(t *testing.T) {
				items := sc.itemFunc()
				itemCount := len(items)

				// Measure batch insertion time
				start := time.Now()
				err := mgr.BatchSet(ctx, items)
				require.NoError(t, err)
				insertDuration := time.Since(start)

				// Measure batch retrieval time
				start = time.Now()
				for _, item := range items {
					_, err := mgr.Get(ctx, item.ID)
					require.NoError(t, err)
				}
				retrieveDuration := time.Since(start)

				// Measure search performance
				start = time.Now()
				results, err := mgr.Search(ctx, items[0].Vector, searchTopK)
				require.NoError(t, err)
				require.LessOrEqual(t, len(results), searchTopK)
				searchDuration := time.Since(start)

				// Measure batch deletion time
				ids := make([]string, len(items))
				for i, item := range items {
					ids[i] = item.ID
				}
				start = time.Now()
				err = mgr.DeleteFromStore(ctx, ids)
				require.NoError(t, err)
				deleteDuration := time.Since(start)

				// Log performance metrics
				t.Logf("\nPerformance Metrics for %s (%d items):", sc.name, itemCount)
				t.Logf("Batch Insert: %v total, %v/item", insertDuration, insertDuration/time.Duration(itemCount))
				t.Logf("Retrieve: %v total, %v/item", retrieveDuration, retrieveDuration/time.Duration(itemCount))
				t.Logf("Vector Search (top-%d): %v", searchTopK, searchDuration)
				t.Logf("Batch Delete: %v total, %v/item", deleteDuration, deleteDuration/time.Duration(itemCount))

				// Assert reasonable performance based on batch size
				switch {
				case itemCount == 1:
					assert.Less(t, insertDuration, 1000*time.Millisecond, "Single item insert should be reasonable")
					assert.Less(t, retrieveDuration, 100*time.Millisecond, "Single item retrieve should be reasonable")
					assert.Less(t, searchDuration, 200*time.Millisecond, "Single item search should be reasonable")
					assert.Less(t, deleteDuration, 1000*time.Millisecond, "Single item delete should be reasonable")
				case itemCount <= 10:
					assert.Less(t, insertDuration/time.Duration(itemCount), 200*time.Millisecond, "Small batch insert per item should be reasonable")
					assert.Less(t, retrieveDuration/time.Duration(itemCount), 50*time.Millisecond, "Small batch retrieve per item should be reasonable")
					assert.Less(t, searchDuration, 200*time.Millisecond, "Small batch search should be reasonable")
					assert.Less(t, deleteDuration/time.Duration(itemCount), 200*time.Millisecond, "Small batch delete per item should be reasonable")
				default:
					assert.Less(t, insertDuration/time.Duration(itemCount), 20*time.Millisecond, "Large batch insert per item should be optimized")
					assert.Less(t, retrieveDuration/time.Duration(itemCount), 10*time.Millisecond, "Large batch retrieve per item should be optimized")
					assert.Less(t, searchDuration, 200*time.Millisecond, "Large batch search should be reasonable")
					assert.Less(t, deleteDuration/time.Duration(itemCount), 20*time.Millisecond, "Large batch delete per item should be optimized")
				}
			})
		}
	})
}