package unit

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/milvus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMilvusStore(t *testing.T) {
	// Create test configuration
	cfg := milvus.Config{
		Host:           "localhost",
		Port:           19530,
		CollectionName: "test_collection",
		Dimension:      768,
		BatchSize:      1000,
		MaxRetries:     3,
		PoolSize:       5,
		Quantization: milvus.QuantizationConfig{
			NumCentroids:     256,
			MaxIterations:    100,
			ConvergenceEps:   1e-6,
			SampleSize:       10000,
			BatchSize:        1000,
			UpdateInterval:   1 * time.Hour,
			NumWorkers:       4,
			QuantizationType: milvus.StandardQuantization,
			NumSubspaces:     8,
			NumBitsPerIdx:    8,
			OptimisticProbe:  8,
		},
	}

	t.Run("Store_Initialization", func(t *testing.T) {
		ctx := context.Background()

		// Test successful initialization
		store, err := milvus.NewMilvusStore(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		defer store.Close()

		// Test health check
		err = store.Health(ctx)
		assert.NoError(t, err)

		// Test invalid configuration
		invalidCfg := cfg
		invalidCfg.Dimension = 0
		_, err = milvus.NewMilvusStore(invalidCfg)
		assert.Error(t, err)

		invalidCfg = cfg
		invalidCfg.BatchSize = 0
		_, err = milvus.NewMilvusStore(invalidCfg)
		assert.Error(t, err)
	})

	t.Run("Basic_Operations", func(t *testing.T) {
		ctx := context.Background()
		store, err := milvus.NewMilvusStore(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		defer store.Close()

		// Create test item
		vector := make([]float32, cfg.Dimension)
		for i := range vector {
			vector[i] = rand.Float32()
		}
		item := &storage.Item{
			ID:     "test_item",
			Vector: vector,
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test content"),
			},
			Metadata: map[string]interface{}{
				"test": "metadata",
			},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(time.Hour),
		}

		// Test insert
		err = store.Insert(ctx, []*storage.Item{item})
		require.NoError(t, err)

		// Test search
		results, err := store.Search(ctx, item.Vector, 1)
		require.NoError(t, err)
		require.NotEmpty(t, results)
		assert.Equal(t, item.ID, results[0].ID)
		assert.Equal(t, item.Content.Type, results[0].Content.Type)
		assert.Equal(t, item.Content.Data, results[0].Content.Data)
		assert.Equal(t, item.Metadata["test"], results[0].Metadata["test"])

		// Test delete
		err = store.DeleteFromStore(ctx, []string{item.ID})
		require.NoError(t, err)

		// Verify deletion
		results, err = store.Search(ctx, item.Vector, 1)
		require.NoError(t, err)
		assert.Empty(t, results)
	})

	t.Run("Batch_Operations", func(t *testing.T) {
		ctx := context.Background()
		store, err := milvus.NewMilvusStore(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		defer store.Close()

		// Create batch of items
		numItems := 100
		items := make([]*storage.Item, numItems)
		for i := range items {
			vector := make([]float32, cfg.Dimension)
			for j := range vector {
				vector[j] = rand.Float32()
			}
			items[i] = &storage.Item{
				ID:     fmt.Sprintf("batch_item_%d", i),
				Vector: vector,
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("batch content %d", i)),
				},
				Metadata: map[string]interface{}{
					"index": i,
				},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(time.Hour),
			}
		}

		// Test batch insert
		err = store.Insert(ctx, items)
		require.NoError(t, err)

		// Test search for each item
		for _, item := range items {
			results, err := store.Search(ctx, item.Vector, 1)
			require.NoError(t, err)
			require.NotEmpty(t, results)
			assert.Equal(t, item.ID, results[0].ID)
		}

		// Test batch delete
		ids := make([]string, len(items))
		for i, item := range items {
			ids[i] = item.ID
		}
		err = store.DeleteFromStore(ctx, ids)
		require.NoError(t, err)

		// Verify deletion
		for _, item := range items {
			results, err := store.Search(ctx, item.Vector, 1)
			require.NoError(t, err)
			assert.Empty(t, results)
		}
	})

	t.Run("Vector_Quantization", func(t *testing.T) {
		ctx := context.Background()
		store, err := milvus.NewMilvusStore(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		defer store.Close()

		// Create test vectors with known similarities
		numVectors := 1000
		baseVector := make([]float32, cfg.Dimension)
		for i := range baseVector {
			baseVector[i] = rand.Float32()
		}

		items := make([]*storage.Item, numVectors)
		for i := range items {
			vector := make([]float32, cfg.Dimension)
			copy(vector, baseVector)

			// Add controlled noise
			noise := float32(i) / float32(numVectors)
			for j := range vector {
				vector[j] += rand.Float32() * noise
			}

			items[i] = &storage.Item{
				ID:     fmt.Sprintf("quant_item_%d", i),
				Vector: vector,
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("quantization content %d", i)),
				},
				Metadata: map[string]interface{}{
					"noise_level": noise,
				},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(time.Hour),
			}
		}

		// Insert vectors
		err = store.Insert(ctx, items)
		require.NoError(t, err)

		// Search with base vector
		results, err := store.Search(ctx, baseVector, 10)
		require.NoError(t, err)
		assert.Len(t, results, 10)

		// Verify results are ordered by similarity (noise level)
		for i := 1; i < len(results); i++ {
			prevNoise := results[i-1].Metadata["noise_level"].(float32)
			currNoise := results[i].Metadata["noise_level"].(float32)
			assert.True(t, prevNoise <= currNoise, "Results should be ordered by similarity")
		}

		// Clean up
		ids := make([]string, len(items))
		for i, item := range items {
			ids[i] = item.ID
		}
		err = store.DeleteFromStore(ctx, ids)
		require.NoError(t, err)
	})

	t.Run("Concurrent_Access", func(t *testing.T) {
		ctx := context.Background()
		store, err := milvus.NewMilvusStore(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		defer store.Close()

		// Create test vectors
		numVectors := 100
		items := make([]*storage.Item, numVectors)
		for i := range items {
			vector := make([]float32, cfg.Dimension)
			for j := range vector {
				vector[j] = rand.Float32()
			}
			items[i] = &storage.Item{
				ID:     fmt.Sprintf("concurrent_item_%d", i),
				Vector: vector,
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("concurrent content %d", i)),
				},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(time.Hour),
			}
		}

		// Insert vectors
		err = store.Insert(ctx, items)
		require.NoError(t, err)

		// Run concurrent operations
		numGoroutines := 10
		var wg sync.WaitGroup
		wg.Add(numGoroutines)
		errChan := make(chan error, numGoroutines)

		for i := 0; i < numGoroutines; i++ {
			go func(routineID int) {
				defer wg.Done()

				// Mix of operations
				for j := 0; j < 10; j++ {
					switch j % 3 {
					case 0:
						// Search operation
						queryVector := make([]float32, cfg.Dimension)
						for k := range queryVector {
							queryVector[k] = rand.Float32()
						}
						results, err := store.Search(ctx, queryVector, 5)
						if err != nil {
							errChan <- fmt.Errorf("search error in routine %d: %w", routineID, err)
							return
						}
						if len(results) == 0 {
							errChan <- fmt.Errorf("no results in routine %d", routineID)
							return
						}
					case 1:
						// Insert operation
						vector := make([]float32, cfg.Dimension)
						for k := range vector {
							vector[k] = rand.Float32()
						}
						item := &storage.Item{
							ID:     fmt.Sprintf("concurrent_new_item_%d_%d", routineID, j),
							Vector: vector,
							Content: storage.Content{
								Type: storage.ContentTypeText,
								Data: []byte(fmt.Sprintf("concurrent new content %d_%d", routineID, j)),
							},
							CreatedAt: time.Now(),
							ExpiresAt: time.Now().Add(time.Hour),
						}
						if err := store.Insert(ctx, []*storage.Item{item}); err != nil {
							errChan <- fmt.Errorf("insert error in routine %d: %w", routineID, err)
							return
						}
					case 2:
						// Delete operation
						id := fmt.Sprintf("concurrent_item_%d", rand.Intn(numVectors))
						if err := store.DeleteFromStore(ctx, []string{id}); err != nil {
							errChan <- fmt.Errorf("delete error in routine %d: %w", routineID, err)
							return
						}
					}
					time.Sleep(time.Millisecond * time.Duration(rand.Intn(10)))
				}
			}(i)
		}

		// Wait for all goroutines to complete
		wg.Wait()
		close(errChan)

		// Check for errors
		for err := range errChan {
			t.Error(err)
		}

		// Clean up
		ids := make([]string, len(items))
		for i, item := range items {
			ids[i] = item.ID
		}
		err = store.DeleteFromStore(ctx, ids)
		require.NoError(t, err)
	})
}
