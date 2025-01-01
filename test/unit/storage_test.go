package unit

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/mock"
	"github.com/objones25/athena/test/testutil"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMain(m *testing.M) {
	// Initialize test logger
	testutil.InitTestLogger()
	// Set default test log level
	testutil.SetLogLevel(testutil.ParseLogLevel(zerolog.WarnLevel))
	m.Run()
}

func TestStorageOperations(t *testing.T) {
	store := mock.NewMockStore()
	ctx := context.Background()

	t.Run("Basic Operations", func(t *testing.T) {
		// Set debug level for this specific test
		cleanup := testutil.TestLogLevel(t, zerolog.DebugLevel)
		defer cleanup()

		// Test item creation
		item := &storage.Item{
			ID: "test1",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test content"),
			},
			Vector: []float32{1.0, 2.0, 3.0},
			Metadata: map[string]interface{}{
				"test": "metadata",
			},
			CreatedAt: time.Now(),
			ExpiresAt: time.Now().Add(24 * time.Hour),
		}

		// Test Set
		err := store.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Test Get
		retrieved, err := store.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.Equal(t, item.ID, retrieved.ID)
		assert.Equal(t, item.Content.Type, retrieved.Content.Type)
		assert.Equal(t, item.Content.Data, retrieved.Content.Data)
		assert.Equal(t, item.Vector, retrieved.Vector)
		assert.Equal(t, item.Metadata["test"], retrieved.Metadata["test"])

		// Test Delete
		err = store.DeleteFromCache(ctx, item.ID)
		require.NoError(t, err)

		// Verify deletion
		retrieved, err = store.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.Nil(t, retrieved)
	})

	t.Run("Batch Operations", func(t *testing.T) {
		items := make(map[string]*storage.Item)
		for i := 0; i < 10; i++ {
			item := &storage.Item{
				ID: fmt.Sprintf("batch%d", i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("content %d", i)),
				},
				Vector:    []float32{float32(i), float32(i + 1)},
				CreatedAt: time.Now(),
				ExpiresAt: time.Now().Add(24 * time.Hour),
			}
			items[item.ID] = item
		}

		// Test BatchSet
		err := store.BatchSet(ctx, items)
		require.NoError(t, err)

		// Test BatchGet
		keys := make([]string, 0, len(items))
		for k := range items {
			keys = append(keys, k)
		}
		retrieved, err := store.BatchGet(ctx, keys)
		require.NoError(t, err)
		assert.Equal(t, len(items), len(retrieved))

		// Verify each item
		for id, item := range items {
			retrieved, ok := retrieved[id]
			assert.True(t, ok)
			assert.Equal(t, item.ID, retrieved.ID)
			assert.Equal(t, item.Content.Data, retrieved.Content.Data)
		}
	})

	t.Run("Vector Operations", func(t *testing.T) {
		vectors := []*storage.Item{
			{
				ID: "vec1",
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte("vector 1"),
				},
				Vector: []float32{1.0, 0.0, 0.0},
			},
			{
				ID: "vec2",
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte("vector 2"),
				},
				Vector: []float32{0.0, 1.0, 0.0},
			},
		}

		// Test Insert
		err := store.Insert(ctx, vectors)
		require.NoError(t, err)

		// Test Search
		results, err := store.Search(ctx, []float32{1.0, 0.0, 0.0}, 2)
		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(results), 1)

		// Test Update
		vectors[0].Content.Data = []byte("updated vector 1")
		err = store.Update(ctx, vectors[:1])
		require.NoError(t, err)

		// Verify update
		results, err = store.Search(ctx, []float32{1.0, 0.0, 0.0}, 1)
		require.NoError(t, err)
		assert.Equal(t, "updated vector 1", string(results[0].Content.Data))

		// Test Delete
		err = store.DeleteFromStore(ctx, []string{"vec1"})
		require.NoError(t, err)

		// Verify deletion
		results, err = store.Search(ctx, []float32{1.0, 0.0, 0.0}, 1)
		require.NoError(t, err)
		for _, result := range results {
			assert.NotEqual(t, "vec1", result.ID)
		}
	})

	t.Run("Error Handling", func(t *testing.T) {
		store.SetFailRate(0.5) // 50% failure rate
		store.SetLatency(10 * time.Millisecond)

		item := &storage.Item{ID: "error_test"}

		// Test multiple operations to ensure some fail
		for i := 0; i < 10; i++ {
			err := store.Set(ctx, item.ID, item)
			if err != nil {
				assert.Contains(t, err.Error(), "simulated failure")
			}
		}

		// Reset failure rate
		store.SetFailRate(0)
	})

	t.Run("Content Types", func(t *testing.T) {
		contentTypes := []storage.ContentType{
			storage.ContentTypeText,
			storage.ContentTypeCode,
			storage.ContentTypeMath,
			storage.ContentTypeJSON,
			storage.ContentTypeMarkdown,
		}

		for _, ct := range contentTypes {
			item := &storage.Item{
				ID: fmt.Sprintf("content_%s", ct),
				Content: storage.Content{
					Type: ct,
					Data: []byte("test content"),
				},
			}

			err := store.Set(ctx, item.ID, item)
			require.NoError(t, err)

			retrieved, err := store.Get(ctx, item.ID)
			require.NoError(t, err)
			assert.Equal(t, ct, retrieved.Content.Type)
		}
	})

	t.Run("Concurrent Operations", func(t *testing.T) {
		const goroutines = 10
		const operationsPerGoroutine = 100

		var wg sync.WaitGroup
		errors := make(chan error, goroutines*operationsPerGoroutine)

		for i := 0; i < goroutines; i++ {
			wg.Add(1)
			go func(routineID int) {
				defer wg.Done()
				for j := 0; j < operationsPerGoroutine; j++ {
					item := &storage.Item{
						ID: fmt.Sprintf("concurrent_%d_%d", routineID, j),
						Content: storage.Content{
							Type: storage.ContentTypeText,
							Data: []byte("concurrent test"),
						},
					}

					if err := store.Set(ctx, item.ID, item); err != nil {
						errors <- err
						continue
					}

					if _, err := store.Get(ctx, item.ID); err != nil {
						errors <- err
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		var errCount int
		for err := range errors {
			t.Logf("Concurrent operation error: %v", err)
			errCount++
		}

		assert.Zero(t, errCount, "Expected no errors in concurrent operations")
	})

	t.Run("Performance", func(t *testing.T) {
		if testing.Short() {
			t.Skip("Skipping performance test in short mode")
		}

		store.SetLatency(0) // Reset latency for performance testing

		// Prepare test data
		const itemCount = 10000
		items := make(map[string]*storage.Item)
		for i := 0; i < itemCount; i++ {
			items[fmt.Sprintf("perf_%d", i)] = &storage.Item{
				ID: fmt.Sprintf("perf_%d", i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte("performance test"),
				},
			}
		}

		// Test batch operations performance
		start := time.Now()
		err := store.BatchSet(ctx, items)
		require.NoError(t, err)
		batchDuration := time.Since(start)

		// Test individual operations performance
		start = time.Now()
		for id, item := range items {
			err := store.Set(ctx, id, item)
			require.NoError(t, err)
		}
		individualDuration := time.Since(start)

		// Assert batch operations are significantly faster
		assert.Less(t, batchDuration, individualDuration/2,
			"Expected batch operations to be at least 2x faster than individual operations")
	})
}
