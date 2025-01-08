package unit

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockConnection struct {
	id        int
	connected bool
	failures  int32
	inUse     bool
	lastUsed  time.Time
}

func newMockConnection(id int) *mockConnection {
	return &mockConnection{
		id:        id,
		connected: false,
		failures:  0,
		inUse:     false,
		lastUsed:  time.Now(),
	}
}

func (c *mockConnection) Connect() error {
	if atomic.LoadInt32(&c.failures) > 3 {
		return fmt.Errorf("connection failed")
	}
	c.connected = true
	c.lastUsed = time.Now()
	return nil
}

func (c *mockConnection) Close() error {
	c.connected = false
	c.inUse = false
	return nil
}

func (c *mockConnection) IsHealthy() bool {
	return c.connected && atomic.LoadInt32(&c.failures) <= 3
}

func (c *mockConnection) MarkFailure() {
	atomic.AddInt32(&c.failures, 1)
}

func TestConnectionPool(t *testing.T) {
	t.Run("PoolInitialization", func(t *testing.T) {
		// Create a pool of mock connections
		poolSize := 5
		connections := make([]*mockConnection, poolSize)
		for i := range connections {
			connections[i] = newMockConnection(i)
			err := connections[i].Connect()
			require.NoError(t, err)
		}

		// Verify all connections are healthy
		for _, conn := range connections {
			assert.True(t, conn.IsHealthy())
		}

		// Cleanup
		for _, conn := range connections {
			err := conn.Close()
			require.NoError(t, err)
		}
	})

	t.Run("ConcurrentConnections", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		var wg sync.WaitGroup
		numRoutines := 10
		iterations := 100
		errors := make(chan error, numRoutines*iterations)

		// Start concurrent operations
		for i := 0; i < numRoutines; i++ {
			wg.Add(1)
			go func(routineID int) {
				defer wg.Done()

				for j := 0; j < iterations; j++ {
					// Create and insert test item
					item := &storage.Item{
						ID: fmt.Sprintf("test_%d_%d", routineID, j),
						Content: storage.Content{
							Type: storage.ContentTypeText,
							Data: []byte(fmt.Sprintf("test_data_%d_%d", routineID, j)),
						},
						Vector: make([]float32, 128),
					}
					for k := range item.Vector {
						item.Vector[k] = float32(routineID*10000 + j*100 + k)
					}

					// Insert item
					if err := store.Insert(context.Background(), []*storage.Item{item}); err != nil {
						errors <- fmt.Errorf("failed to insert item: %w", err)
						continue
					}

					// Search for item
					if _, err := store.Search(context.Background(), item.Vector, 10); err != nil {
						errors <- fmt.Errorf("failed to search: %w", err)
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			assert.NoError(t, err)
		}
	})

	t.Run("ConnectionRecovery", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create test item
		item := &storage.Item{
			ID: "test_item",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test_data"),
			},
			Vector: make([]float32, 128),
		}
		for i := range item.Vector {
			item.Vector[i] = float32(i)
		}

		// Insert item
		err = store.Insert(context.Background(), []*storage.Item{item})
		require.NoError(t, err)

		// Simulate connection failures and recovery
		for i := 0; i < 5; i++ {
			// Force connection failure
			err = store.Health(context.Background())
			if err != nil {
				// Wait for recovery
				time.Sleep(time.Second)
				continue
			}

			// Try operation after recovery
			_, err := store.Search(context.Background(), item.Vector, 10)
			require.NoError(t, err)
			break
		}
	})

	t.Run("ConnectionReuse", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create test items
		numItems := 100
		items := make([]*storage.Item, numItems)
		for i := range items {
			items[i] = &storage.Item{
				ID: fmt.Sprintf("test_%d", i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", i)),
				},
				Vector: make([]float32, 128),
			}
			for j := range items[i].Vector {
				items[i].Vector[j] = float32(i*1000 + j)
			}
		}

		// Insert items
		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Perform multiple searches
		var wg sync.WaitGroup
		errors := make(chan error, numItems)

		for i := 0; i < numItems; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()
				_, err := store.Search(context.Background(), items[idx].Vector, 10)
				if err != nil {
					errors <- fmt.Errorf("search failed: %w", err)
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			assert.NoError(t, err)
		}
	})

	t.Run("ConnectionTimeout", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create test item
		item := &storage.Item{
			ID: "test_timeout",
			Content: storage.Content{
				Type: storage.ContentTypeText,
				Data: []byte("test_data"),
			},
			Vector: make([]float32, 128),
		}
		for i := range item.Vector {
			item.Vector[i] = float32(i)
		}

		// Test with short context timeout
		ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
		defer cancel()

		// Operation should fail due to timeout
		_, err = store.Search(ctx, item.Vector, 10)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "context deadline exceeded")

		// Operation should succeed with normal timeout
		ctx, cancel = context.WithTimeout(context.Background(), time.Second)
		defer cancel()

		_, err = store.Search(ctx, item.Vector, 10)
		assert.NoError(t, err)
	})
}
