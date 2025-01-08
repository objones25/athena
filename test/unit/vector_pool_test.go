package unit

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVectorPool(t *testing.T) {
	t.Run("BufferAllocation", func(t *testing.T) {
		dimension := 128
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Test different buffer sizes
		sizes := []int{64, 128, 256, 512, 1024}
		for _, size := range sizes {
			// Create vectors
			vectors := make([][]float32, size)
			for i := range vectors {
				vectors[i] = make([]float32, dimension)
				for j := range vectors[i] {
					vectors[i][j] = float32(i*dimension + j)
				}
			}

			// Insert vectors
			items := make([]*storage.Item, size)
			for i := range items {
				items[i] = &storage.Item{
					ID:     fmt.Sprintf("test_%d_%d", size, i),
					Vector: vectors[i],
					Content: storage.Content{
						Type: storage.ContentTypeText,
						Data: []byte(fmt.Sprintf("test_data_%d_%d", size, i)),
					},
				}
			}

			err := store.Insert(context.Background(), items)
			require.NoError(t, err)

			// Search with the first vector
			results, err := store.Search(context.Background(), vectors[0], 10)
			require.NoError(t, err)
			assert.NotEmpty(t, results)
		}
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Use a smaller number of operations but maintain concurrency
		numWorkers := runtime.NumCPU() * 2
		opsPerWorker := 1000
		vectorSizes := []int{64, 128, 256}

		var wg sync.WaitGroup
		errors := make(chan error, numWorkers*opsPerWorker)

		// Create worker pool
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()

				// Each worker alternates between different vector sizes
				for i := 0; i < opsPerWorker; i++ {
					size := vectorSizes[i%len(vectorSizes)]
					vector := make([]float32, size)
					for j := range vector {
						vector[j] = rand.Float32()
					}

					// Mix of operations
					switch i % 4 {
					case 0: // Search
						results, err := store.Search(context.Background(), vector, 5)
						if err != nil {
							errors <- fmt.Errorf("search error: %v", err)
						}
						if len(results) > 0 {
							// Use results to prevent optimization
							_ = results[0].ID
						}
					case 1: // Insert
						item := &storage.Item{
							ID:     fmt.Sprintf("test_%d_%d", workerID, i),
							Vector: vector,
							Content: storage.Content{
								Type: storage.ContentTypeText,
								Data: []byte("test"),
							},
						}
						if err := store.Insert(context.Background(), []*storage.Item{item}); err != nil {
							errors <- fmt.Errorf("insert error: %v", err)
						}
					case 2: // Delete
						if err := store.DeleteFromStore(context.Background(), []string{fmt.Sprintf("test_%d_%d", workerID, i-1)}); err != nil {
							errors <- fmt.Errorf("delete error: %v", err)
						}
					case 3: // Update
						item := &storage.Item{
							ID:     fmt.Sprintf("test_%d_%d", workerID, i),
							Vector: vector,
							Content: storage.Content{
								Type: storage.ContentTypeText,
								Data: []byte("updated"),
							},
						}
						if err := store.Update(context.Background(), []*storage.Item{item}); err != nil {
							errors <- fmt.Errorf("update error: %v", err)
						}
					}
				}
			}(w)
		}

		// Use a timeout to prevent hanging
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			close(errors)
			for err := range errors {
				assert.NoError(t, err)
			}
		case <-time.After(30 * time.Second):
			t.Fatal("Test timed out")
		}
	})

	t.Run("SearchPerformance", func(t *testing.T) {
		dimension := 128
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Insert a large number of vectors
		size := 1000
		vectors := make([][]float32, size)
		items := make([]*storage.Item, size)

		for i := range vectors {
			vectors[i] = make([]float32, dimension)
			for j := range vectors[i] {
				vectors[i][j] = float32(i*dimension + j)
			}
			items[i] = &storage.Item{
				ID:     fmt.Sprintf("test_%d", i),
				Vector: vectors[i],
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", i)),
				},
			}
		}

		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Measure search times
		var searchTimes []time.Duration
		for i := 0; i < 100; i++ {
			start := time.Now()
			results, err := store.Search(context.Background(), vectors[i], 10)
			require.NoError(t, err)
			assert.NotEmpty(t, results)
			searchTimes = append(searchTimes, time.Since(start))
		}

		// Calculate average search time
		var totalTime time.Duration
		for _, duration := range searchTimes {
			totalTime += duration
		}
		avgTime := totalTime / time.Duration(len(searchTimes))

		// Search time should be reasonable
		assert.Less(t, avgTime, time.Millisecond*100, "Average search time should be reasonable")
	})

	t.Run("MemoryUsage", func(t *testing.T) {
		dimension := 128

		// Force multiple GC cycles to get a clean baseline
		for i := 0; i < 5; i++ {
			runtime.GC()
			time.Sleep(time.Millisecond * 100)
		}

		// Record initial memory stats
		var m1 runtime.MemStats
		runtime.ReadMemStats(&m1)
		t.Logf("Initial memory state:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m1.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m1.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m1.HeapObjects)

		// Create store
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer func() {
			err := store.Close()
			require.NoError(t, err, "Failed to close store")
		}()

		// Insert and search with a large number of vectors
		size := 1000
		vectors := make([][]float32, size)
		items := make([]*storage.Item, size)

		// Track memory after allocation
		var m1a runtime.MemStats
		runtime.ReadMemStats(&m1a)
		t.Logf("After initial allocation:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m1a.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m1a.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m1a.HeapObjects)

		for i := range vectors {
			vectors[i] = make([]float32, dimension)
			for j := range vectors[i] {
				vectors[i][j] = float32(i*dimension + j)
			}
			items[i] = &storage.Item{
				ID:     fmt.Sprintf("test_%d", i),
				Vector: vectors[i],
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", i)),
				},
				Metadata: make(map[string]interface{}),
			}
		}

		// Track memory after data preparation
		var m1b runtime.MemStats
		runtime.ReadMemStats(&m1b)
		t.Logf("After data preparation:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m1b.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m1b.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m1b.HeapObjects)

		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Track memory after insertion
		var m1c runtime.MemStats
		runtime.ReadMemStats(&m1c)
		t.Logf("After insertion:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m1c.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m1c.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m1c.HeapObjects)

		// Perform searches
		for i := 0; i < 100; i++ {
			results, err := store.Search(context.Background(), vectors[i], 10)
			require.NoError(t, err)
			assert.NotEmpty(t, results)
		}

		// Track memory after searches
		var m1d runtime.MemStats
		runtime.ReadMemStats(&m1d)
		t.Logf("After searches:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m1d.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m1d.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m1d.HeapObjects)

		// Clear references to help GC
		vectors = nil
		items = nil
		runtime.GC()
		time.Sleep(time.Millisecond * 100)

		// Force multiple GC cycles for cleanup
		for i := 0; i < 5; i++ {
			runtime.GC()
			time.Sleep(time.Millisecond * 100)
		}

		// Record final memory stats
		var m2 runtime.MemStats
		runtime.ReadMemStats(&m2)
		t.Logf("Final memory state:")
		t.Logf("  - HeapAlloc: %.2f MB", float64(m2.HeapAlloc)/(1024*1024))
		t.Logf("  - HeapInuse: %.2f MB", float64(m2.HeapInuse)/(1024*1024))
		t.Logf("  - HeapObjects: %d", m2.HeapObjects)

		// Calculate memory increases for different phases
		prepIncrease := m1b.HeapAlloc - m1a.HeapAlloc
		insertIncrease := m1c.HeapAlloc - m1b.HeapAlloc
		searchIncrease := m1d.HeapAlloc - m1c.HeapAlloc
		totalIncrease := m2.HeapAlloc - m1.HeapAlloc

		t.Logf("\nMemory increases by phase:")
		t.Logf("  - Data preparation: %.2f MB", float64(prepIncrease)/(1024*1024))
		t.Logf("  - Insertion: %.2f MB", float64(insertIncrease)/(1024*1024))
		t.Logf("  - Search operations: %.2f MB", float64(searchIncrease)/(1024*1024))
		t.Logf("  - Total increase: %.2f MB", float64(totalIncrease)/(1024*1024))

		// Calculate expected memory components
		vectorMemory := float64(size*dimension*4) / (1024 * 1024) // 4 bytes per float32
		metadataMemory := float64(size*144) / (1024 * 1024)       // 144 bytes per metadata
		cacheMemory := 0.25                                       // 250KB for cache overhead
		poolMemory := 0.5                                         // 500KB for pre-allocated buffers
		runtimeOverhead := 0.25                                   // 250KB for runtime overhead

		totalExpectedMemory := vectorMemory + metadataMemory + cacheMemory + poolMemory + runtimeOverhead

		t.Logf("\nExpected memory components:")
		t.Logf("  - Vector memory: %.2f MB", vectorMemory)
		t.Logf("  - Metadata memory: %.2f MB", metadataMemory)
		t.Logf("  - Cache memory: %.2f MB", cacheMemory)
		t.Logf("  - Pool memory: %.2f MB", poolMemory)
		t.Logf("  - Runtime overhead: %.2f MB", runtimeOverhead)
		t.Logf("  - Total expected: %.2f MB", totalExpectedMemory)

		// Convert everything to bytes for comparison
		memoryIncreaseFloat := float64(totalIncrease) / (1024 * 1024)
		expectedMemoryBytes := totalExpectedMemory

		// Memory should be at least half the expected and no more than 4x
		assert.GreaterOrEqual(t, memoryIncreaseFloat, expectedMemoryBytes/2,
			"Memory usage (%.2f MB) should be at least half the expected memory (%.2f MB)",
			memoryIncreaseFloat, expectedMemoryBytes)
		assert.Less(t, memoryIncreaseFloat, expectedMemoryBytes*4,
			"Memory usage (%.2f MB) should be less than 4x the expected memory (%.2f MB)",
			memoryIncreaseFloat, expectedMemoryBytes)
	})
}
