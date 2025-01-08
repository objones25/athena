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

func TestAdaptiveIndex(t *testing.T) {
	t.Run("IndexSelection", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Test with different dataset sizes
		sizes := []int{100, 1000, 10000}
		for _, size := range sizes {
			t.Run(fmt.Sprintf("Size_%d", size), func(t *testing.T) {
				// Create test items
				items := make([]*storage.Item, size)
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
						items[i].Vector[j] = rand.Float32()
					}
				}

				// Insert items
				err := store.Insert(context.Background(), items)
				require.NoError(t, err)

				// Let the system stabilize
				time.Sleep(time.Second)

				// Perform searches to gather stats
				for i := 0; i < 20; i++ {
					_, err := store.Search(context.Background(), items[rand.Intn(len(items))].Vector, 10)
					require.NoError(t, err)
				}

				// Get a sample item to check its metadata
				result, err := store.Search(context.Background(), items[0].Vector, 1)
				require.NoError(t, err)
				require.NotEmpty(t, result)

				indexType, ok := result[0].Metadata["index_type"].(string)
				require.True(t, ok, "index_type should be present in metadata")

				// Verify index selection is appropriate for dataset size
				switch {
				case size <= 100:
					assert.Equal(t, "hnsw", indexType, "Small datasets should use HNSW")
				case size <= 1000:
					assert.Contains(t, []string{"hnsw", "lsh"}, indexType, "Medium datasets should use HNSW or LSH")
				default:
					assert.Contains(t, []string{"lsh", "quantization"}, indexType, "Large datasets should use LSH or quantization")
				}
			})
		}
	})

	t.Run("AdaptivePerformance", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create initial dataset
		size := 1000
		items := make([]*storage.Item, size)
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
				items[i].Vector[j] = rand.Float32()
			}
		}

		// Insert items
		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Measure performance with different workloads
		testCases := []struct {
			name         string
			searchCount  int
			concurrency  int
			expectedType string
			searchVector []float32
			searchRadius float32
			expectedTime time.Duration
			searchLimit  int
		}{
			{
				name:         "HighPrecision",
				searchCount:  100,
				concurrency:  1,
				searchLimit:  10,
				searchRadius: 0.1,
			},
			{
				name:         "HighThroughput",
				searchCount:  1000,
				concurrency:  runtime.NumCPU(),
				searchLimit:  10,
				searchRadius: 0.5,
			},
			{
				name:         "LargeResults",
				searchCount:  100,
				concurrency:  1,
				searchLimit:  100,
				searchRadius: 0.8,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// Create search vector
				searchVector := make([]float32, 128)
				for i := range searchVector {
					searchVector[i] = rand.Float32()
				}

				// Perform concurrent searches
				var wg sync.WaitGroup
				errors := make(chan error, tc.searchCount)
				start := time.Now()

				for i := 0; i < tc.searchCount; i++ {
					if tc.concurrency > 1 {
						wg.Add(1)
						go func() {
							defer wg.Done()
							_, err := store.Search(context.Background(), searchVector, tc.searchLimit)
							if err != nil {
								errors <- err
							}
						}()
					} else {
						_, err := store.Search(context.Background(), searchVector, tc.searchLimit)
						require.NoError(t, err)
					}
				}

				if tc.concurrency > 1 {
					wg.Wait()
					close(errors)
					for err := range errors {
						assert.NoError(t, err)
					}
				}

				duration := time.Since(start)
				avgLatency := duration / time.Duration(tc.searchCount)

				// Get current index type
				result, err := store.Search(context.Background(), searchVector, 1)
				require.NoError(t, err)
				require.NotEmpty(t, result)

				indexType, ok := result[0].Metadata["index_type"].(string)
				require.True(t, ok, "index_type should be present in metadata")

				t.Logf("Workload: %s, Index: %s, Avg Latency: %v", tc.name, indexType, avgLatency)

				// Verify performance is reasonable
				assert.Less(t, avgLatency, time.Second, "Search latency should be reasonable")
			})

			// Let the system stabilize between test cases
			time.Sleep(time.Second)
		}
	})

	t.Run("IndexTransition", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Start with small dataset
		initialSize := 100
		items := make([]*storage.Item, initialSize)
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
				items[i].Vector[j] = rand.Float32()
			}
		}

		// Insert initial items
		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Let the system stabilize
		time.Sleep(time.Second)

		// Get initial index type
		result, err := store.Search(context.Background(), items[0].Vector, 1)
		require.NoError(t, err)
		require.NotEmpty(t, result)

		initialType, ok := result[0].Metadata["index_type"].(string)
		require.True(t, ok, "index_type should be present in metadata")
		assert.Equal(t, "hnsw", initialType, "Should start with HNSW for small dataset")

		// Add more items to trigger transition
		moreItems := make([]*storage.Item, 10000)
		for i := range moreItems {
			moreItems[i] = &storage.Item{
				ID: fmt.Sprintf("test_%d", initialSize+i),
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", initialSize+i)),
				},
				Vector: make([]float32, 128),
			}
			for j := range moreItems[i].Vector {
				moreItems[i].Vector[j] = rand.Float32()
			}
		}

		// Insert more items
		err = store.Insert(context.Background(), moreItems)
		require.NoError(t, err)

		// Let the system stabilize
		time.Sleep(time.Second)

		// Perform searches to trigger adaptation
		for i := 0; i < 100; i++ {
			_, err := store.Search(context.Background(), moreItems[rand.Intn(len(moreItems))].Vector, 10)
			require.NoError(t, err)
		}

		// Get final index type
		result, err = store.Search(context.Background(), moreItems[0].Vector, 1)
		require.NoError(t, err)
		require.NotEmpty(t, result)

		finalType, ok := result[0].Metadata["index_type"].(string)
		require.True(t, ok, "index_type should be present in metadata")
		assert.NotEqual(t, initialType, finalType, "Index type should change for larger dataset")
		assert.Contains(t, []string{"lsh", "quantization"}, finalType, "Should transition to LSH or quantization for large dataset")
	})

	t.Run("MemoryPressure", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create large dataset
		size := 10000
		items := make([]*storage.Item, size)
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
				items[i].Vector[j] = rand.Float32()
			}
		}

		// Insert items
		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Simulate memory pressure by allocating large arrays
		pressure := make([][]byte, 0)
		defer func() {
			// Clear pressure
			pressure = nil
			runtime.GC()
		}()

		// Add memory pressure gradually
		for i := 0; i < 5; i++ {
			pressure = append(pressure, make([]byte, 100*1024*1024)) // 100MB each
			time.Sleep(time.Second)

			// Perform searches under pressure
			result, err := store.Search(context.Background(), items[0].Vector, 10)
			require.NoError(t, err)
			require.NotEmpty(t, result)

			// Verify index adapts to memory pressure
			indexType, ok := result[0].Metadata["index_type"].(string)
			require.True(t, ok, "index_type should be present in metadata")
			assert.Contains(t, []string{"lsh", "quantization"}, indexType,
				"Under memory pressure, should prefer memory-efficient indexes")
		}
	})

	t.Run("MixedWorkload", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Create dataset
		size := 5000
		items := make([]*storage.Item, size)
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
				items[i].Vector[j] = rand.Float32()
			}
		}

		err = store.Insert(context.Background(), items)
		require.NoError(t, err)

		// Run mixed workload patterns
		patterns := []struct {
			name       string
			operations int
			concurrent bool
			searchSize int
			sleepTime  time.Duration
		}{
			{"BurstTraffic", 100, true, 10, time.Millisecond},
			{"SteadyTraffic", 50, false, 20, time.Millisecond * 100},
			{"LargeResults", 20, false, 100, time.Millisecond * 50},
		}

		for _, pattern := range patterns {
			t.Run(pattern.name, func(t *testing.T) {
				var wg sync.WaitGroup
				start := time.Now()

				for i := 0; i < pattern.operations; i++ {
					if pattern.concurrent {
						wg.Add(1)
						go func(i int) {
							defer wg.Done()
							result, err := store.Search(context.Background(), items[i].Vector, pattern.searchSize)
							require.NoError(t, err)
							require.NotEmpty(t, result)
						}(i)
					} else {
						result, err := store.Search(context.Background(), items[i].Vector, pattern.searchSize)
						require.NoError(t, err)
						require.NotEmpty(t, result)
						time.Sleep(pattern.sleepTime)
					}
				}

				if pattern.concurrent {
					wg.Wait()
				}

				duration := time.Since(start)
				t.Logf("%s completed in %v", pattern.name, duration)

				// Verify index adaptation
				result, err := store.Search(context.Background(), items[0].Vector, 1)
				require.NoError(t, err)
				require.NotEmpty(t, result)

				indexType, ok := result[0].Metadata["index_type"].(string)
				require.True(t, ok, "index_type should be present in metadata")
				t.Logf("%s final index type: %s", pattern.name, indexType)
			})

			// Let system stabilize between patterns
			time.Sleep(time.Second * 2)
		}
	})
}
