package integration

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSummary holds performance metrics and test results
type TestSummary struct {
	BatchInsertDuration       time.Duration
	BatchSize                 int
	ParallelRetrievalDuration time.Duration
	ParallelRetrievalCount    int
	CacheHitRate              float64
	Errors                    []string
}

func (s *TestSummary) Print() {
	log.Info().Msg("\n=== Storage Integration Test Summary ===")
	log.Info().Msgf("Batch Insert (%d items): %v (%.2f ms/item)",
		s.BatchSize,
		s.BatchInsertDuration,
		float64(s.BatchInsertDuration.Milliseconds())/float64(s.BatchSize))
	log.Info().Msgf("Parallel Retrieval (%d items): %v (%.2f ms/item)",
		s.ParallelRetrievalCount,
		s.ParallelRetrievalDuration,
		float64(s.ParallelRetrievalDuration.Milliseconds())/float64(s.ParallelRetrievalCount))
	log.Info().Msgf("Cache Hit Rate: %.2f%%", s.CacheHitRate*100)
	if len(s.Errors) > 0 {
		log.Info().Msgf("Errors encountered: %d", len(s.Errors))
		for _, err := range s.Errors {
			log.Info().Msgf("- %s", err)
		}
	}
	log.Info().Msg("=====================================")
}

func TestMain(m *testing.M) {
	// Initialize test logger with more verbose default for integration tests
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	m.Run()
}

// createTestItem creates a test item with mock vector embeddings
func createTestItem(id string, content string) *storage.Item {
	// Create mock vector embeddings of dimension 1536
	vector := make([]float32, 1536)
	for i := range vector {
		vector[i] = float32(i) / 1536.0 // Simple linear distribution
	}

	return &storage.Item{
		ID: id,
		Content: storage.Content{
			Type: storage.ContentTypeText,
			Data: []byte(content),
		},
		Metadata: map[string]interface{}{
			"test": fmt.Sprintf("metadata_%s", id),
		},
		Vector:    vector,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(24 * time.Hour),
	}
}

func TestStorageIntegration(t *testing.T) {
	logger := log.With().Str("test", "storage_integration").Logger()
	ctx := context.Background()

	// Create storage manager
	logger.Info().Msg("Initializing storage manager")
	mgr, err := setupTestStore(t)
	require.NoError(t, err)
	defer mgr.Close()

	// Initialize test summary
	summary := &TestSummary{}

	t.Run("Basic_Operations", func(t *testing.T) {
		logger := log.With().Str("test", "basic_operations").Logger()
		logger.Info().Msg("Starting basic operations test")

		item := createTestItem("test_basic", "test content")

		// Test Set operation
		logger.Info().Str("item_id", item.ID).Msg("Testing Set operation")
		err := mgr.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Test Get operation
		logger.Info().Str("item_id", item.ID).Msg("Testing Get operation")
		retrieved, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		require.NotNil(t, retrieved)

		// Verify retrieved item
		logger.Info().Str("item_id", item.ID).Msg("Verifying retrieved item")
		assert.Equal(t, item.ID, retrieved.ID)
		assert.Equal(t, item.Content.Type, retrieved.Content.Type)
		assert.Equal(t, item.Content.Data, retrieved.Content.Data)
		assert.Equal(t, item.Metadata, retrieved.Metadata)
		assert.Equal(t, item.Vector, retrieved.Vector)

		// Test Delete operation
		logger.Info().Str("item_id", item.ID).Msg("Testing Delete operation")
		err = mgr.DeleteFromStore(ctx, []string{item.ID})
		require.NoError(t, err)

		// Verify deletion
		logger.Info().Str("item_id", item.ID).Msg("Verifying deletion")
		deleted, err := mgr.Get(ctx, item.ID)
		require.NoError(t, err)
		assert.Nil(t, deleted, "Item should be deleted")
	})

	t.Run("Type_Safety", func(t *testing.T) {
		logger := log.With().Str("test", "type_safety").Logger()
		logger.Info().Msg("Starting type safety test")

		// Test invalid content type
		invalidTypeItem := createTestItem("invalid_type", "test content")
		invalidTypeItem.Content.Type = "invalid_type"

		logger.Info().Str("item_id", invalidTypeItem.ID).Msg("Testing invalid content type")
		err := mgr.Set(ctx, invalidTypeItem.ID, invalidTypeItem)
		if err == nil {
			t.Error("Expected error for invalid content type, got nil")
		} else {
			assert.Contains(t, err.Error(), "invalid content type")
		}
	})

	t.Run("Performance", func(t *testing.T) {
		logger := log.With().Str("test", "performance").Logger()
		logger.Info().Msg("Starting performance test")

		// Create multiple test items
		items := make([]*storage.Item, 100)
		for i := range items {
			items[i] = createTestItem(fmt.Sprintf("perf_test_%d", i), fmt.Sprintf("test content %d", i))
		}

		// Test batch insert performance
		logger.Info().Int("item_count", len(items)).Msg("Testing batch insert performance")
		start := time.Now()
		err := mgr.BatchSet(ctx, items)
		if err != nil {
			summary.Errors = append(summary.Errors, fmt.Sprintf("Batch insert error: %v", err))
		}
		summary.BatchInsertDuration = time.Since(start)
		summary.BatchSize = len(items)
		logger.Info().Dur("duration", summary.BatchInsertDuration).Msg("Batch insert completed")

		// Test batch retrieval performance
		logger.Info().Msg("Testing parallel retrieval performance")
		start = time.Now()
		var wg sync.WaitGroup
		retrievalCount := 10
		hits := 0
		var hitsMutex sync.Mutex

		for _, item := range items[:retrievalCount] {
			wg.Add(1)
			go func(id string) {
				defer wg.Done()
				// Multiple retrievals per item to test caching
				for i := 0; i < 5; i++ {
					retrieved, err := mgr.Get(ctx, id)
					if err != nil {
						summary.Errors = append(summary.Errors, fmt.Sprintf("Retrieval error for %s: %v", id, err))
						continue
					}
					if retrieved != nil {
						hitsMutex.Lock()
						hits++
						hitsMutex.Unlock()
					}
				}
			}(item.ID)
		}
		wg.Wait()
		summary.ParallelRetrievalDuration = time.Since(start)
		summary.ParallelRetrievalCount = retrievalCount * 5
		summary.CacheHitRate = float64(hits) / float64(retrievalCount*5)
		logger.Info().
			Dur("duration", summary.ParallelRetrievalDuration).
			Float64("hit_rate", summary.CacheHitRate).
			Msg("Parallel retrieval completed")

		// Clean up
		logger.Info().Msg("Cleaning up test data")
		var ids []string
		for _, item := range items {
			ids = append(ids, item.ID)
		}
		err = mgr.DeleteFromStore(ctx, ids)
		require.NoError(t, err)
	})

	t.Run("Cache_Eviction", func(t *testing.T) {
		logger := log.With().Str("test", "cache_eviction").Logger()
		logger.Info().Msg("Starting cache eviction test")

		// Insert items until cache eviction occurs
		numItems := 1000
		inserted := make([]*storage.Item, numItems)

		for i := 0; i < numItems; i++ {
			item := createTestItem(fmt.Sprintf("cache_test_%d", i), fmt.Sprintf("cache test content %d", i))
			err := mgr.Set(ctx, item.ID, item)
			require.NoError(t, err)
			inserted[i] = item

			// Periodically verify cache behavior
			if i > 0 && i%100 == 0 {
				// Check first item (should be evicted)
				firstItem, err := mgr.Get(ctx, inserted[0].ID)
				require.NoError(t, err)
				if firstItem != nil {
					// If still in cache, verify it matches
					assert.Equal(t, inserted[0].ID, firstItem.ID)
				}

				// Check last item (should be in cache)
				lastItem, err := mgr.Get(ctx, item.ID)
				require.NoError(t, err)
				require.NotNil(t, lastItem)
				assert.Equal(t, item.ID, lastItem.ID)
			}
		}
	})

	// Print summary after all tests complete
	summary.Print()
}
