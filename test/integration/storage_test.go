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
	"github.com/objones25/athena/test/testutil"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSummary holds performance metrics and test results
type TestSummary struct {
	BatchInsertDuration       time.Duration
	BatchSize                 int
	SearchDuration            time.Duration
	SearchResultCount         int
	ParallelRetrievalDuration time.Duration
	ParallelRetrievalCount    int
	Errors                    []string
}

func (s *TestSummary) Print() {
	log.Info().Msg("\n=== Storage Integration Test Summary ===")
	log.Info().Msgf("Batch Insert (%d items): %v (%.2f ms/item)",
		s.BatchSize,
		s.BatchInsertDuration,
		float64(s.BatchInsertDuration.Milliseconds())/float64(s.BatchSize))
	log.Info().Msgf("Vector Search (%d results): %v",
		s.SearchResultCount,
		s.SearchDuration)
	log.Info().Msgf("Parallel Retrieval (%d items): %v (%.2f ms/item)",
		s.ParallelRetrievalCount,
		s.ParallelRetrievalDuration,
		float64(s.ParallelRetrievalDuration.Milliseconds())/float64(s.ParallelRetrievalCount))
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
	testutil.InitTestLogger()
	// Set default test log level - using INFO for integration tests
	testutil.SetLogLevel(testutil.ParseLogLevel(zerolog.InfoLevel))
	m.Run()
}

func TestStorageIntegration(t *testing.T) {
	logger := log.With().Str("test", "storage_integration").Logger()
	ctx := context.Background()

	// Create storage manager
	logger.Info().Msg("Initializing storage manager configuration")
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

	logger.Info().Interface("config", config).Msg("Creating storage manager")
	mgr, err := manager.NewManager(config)
	require.NoError(t, err)
	defer mgr.Close()

	// Clear existing data
	logger.Info().Msg("Clearing existing test data")
	err = mgr.DeleteFromStore(ctx, []string{"*"})
	require.NoError(t, err)

	// Initialize test summary
	summary := &TestSummary{}

	t.Run("Basic_Operations", func(t *testing.T) {
		logger := log.With().Str("test", "basic_operations").Logger()
		logger.Info().Msg("Starting basic operations test")

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
		logger.Info().Str("item_id", item.ID).Msg("Testing Set operation")
		err := mgr.Set(ctx, item.ID, item)
		require.NoError(t, err)

		// Add a small delay to ensure the item is indexed
		logger.Info().Msg("Waiting for item to be indexed")
		time.Sleep(500 * time.Millisecond)

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
		assert.Equal(t, len(item.Vector), len(retrieved.Vector))
		assert.Equal(t, item.Metadata, retrieved.Metadata)

		// Test Search operation
		logger.Info().Msg("Testing Search operation")
		results, err := mgr.Search(ctx, vector, 1)
		require.NoError(t, err)
		require.Len(t, results, 1)
		assert.Equal(t, item.ID, results[0].ID)

		// Test Delete operation
		logger.Info().Str("item_id", item.ID).Msg("Testing Delete operation")
		err = mgr.DeleteFromStore(ctx, []string{item.ID})
		require.NoError(t, err)

		// Add a small delay to ensure deletion propagates
		time.Sleep(100 * time.Millisecond)

		// Verify deletion with retries
		logger.Info().Str("item_id", item.ID).Msg("Verifying deletion")
		maxRetries := 3
		var deleted *storage.Item
		for i := 0; i < maxRetries; i++ {
			deleted, err = mgr.Get(ctx, item.ID)
			require.NoError(t, err)
			if deleted == nil {
				break
			}
			logger.Warn().
				Str("item_id", item.ID).
				Int("retry", i+1).
				Msg("Item still exists after deletion, retrying verification")
			time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
		}
		assert.Nil(t, deleted, "Item should be deleted")

		// Verify search returns no results
		logger.Info().Msg("Verifying search after deletion")
		results, err = mgr.Search(ctx, vector, 1)
		require.NoError(t, err)
		assert.Empty(t, results, "Search should return no results after deletion")
	})

	t.Run("Type_Safety", func(t *testing.T) {
		logger := log.With().Str("test", "type_safety").Logger()
		logger.Info().Msg("Starting type safety test")

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

		logger.Info().Str("item_id", invalidTypeItem.ID).Msg("Testing invalid content type")
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

		logger.Info().Str("item_id", invalidDimItem.ID).Msg("Testing invalid vector dimension")
		err = mgr.Set(ctx, invalidDimItem.ID, invalidDimItem)
		if err == nil {
			t.Error("Expected error for invalid vector dimension, got nil")
		} else {
			assert.Contains(t, err.Error(), "invalid vector dimension")
		}
	})

	t.Run("Performance", func(t *testing.T) {
		logger := log.With().Str("test", "performance").Logger()
		logger.Info().Msg("Starting performance test")

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
		logger.Info().Int("item_count", len(items)).Msg("Testing batch insert performance")
		start := time.Now()
		err := mgr.BatchSet(ctx, items)
		if err != nil {
			summary.Errors = append(summary.Errors, fmt.Sprintf("Batch insert error: %v", err))
		}
		summary.BatchInsertDuration = time.Since(start)
		summary.BatchSize = len(items)
		logger.Info().Dur("duration", summary.BatchInsertDuration).Msg("Batch insert completed")

		// Test search performance
		logger.Info().Msg("Testing search performance")
		start = time.Now()
		results, err := mgr.Search(ctx, items[0].Vector, 10)
		if err != nil {
			summary.Errors = append(summary.Errors, fmt.Sprintf("Search error: %v", err))
		}
		summary.SearchDuration = time.Since(start)
		summary.SearchResultCount = len(results)
		logger.Info().
			Dur("duration", summary.SearchDuration).
			Int("result_count", len(results)).
			Msg("Search completed")

		// Test batch retrieval performance
		logger.Info().Msg("Testing parallel retrieval performance")
		start = time.Now()
		var wg sync.WaitGroup
		retrievalCount := 10
		for _, item := range items[:retrievalCount] {
			wg.Add(1)
			go func(id string) {
				defer wg.Done()
				retrieved, err := mgr.Get(ctx, id)
				if err != nil {
					summary.Errors = append(summary.Errors, fmt.Sprintf("Retrieval error for %s: %v", id, err))
				}
				require.NotNil(t, retrieved)
			}(item.ID)
		}
		wg.Wait()
		summary.ParallelRetrievalDuration = time.Since(start)
		summary.ParallelRetrievalCount = retrievalCount
		logger.Info().Dur("duration", summary.ParallelRetrievalDuration).Msg("Parallel retrieval completed")

		// Clean up
		logger.Info().Msg("Cleaning up test data")
		var ids []string
		for _, item := range items {
			ids = append(ids, item.ID)
		}
		err = mgr.DeleteFromStore(ctx, ids)
		require.NoError(t, err)
	})

	// Print summary after all tests complete
	summary.Print()
}
