package integration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage/cache"
	"github.com/objones25/athena/internal/storage/manager"
	"github.com/objones25/athena/internal/storage/milvus"
)

// TestTimeout is the default timeout for integration tests
const TestTimeout = 5 * time.Minute

// setupTestStore creates a new storage manager for testing
func setupTestStore(t *testing.T) (*manager.Manager, error) {
	config := manager.Config{
		Cache: struct {
			Config         cache.Config
			WarmupInterval time.Duration
			WarmupQueries  []string
		}{
			Config: cache.Config{
				Host:                 "localhost",
				Port:                 "6379",
				DefaultTTL:           24 * time.Hour,
				PoolSize:             10,
				MinIdleConns:         2,
				MaxRetries:           3,
				CompressionThreshold: 1024,
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
				CollectionName: fmt.Sprintf("test_collection_%d", time.Now().UnixNano()),
				Dimension:      1536,
				BatchSize:      1000,
				MaxRetries:     3,
				PoolSize:       10,
				Quantization: milvus.QuantizationConfig{
					NumCentroids:     256,
					MaxIterations:    50,
					BatchSize:        1000,
					NumWorkers:       4,
					UpdateInterval:   time.Minute,
					ConvergenceEps:   0.01,
					SampleSize:       10000,
					QuantizationType: 1, // Product quantization
					NumSubspaces:     64,
					NumBitsPerIdx:    8,
					OptimisticProbe:  16,
				},
				Graph: milvus.GraphConfig{
					Dimension:      1536,
					MaxNeighbors:   32,
					MaxSearchDepth: 64,
					BatchSize:      1000,
					NumWorkers:     4,
					UpdateInterval: time.Minute,
					MinSimilarity:  0.5,
					PruneThreshold: 30, // Using integer percentage (30%)
				},
			},
		},
		MaxRetries:     3,
		RetryInterval:  time.Second,
		BreakDuration:  5 * time.Second,
		HealthInterval: 30 * time.Second,
	}

	store, err := manager.NewManager(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create test store: %w", err)
	}

	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Logf("failed to close test store: %v", err)
		}
	})

	// Clear any existing data
	err = store.DeleteFromStore(context.Background(), []string{"*"})
	if err != nil {
		return nil, fmt.Errorf("failed to clear test store: %w", err)
	}

	return store, nil
}

// getTestDataPath returns the path to the test data directory
func getTestDataPath() string {
	path := os.Getenv("TEST_DATA_PATH")
	if path == "" {
		// Fallback to default path
		_, currentFile, _, ok := runtime.Caller(0)
		if !ok {
			panic("Failed to get current file path")
		}
		path = filepath.Join(filepath.Dir(filepath.Dir(filepath.Dir(currentFile))), "test", "integration", "testdata")
	}
	return path
}
