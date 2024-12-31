package monitor

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/prometheus/client_golang/prometheus"
)

// ConsistencyVerifier handles data consistency checks between cache and vector store
type ConsistencyVerifier struct {
	cache       storage.Cache
	vectorStore storage.VectorStore
	mu          sync.RWMutex
	config      VerifierConfig
}

type VerifierConfig struct {
	// How often to run full consistency checks
	CheckInterval time.Duration
	// Maximum number of items to check in one batch
	BatchSize int
	// Whether to automatically repair inconsistencies
	AutoRepair bool
	// Maximum time to wait for a consistency check
	Timeout time.Duration
}

type VerificationResult struct {
	Checked    int
	Mismatches int
	Repaired   int
	Errors     []error
	Duration   time.Duration
	StartTime  time.Time
	EndTime    time.Time
}

func NewConsistencyVerifier(cache storage.Cache, vectorStore storage.VectorStore, cfg VerifierConfig) *ConsistencyVerifier {
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 1000
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 5 * time.Minute
	}
	if cfg.CheckInterval == 0 {
		cfg.CheckInterval = 1 * time.Hour
	}

	return &ConsistencyVerifier{
		cache:       cache,
		vectorStore: vectorStore,
		config:      cfg,
	}
}

// StartPeriodicChecks begins periodic consistency verification
func (cv *ConsistencyVerifier) StartPeriodicChecks(ctx context.Context) {
	ticker := time.NewTicker(cv.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			timer := prometheus.NewTimer(ConsistencyCheckLatency)
			result, err := cv.VerifyConsistency(ctx)
			timer.ObserveDuration()

			if err != nil {
				ConsistencyErrors.WithLabelValues("check_failed").Inc()
				continue
			}

			if result.Mismatches > 0 {
				ConsistencyErrors.WithLabelValues("mismatch").Add(float64(result.Mismatches))
			}
		}
	}
}

// VerifyConsistency checks data consistency between cache and vector store
func (cv *ConsistencyVerifier) VerifyConsistency(ctx context.Context) (*VerificationResult, error) {
	ctx, cancel := context.WithTimeout(ctx, cv.config.Timeout)
	defer cancel()

	result := &VerificationResult{
		StartTime: time.Now(),
	}

	// Get all items from vector store
	vectors, err := cv.vectorStore.Search(ctx, nil, cv.config.BatchSize) // nil vector means get all
	if err != nil {
		return result, fmt.Errorf("failed to get vectors: %w", err)
	}

	// Check each vector against cache
	var wg sync.WaitGroup
	errChan := make(chan error, len(vectors))
	mismatchChan := make(chan string, len(vectors))
	repairedChan := make(chan string, len(vectors))

	for _, item := range vectors {
		wg.Add(1)
		go func(item *storage.Item) {
			defer wg.Done()

			// Check cache
			cached, err := cv.cache.Get(ctx, item.ID)
			if err != nil {
				errChan <- fmt.Errorf("failed to get item %s from cache: %w", item.ID, err)
				return
			}

			if !isConsistent(item, cached) {
				mismatchChan <- item.ID

				if cv.config.AutoRepair {
					if err := cv.repair(ctx, item); err != nil {
						errChan <- fmt.Errorf("failed to repair item %s: %w", item.ID, err)
					} else {
						repairedChan <- item.ID
					}
				}
			}
		}(item)
	}

	// Wait for all checks to complete
	wg.Wait()
	close(errChan)
	close(mismatchChan)
	close(repairedChan)

	// Collect results
	for err := range errChan {
		result.Errors = append(result.Errors, err)
	}

	result.Checked = len(vectors)
	result.Mismatches = len(mismatchChan)
	result.Repaired = len(repairedChan)
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)

	return result, nil
}

// repair fixes inconsistencies by updating the cache with vector store data
func (cv *ConsistencyVerifier) repair(ctx context.Context, item *storage.Item) error {
	return cv.cache.Set(ctx, item.ID, item)
}

// isConsistent checks if two items are consistent
func isConsistent(a, b *storage.Item) bool {
	if a == nil || b == nil {
		return a == b
	}

	// Check basic fields
	if a.ID != b.ID {
		return false
	}

	// Check content
	if a.Content.Type != b.Content.Type {
		return false
	}
	if !bytesEqual(a.Content.Data, b.Content.Data) {
		return false
	}

	// Check vector
	if !vectorsEqual(a.Vector, b.Vector) {
		return false
	}

	// Check metadata (shallow comparison is sufficient for our needs)
	if len(a.Metadata) != len(b.Metadata) {
		return false
	}
	for k, v := range a.Metadata {
		if bv, ok := b.Metadata[k]; !ok || bv != v {
			return false
		}
	}

	return true
}

// bytesEqual compares two byte slices
func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// vectorsEqual compares two float32 slices
func vectorsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
