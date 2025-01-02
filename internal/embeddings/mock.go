package embeddings

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// MockService implements the Service interface for testing
type MockService struct {
	mu           sync.RWMutex
	config       ModelConfig
	failureRate  float64
	latency      time.Duration
	embeddings   map[string][]float32
	healthStatus bool
	callCount    map[string]int
	errorTypes   map[string]error
}

// NewMockService creates a new mock embedding service
func NewMockService(cfg ModelConfig) *MockService {
	return &MockService{
		config:       cfg,
		embeddings:   make(map[string][]float32),
		healthStatus: true,
		callCount:    make(map[string]int),
		errorTypes:   make(map[string]error),
	}
}

// SetLatency sets artificial latency for operations
func (m *MockService) SetLatency(d time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.latency = d
}

// SetFailureRate sets the percentage of operations that should fail
func (m *MockService) SetFailureRate(rate float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failureRate = rate
}

// SetHealthStatus sets the mock health status
func (m *MockService) SetHealthStatus(healthy bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.healthStatus = healthy
}

// SetError sets a specific error for an operation
func (m *MockService) SetError(operation string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.errorTypes[operation] = err
}

// GetCallCount returns the number of calls for an operation
func (m *MockService) GetCallCount(operation string) int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.callCount[operation]
}

// simulateOperation adds latency and tracks calls
func (m *MockService) simulateOperation(operation string) error {
	m.mu.Lock()
	m.callCount[operation]++
	m.mu.Unlock()

	if m.latency > 0 {
		time.Sleep(m.latency)
	}

	if err, ok := m.errorTypes[operation]; ok {
		return err
	}

	if m.failureRate > 0 {
		if rand.Float64() < m.failureRate {
			return fmt.Errorf("simulated failure for %s", operation)
		}
	}

	return nil
}

// Generate implements the Service interface
func (m *MockService) Generate(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResult, error) {
	if err := m.simulateOperation("generate"); err != nil {
		return nil, err
	}

	// Generate deterministic mock embedding based on content
	vector := make([]float32, m.config.Dimension)
	for i := range vector {
		// Use content length and character values to generate deterministic vectors
		sum := 0
		for _, c := range req.Content {
			sum += int(c)
		}
		vector[i] = float32(sum) / float32(i+1)
	}

	return &EmbeddingResult{
		Vector:    vector,
		Model:     req.Model,
		Dimension: m.config.Dimension,
		Duration:  m.latency,
		Metadata:  req.Metadata,
	}, nil
}

// BatchGenerate implements the Service interface
func (m *MockService) BatchGenerate(ctx context.Context, reqs []*EmbeddingRequest) ([]*EmbeddingResult, error) {
	if err := m.simulateOperation("batch_generate"); err != nil {
		return nil, err
	}

	results := make([]*EmbeddingResult, len(reqs))
	for i, req := range reqs {
		result, err := m.Generate(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// GetModelInfo implements the Service interface
func (m *MockService) GetModelInfo(modelType ModelType) (*ModelConfig, error) {
	if err := m.simulateOperation("get_model_info"); err != nil {
		return nil, err
	}
	return &m.config, nil
}

// Health implements the Service interface
func (m *MockService) Health(ctx context.Context) error {
	if err := m.simulateOperation("health"); err != nil {
		return err
	}

	m.mu.RLock()
	healthy := m.healthStatus
	m.mu.RUnlock()

	if !healthy {
		return fmt.Errorf("mock service unhealthy")
	}
	return nil
}

// Close implements the Service interface
func (m *MockService) Close() error {
	if err := m.simulateOperation("close"); err != nil {
		return err
	}
	return nil
}
