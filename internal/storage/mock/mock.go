package mock

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/objones25/athena/internal/storage"
)

// MockStore implements both Cache and VectorStore interfaces for testing
type MockStore struct {
	mu       sync.RWMutex
	items    map[string]*storage.Item
	errors   map[string]error // Simulate specific errors for testing
	latency  time.Duration    // Simulate network latency
	failRate float64          // Percentage of operations that should fail
}

func NewMockStore() *MockStore {
	return &MockStore{
		items:  make(map[string]*storage.Item),
		errors: make(map[string]error),
	}
}

// SetLatency sets artificial latency for operations
func (m *MockStore) SetLatency(d time.Duration) {
	m.latency = d
}

// SetFailRate sets the percentage of operations that should fail
func (m *MockStore) SetFailRate(rate float64) {
	m.failRate = rate
}

// SetError sets a specific error for an operation
func (m *MockStore) SetError(operation string, err error) {
	m.errors[operation] = err
}

// simulateLatencyAndFailure adds artificial latency and simulates failures
func (m *MockStore) simulateLatencyAndFailure(operation string) error {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}

	if err, ok := m.errors[operation]; ok {
		return err
	}

	if m.failRate > 0 {
		if time.Now().UnixNano()%100 < int64(m.failRate*100) {
			return fmt.Errorf("simulated failure for %s", operation)
		}
	}

	return nil
}

// Cache interface implementation

func (m *MockStore) Set(ctx context.Context, key string, item *storage.Item) error {
	if err := m.simulateLatencyAndFailure("set"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.items[key] = item
	return nil
}

func (m *MockStore) Get(ctx context.Context, key string) (*storage.Item, error) {
	if err := m.simulateLatencyAndFailure("get"); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	item, ok := m.items[key]
	if !ok {
		return nil, nil
	}
	return item, nil
}

func (m *MockStore) DeleteFromCache(ctx context.Context, key string) error {
	if err := m.simulateLatencyAndFailure("delete"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.items, key)
	return nil
}

func (m *MockStore) Clear(ctx context.Context) error {
	if err := m.simulateLatencyAndFailure("clear"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.items = make(map[string]*storage.Item)
	return nil
}

func (m *MockStore) BatchSet(ctx context.Context, items map[string]*storage.Item) error {
	if err := m.simulateLatencyAndFailure("batchset"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for k, v := range items {
		m.items[k] = v
	}
	return nil
}

func (m *MockStore) BatchGet(ctx context.Context, keys []string) (map[string]*storage.Item, error) {
	if err := m.simulateLatencyAndFailure("batchget"); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*storage.Item)
	for _, key := range keys {
		if item, ok := m.items[key]; ok {
			result[key] = item
		}
	}
	return result, nil
}

// VectorStore interface implementation

func (m *MockStore) Insert(ctx context.Context, items []*storage.Item) error {
	if err := m.simulateLatencyAndFailure("insert"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, item := range items {
		m.items[item.ID] = item
	}
	return nil
}

func (m *MockStore) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	if err := m.simulateLatencyAndFailure("search"); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// For testing, return items sorted by vector similarity if vector is provided
	result := make([]*storage.Item, 0, limit)
	if vector != nil {
		// Create a slice of items for sorting
		items := make([]*storage.Item, 0, len(m.items))
		for _, item := range m.items {
			items = append(items, item)
		}

		// Sort by vector similarity (simple dot product for testing)
		sort.Slice(items, func(i, j int) bool {
			return dotProduct(items[i].Vector, vector) > dotProduct(items[j].Vector, vector)
		})

		// Take top k results
		for i := 0; i < len(items) && i < limit; i++ {
			result = append(result, items[i])
		}
	} else {
		// If no vector provided, just return first limit items
		for _, item := range m.items {
			if len(result) >= limit {
				break
			}
			result = append(result, item)
		}
	}
	return result, nil
}

// dotProduct calculates the dot product of two vectors
func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func (m *MockStore) DeleteFromStore(ctx context.Context, ids []string) error {
	if err := m.simulateLatencyAndFailure("deletefromstore"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, id := range ids {
		delete(m.items, id)
	}
	return nil
}

func (m *MockStore) Update(ctx context.Context, items []*storage.Item) error {
	if err := m.simulateLatencyAndFailure("update"); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, item := range items {
		m.items[item.ID] = item
	}
	return nil
}

// Helper methods for testing

func (m *MockStore) GetItemCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.items)
}

func (m *MockStore) GetAllItems() map[string]*storage.Item {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*storage.Item)
	for k, v := range m.items {
		result[k] = v
	}
	return result
}
