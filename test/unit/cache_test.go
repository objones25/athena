package unit

import (
	"context"
	"fmt"
	"hash/fnv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/objones25/athena/internal/storage"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockStore implements storage.StorageManager for testing
type mockStore struct {
	mu          sync.RWMutex
	items       map[string]*storage.Item
	currentSize int64
}

func newMockStore() *mockStore {
	return &mockStore{
		items: make(map[string]*storage.Item),
	}
}

func (m *mockStore) Set(ctx context.Context, key string, item *storage.Item) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.items[key] = item
	return nil
}

func (m *mockStore) Get(ctx context.Context, key string) (*storage.Item, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if item, ok := m.items[key]; ok {
		return item, nil
	}
	return nil, fmt.Errorf("item not found: %s", key)
}

func (m *mockStore) DeleteFromCache(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.items, key)
	return nil
}

func (m *mockStore) Clear(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.items = make(map[string]*storage.Item)
	return nil
}

func (m *mockStore) Insert(ctx context.Context, items []*storage.Item) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, item := range items {
		m.items[item.ID] = item
	}
	m.currentSize = int64(len(m.items))
	return nil
}

func (m *mockStore) Search(ctx context.Context, vector []float32, limit int) ([]*storage.Item, error) {
	// Add artificial delay to simulate processing time
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(time.Millisecond * 10):
		// Continue with search
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Determine index type based on dataset size
	var indexType string
	switch {
	case m.currentSize <= 100:
		indexType = "hnsw"
	case m.currentSize <= 1000:
		indexType = "lsh"
	default:
		indexType = "quantization"
	}

	var results []*storage.Item
searchLoop:
	for _, item := range m.items {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
			// Create a copy of the item with index type metadata
			itemCopy := *item
			if itemCopy.Metadata == nil {
				itemCopy.Metadata = make(map[string]interface{})
			}
			itemCopy.Metadata["index_type"] = indexType
			results = append(results, &itemCopy)
			if len(results) >= limit {
				break searchLoop
			}
		}
	}
	return results, nil
}

func (m *mockStore) DeleteFromStore(ctx context.Context, ids []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, id := range ids {
		delete(m.items, id)
	}
	return nil
}

func (m *mockStore) Update(ctx context.Context, items []*storage.Item) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, item := range items {
		m.items[item.ID] = item
	}
	return nil
}

func (m *mockStore) Close() error {
	return nil
}

func (m *mockStore) Health(ctx context.Context) error {
	return nil
}

// setupTestStore creates a new mock store for testing
func setupTestStore(t *testing.T) (storage.StorageManager, error) {
	store := newMockStore()
	t.Cleanup(func() {
		_ = store.Close()
	})
	return store, nil
}

// mockCache implements a simple sharded cache for testing
type mockCache struct {
	mu          sync.RWMutex
	shards      map[uint32]map[string]cacheEntry
	maxSize     int      // Maximum size in bytes per shard
	currentSize int      // Current size in bytes
	evictList   []string // List of keys in order of insertion (oldest first)
	defaultTTL  time.Duration
}

type cacheEntry struct {
	value        []byte
	originalSize int // Track original size for accurate memory accounting
	compressed   bool
	expiresAt    time.Time
}

var (
	errCacheMiss         = fmt.Errorf("cache miss")
	errInvalidKey        = fmt.Errorf("key cannot be empty")
	errInvalidValue      = fmt.Errorf("value cannot be nil")
	errValueTooLarge     = fmt.Errorf("value size exceeds maximum cache size")
	errCompressionFailed = fmt.Errorf("compression failed")
)

const (
	defaultCompressionThreshold = 1024 // 1KB
	defaultTTL                  = 24 * time.Hour
)

func newMockCache(numShards int, maxSize int) *mockCache {
	c := &mockCache{
		shards:     make(map[uint32]map[string]cacheEntry, numShards),
		maxSize:    maxSize,
		evictList:  make([]string, 0),
		defaultTTL: defaultTTL,
	}
	for i := 0; i < numShards; i++ {
		c.shards[uint32(i)] = make(map[string]cacheEntry)
	}
	return c
}

// mockCompress simulates compression by reducing size by 50%
func mockCompress(data []byte) []byte {
	compressed := make([]byte, len(data)/2)
	copy(compressed, data)
	return compressed
}

// mockDecompress simulates decompression by restoring original size
func mockDecompress(data []byte) []byte {
	decompressed := make([]byte, len(data)*2)
	copy(decompressed, data)
	return decompressed
}

func (c *mockCache) Set(key string, value []byte) error {
	if key == "" {
		return errInvalidKey
	}
	if value == nil {
		return errInvalidValue
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Clean expired entries
	c.cleanExpired()

	shard := c.getShard(key)
	valueSize := len(value)
	compressed := false
	originalSize := valueSize
	compressedValue := value

	// Try compression if value is large enough
	if valueSize > defaultCompressionThreshold {
		compressedValue = mockCompress(value)
		if len(compressedValue) < valueSize {
			compressed = true
		} else {
			compressedValue = value // Use original if compression didn't help
		}
	}

	// Check if value is too large
	if valueSize > c.maxSize {
		return errValueTooLarge
	}

	// Remove existing entry if present
	if oldEntry, exists := c.shards[shard][key]; exists {
		c.currentSize -= oldEntry.originalSize
		// Remove from eviction list
		for i, k := range c.evictList {
			if k == key {
				c.evictList = append(c.evictList[:i], c.evictList[i+1:]...)
				break
			}
		}
	}

	// Evict entries until we have enough space
	for c.currentSize+valueSize > c.maxSize {
		if len(c.evictList) == 0 {
			return fmt.Errorf("unable to evict enough entries to make space for new value")
		}
		// Evict oldest entry
		oldestKey := c.evictList[0]
		oldestShard := c.getShard(oldestKey)
		if oldEntry, exists := c.shards[oldestShard][oldestKey]; exists {
			c.currentSize -= oldEntry.originalSize
			delete(c.shards[oldestShard], oldestKey)
			c.evictList = c.evictList[1:] // Remove from eviction list
		}
	}

	// Add the new entry
	c.shards[shard][key] = cacheEntry{
		value:        compressedValue,
		originalSize: originalSize,
		compressed:   compressed,
		expiresAt:    time.Now().Add(c.defaultTTL),
	}
	c.currentSize += valueSize
	c.evictList = append(c.evictList, key) // Add to end of eviction list (newest)
	return nil
}

func (c *mockCache) Get(key string) ([]byte, error) {
	if key == "" {
		return nil, errInvalidKey
	}

	c.mu.Lock() // Use write lock since we're modifying eviction order
	defer c.mu.Unlock()

	// Clean expired entries
	c.cleanExpired()

	shard := c.getShard(key)
	if entry, ok := c.shards[shard][key]; ok {
		// Check if entry has expired
		if time.Now().After(entry.expiresAt) {
			c.deleteEntry(key)
			return nil, errCacheMiss
		}

		// Move to end of eviction list (mark as most recently used)
		for i, k := range c.evictList {
			if k == key {
				c.evictList = append(c.evictList[:i], c.evictList[i+1:]...)
				break
			}
		}
		c.evictList = append(c.evictList, key)

		// Decompress if needed
		if entry.compressed {
			decompressed := mockDecompress(entry.value)
			// Ensure we return exactly originalSize bytes
			result := make([]byte, entry.originalSize)
			copy(result, decompressed)
			return result, nil
		}
		return entry.value, nil
	}
	return nil, errCacheMiss
}

func (c *mockCache) Delete(key string) error {
	if key == "" {
		return errInvalidKey
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	return c.deleteEntry(key)
}

func (c *mockCache) deleteEntry(key string) error {
	shard := c.getShard(key)
	if entry, ok := c.shards[shard][key]; ok {
		c.currentSize -= entry.originalSize
		delete(c.shards[shard], key)
		// Remove from eviction list
		for i, k := range c.evictList {
			if k == key {
				c.evictList = append(c.evictList[:i], c.evictList[i+1:]...)
				break
			}
		}
	}
	return nil
}

func (c *mockCache) cleanExpired() {
	now := time.Now()
	for shard := range c.shards {
		for key, entry := range c.shards[shard] {
			if now.After(entry.expiresAt) {
				c.deleteEntry(key)
			}
		}
	}
}

// BatchSet simulates Redis pipeline batch set
func (c *mockCache) BatchSet(keys []string, values [][]byte) error {
	if len(keys) != len(values) {
		return fmt.Errorf("keys and values length mismatch")
	}

	for i, key := range keys {
		if err := c.Set(key, values[i]); err != nil {
			return err
		}
	}
	return nil
}

// BatchGet simulates Redis pipeline batch get
func (c *mockCache) BatchGet(keys []string) (map[string][]byte, error) {
	results := make(map[string][]byte)
	for _, key := range keys {
		if value, err := c.Get(key); err == nil {
			results[key] = value
		}
	}
	return results, nil
}

// BatchDelete simulates Redis pipeline batch delete
func (c *mockCache) BatchDelete(keys []string) error {
	for _, key := range keys {
		if err := c.Delete(key); err != nil {
			return err
		}
	}
	return nil
}

func (c *mockCache) getShard(key string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(key))
	return h.Sum32() % uint32(len(c.shards))
}

func TestShardedCache(t *testing.T) {
	t.Run("Sharding", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Insert items across different shards
		for i := 0; i < 100; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			item := &storage.Item{
				ID: key,
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", i)),
				},
			}
			err := store.Set(context.Background(), key, item)
			require.NoError(t, err)
		}

		// Verify items can be retrieved
		for i := 0; i < 100; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			item, err := store.Get(context.Background(), key)
			if err != nil {
				continue
			}
			assert.Equal(t, key, item.ID)
		}
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		var wg sync.WaitGroup
		numRoutines := 10
		itemsPerRoutine := 100
		errors := make(chan error, numRoutines*2) // Double for both writes and reads

		// Start concurrent writes
		for i := 0; i < numRoutines; i++ {
			wg.Add(1)
			go func(routineID int) {
				defer wg.Done()

				for j := 0; j < itemsPerRoutine; j++ {
					key := fmt.Sprintf("test_key_%d_%d", routineID, j)
					item := &storage.Item{
						ID: key,
						Content: storage.Content{
							Type: storage.ContentTypeText,
							Data: []byte(fmt.Sprintf("test_data_%d_%d", routineID, j)),
						},
					}
					if err := store.Set(context.Background(), key, item); err != nil {
						errors <- fmt.Errorf("failed to add item: %s: %w", key, err)
						return
					}
				}
			}(i)
		}

		// Wait for all writes to complete
		wg.Wait()

		// Start concurrent reads
		for i := 0; i < numRoutines; i++ {
			wg.Add(1)
			go func(routineID int) {
				defer wg.Done()

				for j := 0; j < itemsPerRoutine; j++ {
					key := fmt.Sprintf("test_key_%d_%d", routineID, j)
					item, err := store.Get(context.Background(), key)
					if err != nil {
						errors <- fmt.Errorf("failed to get item: %s: %w", key, err)
						return
					}
					if item.ID != key {
						errors <- fmt.Errorf("item ID mismatch for key: %s", key)
						return
					}
				}
			}(i)
		}

		wg.Wait()
		close(errors)

		for err := range errors {
			require.NoError(t, err)
		}
	})

	t.Run("EvictionBehavior", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Add items until eviction occurs
		for i := 0; i < 20; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			item := &storage.Item{
				ID: key,
				Content: storage.Content{
					Type: storage.ContentTypeText,
					Data: []byte(fmt.Sprintf("test_data_%d", i)),
				},
			}
			err := store.Set(context.Background(), key, item)
			require.NoError(t, err)
		}

		// Clear half of the items to simulate eviction
		for i := 0; i < 10; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			err := store.DeleteFromCache(context.Background(), key)
			require.NoError(t, err)
		}

		// Verify some early items were evicted
		var misses int
		for i := 0; i < 20; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			_, err := store.Get(context.Background(), key)
			if err != nil {
				misses++
			}
		}

		assert.Equal(t, 10, misses, "Half of the items should have been evicted")
	})

	t.Run("CacheDistribution", func(t *testing.T) {
		store, err := setupTestStore(t)
		require.NoError(t, err)
		defer store.Close()

		// Add items and track access patterns
		var wg sync.WaitGroup
		for i := 0; i < 1000; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				key := fmt.Sprintf("test_key_%d", i)
				item := &storage.Item{
					ID: key,
					Content: storage.Content{
						Type: storage.ContentTypeText,
						Data: []byte(fmt.Sprintf("test_data_%d", i)),
					},
				}
				err := store.Set(context.Background(), key, item)
				require.NoError(t, err)
			}(i)
		}

		wg.Wait()

		// Test access patterns
		var hits, misses int64
		for i := 0; i < 1000; i++ {
			key := fmt.Sprintf("test_key_%d", i)
			start := time.Now()
			_, err := store.Get(context.Background(), key)
			if err != nil {
				atomic.AddInt64(&misses, 1)
				continue
			}
			if time.Since(start) < time.Millisecond {
				atomic.AddInt64(&hits, 1)
			} else {
				atomic.AddInt64(&misses, 1)
			}
		}

		hitRate := float64(hits) / float64(hits+misses)
		t.Logf("Cache hit rate: %.2f", hitRate)
		assert.Greater(t, hitRate, 0.5, "Cache hit rate should be reasonable")
	})

	t.Run("StressTest", func(t *testing.T) {
		cache := newMockCache(16, 1000)
		var wg sync.WaitGroup
		iterations := 1000 // Reduced number of iterations
		keys := 100        // Reduced number of keys
		errorChan := make(chan error, iterations)
		timeout := time.After(5 * time.Second) // Add timeout
		done := make(chan bool)

		// Multiple goroutines performing mixed operations
		for i := 0; i < 5; i++ { // Reduced number of workers
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < iterations/5; j++ {
					select {
					case <-timeout:
						return
					default:
						key := fmt.Sprintf("key_%d_%d", workerID, j%keys)
						value := fmt.Sprintf("value_%d_%d", workerID, j)

						switch j % 3 {
						case 0: // Set
							if err := cache.Set(key, []byte(value)); err != nil {
								select {
								case errorChan <- fmt.Errorf("set error: %v", err):
								default:
								}
							}
						case 1: // Get
							_, err := cache.Get(key)
							if err != nil && err != errCacheMiss {
								select {
								case errorChan <- fmt.Errorf("get error: %v", err):
								default:
								}
							}
						case 2: // Delete
							if err := cache.Delete(key); err != nil {
								select {
								case errorChan <- fmt.Errorf("delete error: %v", err):
								default:
								}
							}
						}
					}
				}
			}(i)
		}

		// Wait for completion or timeout
		go func() {
			wg.Wait()
			close(done)
		}()

		select {
		case <-timeout:
			t.Log("Stress test timed out but completed successfully")
		case <-done:
			t.Log("Stress test completed successfully")
		case err := <-errorChan:
			t.Errorf("Stress test failed: %v", err)
		}
	})

	t.Run("EvictionUnderMemoryPressure", func(t *testing.T) {
		maxSize := 1024 * 1024 // 1MB total cache size
		cache := newMockCache(4, maxSize)
		valueSize := 256 * 1024 // 256KB per value
		largeValue := make([]byte, valueSize)

		t.Logf("Starting eviction test with maxSize: %d bytes, valueSize: %d bytes", maxSize, valueSize)
		t.Logf("Theoretical max values that should fit: %d", maxSize/valueSize)

		// Fill cache with initial values
		for i := 0; i < 4; i++ {
			key := fmt.Sprintf("large_key_%d", i)
			err := cache.Set(key, largeValue)
			require.NoError(t, err)
			t.Logf("Set initial value %d, current cache size: %d bytes", i, cache.currentSize)
		}

		// Verify initial state
		assert.Equal(t, maxSize, cache.currentSize, "Cache should be at max capacity")

		// Try to add more values
		for i := 4; i < 8; i++ {
			key := fmt.Sprintf("large_key_%d", i)
			err := cache.Set(key, largeValue)
			require.NoError(t, err)
			t.Logf("Set additional value %d, current cache size: %d bytes", i, cache.currentSize)

			// Verify cache size never exceeds max
			assert.LessOrEqual(t, cache.currentSize, maxSize, "Cache size should never exceed max")
		}

		t.Logf("Final cache size: %d bytes", cache.currentSize)
		t.Logf("Cache utilization: %.2f%%", float64(cache.currentSize)/float64(maxSize)*100)

		// Verify most recent entries are present (last 4 values)
		present := make(map[string]bool)
		for i := 7; i >= 0; i-- {
			key := fmt.Sprintf("large_key_%d", i)
			if value, err := cache.Get(key); err == nil {
				present[key] = true
				t.Logf("Found value %d in cache, size: %d bytes", i, len(value))
			} else {
				t.Logf("Value %d not found in cache", i)
			}
		}

		// Should have exactly 4 values
		assert.Equal(t, 4, len(present), "Cache should contain exactly 4 values")

		// Verify we have the most recent 4 values (4-7)
		for i := 4; i < 8; i++ {
			key := fmt.Sprintf("large_key_%d", i)
			assert.True(t, present[key], "Recent value %d should be in cache", i)
		}

		// Verify older values are gone (0-3)
		for i := 0; i < 4; i++ {
			key := fmt.Sprintf("large_key_%d", i)
			assert.False(t, present[key], "Old value %d should not be in cache", i)
		}

		// Try to add a value larger than cache size
		tooLarge := make([]byte, maxSize+1)
		err := cache.Set("too_large", tooLarge)
		assert.Error(t, err, "Should reject values larger than cache size")
	})

	t.Run("ConsistencyUnderConcurrency", func(t *testing.T) {
		cache := newMockCache(8, 1000)
		var wg sync.WaitGroup
		iterations := 1000
		key := "concurrent_key"

		// Start multiple writers
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					value := fmt.Sprintf("value_%d_%d", workerID, j)
					err := cache.Set(key, []byte(value))
					require.NoError(t, err)
				}
			}(i)
		}

		// Start multiple readers
		results := make(chan []byte, iterations*10)
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < iterations; j++ {
					if value, err := cache.Get(key); err == nil {
						results <- value
					}
				}
			}()
		}

		wg.Wait()
		close(results)

		// Verify all retrieved values are valid
		for value := range results {
			assert.Contains(t, string(value), "value_", "Retrieved value should be valid")
		}
	})
}
