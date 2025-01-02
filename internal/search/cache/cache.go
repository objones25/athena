package cache

import (
	"container/list"
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/objones25/athena/internal/search/similarity"
)

// Entry represents a cache entry with metadata
type Entry struct {
	Vector     []float32
	Metadata   map[string]interface{}
	Scores     map[string]similarity.Metrics
	LastAccess time.Time
}

// Config holds cache configuration
type Config struct {
	MaxSize           int           // Maximum number of entries
	EvictionBatchSize int           // Number of entries to evict at once
	TTL               time.Duration // Time-to-live for entries
	CleanupInterval   time.Duration // Interval for cleanup routine
}

// DefaultConfig returns default cache configuration
func DefaultConfig() Config {
	return Config{
		MaxSize:           100000,
		EvictionBatchSize: 1000,
		TTL:               24 * time.Hour,
		CleanupInterval:   1 * time.Hour,
	}
}

// Cache provides thread-safe in-memory caching with LRU eviction
type Cache struct {
	entries    map[string]*Entry
	lru        *list.List
	lruIndex   map[string]*list.Element
	dimensions int
	config     Config
	mu         sync.RWMutex
	done       chan struct{}
}

// New creates a new cache instance
func New(dimensions int, cfg Config) *Cache {
	c := &Cache{
		entries:    make(map[string]*Entry),
		lru:        list.New(),
		lruIndex:   make(map[string]*list.Element),
		dimensions: dimensions,
		config:     cfg,
		done:       make(chan struct{}),
	}

	// Start cleanup routine
	go c.cleanupRoutine()

	return c
}

// Get retrieves a vector from cache
func (c *Cache) Get(ctx context.Context, key string) ([]float32, error) {
	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Update access time and LRU
	c.mu.Lock()
	entry.LastAccess = time.Now()
	if elem, ok := c.lruIndex[key]; ok {
		c.lru.MoveToFront(elem)
	}
	c.mu.Unlock()

	return entry.Vector, nil
}

// GetWithMetadata retrieves a vector and its metadata
func (c *Cache) GetWithMetadata(ctx context.Context, key string) ([]float32, map[string]interface{}, error) {
	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()

	if !ok {
		return nil, nil, fmt.Errorf("key not found: %s", key)
	}

	// Update access time and LRU
	c.mu.Lock()
	entry.LastAccess = time.Now()
	if elem, ok := c.lruIndex[key]; ok {
		c.lru.MoveToFront(elem)
	}
	c.mu.Unlock()

	return entry.Vector, entry.Metadata, nil
}

// Set stores a vector in cache
func (c *Cache) Set(ctx context.Context, key string, vector []float32, metadata map[string]interface{}) error {
	if len(vector) != c.dimensions {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", c.dimensions, len(vector))
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if we need to evict
	if len(c.entries) >= c.config.MaxSize {
		c.evict(c.config.EvictionBatchSize)
	}

	// Create or update entry
	entry := &Entry{
		Vector:     vector,
		Metadata:   metadata,
		Scores:     make(map[string]similarity.Metrics),
		LastAccess: time.Now(),
	}
	c.entries[key] = entry

	// Update LRU
	if elem, ok := c.lruIndex[key]; ok {
		c.lru.MoveToFront(elem)
	} else {
		elem := c.lru.PushFront(key)
		c.lruIndex[key] = elem
	}

	return nil
}

// SetBatch stores multiple vectors in batch
func (c *Cache) SetBatch(ctx context.Context, keys []string, vectors [][]float32, metadata []map[string]interface{}) error {
	if len(keys) != len(vectors) || len(keys) != len(metadata) {
		return fmt.Errorf("keys, vectors, and metadata length mismatch")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if we need to evict
	if len(c.entries)+len(keys) > c.config.MaxSize {
		c.evict(len(keys))
	}

	// Add all entries
	for i, key := range keys {
		if len(vectors[i]) != c.dimensions {
			return fmt.Errorf("vector dimension mismatch for key %s: expected %d, got %d",
				key, c.dimensions, len(vectors[i]))
		}

		entry := &Entry{
			Vector:     vectors[i],
			Metadata:   metadata[i],
			Scores:     make(map[string]similarity.Metrics),
			LastAccess: time.Now(),
		}
		c.entries[key] = entry

		// Update LRU
		if elem, ok := c.lruIndex[key]; ok {
			c.lru.MoveToFront(elem)
		} else {
			elem := c.lru.PushFront(key)
			c.lruIndex[key] = elem
		}
	}

	return nil
}

// Remove removes entries from cache
func (c *Cache) Remove(ctx context.Context, keys ...string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, key := range keys {
		delete(c.entries, key)
		if elem, ok := c.lruIndex[key]; ok {
			c.lru.Remove(elem)
			delete(c.lruIndex, key)
		}
	}
}

// Clear removes all entries from cache
func (c *Cache) Clear(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.entries = make(map[string]*Entry)
	c.lru = list.New()
	c.lruIndex = make(map[string]*list.Element)
}

// GetSimilarityScore gets cached similarity score between two vectors
func (c *Cache) GetSimilarityScore(key1, key2 string) (similarity.Metrics, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry1, ok1 := c.entries[key1]
	if !ok1 {
		return similarity.Metrics{}, false
	}

	metrics, ok := entry1.Scores[key2]
	return metrics, ok
}

// SetSimilarityScore caches similarity score between two vectors
func (c *Cache) SetSimilarityScore(key1, key2 string, metrics similarity.Metrics) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry1, ok := c.entries[key1]; ok {
		entry1.Scores[key2] = metrics
	}
	if entry2, ok := c.entries[key2]; ok {
		entry2.Scores[key1] = metrics
	}
}

// evict removes the specified number of least recently used entries
func (c *Cache) evict(count int) {
	for i := 0; i < count && c.lru.Len() > 0; i++ {
		elem := c.lru.Back()
		if elem == nil {
			break
		}
		key := elem.Value.(string)
		delete(c.entries, key)
		delete(c.lruIndex, key)
		c.lru.Remove(elem)
	}
}

// cleanupRoutine periodically removes expired entries
func (c *Cache) cleanupRoutine() {
	ticker := time.NewTicker(c.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanup()
		case <-c.done:
			return
		}
	}
}

// cleanup removes expired entries
func (c *Cache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	var expired []string

	for key, entry := range c.entries {
		if now.Sub(entry.LastAccess) > c.config.TTL {
			expired = append(expired, key)
		}
	}

	for _, key := range expired {
		delete(c.entries, key)
		if elem, ok := c.lruIndex[key]; ok {
			c.lru.Remove(elem)
			delete(c.lruIndex, key)
		}
	}
}

// Close stops the cleanup routine
func (c *Cache) Close() {
	close(c.done)
}
