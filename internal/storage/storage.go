package storage

import (
	"context"
	"time"
)

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeCode     ContentType = "code"
	ContentTypeMath     ContentType = "math"
	ContentTypeJSON     ContentType = "json"
	ContentTypeMarkdown ContentType = "markdown"
)

type Content struct {
	Type   ContentType       `json:"type"`
	Data   []byte            `json:"data"`
	Format map[string]string `json:"format,omitempty"` // Additional formatting metadata
}

// Item represents a storable item with its metadata
type Item struct {
	ID        string
	Content   Content
	Vector    []float32
	Metadata  map[string]interface{}
	CreatedAt time.Time
	ExpiresAt time.Time
}

// Cache defines the interface for caching operations
type Cache interface {
	// Set stores an item in the cache with an optional expiration
	Set(ctx context.Context, key string, item *Item) error

	// Get retrieves an item from the cache
	Get(ctx context.Context, key string) (*Item, error)

	// Delete removes an item from the cache
	DeleteFromCache(ctx context.Context, key string) error

	// Clear removes all items from the cache
	Clear(ctx context.Context) error
}

// VectorStore defines the interface for vector storage and similarity search
type VectorStore interface {
	// Insert stores a vector with its associated data
	Insert(ctx context.Context, items []*Item) error

	// Search performs a similarity search using the provided vector
	Search(ctx context.Context, vector []float32, limit int) ([]*Item, error)

	// Delete removes items by their IDs
	DeleteFromStore(ctx context.Context, ids []string) error

	// Update updates existing items
	Update(ctx context.Context, items []*Item) error
}

// StorageManager coordinates cache and vector storage operations
type StorageManager interface {
	Cache
	VectorStore

	// Close cleanly shuts down all storage connections
	Close() error

	// Health checks the health of storage connections
	Health(ctx context.Context) error
}
