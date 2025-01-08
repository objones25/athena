package storage

import (
	"context"
)

// Store defines the interface for vector storage backends
type Store interface {
	// Insert adds items to the store
	Insert(ctx context.Context, items []*Item) error

	// Search finds similar items to the query vector
	Search(ctx context.Context, vector []float32, limit int) ([]*Item, error)

	// DeleteFromStore removes items from the store
	DeleteFromStore(ctx context.Context, ids []string) error

	// GetFrequentItems returns frequently accessed items for cache warming
	GetFrequentItems(ctx context.Context, maxItems, minAccessCount int) ([]*Item, error)

	// Health checks the health of the store
	Health(ctx context.Context) error

	// Close releases any resources held by the store
	Close() error
}
