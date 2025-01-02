package cache

import "context"

// Cache defines the interface for embedding caches
type Cache interface {
	// Get retrieves an embedding from cache
	Get(ctx context.Context, key string) ([]float32, error)

	// Set stores an embedding in cache
	Set(ctx context.Context, key string, embedding []float32) error

	// MGet retrieves multiple embeddings from cache
	MGet(ctx context.Context, keys []string) ([][]float32, error)

	// MSet stores multiple embeddings in cache
	MSet(ctx context.Context, items map[string][]float32) error

	// Delete removes embeddings from cache
	Delete(ctx context.Context, keys []string) error

	// Clear removes all embeddings from cache
	Clear(ctx context.Context) error

	// Close closes the cache connection
	Close() error

	// Health checks if the cache is healthy
	Health(ctx context.Context) error
}

// KeyGenerator generates cache keys for embeddings
type KeyGenerator interface {
	// GenerateKey generates a cache key for the given content and model
	GenerateKey(content string, model string) string

	// GenerateKeys generates cache keys for multiple contents and model
	GenerateKeys(contents []string, model string) []string
}
