package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/rs/zerolog/log"
)

const (
	defaultTTL = 24 * time.Hour
)

// RedisCache implements caching for embeddings using Redis
type RedisCache struct {
	client *redis.Client
	ttl    time.Duration
}

// Config holds Redis configuration
type Config struct {
	Addr     string
	Password string
	DB       int
	TTL      time.Duration
}

// NewRedisCache creates a new Redis cache client
func NewRedisCache(cfg Config) (*RedisCache, error) {
	if cfg.Addr == "" {
		return nil, fmt.Errorf("Redis address is required")
	}

	if cfg.TTL == 0 {
		cfg.TTL = defaultTTL
	}

	client := redis.NewClient(&redis.Options{
		Addr:     cfg.Addr,
		Password: cfg.Password,
		DB:       cfg.DB,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisCache{
		client: client,
		ttl:    cfg.TTL,
	}, nil
}

// Get retrieves embeddings from cache
func (c *RedisCache) Get(ctx context.Context, key string) ([]float32, error) {
	val, err := c.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return nil, nil // Cache miss
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get from cache: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal([]byte(val), &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
	}

	log.Debug().Str("key", key).Msg("Cache hit")
	return embedding, nil
}

// Set stores embeddings in cache
func (c *RedisCache) Set(ctx context.Context, key string, embedding []float32) error {
	data, err := json.Marshal(embedding)
	if err != nil {
		return fmt.Errorf("failed to marshal embedding: %w", err)
	}

	if err := c.client.Set(ctx, key, string(data), c.ttl).Err(); err != nil {
		return fmt.Errorf("failed to set in cache: %w", err)
	}

	log.Debug().Str("key", key).Msg("Cached embedding")
	return nil
}

// MGet retrieves multiple embeddings from cache
func (c *RedisCache) MGet(ctx context.Context, keys []string) ([][]float32, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	vals, err := c.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get from cache: %w", err)
	}

	results := make([][]float32, len(keys))
	for i, val := range vals {
		if val == nil {
			continue // Cache miss
		}

		var embedding []float32
		if err := json.Unmarshal([]byte(val.(string)), &embedding); err != nil {
			return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
		}
		results[i] = embedding
	}

	return results, nil
}

// MSet stores multiple embeddings in cache
func (c *RedisCache) MSet(ctx context.Context, items map[string][]float32) error {
	if len(items) == 0 {
		return nil
	}

	pipe := c.client.Pipeline()
	for key, embedding := range items {
		data, err := json.Marshal(embedding)
		if err != nil {
			return fmt.Errorf("failed to marshal embedding: %w", err)
		}
		pipe.Set(ctx, key, string(data), c.ttl)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to set in cache: %w", err)
	}

	log.Debug().Int("count", len(items)).Msg("Cached embeddings")
	return nil
}

// Delete removes embeddings from cache
func (c *RedisCache) Delete(ctx context.Context, keys []string) error {
	if len(keys) == 0 {
		return nil
	}

	if err := c.client.Del(ctx, keys...).Err(); err != nil {
		return fmt.Errorf("failed to delete from cache: %w", err)
	}

	log.Debug().Int("count", len(keys)).Msg("Deleted cached embeddings")
	return nil
}

// Clear removes all embeddings from cache
func (c *RedisCache) Clear(ctx context.Context) error {
	if err := c.client.FlushDB(ctx).Err(); err != nil {
		return fmt.Errorf("failed to clear cache: %w", err)
	}

	log.Debug().Msg("Cleared cache")
	return nil
}

// Close closes the Redis connection
func (c *RedisCache) Close() error {
	return c.client.Close()
}

// Health checks if Redis is healthy
func (c *RedisCache) Health(ctx context.Context) error {
	return c.client.Ping(ctx).Err()
}
