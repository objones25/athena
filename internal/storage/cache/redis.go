package cache

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/compression"
	"github.com/rs/zerolog"
)

// Custom errors for better error handling
var (
	ErrInvalidContentType = fmt.Errorf("invalid content type")
	ErrNilItem            = fmt.Errorf("item cannot be nil")
	ErrEmptyKey           = fmt.Errorf("key cannot be empty")
	ErrCompression        = fmt.Errorf("compression failed")
	ErrDecompression      = fmt.Errorf("decompression failed")
)

const (
	defaultCompressionThreshold = 1024 // Compress items larger than 1KB
	defaultMaxRetries           = 3
	defaultPoolSize             = 10
	defaultMinIdleConns         = 5
)

// RedisCache implements the Cache interface using Redis
type RedisCache struct {
	client               *redis.Client
	defaultTTL           time.Duration
	compressionThreshold int
	config               Config
	keyPrefix            string
	logger               zerolog.Logger
	compressor           *compression.Compressor
}

type Config struct {
	Host                 string
	Port                 string
	Password             string
	DB                   int
	DefaultTTL           time.Duration
	PoolSize             int
	MinIdleConns         int
	MaxRetries           int
	CompressionThreshold int
}

// compress compresses data using gzip
func compress(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)

	if _, err := gz.Write(data); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCompression, err)
	}

	if err := gz.Close(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrCompression, err)
	}

	return buf.Bytes(), nil
}

// decompress decompresses gzipped data
func decompress(data []byte) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrDecompression, err)
	}
	defer gz.Close()

	decompressed, err := io.ReadAll(gz)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrDecompression, err)
	}

	return decompressed, nil
}

func NewRedisCache(cfg Config) (*RedisCache, error) {
	// Set defaults for optional configuration
	if cfg.PoolSize <= 0 {
		cfg.PoolSize = defaultPoolSize
	}
	if cfg.MinIdleConns <= 0 {
		cfg.MinIdleConns = defaultMinIdleConns
	}
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = defaultMaxRetries
	}
	if cfg.CompressionThreshold <= 0 {
		cfg.CompressionThreshold = defaultCompressionThreshold
	}
	if cfg.DefaultTTL == 0 {
		cfg.DefaultTTL = 24 * time.Hour
	}

	// Input validation
	if cfg.Host == "" {
		return nil, fmt.Errorf("host cannot be empty")
	}
	if cfg.Port == "" {
		return nil, fmt.Errorf("port cannot be empty")
	}

	client := redis.NewClient(&redis.Options{
		Addr:         cfg.Host + ":" + cfg.Port,
		Password:     cfg.Password,
		DB:           cfg.DB,
		PoolSize:     cfg.PoolSize,
		MinIdleConns: cfg.MinIdleConns,
		MaxRetries:   cfg.MaxRetries,
		ReadTimeout:  2 * time.Second,
		WriteTimeout: 2 * time.Second,
		PoolTimeout:  4 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisCache{
		client:               client,
		defaultTTL:           cfg.DefaultTTL,
		compressionThreshold: cfg.CompressionThreshold,
		config:               cfg,
		keyPrefix:            "cache:",
		logger:               zerolog.New(os.Stdout).With().Timestamp().Logger(),
		compressor:           &compression.Compressor{Threshold: cfg.CompressionThreshold},
	}, nil
}

func (rc *RedisCache) Set(ctx context.Context, key string, item *storage.Item) error {
	if key == "" {
		return ErrEmptyKey
	}
	if item == nil {
		return ErrNilItem
	}
	if !isValidContentType(item.Content.Type) {
		return ErrInvalidContentType
	}

	data, err := json.Marshal(item)
	if err != nil {
		return fmt.Errorf("failed to marshal item: %w", err)
	}

	// Compress if data exceeds threshold
	if len(data) > rc.compressionThreshold {
		compressed, err := compress(data)
		if err != nil {
			return err
		}
		data = compressed
		// Add compression flag to key
		key = "compressed:" + key
	}

	expiration := time.Until(item.ExpiresAt)
	if expiration <= 0 {
		expiration = rc.defaultTTL
	}

	return rc.client.Set(ctx, key, data, expiration).Err()
}

func (rc *RedisCache) Get(ctx context.Context, key string) (*storage.Item, error) {
	if key == "" {
		return nil, ErrEmptyKey
	}

	// Try compressed key first
	compressedKey := "compressed:" + key
	data, err := rc.client.Get(ctx, compressedKey).Bytes()
	if err == nil {
		// Data was compressed
		decompressed, err := decompress(data)
		if err != nil {
			return nil, err
		}
		data = decompressed
	} else if err != redis.Nil {
		// Try uncompressed key
		data, err = rc.client.Get(ctx, key).Bytes()
		if err != nil {
			if err == redis.Nil {
				return nil, nil // Cache miss
			}
			return nil, fmt.Errorf("failed to get item from Redis: %w", err)
		}
	} else {
		return nil, nil // Cache miss
	}

	var item storage.Item
	if err := json.Unmarshal(data, &item); err != nil {
		return nil, fmt.Errorf("failed to unmarshal item: %w", err)
	}

	return &item, nil
}

// Pipeline returns a Redis pipeline for batch operations
func (rc *RedisCache) Pipeline() redis.Pipeliner {
	return rc.client.Pipeline()
}

// BatchSet stores multiple items efficiently using pipelining
func (rc *RedisCache) BatchSet(ctx context.Context, items map[string]*storage.Item) error {
	if len(items) == 0 {
		return nil
	}

	pipe := rc.client.Pipeline()
	for key, item := range items {
		data, err := json.Marshal(item)
		if err != nil {
			continue
		}

		// Try compression if data is large enough
		if len(data) > rc.compressionThreshold {
			compressed, err := compress(data)
			if err == nil && len(compressed) < len(data) {
				pipe.Set(ctx, "compressed:"+key, compressed, rc.defaultTTL)
				continue
			}
		}

		pipe.Set(ctx, key, data, rc.defaultTTL)
	}

	_, err := pipe.Exec(ctx)
	return err
}

// BatchDelete removes multiple items efficiently using pipelining
func (rc *RedisCache) BatchDelete(ctx context.Context, keys []string) error {
	if len(keys) == 0 {
		return nil
	}

	pipe := rc.client.Pipeline()
	for _, key := range keys {
		pipe.Del(ctx, key)
		pipe.Del(ctx, "compressed:"+key)
	}

	_, err := pipe.Exec(ctx)
	return err
}

// BatchGet retrieves multiple items efficiently using pipelining
func (rc *RedisCache) BatchGet(ctx context.Context, keys []string) (map[string]*storage.Item, error) {
	if len(keys) == 0 {
		return nil, nil
	}

	// Try both compressed and uncompressed keys
	pipe := rc.client.Pipeline()
	keyMap := make(map[string]string, len(keys)*2)

	for _, key := range keys {
		if key == "" {
			continue
		}
		compressedKey := "compressed:" + key
		keyMap[compressedKey] = key
		keyMap[key] = key
		pipe.Get(ctx, compressedKey)
		pipe.Get(ctx, key)
	}

	cmds, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		return nil, fmt.Errorf("failed to execute batch get: %w", err)
	}

	results := make(map[string]*storage.Item)
	for _, cmd := range cmds {
		if cmd.Err() == nil {
			getCmd := cmd.(*redis.StringCmd)
			key := getCmd.Args()[1].(string)
			originalKey := keyMap[key]

			data, err := getCmd.Bytes()
			if err != nil {
				continue
			}

			// Decompress if needed
			if strings.HasPrefix(key, "compressed:") {
				data, err = decompress(data)
				if err != nil {
					continue
				}
			}

			var item storage.Item
			if err := json.Unmarshal(data, &item); err != nil {
				continue
			}
			results[originalKey] = &item
		}
	}

	return results, nil
}

func (rc *RedisCache) DeleteFromCache(ctx context.Context, key string) error {
	if key == "" {
		return ErrEmptyKey
	}
	pipe := rc.client.Pipeline()
	pipe.Del(ctx, key)
	pipe.Del(ctx, "compressed:"+key)
	_, err := pipe.Exec(ctx)
	return err
}

func (rc *RedisCache) Clear(ctx context.Context) error {
	return rc.client.FlushAll(ctx).Err()
}

// Close implements proper resource cleanup
func (rc *RedisCache) Close() error {
	return rc.client.Close()
}

// Health checks the health of the Redis connection
func (rc *RedisCache) Health(ctx context.Context) error {
	return rc.client.Ping(ctx).Err()
}

func isValidContentType(ct storage.ContentType) bool {
	// Special case: search results don't have a content type
	if ct == "" {
		return true
	}

	switch ct {
	case storage.ContentTypeText,
		storage.ContentTypeCode,
		storage.ContentTypeMath,
		storage.ContentTypeJSON,
		storage.ContentTypeMarkdown:
		return true
	default:
		return false
	}
}

// WarmupConfig defines parameters for cache warming
type WarmupConfig struct {
	// Batch size for warming up items
	BatchSize int
	// Maximum number of items to warm up
	MaxItems int
	// Minimum access count for an item to be considered for warmup
	MinAccessCount int
}

// WarmUp pre-populates the cache with frequently accessed items
func (c *RedisCache) WarmUp(ctx context.Context, store storage.Store, cfg WarmupConfig) error {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1000
	}
	if cfg.MaxItems <= 0 {
		cfg.MaxItems = 10000
	}
	if cfg.MinAccessCount <= 0 {
		cfg.MinAccessCount = 5
	}

	// Get frequently accessed items from store
	items, err := store.GetFrequentItems(ctx, cfg.MaxItems, cfg.MinAccessCount)
	if err != nil {
		return fmt.Errorf("failed to get frequent items: %w", err)
	}

	// Process items in batches
	for i := 0; i < len(items); i += cfg.BatchSize {
		end := i + cfg.BatchSize
		if end > len(items) {
			end = len(items)
		}
		batch := items[i:end]

		// Set items in cache
		pipe := c.client.Pipeline()
		for _, item := range batch {
			data, err := c.marshal(item)
			if err != nil {
				c.logger.Error().Err(err).Str("item_id", item.ID).Msg("Failed to marshal item during warmup")
				continue
			}

			pipe.Set(ctx, c.keyPrefix+item.ID, data, c.defaultTTL)
		}

		// Execute pipeline
		if _, err := pipe.Exec(ctx); err != nil {
			return fmt.Errorf("failed to execute pipeline during warmup: %w", err)
		}

		c.logger.Info().
			Int("batch_size", len(batch)).
			Int("total_processed", end).
			Int("total_items", len(items)).
			Msg("Cache warmup batch completed")
	}

	c.logger.Info().
		Int("total_items", len(items)).
		Msg("Cache warmup completed")
	return nil
}

// marshal serializes an item for storage in Redis
func (c *RedisCache) marshal(item *storage.Item) ([]byte, error) {
	data, err := json.Marshal(item)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal item: %w", err)
	}

	// Compress if data exceeds threshold
	if len(data) > c.compressor.Threshold {
		compressed, err := c.compressor.Compress(data)
		if err != nil {
			return nil, fmt.Errorf("failed to compress item: %w", err)
		}
		return compressed, nil
	}

	return data, nil
}

// Store defines the interface for the backing store used in cache warming
type Store interface {
	// GetFrequentItems returns frequently accessed items for cache warming
	GetFrequentItems(ctx context.Context, maxItems, minAccessCount int) ([]*storage.Item, error)
}
