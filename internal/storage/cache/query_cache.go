package cache

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/objones25/athena/internal/storage"
)

// QueryKey represents a search query for caching
type QueryKey struct {
	Vector []float32
	Limit  int
}

// QueryResult represents cached search results
type QueryResult struct {
	Items     []*storage.Item
	Timestamp time.Time
}

// QueryCache implements caching for search queries
type QueryCache struct {
	redis      *RedisCache
	ttl        time.Duration
	similarity float32 // Similarity threshold for query matching
}

// NewQueryCache creates a new query cache
func NewQueryCache(redis *RedisCache, ttl time.Duration, similarity float32) *QueryCache {
	if ttl == 0 {
		ttl = time.Minute * 5 // Default TTL for query results
	}
	if similarity == 0 {
		similarity = 0.95 // Default similarity threshold
	}

	return &QueryCache{
		redis:      redis,
		ttl:        ttl,
		similarity: similarity,
	}
}

// generateKey creates a unique key for a query
func (qc *QueryCache) generateKey(query *QueryKey) (string, error) {
	if query == nil {
		return "", fmt.Errorf("query cannot be nil")
	}
	if len(query.Vector) == 0 {
		return "", fmt.Errorf("query vector cannot be empty")
	}
	if query.Limit <= 0 {
		return "", fmt.Errorf("query limit must be positive")
	}

	// Normalize vector for consistent keys
	norm := float32(0)
	for _, v := range query.Vector {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	normalized := make([]float32, len(query.Vector))
	for i, v := range query.Vector {
		normalized[i] = v / norm
	}

	// Create JSON representation
	data := struct {
		Vector []float32 `json:"v"`
		Limit  int       `json:"l"`
	}{
		Vector: normalized,
		Limit:  query.Limit,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("failed to marshal query data: %w", err)
	}

	// Generate hash
	hash := sha256.Sum256(jsonData)
	return "query:" + hex.EncodeToString(hash[:]), nil
}

// Get retrieves cached results for a query
func (qc *QueryCache) Get(ctx context.Context, query *QueryKey) (*QueryResult, error) {
	key, err := qc.generateKey(query)
	if err != nil {
		return nil, err
	}

	item, err := qc.redis.Get(ctx, key)
	if err != nil {
		return nil, err // Return the actual error
	}
	if item == nil {
		return nil, nil // Cache miss
	}

	// Unmarshal cached results
	var result QueryResult
	if err := json.Unmarshal(item.Content.Data, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal query result: %w", err)
	}

	return &result, nil
}

// Set stores search results for a query
func (qc *QueryCache) Set(ctx context.Context, query *QueryKey, items []*storage.Item) error {
	key, err := qc.generateKey(query)
	if err != nil {
		return err
	}

	result := &QueryResult{
		Items:     items,
		Timestamp: time.Now(),
	}

	// Marshal results
	data, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("failed to marshal query result: %w", err)
	}

	// Store in cache
	item := &storage.Item{
		ID: key,
		Content: storage.Content{
			Type: storage.ContentTypeJSON,
			Data: data,
		},
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(qc.ttl),
	}

	return qc.redis.Set(ctx, key, item)
}

// FindSimilarQuery looks for a cached query similar to the input query
func (qc *QueryCache) FindSimilarQuery(ctx context.Context, query *QueryKey) (*QueryKey, error) {
	// This would typically involve:
	// 1. Maintaining an index of recent queries
	// 2. Computing cosine similarity between query vectors
	// 3. Returning the most similar query above the threshold

	// For now, we'll just use exact matching
	// In a production system, this would be replaced with an approximate nearest neighbor search
	return nil, nil
}

// cosineSimilarity computes the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}
