package embedding

import (
	"context"
	"errors"
	"os"

	"github.com/joho/godotenv"
)

// Common errors
var (
	ErrInvalidInput    = errors.New("invalid input for embedding")
	ErrModelNotFound   = errors.New("embedding model not found")
	ErrAPIKeyNotSet    = errors.New("HuggingFace API key not set")
	ErrEmbeddingFailed = errors.New("failed to generate embeddings")
)

// ModelType represents supported embedding models
type ModelType string

const (
	ModelMPNetBaseV2    ModelType = "sentence-transformers/all-mpnet-base-v2"
	ModelMiniLML6V2     ModelType = "sentence-transformers/all-MiniLM-L6-v2"
	ModelRobertaLargeV1 ModelType = "sentence-transformers/all-roberta-large-v1"
)

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	// Try loading from different possible locations
	envFiles := []string{
		".env",          // Current directory
		"../../.env",    // Project root when running from internal/embedding
		"../../../.env", // Project root when running from test/unit
	}

	var loadErr error
	for _, envFile := range envFiles {
		err := godotenv.Load(envFile)
		if err == nil {
			loadErr = nil
			break
		}
		loadErr = err
	}

	// It's okay if no .env file is found, we'll use environment variables
	if loadErr != nil && !os.IsNotExist(loadErr) {
		return nil, loadErr
	}

	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		return nil, ErrAPIKeyNotSet
	}

	return &Config{
		APIKey:       apiKey,
		DefaultModel: ModelMPNetBaseV2,
		MaxBatchSize: 32,
		Timeout:      30,
	}, nil
}

// Config holds the configuration for the embedding service
type Config struct {
	APIKey       string
	DefaultModel ModelType
	MaxBatchSize int
	Timeout      int // in seconds
}

// EmbeddingResult represents the result of an embedding operation
type EmbeddingResult struct {
	Vector []float32
	Model  ModelType
}

// Service defines the interface for embedding operations
type Service interface {
	// Embed generates embeddings for a single text input
	Embed(ctx context.Context, text string) (*EmbeddingResult, error)

	// EmbedBatch generates embeddings for multiple text inputs
	EmbedBatch(ctx context.Context, texts []string) ([]*EmbeddingResult, error)

	// EmbedWithModel generates embeddings using a specific model
	EmbedWithModel(ctx context.Context, text string, model ModelType) (*EmbeddingResult, error)

	// GetSupportedModels returns the list of supported models
	GetSupportedModels() []ModelType

	// GetConfig returns the current configuration
	GetConfig() Config
}
