package embeddings

import (
	"context"
	"fmt"
	"time"
)

// ModelType represents different embedding models
type ModelType string

const (
	ModelMiniLML6V2     ModelType = "all-MiniLM-L6-v2"
	ModelMPNetBase      ModelType = "all-mpnet-base-v2"
	ModelMultilingualLM ModelType = "paraphrase-multilingual-MiniLM-L12-v2"
)

// ContentType represents different types of content to embed
type ContentType string

const (
	ContentTypeCode ContentType = "code"
	ContentTypeText ContentType = "text"
	ContentTypeMath ContentType = "math"
)

// Config holds configuration for the embedding service
type Config struct {
	// Maximum batch size for embedding generation
	BatchSize int
	// Timeout for embedding operations
	Timeout time.Duration
	// Model-specific configurations
	Models map[ModelType]ModelConfig
	// Content type to model mapping
	ContentTypeModels map[ContentType]ModelType
	// Path to ONNX Runtime library
	ONNXRuntimePath string
}

// ModelConfig holds configuration for a specific model
type ModelConfig struct {
	// Path to local ONNX model
	ModelPath string
	// Dimension of embeddings produced by this model
	Dimension int
	// Maximum input length
	MaxLength int
	// Whether to use GPU acceleration
	UseGPU bool
	// Model-specific tokenizer configuration file
	TokenizerConfigPath string
}

// EmbeddingRequest represents a request to generate embeddings
type EmbeddingRequest struct {
	// Content to embed
	Content string
	// Type of content
	ContentType ContentType
	// Optional specific model to use (if not specified, use mapped model)
	Model ModelType
	// Optional metadata
	Metadata map[string]interface{}
}

// EmbeddingResult represents the result of embedding generation
type EmbeddingResult struct {
	// Generated embedding vector
	Vector []float32
	// Model used to generate the embedding
	Model ModelType
	// Dimension of the embedding
	Dimension int
	// Time taken to generate the embedding
	Duration time.Duration
	// Any additional metadata
	Metadata map[string]interface{}
}

// Service defines the interface for embedding generation
type Service interface {
	// Generate creates embeddings for a single piece of content
	Generate(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResult, error)

	// BatchGenerate creates embeddings for multiple pieces of content
	BatchGenerate(ctx context.Context, reqs []*EmbeddingRequest) ([]*EmbeddingResult, error)

	// GetModelInfo returns information about a specific model
	GetModelInfo(modelType ModelType) (*ModelConfig, error)

	// Health checks the health of the embedding service
	Health(ctx context.Context) error

	// Close cleans up any resources
	Close() error
}

// Common errors
var (
	ErrInvalidContent = fmt.Errorf("invalid content")
	ErrInvalidModel   = fmt.Errorf("invalid model")
	ErrModelNotFound  = fmt.Errorf("model not found")
	ErrTimeout        = fmt.Errorf("operation timed out")
)
