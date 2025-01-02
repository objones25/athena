package embeddings

import (
	"context"
	"fmt"
	"math"

	"github.com/rs/zerolog/log"
)

// EnsembleStrategy defines different methods of combining model predictions
type EnsembleStrategy int

const (
	WeightedAverage EnsembleStrategy = iota
	MaxScore
	MinScore
	MajorityVoting
	Concatenate
)

// String implements fmt.Stringer
func (s EnsembleStrategy) String() string {
	switch s {
	case WeightedAverage:
		return "WeightedAverage"
	case MaxScore:
		return "MaxScore"
	case MinScore:
		return "MinScore"
	case MajorityVoting:
		return "MajorityVoting"
	case Concatenate:
		return "Concatenate"
	default:
		return "Unknown"
	}
}

// EnsembleConfig holds configuration for the ensemble service
type EnsembleConfig struct {
	Models    []ModelType      // List of models to use
	Weights   []float32        // Weights for each model (for weighted average)
	Strategy  EnsembleStrategy // Which ensemble strategy to use
	Threshold float32          // Threshold for majority voting (default: 0.5)
}

// EnsembleService combines multiple embedding models for improved accuracy
type EnsembleService struct {
	config         Config
	ensembleConfig EnsembleConfig
	clients        map[ModelType]*ONNXClient
}

// NewEnsembleService creates a new ensemble service
func NewEnsembleService(cfg Config) (*EnsembleService, error) {
	// Create service
	service := &EnsembleService{
		config:  cfg,
		clients: make(map[ModelType]*ONNXClient),
	}

	// Create clients for each model type
	modelTypes := make([]ModelType, 0, len(cfg.Models))
	for modelType, modelCfg := range cfg.Models {
		// Create client
		client, err := NewONNXClient(modelCfg)
		if err != nil {
			return nil, fmt.Errorf("failed to create client for model %s: %w", modelType, err)
		}

		// Initialize client
		if err := client.Initialize(context.Background()); err != nil {
			return nil, fmt.Errorf("failed to initialize client for model %s: %w", modelType, err)
		}

		service.clients[modelType] = client
		modelTypes = append(modelTypes, modelType)
	}

	// Set up default ensemble configuration
	service.ensembleConfig = EnsembleConfig{
		Models:    modelTypes,
		Weights:   make([]float32, len(modelTypes)), // Initialize with equal weights
		Strategy:  WeightedAverage,                  // Default to weighted average
		Threshold: 0.5,                              // Default threshold for majority voting
	}

	// Set equal weights for all models
	weight := float32(1.0) / float32(len(modelTypes))
	for i := range service.ensembleConfig.Weights {
		service.ensembleConfig.Weights[i] = weight
	}

	return service, nil
}

// Generate implements Service.Generate
func (s *EnsembleService) Generate(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResult, error) {
	log.Debug().
		Int("num_models", len(s.clients)).
		Interface("model_types", s.ensembleConfig.Models).
		Interface("weights", s.ensembleConfig.Weights).
		Str("strategy", s.ensembleConfig.Strategy.String()).
		Msg("Starting ensemble embedding generation")

	// Get embeddings from each model
	i := 0
	embeddings := make([][]float32, len(s.clients))
	for modelType, client := range s.clients {
		log.Debug().
			Str("model_type", string(modelType)).
			Str("content", req.Content).
			Msg("Generating embedding for model")

		embedding, err := client.GenerateEmbedding(ctx, req.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for model %s: %w", modelType, err)
		}

		// Log first few values and norm of embedding
		norm := vectorNorm(embedding)
		firstValues := embedding
		if len(embedding) > 5 {
			firstValues = embedding[:5]
		}

		log.Debug().
			Str("model_type", string(modelType)).
			Int("dimension", len(embedding)).
			Float64("norm", norm).
			Interface("first_values", firstValues).
			Msg("Generated embedding")

		embeddings[i] = embedding
		i++
	}

	// Combine embeddings based on strategy
	var result *EmbeddingResult
	var err error
	switch s.ensembleConfig.Strategy {
	case WeightedAverage:
		result, err = s.weightedAverage(embeddings)
	case Concatenate:
		result, err = s.concatenate(embeddings)
	default:
		return nil, fmt.Errorf("unsupported ensemble strategy: %s", s.ensembleConfig.Strategy)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to combine embeddings: %w", err)
	}

	// Log combined embedding details
	norm := vectorNorm(result.Vector)
	firstValues := result.Vector
	if len(result.Vector) > 5 {
		firstValues = result.Vector[:5]
	}

	log.Debug().
		Str("strategy", s.ensembleConfig.Strategy.String()).
		Int("dimension", result.Dimension).
		Float64("norm", norm).
		Interface("first_values", firstValues).
		Msg("Combined embedding result")

	return result, nil
}

// weightedAverage combines embeddings using weighted average
func (s *EnsembleService) weightedAverage(embeddings [][]float32) (*EmbeddingResult, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings to combine")
	}
	if len(embeddings) != len(s.ensembleConfig.Weights) {
		return nil, fmt.Errorf("number of embeddings (%d) does not match number of weights (%d)",
			len(embeddings), len(s.ensembleConfig.Weights))
	}

	// Get dimension from first embedding
	dimension := len(embeddings[0])
	result := make([]float32, dimension)

	// Compute weighted average
	for i, embedding := range embeddings {
		if len(embedding) != dimension {
			return nil, fmt.Errorf("embedding %d has inconsistent dimension: expected %d, got %d",
				i, dimension, len(embedding))
		}
		weight := s.ensembleConfig.Weights[i]
		for j, value := range embedding {
			result[j] += value * weight
		}
	}

	// Normalize the result
	norm := float32(math.Sqrt(float64(vectorNorm(result))))
	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return &EmbeddingResult{
		Vector:    result,
		Model:     "ensemble",
		Dimension: dimension,
	}, nil
}

// concatenate combines embeddings by concatenating them
func (s *EnsembleService) concatenate(embeddings [][]float32) (*EmbeddingResult, error) {
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings to combine")
	}

	// Calculate total dimension
	totalDim := 0
	for i, embedding := range embeddings {
		if len(embedding) == 0 {
			return nil, fmt.Errorf("embedding %d is empty", i)
		}
		totalDim += len(embedding)
	}

	// Concatenate embeddings
	result := make([]float32, 0, totalDim)
	for _, embedding := range embeddings {
		result = append(result, embedding...)
	}

	// Normalize the concatenated vector
	norm := float32(math.Sqrt(float64(vectorNorm(result))))
	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return &EmbeddingResult{
		Vector:    result,
		Model:     "ensemble",
		Dimension: totalDim,
	}, nil
}

// BatchGenerate implements Service.BatchGenerate
func (s *EnsembleService) BatchGenerate(ctx context.Context, reqs []*EmbeddingRequest) ([]*EmbeddingResult, error) {
	results := make([]*EmbeddingResult, len(reqs))
	for i, req := range reqs {
		result, err := s.Generate(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embedding for request %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// GetModelInfo implements Service.GetModelInfo
func (s *EnsembleService) GetModelInfo(modelType ModelType) (*ModelConfig, error) {
	client, ok := s.clients[modelType]
	if !ok {
		return nil, fmt.Errorf("model %s not found", modelType)
	}
	info, err := client.GetModelInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get model info for %s: %w", modelType, err)
	}
	return info, nil
}

// Health implements Service.Health
func (s *EnsembleService) Health(ctx context.Context) error {
	for modelType, client := range s.clients {
		if err := client.Health(ctx); err != nil {
			return fmt.Errorf("model %s is unhealthy: %w", modelType, err)
		}
	}
	return nil
}

// Close implements Service.Close
func (s *EnsembleService) Close() error {
	var lastErr error
	for name, client := range s.clients {
		if err := client.Close(); err != nil {
			lastErr = fmt.Errorf("failed to close model %s: %w", name, err)
		}
	}
	return lastErr
}
