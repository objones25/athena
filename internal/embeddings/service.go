package embeddings

import (
	"fmt"
	"os"
	"path/filepath"
)

// InitService initializes the embedding service with the given configuration
func InitService(cfg Config) (Service, error) {
	// Validate configuration
	if err := validateServiceConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Create ONNX service
	service, err := NewONNXService(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX service: %w", err)
	}

	return service, nil
}

// validateServiceConfig validates the service configuration
func validateServiceConfig(cfg Config) error {
	if cfg.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive")
	}

	if len(cfg.Models) == 0 {
		return fmt.Errorf("no models configured")
	}

	if len(cfg.ContentTypeModels) == 0 {
		return fmt.Errorf("no content type mappings configured")
	}

	// Validate each model configuration
	for modelType, modelCfg := range cfg.Models {
		// Check model path
		if modelCfg.ModelPath == "" {
			return fmt.Errorf("model path not specified for %s", modelType)
		}
		if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
			return fmt.Errorf("model file not found at %s: %w", modelCfg.ModelPath, err)
		}

		// Check tokenizer config
		if modelCfg.TokenizerConfigPath == "" {
			return fmt.Errorf("tokenizer config path not specified for %s", modelType)
		}
		if _, err := os.Stat(modelCfg.TokenizerConfigPath); os.IsNotExist(err) {
			return fmt.Errorf("tokenizer config file not found at %s: %w", modelCfg.TokenizerConfigPath, err)
		}

		// Validate dimensions
		if modelCfg.Dimension <= 0 {
			return fmt.Errorf("invalid dimension for %s: must be positive", modelType)
		}

		// Validate max length
		if modelCfg.MaxLength <= 0 {
			return fmt.Errorf("invalid max length for %s: must be positive", modelType)
		}
	}

	// Validate content type mappings
	for contentType, modelType := range cfg.ContentTypeModels {
		if _, ok := cfg.Models[modelType]; !ok {
			return fmt.Errorf("content type %s mapped to undefined model %s", contentType, modelType)
		}
	}

	return nil
}

// EnsureModelDirectories ensures that all necessary model directories exist
func EnsureModelDirectories() error {
	dirs := []string{
		"models",
		"models/tokenizers",
		"models/configs",
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}

// GetModelPath returns the absolute path to a model file
func GetModelPath(name string) string {
	return filepath.Join("models", name)
}

// GetTokenizerPath returns the absolute path to a tokenizer file
func GetTokenizerPath(name string) string {
	return filepath.Join("models", "tokenizers", name)
}

// GetConfigPath returns the absolute path to a config file
func GetConfigPath(name string) string {
	return filepath.Join("models", "configs", name)
}
