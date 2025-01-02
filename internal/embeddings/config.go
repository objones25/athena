package embeddings

import (
	"path/filepath"
	"time"
)

// DefaultMiniLMConfig returns the default configuration for the all-MiniLM-L6-v2 model
func DefaultMiniLMConfig() ModelConfig {
	modelName := "all-MiniLM-L6-v2"
	return ModelConfig{
		ModelPath:           filepath.Join("models", modelName+".onnx"),
		TokenizerConfigPath: filepath.Join("models", modelName+"_tokenizer"),
		Dimension:           384,
		MaxLength:           512,
		UseGPU:              false,
	}
}

// DefaultMPNetConfig returns the default configuration for the all-mpnet-base-v2 model
func DefaultMPNetConfig() ModelConfig {
	modelName := "all-mpnet-base-v2"
	return ModelConfig{
		ModelPath:           filepath.Join("models", modelName+".onnx"),
		TokenizerConfigPath: filepath.Join("models", modelName+"_tokenizer"),
		Dimension:           768,
		MaxLength:           512,
		UseGPU:              false,
	}
}

// DefaultMultilingualConfig returns the default configuration for the paraphrase-multilingual-MiniLM-L12-v2 model
func DefaultMultilingualConfig() ModelConfig {
	modelName := "paraphrase-multilingual-MiniLM-L12-v2"
	return ModelConfig{
		ModelPath:           filepath.Join("models", modelName+".onnx"),
		TokenizerConfigPath: filepath.Join("models", modelName+"_tokenizer"),
		Dimension:           384,
		MaxLength:           512,
		UseGPU:              false,
	}
}

// DefaultConfig returns the default configuration for the embedding service
func DefaultConfig() Config {
	return Config{
		BatchSize: 32,
		Timeout:   30 * time.Second,
		Models:    DefaultModelConfigs(),
		ContentTypeModels: map[ContentType]ModelType{
			ContentTypeCode: ModelMPNetBase,
			ContentTypeText: ModelMultilingualLM,
			ContentTypeMath: ModelMiniLML6V2,
		},
	}
}

// DefaultModelConfigs returns the default configuration for all models
func DefaultModelConfigs() map[ModelType]ModelConfig {
	return map[ModelType]ModelConfig{
		ModelMiniLML6V2:     DefaultMiniLMConfig(),
		ModelMPNetBase:      DefaultMPNetConfig(),
		ModelMultilingualLM: DefaultMultilingualConfig(),
	}
}
