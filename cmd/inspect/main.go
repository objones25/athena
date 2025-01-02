package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/rs/zerolog"
	"github.com/yalue/onnxruntime_go"
)

// ModelConfig holds basic ONNX model configuration
type ModelConfig struct {
	ModelPath string
	Dimension int64 // Changed to int64 to match tensor requirements
	MaxLength int64 // Changed to int64 to match tensor requirements
	UseGPU    bool
}

// ModelArchitecture holds information about the model's architecture
type ModelArchitecture struct {
	NumLayers       int
	HiddenDimension int
	NumHeads        int
	TotalParams     int64
}

func analyzeModel(modelPath string, config ModelConfig) ModelArchitecture {
	arch := ModelArchitecture{}

	// Get input/output info
	_, outputInfo, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return arch
	}

	// Get hidden dimension from output shape or config
	for _, info := range outputInfo {
		if info.Name == "last_hidden_state" {
			if len(info.Dimensions) >= 3 {
				if info.Dimensions[2] > 0 {
					arch.HiddenDimension = int(info.Dimensions[2])
				} else {
					// Use config dimension if shape is dynamic
					arch.HiddenDimension = int(config.Dimension)
				}
				break
			}
		}
	}

	// If hidden dimension is still not set, use config
	if arch.HiddenDimension <= 0 {
		arch.HiddenDimension = int(config.Dimension)
	}

	// Infer number of layers from model name
	if strings.Contains(modelPath, "L6-v2") {
		arch.NumLayers = 6
	} else if strings.Contains(modelPath, "L12-v2") {
		arch.NumLayers = 12
	} else {
		// Default to 12 layers for base models
		arch.NumLayers = 12
	}

	// Estimate number of attention heads (typically hidden_dim / 64)
	arch.NumHeads = arch.HiddenDimension / 64

	// Calculate approximate total parameters
	// This is a rough estimate based on standard transformer architecture
	vocabSize := int64(30522) // Standard BERT vocabulary size
	arch.TotalParams = estimateParameters(int64(arch.NumLayers), int64(arch.HiddenDimension), vocabSize)

	return arch
}

func estimateParameters(numLayers, hiddenDim, vocabSize int64) int64 {
	// Embedding parameters
	params := vocabSize * hiddenDim // Token embeddings
	params += 512 * hiddenDim       // Position embeddings

	// Each transformer layer parameters
	paramsPerLayer := int64(4) * hiddenDim * hiddenDim // Self-attention
	paramsPerLayer += 2 * hiddenDim * hiddenDim        // Feed-forward
	paramsPerLayer += 8 * hiddenDim                    // Layer norms and biases

	// Total parameters
	totalParams := params + (numLayers * paramsPerLayer)
	return totalParams
}

func main() {
	// Setup logging
	log := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Explicitly set DYLD_LIBRARY_PATH
	onnxLibPath, err := filepath.Abs(filepath.Join("..", "..", "onnxruntime-osx-arm64-1.14.0", "lib"))
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to get absolute path for ONNX Runtime library")
	}
	os.Setenv("DYLD_LIBRARY_PATH", onnxLibPath)

	// Set the ONNX Runtime shared library path
	libPath := filepath.Join(onnxLibPath, "libonnxruntime.1.14.0.dylib")
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		log.Fatal().Err(err).Str("path", libPath).Msg("ONNX Runtime library not found")
	}
	onnxruntime_go.SetSharedLibraryPath(libPath)

	// Initialize ONNX Runtime environment
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize ONNX Runtime environment")
	}
	defer onnxruntime_go.DestroyEnvironment()

	// Default models to analyze
	models := []ModelConfig{
		{
			ModelPath: filepath.Join("..", "..", "models", "all-MiniLM-L6-v2.onnx"),
			Dimension: 384,
			MaxLength: 512,
			UseGPU:    false,
		},
		{
			ModelPath: filepath.Join("..", "..", "models", "all-mpnet-base-v2.onnx"),
			Dimension: 768,
			MaxLength: 512,
			UseGPU:    false,
		},
		{
			ModelPath: filepath.Join("..", "..", "models", "paraphrase-multilingual-MiniLM-L12-v2.onnx"),
			Dimension: 384,
			MaxLength: 512,
			UseGPU:    false,
		},
	}

	// Check if a specific model was requested
	if modelPath := os.Getenv("MODEL_PATH"); modelPath != "" {
		// Create a single model config
		config := ModelConfig{
			ModelPath: modelPath,
			Dimension: 384, // Default dimension, will be updated during analysis
			MaxLength: 512,
			UseGPU:    false,
		}
		models = []ModelConfig{config}
	}

	// Print system information
	fmt.Println("\nSystem Information:")
	fmt.Println("-----------------")
	fmt.Printf("OS: %s\n", runtime.GOOS)
	fmt.Printf("Architecture: %s\n", runtime.GOARCH)
	fmt.Printf("Go Version: %s\n", runtime.Version())
	fmt.Printf("ONNX Runtime Path: %s\n", libPath)

	// Analyze each model
	for _, config := range models {
		fmt.Printf("\nAnalyzing model: %s\n", filepath.Base(config.ModelPath))
		fmt.Println(strings.Repeat("-", len(filepath.Base(config.ModelPath))+15))

		absModelPath, err := filepath.Abs(config.ModelPath)
		if err != nil {
			log.Error().Err(err).Str("path", config.ModelPath).Msg("Failed to get absolute path")
			continue
		}

		// Check if model exists
		if _, err := os.Stat(absModelPath); os.IsNotExist(err) {
			fmt.Printf("❌ Model file not found: %s\n", absModelPath)
			continue
		}

		// Print file info
		if info, err := os.Stat(absModelPath); err == nil {
			fmt.Printf("Model file size: %.2f MB\n", float64(info.Size())/1024/1024)
		}

		// Analyze model architecture
		arch := analyzeModel(absModelPath, config)

		fmt.Println("\nModel Architecture:")
		fmt.Printf("Number of Layers: %d\n", arch.NumLayers)
		fmt.Printf("Hidden Dimension: %d\n", arch.HiddenDimension)
		fmt.Printf("Number of Attention Heads: %d\n", arch.NumHeads)
		fmt.Printf("Approximate Parameters: %.2fM\n", float64(arch.TotalParams)/1_000_000)

		// Get and print input/output info
		inputInfo, outputInfo, err := onnxruntime_go.GetInputOutputInfo(absModelPath)
		if err != nil {
			log.Error().Err(err).Msg("Failed to get model input/output info")
			continue
		}

		fmt.Println("\nInput Tensors:")
		for i, info := range inputInfo {
			fmt.Printf("%d. Name: %s\n   Type: %s\n   Dimensions: %v\n",
				i+1, info.Name, info.DataType, info.Dimensions)
		}

		fmt.Println("\nOutput Tensors:")
		for i, info := range outputInfo {
			fmt.Printf("%d. Name: %s\n   Type: %s\n   Dimensions: %v\n",
				i+1, info.Name, info.DataType, info.Dimensions)
		}
	}
}
