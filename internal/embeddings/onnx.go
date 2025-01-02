package embeddings

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/rs/zerolog/log"
	ort "github.com/yalue/onnxruntime_go"

	"github.com/objones25/athena/internal/embeddings/preprocess"
	"github.com/objones25/athena/internal/embeddings/tokenizer"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	runtimeMu     sync.Mutex
	runtimeInited bool
	runtimePID    int // Track which process initialized the runtime
)

// ensureRuntime ensures the ONNX runtime is initialized
func ensureRuntime(libPath string) error {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	currentPID := os.Getpid()
	log.Debug().
		Int("current_pid", currentPID).
		Int("runtime_pid", runtimePID).
		Bool("is_initialized", runtimeInited).
		Bool("ort_initialized", ort.IsInitialized()).
		Str("lib_path", libPath).
		Msg("Checking ONNX runtime state")

	if runtimeInited {
		if currentPID != runtimePID {
			log.Warn().
				Int("current_pid", currentPID).
				Int("runtime_pid", runtimePID).
				Msg("ONNX runtime was initialized in a different process")
			// Reset our tracking state since we're in a new process
			runtimeInited = false
			runtimePID = 0
		} else {
			log.Debug().Msg("ONNX runtime already initialized in current process")
			return nil
		}
	}

	log.Debug().
		Str("lib_path", libPath).
		Msg("Initializing ONNX Runtime")

	// Check if the library exists and is readable
	info, err := os.Stat(libPath)
	if err != nil {
		log.Error().
			Err(err).
			Str("lib_path", libPath).
			Msg("Failed to access ONNX runtime library")
		return fmt.Errorf("failed to access library: %w", err)
	}
	log.Debug().
		Str("lib_path", libPath).
		Int64("size", info.Size()).
		Str("mode", info.Mode().String()).
		Msg("ONNX Runtime library info")

	// Set the library path
	ort.SetSharedLibraryPath(libPath)

	// Get ONNX runtime version before initialization
	version := ort.GetVersion()
	log.Debug().
		Str("version", version).
		Msg("ONNX Runtime version info")

	// Initialize the environment with more detailed error handling
	if err := ort.InitializeEnvironment(); err != nil {
		// Log the full error for debugging
		log.Error().
			Err(err).
			Str("lib_path", libPath).
			Str("version", version).
			Bool("was_initialized", ort.IsInitialized()).
			Msg("Failed to initialize ONNX runtime environment")

		return fmt.Errorf("failed to initialize environment: %w", err)
	}

	// Check if runtime is initialized
	if !ort.IsInitialized() {
		log.Error().
			Str("lib_path", libPath).
			Str("version", version).
			Msg("Runtime not initialized after successful initialization call")
		return fmt.Errorf("runtime not initialized after successful initialization call")
	}

	runtimeInited = true
	runtimePID = currentPID
	log.Debug().
		Str("version", version).
		Bool("initialized", ort.IsInitialized()).
		Int("pid", runtimePID).
		Msg("ONNX Runtime initialized successfully")

	return nil
}

// ResetRuntime is used for testing to reset runtime state
func ResetRuntime() {
	runtimeMu.Lock()
	defer runtimeMu.Unlock()

	currentPID := os.Getpid()
	log.Debug().
		Int("current_pid", currentPID).
		Int("runtime_pid", runtimePID).
		Bool("is_initialized", runtimeInited).
		Bool("ort_initialized", ort.IsInitialized()).
		Msg("ResetRuntime called")

	if ort.IsInitialized() {
		// Destroy all existing sessions and tensors
		ort.DestroyEnvironment()
		log.Debug().Msg("ONNX runtime environment destroyed")
	}

	// Reset our tracking state
	runtimeInited = false
	runtimePID = 0

	log.Debug().
		Bool("ort_initialized", ort.IsInitialized()).
		Msg("Runtime reset complete")
}

// ONNXClient implements LocalModelClient for ONNX model inference
type ONNXClient struct {
	mu         sync.RWMutex
	model      *ort.AdvancedSession
	config     ModelConfig
	tokenizer  *tokenizer.Tokenizer
	inputs     []*ort.Tensor[int64]
	outputs    []*ort.Tensor[float32]
	inputNames []string // Store input names for reference
}

// NewONNXClient creates a new ONNX model client
func NewONNXClient(cfg ModelConfig) (*ONNXClient, error) {
	log.Debug().
		Str("model_path", cfg.ModelPath).
		Str("tokenizer_path", cfg.TokenizerConfigPath).
		Int("dimension", cfg.Dimension).
		Int("max_length", cfg.MaxLength).
		Bool("use_gpu", cfg.UseGPU).
		Msg("Creating new ONNX client")

	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}

	if cfg.TokenizerConfigPath == "" {
		return nil, fmt.Errorf("tokenizer config path is required")
	}

	// Validate model path exists
	if _, err := os.Stat(cfg.ModelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found at %s: %w", cfg.ModelPath, err)
	}

	// Validate tokenizer path exists
	if _, err := os.Stat(cfg.TokenizerConfigPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer config not found at %s: %w", cfg.TokenizerConfigPath, err)
	}

	// Create tokenizer
	tokenizerCfg := tokenizer.TokenizerConfig{
		MaxLength:         cfg.MaxLength,
		PreprocessOptions: preprocess.DefaultTextPreprocessOptions(),
		ModelName:         filepath.Base(cfg.TokenizerConfigPath), // Use the tokenizer directory name
	}

	tok, err := tokenizer.NewTokenizer(tokenizerCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	client := &ONNXClient{
		config:    cfg,
		tokenizer: tok,
	}

	log.Debug().Msg("ONNX client created successfully")
	return client, nil
}

// LoadModel implements LocalModelClient
func (c *ONNXClient) LoadModel(path string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Debug().
		Str("model_path", path).
		Bool("has_existing_model", c.model != nil).
		Msg("Starting model load")

	timer := prometheus.NewTimer(ModelLoadDuration.WithLabelValues(filepath.Base(path)))
	defer timer.ObserveDuration()

	// Get ONNX Runtime library path from environment variable
	libPath := os.Getenv("DYLD_LIBRARY_PATH")
	if libPath == "" {
		// Fallback to project root directory
		_, currentFile, _, ok := runtime.Caller(0)
		if !ok {
			return fmt.Errorf("failed to get current file path")
		}
		// Go up three levels from internal/embeddings/onnx.go to get to the project root
		rootDir := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
		libPath = filepath.Join(rootDir, "onnxruntime-osx-arm64-1.14.0", "lib")
	}
	libPath = filepath.Join(libPath, "libonnxruntime.1.14.0.dylib")

	log.Debug().
		Str("lib_path", libPath).
		Msg("Using ONNX Runtime library")

	// Initialize ONNX runtime
	if err := ensureRuntime(libPath); err != nil {
		log.Error().
			Err(err).
			Str("lib_path", libPath).
			Str("model_path", path).
			Msg("Failed to initialize ONNX runtime")
		return fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Get input/output info
	inputInfo, outputInfo, err := ort.GetInputOutputInfo(path)
	if err != nil {
		log.Error().
			Err(err).
			Str("model_path", path).
			Msg("Failed to get model input/output info")
		return fmt.Errorf("failed to get model info: %w", err)
	}

	// Log model info for debugging
	for _, info := range inputInfo {
		log.Debug().
			Str("name", info.Name).
			Str("type", info.OrtValueType.String()).
			Interface("dimensions", info.Dimensions).
			Str("data_type", info.DataType.String()).
			Msg("Model input info")
	}
	for _, info := range outputInfo {
		log.Debug().
			Str("name", info.Name).
			Str("type", info.OrtValueType.String()).
			Interface("dimensions", info.Dimensions).
			Str("data_type", info.DataType.String()).
			Msg("Model output info")
	}

	// Extract input/output names
	inputNames := make([]string, len(inputInfo))
	outputNames := make([]string, len(outputInfo))
	for i, info := range inputInfo {
		inputNames[i] = info.Name
	}
	for i, info := range outputInfo {
		outputNames[i] = info.Name
	}

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Try to enable CUDA provider if available
	if c.config.UseGPU {
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			log.Warn().Err(err).Msg("Failed to create CUDA provider options, falling back to CPU")
		} else {
			defer cudaOpts.Destroy()

			// Configure CUDA options
			err = cudaOpts.Update(map[string]string{
				"device_id":                 "0", // Use first GPU
				"arena_extend_strategy":     "kNextPowerOfTwo",
				"gpu_mem_limit":             "0", // No limit
				"cudnn_conv_algo_search":    "EXHAUSTIVE",
				"do_copy_in_default_stream": "1",
			})
			if err != nil {
				log.Warn().Err(err).Msg("Failed to update CUDA provider options")
			} else {
				// Append CUDA provider
				err = options.AppendExecutionProviderCUDA(cudaOpts)
				if err != nil {
					log.Warn().Err(err).Msg("Failed to append CUDA provider")
				} else {
					log.Info().Msg("CUDA provider enabled")
				}
			}
		}
	}

	// Create input tensors with appropriate sizes
	batchSize := int64(1)
	seqLength := int64(c.config.MaxLength)

	// Create input tensors based on model requirements
	c.inputs = make([]*ort.Tensor[int64], len(inputNames))
	for i, name := range inputNames {
		var tensor *ort.Tensor[int64]
		var err error

		switch name {
		case "input_ids", "attention_mask":
			tensor, err = ort.NewTensor[int64]([]int64{batchSize, seqLength}, make([]int64, batchSize*seqLength))
		case "token_type_ids":
			tensor, err = ort.NewTensor[int64]([]int64{batchSize, seqLength}, make([]int64, batchSize*seqLength))
		default:
			log.Warn().
				Str("name", name).
				Msg("Unknown input tensor name")
			tensor, err = ort.NewTensor[int64]([]int64{batchSize, seqLength}, make([]int64, batchSize*seqLength))
		}

		if err != nil {
			// Clean up any tensors we've already created
			for j := 0; j < i; j++ {
				c.inputs[j].Destroy()
			}
			return fmt.Errorf("failed to create input tensor %s: %w", name, err)
		}
		c.inputs[i] = tensor
	}

	// Create output tensors
	c.outputs = make([]*ort.Tensor[float32], len(outputNames))
	for i := range outputNames {
		var tensor *ort.Tensor[float32]
		var err error

		switch outputNames[i] {
		case "last_hidden_state":
			// Shape: [batch_size, sequence_length, hidden_size]
			tensor, err = ort.NewTensor[float32]([]int64{batchSize, seqLength, int64(c.config.Dimension)}, make([]float32, batchSize*seqLength*int64(c.config.Dimension)))
		case "pooler_output":
			// Shape: [batch_size, hidden_size]
			tensor, err = ort.NewTensor[float32]([]int64{batchSize, int64(c.config.Dimension)}, make([]float32, batchSize*int64(c.config.Dimension)))
		default:
			log.Warn().Str("name", outputNames[i]).Msg("Unknown output tensor name")
			continue
		}

		if err != nil {
			// Clean up any tensors we've already created
			for j := 0; j < i; j++ {
				c.outputs[j].Destroy()
			}
			for _, input := range c.inputs {
				input.Destroy()
			}
			return fmt.Errorf("failed to create output tensor: %w", err)
		}
		c.outputs[i] = tensor
	}

	// Create the session
	inputs := make([]ort.ArbitraryTensor, len(c.inputs))
	outputs := make([]ort.ArbitraryTensor, len(c.outputs))
	for i, tensor := range c.inputs {
		inputs[i] = tensor
	}
	for i, tensor := range c.outputs {
		outputs[i] = tensor
	}

	session, err := ort.NewAdvancedSession(path, inputNames, outputNames, inputs, outputs, options)
	if err != nil {
		// Clean up all tensors
		for _, input := range c.inputs {
			input.Destroy()
		}
		for _, output := range c.outputs {
			output.Destroy()
		}
		log.Error().
			Err(err).
			Str("model_path", path).
			Msg("Failed to create ONNX session")
		return fmt.Errorf("failed to create session: %w", err)
	}

	c.model = session
	c.inputNames = inputNames // Store input names for later use

	log.Info().
		Str("model_path", path).
		Int("inputs", len(inputNames)).
		Int("outputs", len(outputNames)).
		Interface("input_names", inputNames).
		Msg("ONNX model loaded successfully")

	return nil
}

// Initialize implements ModelClient
func (c *ONNXClient) Initialize(ctx context.Context) error {
	if err := c.LoadModel(c.config.ModelPath); err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	return nil
}

// GenerateEmbedding generates an embedding for a single piece of content
func (c *ONNXClient) GenerateEmbedding(ctx context.Context, content string) ([]float32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.model == nil {
		EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "model_not_loaded").Inc()
		return nil, fmt.Errorf("model not loaded")
	}

	// Record input length
	InputLength.WithLabelValues(filepath.Base(c.config.ModelPath), "text").Observe(float64(len(content)))

	// Start tokenization timer
	tokenTimer := prometheus.NewTimer(TokenizationDuration.WithLabelValues(filepath.Base(c.config.ModelPath)))
	log.Debug().
		Str("content", content).
		Str("model", c.config.ModelPath).
		Msg("Starting tokenization")
	tokenIds, err := c.tokenizer.Tokenize(content)
	tokenTimer.ObserveDuration()

	if err != nil {
		log.Error().
			Err(err).
			Str("content", content).
			Msg("Tokenization failed")
		EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "tokenization_failed").Inc()
		return nil, fmt.Errorf("failed to tokenize input: %w", err)
	}

	log.Debug().
		Int("token_count", len(tokenIds)).
		Interface("token_ids", tokenIds).
		Msg("Tokenization complete")

	// Record token length
	TokenLength.WithLabelValues(filepath.Base(c.config.ModelPath)).Observe(float64(len(tokenIds)))

	// Create attention mask (1 for real tokens, 0 for padding)
	attentionMask := make([]int, c.config.MaxLength)
	for i := range tokenIds {
		attentionMask[i] = 1
	}

	// Convert to int64 for ONNX
	tokenIds64 := make([]int64, c.config.MaxLength)
	attentionMask64 := make([]int64, c.config.MaxLength)

	// Copy tokens into the padded array
	for i, id := range tokenIds {
		tokenIds64[i] = int64(id)
	}

	// Copy attention mask
	for i, mask := range attentionMask {
		attentionMask64[i] = int64(mask)
	}

	log.Debug().
		Int("max_length", c.config.MaxLength).
		Int("token_count", len(tokenIds)).
		Int("padded_length", len(tokenIds64)).
		Interface("first_tokens", tokenIds64[:min(5, len(tokenIds64))]).
		Interface("first_mask", attentionMask64[:min(5, len(attentionMask64))]).
		Msg("Prepared model inputs")

	// Prepare input tensors
	for i, tensor := range c.inputs {
		data := tensor.GetData()
		inputName := c.inputNames[i]
		switch inputName {
		case "input_ids":
			// Create a new slice with the same length as data
			copy(data, tokenIds64)
			log.Debug().
				Int("length", len(tokenIds64)).
				Interface("tokens", tokenIds64[:min(10, len(tokenIds64))]).
				Interface("original_tokens", tokenIds).
				Msg("Setting input_ids")
		case "attention_mask":
			copy(data, attentionMask64)
			log.Debug().
				Int("length", len(attentionMask64)).
				Interface("mask", attentionMask64[:min(10, len(attentionMask64))]).
				Msg("Setting attention_mask")
		case "token_type_ids":
			// All zeros for single sequence
			for j := range data {
				data[j] = 0
			}
			log.Debug().
				Int("length", len(data)).
				Msg("Setting token_type_ids")
		default:
			log.Warn().
				Str("name", inputName).
				Msg("Unknown input tensor name")
			// For any other input, use zeros
			for j := range data {
				data[j] = 0
			}
		}
	}

	// Start embedding timer
	timer := prometheus.NewTimer(EmbeddingRequestDuration.WithLabelValues(filepath.Base(c.config.ModelPath), "text"))
	defer timer.ObserveDuration()

	// Run inference
	err = c.model.Run()
	if err != nil {
		EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "inference_failed").Inc()
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Get output embedding and verify it's not all zeros
	outputData := c.outputs[0].GetData()

	// Calculate mean pooling over token embeddings
	var embedding []float32
	if len(tokenIds) > 0 {
		// Get the actual sequence length (number of non-zero attention mask values)
		seqLen := 0
		for _, mask := range attentionMask64 {
			if mask == 1 {
				seqLen++
			}
		}

		// Initialize embedding with zeros
		embedding = make([]float32, c.config.Dimension)

		// Sum token embeddings weighted by attention mask
		for i := 0; i < seqLen; i++ {
			for j := 0; j < c.config.Dimension; j++ {
				embedding[j] += outputData[i*c.config.Dimension+j]
			}
		}

		// Average by dividing by sequence length
		for j := range embedding {
			embedding[j] /= float32(seqLen)
		}

		// Normalize the embedding (L2 norm)
		norm := float32(0)
		for _, v := range embedding {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for i := range embedding {
				embedding[i] /= norm
			}
		}
	} else {
		// If no tokens, return zero vector
		embedding = make([]float32, c.config.Dimension)
	}

	// Check if output is all zeros
	allZeros := true
	for _, v := range embedding {
		if v != 0 {
			allZeros = false
			break
		}
	}
	if allZeros {
		log.Warn().
			Str("model", c.config.ModelPath).
			Int("dimension", c.config.Dimension).
			Interface("first_output", outputData[:min(10, len(outputData))]).
			Interface("first_tokens", tokenIds64[:min(10, len(tokenIds64))]).
			Interface("first_mask", attentionMask64[:min(10, len(attentionMask64))]).
			Msg("Model produced all zero embeddings")
	} else {
		log.Debug().
			Str("model", c.config.ModelPath).
			Int("dimension", c.config.Dimension).
			Interface("first_values", embedding[:5]).
			Float64("norm", vectorNorm(embedding)).
			Msg("Generated embedding")
	}

	// Increment success counter
	EmbeddingRequestsTotal.WithLabelValues(filepath.Base(c.config.ModelPath), "text").Inc()

	return embedding, nil
}

// BatchGenerateEmbeddings generates embeddings for multiple pieces of content
func (c *ONNXClient) BatchGenerateEmbeddings(ctx context.Context, contents []string) ([][]float32, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.model == nil {
		EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "model_not_loaded").Inc()
		return nil, fmt.Errorf("model not loaded")
	}

	// Record batch size
	EmbeddingBatchSize.WithLabelValues(filepath.Base(c.config.ModelPath)).Observe(float64(len(contents)))

	// Process each content sequentially for now
	// TODO: Implement true batching when ONNX Runtime supports dynamic batching
	results := make([][]float32, len(contents))
	for i, content := range contents {
		embedding, err := c.GenerateEmbedding(ctx, content)
		if err != nil {
			EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "batch_processing_failed").Inc()
			return nil, fmt.Errorf("failed to generate embedding for content %d: %w", i, err)
		}
		results[i] = embedding
	}

	return results, nil
}

// Health implements Service.Health
func (c *ONNXClient) Health(ctx context.Context) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.model == nil {
		EmbeddingErrors.WithLabelValues(filepath.Base(c.config.ModelPath), "health_check_failed").Inc()
		return fmt.Errorf("model not loaded")
	}

	// Update GPU memory usage if available
	if c.config.UseGPU {
		if usage, err := c.GetGPUMemoryUsage(); err == nil {
			GPUMemoryUsage.WithLabelValues(filepath.Base(c.config.ModelPath), "0").Set(float64(usage))
		}
	}

	return nil
}

// Close implements ModelClient
func (c *ONNXClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Debug().
		Bool("has_model", c.model != nil).
		Bool("has_inputs", c.inputs != nil).
		Bool("has_outputs", c.outputs != nil).
		Bool("has_tokenizer", c.tokenizer != nil).
		Msg("Starting client cleanup")

	var errs []error

	// Clean up model session
	if c.model != nil {
		if err := c.model.Destroy(); err != nil {
			log.Error().
				Err(err).
				Msg("Failed to destroy model session")
			errs = append(errs, fmt.Errorf("failed to destroy model session: %w", err))
		} else {
			log.Debug().Msg("Model session destroyed successfully")
		}
		c.model = nil
	}

	// Clean up input tensors
	if c.inputs != nil {
		for i, tensor := range c.inputs {
			if tensor != nil {
				if err := tensor.Destroy(); err != nil {
					log.Error().
						Err(err).
						Int("tensor_index", i).
						Msg("Failed to destroy input tensor")
					errs = append(errs, fmt.Errorf("failed to destroy input tensor %d: %w", i, err))
				}
			}
		}
		c.inputs = nil
		log.Debug().Msg("Input tensors cleaned up")
	}

	// Clean up output tensors
	if c.outputs != nil {
		for i, tensor := range c.outputs {
			if tensor != nil {
				if err := tensor.Destroy(); err != nil {
					log.Error().
						Err(err).
						Int("tensor_index", i).
						Msg("Failed to destroy output tensor")
					errs = append(errs, fmt.Errorf("failed to destroy output tensor %d: %w", i, err))
				}
			}
		}
		c.outputs = nil
		log.Debug().Msg("Output tensors cleaned up")
	}

	// Clean up tokenizer
	if c.tokenizer != nil {
		if err := c.tokenizer.Close(); err != nil {
			log.Error().
				Err(err).
				Msg("Failed to close tokenizer")
			errs = append(errs, fmt.Errorf("failed to close tokenizer: %w", err))
		} else {
			log.Debug().Msg("Tokenizer closed successfully")
		}
		c.tokenizer = nil
	}

	if len(errs) > 0 {
		return fmt.Errorf("errors during cleanup: %v", errs)
	}

	log.Debug().Msg("Client cleanup completed successfully")
	return nil
}

// GetModelInfo returns information about the loaded model
func (c *ONNXClient) GetModelInfo() (*ModelConfig, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.model == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	return &c.config, nil
}

// GetGPUMemoryUsage returns the current GPU memory usage in bytes
func (c *ONNXClient) GetGPUMemoryUsage() (uint64, error) {
	// TODO: Implement GPU memory usage tracking
	return 0, nil
}

// min returns the smaller of x or y
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// vectorNorm calculates the L2 norm of a vector
func vectorNorm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x * x)
	}
	return math.Sqrt(sum)
}
