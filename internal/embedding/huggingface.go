package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	huggingFaceAPIURL = "https://api-inference.huggingface.co/pipeline/feature-extraction/%s"
	defaultTimeout    = 30 // seconds
	defaultBatchSize  = 32
)

// HuggingFaceService implements the Service interface using HuggingFace's API
type HuggingFaceService struct {
	config       Config
	httpClient   *http.Client
	preprocessor *TextPreprocessor
	mu           sync.RWMutex
}

// NewHuggingFaceService creates a new instance of HuggingFaceService
func NewHuggingFaceService(cfg Config) (*HuggingFaceService, error) {
	// If no config provided, try to load from environment
	if cfg.APIKey == "" {
		loadedCfg, err := LoadConfig()
		if err != nil {
			return nil, err
		}
		cfg = *loadedCfg
	}

	// Set defaults if not provided
	if cfg.DefaultModel == "" {
		cfg.DefaultModel = ModelMPNetBaseV2
	}

	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultTimeout
	}

	if cfg.MaxBatchSize <= 0 {
		cfg.MaxBatchSize = defaultBatchSize
	}

	return &HuggingFaceService{
		config: cfg,
		httpClient: &http.Client{
			Timeout: time.Duration(cfg.Timeout) * time.Second,
		},
		preprocessor: NewDefaultPreprocessor(),
	}, nil
}

// Embed implements the Service interface
func (s *HuggingFaceService) Embed(ctx context.Context, text string) (*EmbeddingResult, error) {
	return s.EmbedWithModel(ctx, text, s.config.DefaultModel)
}

// EmbedBatch implements the Service interface with optimized parallel processing
func (s *HuggingFaceService) EmbedBatch(ctx context.Context, texts []string) ([]*EmbeddingResult, error) {
	if len(texts) == 0 {
		return nil, ErrInvalidInput
	}

	// For small batches, just use the direct API call
	if len(texts) <= s.config.MaxBatchSize {
		return s.processSingleBatch(ctx, texts)
	}

	return s.processParallelBatches(ctx, texts)
}

// processSingleBatch handles a single batch of texts
func (s *HuggingFaceService) processSingleBatch(ctx context.Context, texts []string) ([]*EmbeddingResult, error) {
	// Preprocess all texts in the batch
	texts = s.preprocessor.ProcessBatch(texts)

	// Filter out any empty strings after preprocessing
	validTexts := make([]string, 0, len(texts))
	for _, text := range texts {
		if text != "" {
			validTexts = append(validTexts, text)
		}
	}

	if len(validTexts) == 0 {
		return nil, ErrInvalidInput
	}

	embeddings, err := s.callAPI(ctx, validTexts, s.config.DefaultModel)
	if err != nil {
		return nil, fmt.Errorf("batch embedding failed: %w", err)
	}

	results := make([]*EmbeddingResult, len(embeddings))
	for i, embedding := range embeddings {
		results[i] = &EmbeddingResult{
			Vector: embedding,
			Model:  s.config.DefaultModel,
		}
	}

	return results, nil
}

// processParallelBatches processes large batches in parallel
func (s *HuggingFaceService) processParallelBatches(ctx context.Context, texts []string) ([]*EmbeddingResult, error) {
	// Filter out empty strings first
	validTexts := make([]string, 0, len(texts))
	for _, text := range texts {
		if text != "" {
			validTexts = append(validTexts, text)
		}
	}

	if len(validTexts) == 0 {
		return nil, ErrInvalidInput
	}

	numBatches := (len(validTexts) + s.config.MaxBatchSize - 1) / s.config.MaxBatchSize
	results := make([]*EmbeddingResult, len(validTexts))
	errChan := make(chan error, numBatches)
	var wg sync.WaitGroup

	// Process batches in parallel
	for i := 0; i < len(validTexts); i += s.config.MaxBatchSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()

			end := start + s.config.MaxBatchSize
			if end > len(validTexts) {
				end = len(validTexts)
			}

			batch := validTexts[start:end]
			batchResults, err := s.processSingleBatch(ctx, batch)
			if err != nil {
				errChan <- fmt.Errorf("batch %d-%d failed: %w", start, end, err)
				return
			}

			// Copy results to the correct positions
			for j, result := range batchResults {
				results[start+j] = result
			}
		}(i)
	}

	// Wait for all batches to complete
	wg.Wait()
	close(errChan)

	// Check for any errors
	if len(errChan) > 0 {
		var errMsgs []string
		for err := range errChan {
			errMsgs = append(errMsgs, err.Error())
		}
		return nil, fmt.Errorf("batch processing errors: %s", strings.Join(errMsgs, "; "))
	}

	return results, nil
}

// EmbedWithModel implements the Service interface
func (s *HuggingFaceService) EmbedWithModel(ctx context.Context, text string, model ModelType) (*EmbeddingResult, error) {
	if text == "" {
		return nil, ErrInvalidInput
	}

	// Preprocess the text
	text = s.preprocessor.Process(text)
	if text == "" {
		return nil, ErrInvalidInput
	}

	embeddings, err := s.callAPI(ctx, []string{text}, model)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	if len(embeddings) == 0 {
		return nil, ErrEmbeddingFailed
	}

	return &EmbeddingResult{
		Vector: embeddings[0],
		Model:  model,
	}, nil
}

// GetSupportedModels implements the Service interface
func (s *HuggingFaceService) GetSupportedModels() []ModelType {
	return []ModelType{
		ModelMPNetBaseV2,
		ModelMiniLML6V2,
		ModelRobertaLargeV1,
	}
}

// GetConfig returns a copy of the current configuration
func (s *HuggingFaceService) GetConfig() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.config
}

// callAPI makes the actual API call to HuggingFace
func (s *HuggingFaceService) callAPI(ctx context.Context, texts []string, model ModelType) ([][]float32, error) {
	url := fmt.Sprintf(huggingFaceAPIURL, model)

	body, err := json.Marshal(map[string]interface{}{
		"inputs": texts,
		"options": map[string]interface{}{
			"wait_for_model": true,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", s.config.APIKey))
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API returned error status %d: %s", resp.StatusCode, string(body))
	}

	var embeddings [][]float32
	if err := json.NewDecoder(resp.Body).Decode(&embeddings); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return embeddings, nil
}
