package embeddings

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ONNXService implements the Service interface using local ONNX models
type ONNXService struct {
	mu      sync.RWMutex
	config  Config
	clients map[ModelType]*ONNXClient
}

// NewONNXService creates a new ONNX-based embedding service
func NewONNXService(cfg Config) (*ONNXService, error) {
	if len(cfg.Models) == 0 {
		return nil, fmt.Errorf("no models configured")
	}

	service := &ONNXService{
		config:  cfg,
		clients: make(map[ModelType]*ONNXClient),
	}

	// Initialize each model
	for modelType, modelConfig := range cfg.Models {
		client, err := NewONNXClient(modelConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create client for model %s: %w", modelType, err)
		}

		if err := client.LoadModel(modelConfig.ModelPath); err != nil {
			return nil, fmt.Errorf("failed to load model %s: %w", modelType, err)
		}

		service.clients[modelType] = client
	}

	return service, nil
}

// Generate implements Service.Generate
func (s *ONNXService) Generate(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Determine which model to use
	modelType := req.Model
	if modelType == "" {
		var ok bool
		modelType, ok = s.config.ContentTypeModels[req.ContentType]
		if !ok {
			return nil, fmt.Errorf("no model mapped for content type %s", req.ContentType)
		}
	}

	client, ok := s.clients[modelType]
	if !ok {
		return nil, fmt.Errorf("model %s not found", modelType)
	}

	// Set timeout if configured
	if s.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.config.Timeout)
		defer cancel()
	}

	start := time.Now()
	vector, err := client.GenerateEmbedding(ctx, req.Content)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	return &EmbeddingResult{
		Vector:    vector,
		Model:     modelType,
		Dimension: len(vector),
		Duration:  time.Since(start),
		Metadata:  req.Metadata,
	}, nil
}

// BatchGenerate implements Service.BatchGenerate
func (s *ONNXService) BatchGenerate(ctx context.Context, reqs []*EmbeddingRequest) ([]*EmbeddingResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(reqs) == 0 {
		return nil, nil
	}

	// Validate requests
	for i, req := range reqs {
		if req == nil {
			return nil, fmt.Errorf("request at index %d is nil", i)
		}
		if req.Content == "" {
			return nil, fmt.Errorf("request at index %d has empty content", i)
		}
	}

	// Group requests by model and track original indices
	type indexedRequest struct {
		originalIndex int
		request       *EmbeddingRequest
	}
	requestsByModel := make(map[ModelType][]indexedRequest)
	for i, req := range reqs {
		modelType := req.Model
		if modelType == "" {
			var ok bool
			modelType, ok = s.config.ContentTypeModels[req.ContentType]
			if !ok {
				return nil, fmt.Errorf("no model mapped for content type %s", req.ContentType)
			}
		}
		requestsByModel[modelType] = append(requestsByModel[modelType], indexedRequest{i, req})
	}

	// Process each group in parallel
	var wg sync.WaitGroup
	results := make([]*EmbeddingResult, len(reqs))
	errChan := make(chan error, len(reqs))

	for modelType, modelReqs := range requestsByModel {
		client, ok := s.clients[modelType]
		if !ok {
			return nil, fmt.Errorf("model %s not found", modelType)
		}

		wg.Add(1)
		go func(mt ModelType, reqs []indexedRequest, c *ONNXClient) {
			defer wg.Done()

			contents := make([]string, len(reqs))
			for i, req := range reqs {
				contents[i] = req.request.Content
			}

			start := time.Now()
			vectors, err := c.BatchGenerateEmbeddings(ctx, contents)
			if err != nil {
				errChan <- fmt.Errorf("failed to generate embeddings for model %s: %w", mt, err)
				return
			}

			duration := time.Since(start)
			for i, vector := range vectors {
				results[reqs[i].originalIndex] = &EmbeddingResult{
					Vector:    vector,
					Model:     mt,
					Dimension: len(vector),
					Duration:  duration,
					Metadata:  reqs[i].request.Metadata,
				}
			}
		}(modelType, modelReqs, client)
	}

	// Wait for all goroutines to complete
	wg.Wait()
	close(errChan)

	// Check for all errors
	var errs []error
	for err := range errChan {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return nil, fmt.Errorf("batch processing errors: %v", errs)
	}

	// Verify all results are populated
	for i, result := range results {
		if result == nil {
			return nil, fmt.Errorf("missing result for request at index %d", i)
		}
	}

	return results, nil
}

// GetModelInfo implements Service.GetModelInfo
func (s *ONNXService) GetModelInfo(modelType ModelType) (*ModelConfig, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	cfg, ok := s.config.Models[modelType]
	if !ok {
		return nil, ErrModelNotFound
	}

	return &cfg, nil
}

// Health implements Service.Health
func (s *ONNXService) Health(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check if service is closed
	if s.clients == nil {
		return fmt.Errorf("service is closed")
	}

	// Check if any clients exist
	if len(s.clients) == 0 {
		return fmt.Errorf("no models loaded")
	}

	// Check health of each client
	for modelType, client := range s.clients {
		if client == nil {
			return fmt.Errorf("model %s client is nil", modelType)
		}
		if err := client.Health(ctx); err != nil {
			return fmt.Errorf("model %s is unhealthy: %w", modelType, err)
		}
	}

	return nil
}

// Close implements Service.Close
func (s *ONNXService) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error

	// Clean up all clients
	for modelType, client := range s.clients {
		if client != nil {
			if err := client.Close(); err != nil {
				errs = append(errs, fmt.Errorf("failed to close model %s: %w", modelType, err))
			}
		}
	}

	// Clear the clients map
	s.clients = nil

	if len(errs) > 0 {
		return fmt.Errorf("errors during service cleanup: %v", errs)
	}

	return nil
}
