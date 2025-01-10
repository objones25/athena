package unit

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/objones25/athena/internal/embedding"
)

func TestLoadConfig(t *testing.T) {
	// Save original env var and restore after test
	originalKey := os.Getenv("HUGGINGFACE_API_KEY")
	defer os.Setenv("HUGGINGFACE_API_KEY", originalKey)

	tests := []struct {
		name        string
		setupEnv    func()
		wantErr     bool
		wantErrType error
		checkConfig func(*testing.T, *embedding.Config)
	}{
		{
			name: "valid config from environment",
			setupEnv: func() {
				os.Setenv("HUGGINGFACE_API_KEY", "test-key")
			},
			wantErr: false,
			checkConfig: func(t *testing.T, cfg *embedding.Config) {
				assert.Equal(t, "test-key", cfg.APIKey)
				assert.Equal(t, embedding.ModelMPNetBaseV2, cfg.DefaultModel)
				assert.Equal(t, 32, cfg.MaxBatchSize)
				assert.Equal(t, 30, cfg.Timeout)
			},
		},
		{
			name: "missing API key",
			setupEnv: func() {
				os.Unsetenv("HUGGINGFACE_API_KEY")
			},
			wantErr:     true,
			wantErrType: embedding.ErrAPIKeyNotSet,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupEnv()

			cfg, err := embedding.LoadConfig()
			if tt.wantErr {
				assert.Error(t, err)
				if tt.wantErrType != nil {
					assert.ErrorIs(t, err, tt.wantErrType)
				}
				return
			}

			require.NoError(t, err)
			require.NotNil(t, cfg)
			if tt.checkConfig != nil {
				tt.checkConfig(t, cfg)
			}
		})
	}
}

func TestNewHuggingFaceService(t *testing.T) {
	// Save original env var and restore after test
	originalKey := os.Getenv("HUGGINGFACE_API_KEY")
	defer os.Setenv("HUGGINGFACE_API_KEY", originalKey)

	tests := []struct {
		name         string
		setupEnv     func()
		config       embedding.Config
		wantErr      bool
		wantErrType  error
		checkService func(*testing.T, *embedding.HuggingFaceService)
	}{
		{
			name: "explicit config",
			setupEnv: func() {
				os.Unsetenv("HUGGINGFACE_API_KEY")
			},
			config: embedding.Config{
				APIKey:       "explicit-key",
				DefaultModel: embedding.ModelMiniLML6V2,
				MaxBatchSize: 64,
				Timeout:      60,
			},
			wantErr: false,
			checkService: func(t *testing.T, s *embedding.HuggingFaceService) {
				cfg := s.GetConfig()
				assert.Equal(t, "explicit-key", cfg.APIKey)
				assert.Equal(t, embedding.ModelMiniLML6V2, cfg.DefaultModel)
				assert.Equal(t, 64, cfg.MaxBatchSize)
				assert.Equal(t, 60, cfg.Timeout)
			},
		},
		{
			name: "load from environment",
			setupEnv: func() {
				os.Setenv("HUGGINGFACE_API_KEY", "env-key")
			},
			config:  embedding.Config{},
			wantErr: false,
			checkService: func(t *testing.T, s *embedding.HuggingFaceService) {
				cfg := s.GetConfig()
				assert.Equal(t, "env-key", cfg.APIKey)
				assert.Equal(t, embedding.ModelMPNetBaseV2, cfg.DefaultModel)
				assert.Equal(t, 32, cfg.MaxBatchSize)
				assert.Equal(t, 30, cfg.Timeout)
			},
		},
		{
			name: "missing API key",
			setupEnv: func() {
				os.Unsetenv("HUGGINGFACE_API_KEY")
			},
			config:      embedding.Config{},
			wantErr:     true,
			wantErrType: embedding.ErrAPIKeyNotSet,
		},
		{
			name: "override defaults",
			setupEnv: func() {
				os.Setenv("HUGGINGFACE_API_KEY", "env-key")
			},
			config: embedding.Config{
				DefaultModel: embedding.ModelRobertaLargeV1,
				MaxBatchSize: 128,
				Timeout:      45,
			},
			wantErr: false,
			checkService: func(t *testing.T, s *embedding.HuggingFaceService) {
				cfg := s.GetConfig()
				assert.Equal(t, "env-key", cfg.APIKey)
				assert.Equal(t, embedding.ModelRobertaLargeV1, cfg.DefaultModel)
				assert.Equal(t, 128, cfg.MaxBatchSize)
				assert.Equal(t, 45, cfg.Timeout)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setupEnv()

			service, err := embedding.NewHuggingFaceService(tt.config)
			if tt.wantErr {
				assert.Error(t, err)
				if tt.wantErrType != nil {
					assert.ErrorIs(t, err, tt.wantErrType)
				}
				return
			}

			require.NoError(t, err)
			require.NotNil(t, service)
			if tt.checkService != nil {
				tt.checkService(t, service)
			}
		})
	}
}

func TestHuggingFaceService_GetSupportedModels(t *testing.T) {
	service, err := embedding.NewHuggingFaceService(embedding.Config{APIKey: "test-key"})
	require.NoError(t, err)

	models := service.GetSupportedModels()
	assert.Len(t, models, 3)
	assert.Contains(t, models, embedding.ModelMPNetBaseV2)
	assert.Contains(t, models, embedding.ModelMiniLML6V2)
	assert.Contains(t, models, embedding.ModelRobertaLargeV1)
}

func TestHuggingFaceService_Embed(t *testing.T) {
	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: HUGGINGFACE_API_KEY not set")
	}

	ctx := context.Background()
	service, err := embedding.NewHuggingFaceService(embedding.Config{
		APIKey:       apiKey,
		DefaultModel: embedding.ModelMPNetBaseV2,
		Timeout:      30,
	})
	require.NoError(t, err)

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "valid input",
			input:   "This is a test sentence for embedding.",
			wantErr: false,
		},
		{
			name:    "empty input",
			input:   "",
			wantErr: true,
		},
		{
			name:    "very long input",
			input:   "This is a very long sentence " + strings.Repeat("that repeats ", 100),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := service.Embed(ctx, tt.input)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			assert.NotEmpty(t, result.Vector)
			assert.Equal(t, embedding.ModelMPNetBaseV2, result.Model)
		})
	}
}

func TestHuggingFaceService_EmbedBatch(t *testing.T) {
	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: HUGGINGFACE_API_KEY not set")
	}

	ctx := context.Background()
	service, err := embedding.NewHuggingFaceService(embedding.Config{
		APIKey:       apiKey,
		DefaultModel: embedding.ModelMPNetBaseV2,
		MaxBatchSize: 2, // Small batch size to test batching
		Timeout:      30,
	})
	require.NoError(t, err)

	tests := []struct {
		name          string
		inputs        []string
		wantErr       bool
		expectedCount int // Expected number of results after filtering empty strings
	}{
		{
			name: "small batch",
			inputs: []string{
				"First test sentence.",
				"Second test sentence.",
			},
			wantErr:       false,
			expectedCount: 2,
		},
		{
			name: "large batch",
			inputs: []string{
				"First sentence.",
				"Second sentence.",
				"Third sentence.",
				"Fourth sentence.",
			},
			wantErr:       false,
			expectedCount: 4,
		},
		{
			name:          "empty batch",
			inputs:        []string{},
			wantErr:       true,
			expectedCount: 0,
		},
		{
			name: "batch with empty string",
			inputs: []string{
				"Valid sentence",
				"",
				"Another valid sentence",
			},
			wantErr:       false,
			expectedCount: 2, // Only non-empty strings should be processed
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := service.EmbedBatch(ctx, tt.inputs)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, results)
			assert.Len(t, results, tt.expectedCount, "unexpected number of results")

			for _, result := range results {
				assert.NotEmpty(t, result.Vector)
				assert.Equal(t, embedding.ModelMPNetBaseV2, result.Model)
			}
		})
	}
}

func TestHuggingFaceService_EmbedWithModel(t *testing.T) {
	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: HUGGINGFACE_API_KEY not set")
	}

	ctx := context.Background()
	service, err := embedding.NewHuggingFaceService(embedding.Config{
		APIKey:  apiKey,
		Timeout: 30,
	})
	require.NoError(t, err)

	tests := []struct {
		name    string
		input   string
		model   embedding.ModelType
		wantErr bool
	}{
		{
			name:    "MPNet model",
			input:   "Test sentence for MPNet.",
			model:   embedding.ModelMPNetBaseV2,
			wantErr: false,
		},
		{
			name:    "MiniLM model",
			input:   "Test sentence for MiniLM.",
			model:   embedding.ModelMiniLML6V2,
			wantErr: false,
		},
		{
			name:    "invalid model",
			input:   "Test sentence.",
			model:   "invalid-model",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := service.EmbedWithModel(ctx, tt.input, tt.model)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, result)
			assert.NotEmpty(t, result.Vector)
			assert.Equal(t, tt.model, result.Model)
		})
	}
}

func TestHuggingFaceService_Concurrent(t *testing.T) {
	apiKey := os.Getenv("HUGGINGFACE_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: HUGGINGFACE_API_KEY not set")
	}

	ctx := context.Background()
	service, err := embedding.NewHuggingFaceService(embedding.Config{
		APIKey:       apiKey,
		DefaultModel: embedding.ModelMPNetBaseV2,
		Timeout:      30,
	})
	require.NoError(t, err)

	// Run multiple embedding requests concurrently
	concurrentRequests := 5
	done := make(chan bool)
	errors := make(chan error)

	for i := 0; i < concurrentRequests; i++ {
		go func(i int) {
			_, err := service.Embed(ctx, "Concurrent test sentence.")
			if err != nil {
				errors <- err
				return
			}
			done <- true
		}(i)
	}

	// Wait for all requests to complete
	timeout := time.After(time.Second * 30)
	for i := 0; i < concurrentRequests; i++ {
		select {
		case err := <-errors:
			t.Errorf("Concurrent request failed: %v", err)
		case <-done:
			// Request completed successfully
		case <-timeout:
			t.Error("Timeout waiting for concurrent requests")
			return
		}
	}
}
