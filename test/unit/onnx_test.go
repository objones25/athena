package unit

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog/log"
	"github.com/stretchr/testify/require"

	"github.com/objones25/athena/internal/embeddings"
)

func getTestConfig(t *testing.T) embeddings.Config {
	// Get models directory
	modelsDir := os.Getenv("TEST_MODELS_DIR")
	if modelsDir == "" {
		// Default to project root models directory
		rootDir := filepath.Join("..", "..")
		modelsDir = filepath.Join(rootDir, "models")
	}

	// Create test configuration
	cfg := embeddings.Config{
		BatchSize: 32,
		Models: map[embeddings.ModelType]embeddings.ModelConfig{
			embeddings.ModelMiniLML6V2: {
				ModelPath:           filepath.Join(modelsDir, "all-MiniLM-L6-v2.onnx"),
				TokenizerConfigPath: filepath.Join(modelsDir, "all-MiniLM-L6-v2_tokenizer"),
				Dimension:           384,
				MaxLength:           512,
				UseGPU:              true,
			},
			embeddings.ModelMPNetBase: {
				ModelPath:           filepath.Join(modelsDir, "all-mpnet-base-v2.onnx"),
				TokenizerConfigPath: filepath.Join(modelsDir, "all-mpnet-base-v2_tokenizer"),
				Dimension:           768,
				MaxLength:           512,
				UseGPU:              true,
			},
			embeddings.ModelMultilingualLM: {
				ModelPath:           filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2.onnx"),
				TokenizerConfigPath: filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2_tokenizer"),
				Dimension:           384,
				MaxLength:           512,
				UseGPU:              true,
			},
		},
		ContentTypeModels: map[embeddings.ContentType]embeddings.ModelType{
			embeddings.ContentTypeCode: embeddings.ModelMPNetBase,
			embeddings.ContentTypeText: embeddings.ModelMultilingualLM,
			embeddings.ContentTypeMath: embeddings.ModelMiniLML6V2,
		},
	}

	// Skip if model files don't exist
	for modelType, modelCfg := range cfg.Models {
		if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
			t.Skipf("Model %s file not found at %s, skipping test", modelType, modelCfg.ModelPath)
		}
		if _, err := os.Stat(modelCfg.TokenizerConfigPath); os.IsNotExist(err) {
			t.Skipf("Model %s tokenizer config not found at %s, skipping test", modelType, modelCfg.TokenizerConfigPath)
		}
	}

	return cfg
}

func TestONNXService(t *testing.T) {
	modelsDir := os.Getenv("TEST_MODELS_DIR")
	if modelsDir == "" {
		t.Skip("TEST_MODELS_DIR not set")
	}

	testConfigs := map[string]embeddings.ModelConfig{
		"all-MiniLM-L6-v2": {
			ModelPath:           filepath.Join(modelsDir, "all-MiniLM-L6-v2.onnx"),
			TokenizerConfigPath: filepath.Join(modelsDir, "all-MiniLM-L6-v2_tokenizer"),
			Dimension:           384,
			MaxLength:           512,
			UseGPU:              false,
		},
		"all-mpnet-base-v2": {
			ModelPath:           filepath.Join(modelsDir, "all-mpnet-base-v2.onnx"),
			TokenizerConfigPath: filepath.Join(modelsDir, "all-mpnet-base-v2_tokenizer"),
			Dimension:           768,
			MaxLength:           512,
			UseGPU:              false,
		},
		"paraphrase-multilingual-MiniLM-L12-v2": {
			ModelPath:           filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2.onnx"),
			TokenizerConfigPath: filepath.Join(modelsDir, "paraphrase-multilingual-MiniLM-L12-v2_tokenizer"),
			Dimension:           384,
			MaxLength:           512,
			UseGPU:              false,
		},
	}

	log.Debug().
		Int("pid", os.Getpid()).
		Msg("Starting ONNX client test")

	t.Run("Client Initialization", func(t *testing.T) {
		for modelName, modelCfg := range testConfigs {
			t.Run(modelName, func(t *testing.T) {
				// Check if model exists
				if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
					t.Skipf("Model %s not found at %s, skipping test", modelName, modelCfg.ModelPath)
				}

				// Create client
				client, err := embeddings.NewONNXClient(modelCfg)
				require.NoError(t, err)
				defer client.Close()

				// Initialize client
				err = client.Initialize(context.Background())
				require.NoError(t, err)
			})
		}
	})

	t.Run("Single Embedding", func(t *testing.T) {
		for modelName, modelCfg := range testConfigs {
			t.Run(modelName, func(t *testing.T) {
				// Check if model exists
				if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
					t.Skipf("Model %s not found at %s, skipping test", modelName, modelCfg.ModelPath)
				}

				// Create client
				client, err := embeddings.NewONNXClient(modelCfg)
				require.NoError(t, err)
				defer client.Close()

				// Initialize client
				err = client.Initialize(context.Background())
				require.NoError(t, err)

				testCases := []struct {
					name    string
					input   string
					wantErr bool
				}{
					{
						name:    "Code Content",
						input:   "func main() { fmt.Println(\"Hello World\") }",
						wantErr: false,
					},
					{
						name:    "Multilingual Text",
						input:   "Hello World! ¡Hola Mundo! Bonjour le monde!",
						wantErr: false,
					},
					{
						name:    "Math Content",
						input:   "E = mc^2 and the Pythagorean theorem states that a^2 + b^2 = c^2",
						wantErr: false,
					},
					{
						name:    "Empty Content",
						input:   "",
						wantErr: true,
					},
				}

				for _, tc := range testCases {
					t.Run(tc.name, func(t *testing.T) {
						embedding, err := client.GenerateEmbedding(context.Background(), tc.input)
						if tc.wantErr {
							require.Error(t, err)
							return
						}
						require.NoError(t, err)
						require.NotNil(t, embedding)
						require.Equal(t, modelCfg.Dimension, len(embedding))
					})
				}
			})
		}
	})

	t.Run("Batch Embedding", func(t *testing.T) {
		for modelName, modelCfg := range testConfigs {
			t.Run(modelName, func(t *testing.T) {
				// Check if model exists
				if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
					t.Skipf("Model %s not found at %s, skipping test", modelName, modelCfg.ModelPath)
				}

				// Create client
				client, err := embeddings.NewONNXClient(modelCfg)
				require.NoError(t, err)
				defer client.Close()

				// Initialize client
				err = client.Initialize(context.Background())
				require.NoError(t, err)

				inputs := []string{
					"func main() { fmt.Println(\"Hello World\") }",
					"Hello World! ¡Hola Mundo! Bonjour le monde!",
					"E = mc^2 and the Pythagorean theorem states that a^2 + b^2 = c^2",
				}

				embeddings, err := client.BatchGenerateEmbeddings(context.Background(), inputs)
				require.NoError(t, err)
				require.NotNil(t, embeddings)
				require.Equal(t, len(inputs), len(embeddings))

				for _, embedding := range embeddings {
					require.Equal(t, modelCfg.Dimension, len(embedding))
				}
			})
		}
	})

	t.Run("Model Info", func(t *testing.T) {
		for modelName, modelCfg := range testConfigs {
			t.Run(modelName, func(t *testing.T) {
				// Check if model exists
				if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
					t.Skipf("Model %s not found at %s, skipping test", modelName, modelCfg.ModelPath)
				}

				// Create client
				client, err := embeddings.NewONNXClient(modelCfg)
				require.NoError(t, err)
				defer client.Close()

				// Initialize client
				err = client.Initialize(context.Background())
				require.NoError(t, err)

				info, err := client.GetModelInfo()
				require.NoError(t, err)
				require.NotNil(t, info)
				require.Equal(t, modelCfg.ModelPath, info.ModelPath)
				require.Equal(t, modelCfg.Dimension, info.Dimension)
				require.Equal(t, modelCfg.MaxLength, info.MaxLength)
			})
		}
	})

	t.Run("Health Check", func(t *testing.T) {
		for modelName, modelCfg := range testConfigs {
			t.Run(modelName, func(t *testing.T) {
				// Check if model exists
				if _, err := os.Stat(modelCfg.ModelPath); os.IsNotExist(err) {
					t.Skipf("Model %s not found at %s, skipping test", modelName, modelCfg.ModelPath)
				}

				// Create client
				client, err := embeddings.NewONNXClient(modelCfg)
				require.NoError(t, err)
				defer client.Close()

				// Initialize client
				err = client.Initialize(context.Background())
				require.NoError(t, err)

				err = client.Health(context.Background())
				require.NoError(t, err)
			})
		}
	})
}
