package unit

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/objones25/athena/internal/embedding"
	"github.com/objones25/athena/internal/storage"
	"github.com/objones25/athena/internal/storage/milvus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockEmbeddingService implements embedding.Service for testing
type mockEmbeddingService struct {
	dimension int
}

// normalizeVector normalizes a vector to unit length
func normalizeVector(vector []float32) []float32 {
	var sum float32
	for _, v := range vector {
		sum += v * v
	}
	magnitude := float32(math.Sqrt(float64(sum)))

	normalized := make([]float32, len(vector))
	if magnitude > 0 {
		for i, v := range vector {
			normalized[i] = v / magnitude
		}
	}
	return normalized
}

func (m *mockEmbeddingService) Embed(ctx context.Context, text string) (*embedding.EmbeddingResult, error) {
	// Generate a mock vector of the correct dimension
	vector := make([]float32, m.dimension)
	for i := range vector {
		vector[i] = float32(i % 10) // Use smaller numbers that repeat
	}

	// Normalize the vector before returning
	normalizedVector := normalizeVector(vector)

	return &embedding.EmbeddingResult{
		Vector: normalizedVector,
		Model:  embedding.ModelMPNetBaseV2,
	}, nil
}

func (m *mockEmbeddingService) EmbedBatch(ctx context.Context, texts []string) ([]*embedding.EmbeddingResult, error) {
	results := make([]*embedding.EmbeddingResult, len(texts))
	for i := range texts {
		vector := make([]float32, m.dimension)
		// Generate distinct vectors for each item
		for j := range vector {
			// Use a unique pattern for each item
			switch i {
			case 0:
				// First item: increasing values
				vector[j] = float32(j % 10)
			case 1:
				// Second item: decreasing values
				vector[j] = float32(9 - (j % 10))
			case 2:
				// Third item: alternating values
				vector[j] = float32(j % 2 * 9)
			case 3:
				// Fourth item: step pattern
				vector[j] = float32((j % 3) * 3)
			case 4:
				// Fifth item: wave pattern
				vector[j] = float32(4 + (j%2)*2)
			default:
				// Other items: unique offset pattern
				vector[j] = float32(i + (j % 5))
			}
		}

		// Normalize the vector before returning
		normalizedVector := normalizeVector(vector)

		results[i] = &embedding.EmbeddingResult{
			Vector: normalizedVector,
			Model:  embedding.ModelMPNetBaseV2,
		}
	}
	return results, nil
}

func (m *mockEmbeddingService) EmbedWithModel(ctx context.Context, text string, model embedding.ModelType) (*embedding.EmbeddingResult, error) {
	return m.Embed(ctx, text)
}

func (m *mockEmbeddingService) GetSupportedModels() []embedding.ModelType {
	return []embedding.ModelType{embedding.ModelMPNetBaseV2}
}

func (m *mockEmbeddingService) GetConfig() embedding.Config {
	return embedding.Config{
		DefaultModel: embedding.ModelMPNetBaseV2,
		MaxBatchSize: 32,
		Timeout:      30,
	}
}

func TestMilvusStore_InsertWithEmbeddings(t *testing.T) {
	t.Log("Starting TestMilvusStore_InsertWithEmbeddings")
	// Create a mock embedding service
	mockEmbedder := &mockEmbeddingService{dimension: 1536}
	t.Log("Created mock embedder")

	// Create store config
	cfg := milvus.Config{
		Host:             "localhost",
		Port:             19530,
		CollectionName:   "test_collection",
		Dimension:        1536,
		BatchSize:        100,
		MaxRetries:       3,
		PoolSize:         5,
		EmbeddingService: mockEmbedder,
	}
	t.Log("Created store config")

	// Create store
	t.Log("Creating new MilvusStore...")
	store, err := milvus.NewMilvusStore(cfg)
	if err != nil {
		t.Fatalf("Failed to create store: %v", err)
	}
	t.Log("Successfully created MilvusStore")
	defer store.Close()

	ctx := context.Background()

	tests := []struct {
		name        string
		items       []*storage.Item
		expectError bool
	}{
		{
			name: "Insert text items",
			items: []*storage.Item{
				{
					ID: "1",
					Content: storage.Content{
						Type: storage.ContentTypeText,
						Data: []byte("test text 1"),
					},
					Metadata:  map[string]interface{}{"key": "value"},
					CreatedAt: time.Now(),
				},
				{
					ID: "2",
					Content: storage.Content{
						Type: storage.ContentTypeText,
						Data: []byte("test text 2"),
					},
					Metadata:  map[string]interface{}{"key": "value2"},
					CreatedAt: time.Now(),
				},
			},
			expectError: false,
		},
		{
			name: "Insert code items",
			items: []*storage.Item{
				{
					ID: "3",
					Content: storage.Content{
						Type: storage.ContentTypeCode,
						Data: []byte("func test() {}"),
					},
					Metadata:  map[string]interface{}{"language": "go"},
					CreatedAt: time.Now(),
				},
			},
			expectError: false,
		},
		{
			name: "Insert mixed content types",
			items: []*storage.Item{
				{
					ID: "4",
					Content: storage.Content{
						Type: storage.ContentTypeText,
						Data: []byte("text content"),
					},
					CreatedAt: time.Now(),
				},
				{
					ID: "5",
					Content: storage.Content{
						Type: storage.ContentTypeCode,
						Data: []byte("code content"),
					},
					CreatedAt: time.Now(),
				},
				{
					ID: "6",
					Content: storage.Content{
						Type: storage.ContentTypeMarkdown,
						Data: []byte("# markdown content"),
					},
					CreatedAt: time.Now(),
				},
			},
			expectError: false,
		},
		{
			name:        "Insert empty batch",
			items:       []*storage.Item{},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Logf("Starting subtest: %s", tt.name)
			t.Logf("Inserting %d items", len(tt.items))

			err := store.Insert(ctx, tt.items)
			if tt.expectError {
				assert.Error(t, err)
				t.Log("Expected error received")
				return
			}
			if err != nil {
				t.Fatalf("Unexpected error during insert: %v", err)
			}
			t.Log("Insert completed successfully")

			// Verify vectors were generated
			for i, item := range tt.items {
				t.Logf("Verifying vector for item %d", i)
				assert.NotNil(t, item.Vector)
				assert.Equal(t, cfg.Dimension, len(item.Vector))
			}
			t.Log("Vector verification completed")

			// Test search functionality
			if len(tt.items) > 0 {
				t.Log("Starting search verification")
				// Get the expected item based on the test case
				var expectedItem *storage.Item
				switch tt.name {
				case "Insert text items":
					expectedItem = tt.items[0] // First item for text items
				case "Insert code items":
					expectedItem = tt.items[0] // The code item
				case "Insert mixed content types":
					expectedItem = tt.items[0] // First item of mixed content
				default:
					expectedItem = tt.items[0]
				}

				// Generate a query vector for the expected item
				queryVector := make([]float32, cfg.Dimension)
				for j := range queryVector {
					switch expectedItem.ID {
					case "1":
						queryVector[j] = float32(j % 10)
					case "2":
						queryVector[j] = float32(9 - (j % 10))
					case "3":
						queryVector[j] = float32(j % 2 * 9)
					case "4":
						queryVector[j] = float32((j % 3) * 3)
					case "5":
						queryVector[j] = float32(4 + (j%2)*2)
					default:
						queryVector[j] = float32(0)
					}
				}

				// Normalize the query vector
				queryVector = normalizeVector(queryVector)

				results, err := store.Search(ctx, queryVector, 5)
				assert.NoError(t, err)
				assert.NotEmpty(t, results)
				assert.Equal(t, expectedItem.ID, results[0].ID)
				t.Log("Search verification completed")
			}
			t.Log("Subtest completed successfully")
		})
	}
	t.Log("Test completed successfully")
}

func TestMilvusStore_UpdateWithEmbeddings(t *testing.T) {
	// Create a mock embedding service
	mockEmbedder := &mockEmbeddingService{dimension: 1536}

	// Create store config
	cfg := milvus.Config{
		Host:             "localhost",
		Port:             19530,
		CollectionName:   "test_collection",
		Dimension:        1536,
		BatchSize:        100,
		MaxRetries:       3,
		PoolSize:         5,
		EmbeddingService: mockEmbedder,
	}

	// Create store
	store, err := milvus.NewMilvusStore(cfg)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	// First, insert an item
	originalItem := &storage.Item{
		ID: "test_update",
		Content: storage.Content{
			Type: storage.ContentTypeText,
			Data: []byte("original text"),
		},
		Metadata:  map[string]interface{}{"version": 1},
		CreatedAt: time.Now(),
	}

	err = store.Insert(ctx, []*storage.Item{originalItem})
	require.NoError(t, err)

	// Update the item
	updatedItem := &storage.Item{
		ID: "test_update",
		Content: storage.Content{
			Type: storage.ContentTypeText,
			Data: []byte("updated text"),
		},
		Metadata:  map[string]interface{}{"version": 2},
		CreatedAt: time.Now(),
	}

	err = store.Update(ctx, []*storage.Item{updatedItem})
	require.NoError(t, err)

	// Search for the updated item
	results, err := store.Search(ctx, updatedItem.Vector, 1)
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Verify the update
	assert.Equal(t, updatedItem.ID, results[0].ID)
	assert.Equal(t, "updated text", string(results[0].Content.Data))
	assert.Equal(t, float64(2), results[0].Metadata["version"])
}
