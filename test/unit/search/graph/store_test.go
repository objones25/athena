package graph_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/objones25/athena/internal/search/graph"
	"github.com/stretchr/testify/assert"
)

func TestStore(t *testing.T) {
	ctx := context.Background()

	t.Run("basic operations", func(t *testing.T) {
		cfg := graph.DefaultConfig()
		store := graph.New(cfg)

		// Add edge
		err := store.AddEdge(ctx, "vec1", "vec2", 0.8)
		assert.NoError(t, err)

		// Get score
		score, err := store.GetScore(ctx, "vec1", "vec2")
		assert.NoError(t, err)
		assert.Equal(t, 0.8, score)

		// Get edges
		edges, err := store.GetEdges(ctx, "vec1")
		assert.NoError(t, err)
		assert.Len(t, edges, 1)
		assert.Equal(t, "vec2", edges[0].To)
		assert.Equal(t, 0.8, edges[0].Score)
	})

	t.Run("edge expiration", func(t *testing.T) {
		cfg := graph.Config{
			MaxEdgesPerVector: 100,
			EdgeTTL:           10 * time.Millisecond,
			BatchSize:         1000,
		}
		store := graph.New(cfg)

		// Add edge
		err := store.AddEdge(ctx, "vec1", "vec2", 0.8)
		assert.NoError(t, err)

		// Wait for edge to expire
		time.Sleep(20 * time.Millisecond)

		// Try to get expired edge
		_, err = store.GetScore(ctx, "vec1", "vec2")
		assert.Error(t, err)

		// Cleanup should remove expired edges
		err = store.Cleanup(ctx)
		assert.NoError(t, err)

		edges, err := store.GetEdges(ctx, "vec1")
		assert.Error(t, err) // Should error as all edges are removed
	})

	t.Run("edge pruning", func(t *testing.T) {
		cfg := graph.Config{
			MaxEdgesPerVector: 2,
			EdgeTTL:           time.Hour,
			BatchSize:         1000,
		}
		store := graph.New(cfg)

		// Add edges
		err := store.AddEdge(ctx, "vec1", "vec2", 0.8)
		assert.NoError(t, err)
		time.Sleep(time.Millisecond) // Ensure different timestamps
		err = store.AddEdge(ctx, "vec1", "vec3", 0.9)
		assert.NoError(t, err)
		time.Sleep(time.Millisecond)
		err = store.AddEdge(ctx, "vec1", "vec4", 0.7)
		assert.NoError(t, err)

		// Should only keep newest 2 edges
		edges, err := store.GetEdges(ctx, "vec1")
		assert.NoError(t, err)
		assert.Len(t, edges, 2)

		// vec2 should be pruned as it's oldest
		_, err = store.GetScore(ctx, "vec1", "vec2")
		assert.Error(t, err)
	})

	t.Run("batch operations", func(t *testing.T) {
		cfg := graph.Config{
			MaxEdgesPerVector: 100,
			EdgeTTL:           time.Hour,
			BatchSize:         2, // Small batch size for testing
		}
		store := graph.New(cfg)

		// Create batch of edges
		edges := []graph.Edge{
			{From: "vec1", To: "vec2", Score: 0.8, Timestamp: time.Now()},
			{From: "vec1", To: "vec3", Score: 0.9, Timestamp: time.Now()},
			{From: "vec2", To: "vec3", Score: 0.7, Timestamp: time.Now()},
		}

		// Add edges in batch
		err := store.AddEdgeBatch(ctx, edges)
		assert.NoError(t, err)

		// Verify all edges were added
		for _, edge := range edges {
			score, err := store.GetScore(ctx, edge.From, edge.To)
			assert.NoError(t, err)
			assert.Equal(t, edge.Score, score)
		}
	})

	t.Run("concurrent operations", func(t *testing.T) {
		cfg := graph.DefaultConfig()
		store := graph.New(cfg)

		// Run concurrent operations
		done := make(chan bool)
		for i := 0; i < 10; i++ {
			go func(i int) {
				from := fmt.Sprintf("vec%d", i)
				to := fmt.Sprintf("vec%d", i+1)
				err := store.AddEdge(ctx, from, to, 0.8)
				assert.NoError(t, err)

				score, err := store.GetScore(ctx, from, to)
				assert.NoError(t, err)
				assert.Equal(t, 0.8, score)

				edges, err := store.GetEdges(ctx, from)
				assert.NoError(t, err)
				assert.NotEmpty(t, edges)

				done <- true
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < 10; i++ {
			<-done
		}
	})
}

func BenchmarkStore(b *testing.B) {
	ctx := context.Background()
	cfg := graph.DefaultConfig()
	store := graph.New(cfg)

	b.Run("add edge", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			from := fmt.Sprintf("vec%d", i)
			to := fmt.Sprintf("vec%d", i+1)
			_ = store.AddEdge(ctx, from, to, 0.8)
		}
	})

	b.Run("get score", func(b *testing.B) {
		// Add some edges first
		for i := 0; i < 1000; i++ {
			from := fmt.Sprintf("vec%d", i)
			to := fmt.Sprintf("vec%d", i+1)
			_ = store.AddEdge(ctx, from, to, 0.8)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			from := fmt.Sprintf("vec%d", i%1000)
			to := fmt.Sprintf("vec%d", (i%1000)+1)
			_, _ = store.GetScore(ctx, from, to)
		}
	})

	b.Run("batch add", func(b *testing.B) {
		batchSize := 100
		edges := make([]graph.Edge, batchSize)
		for i := range edges {
			edges[i] = graph.Edge{
				From:      fmt.Sprintf("vec%d", i),
				To:        fmt.Sprintf("vec%d", i+1),
				Score:     0.8,
				Timestamp: time.Now(),
			}
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = store.AddEdgeBatch(ctx, edges)
		}
	})
}
