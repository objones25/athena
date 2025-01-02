package search

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/objones25/athena/internal/search/similarity"
)

// SearchResult represents a single search result with metadata
type SearchResult struct {
	Key           string
	Score         float64
	Confidence    float64
	Significance  float64
	RelativeScore float64
	Metadata      map[string]interface{}
}

// SearchOptions configures the search behavior
type SearchOptions struct {
	// Scoring thresholds
	MinScore        float64
	MinConfidence   float64
	MinSignificance float64

	// Result limits
	MaxResults  int
	MaxParallel int

	// Search behavior
	IncludeMetadata bool
	SortBy          SortCriteria
	FilterFn        FilterFunc

	// Similarity context
	SimilarityCtx similarity.Context
}

// SortCriteria defines how to sort search results
type SortCriteria int

const (
	SortByScore SortCriteria = iota
	SortByConfidence
	SortBySignificance
	SortByRelative
)

// FilterFunc is a function type for custom result filtering
type FilterFunc func(SearchResult) bool

// DefaultSearchOptions returns sensible default search options
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		MinScore:        0.7,
		MinConfidence:   0.6,
		MinSignificance: 0.5,
		MaxResults:      100,
		MaxParallel:     10,
		IncludeMetadata: true,
		SortBy:          SortByScore,
		SimilarityCtx:   similarity.DefaultContext(),
	}
}

// Search performs a semantic search using the provided query embedding
type Search struct {
	store      Store
	similarity *similarity.Context
	mu         sync.RWMutex
}

// Store defines the interface for embedding storage and retrieval
type Store interface {
	Get(ctx context.Context, key string) ([]float32, error)
	MGet(ctx context.Context, keys []string) ([][]float32, error)
	Keys(ctx context.Context) ([]string, error)
	GetMetadata(ctx context.Context, key string) (map[string]interface{}, error)
}

// New creates a new search instance
func New(store Store, simCtx *similarity.Context) *Search {
	if simCtx == nil {
		simCtx = &similarity.Context{
			TopicalWeight:   0.35,
			SemanticWeight:  0.30,
			SyntacticWeight: 0.20,
			LanguageWeight:  0.15,
			BatchSize:       1000,
			UseConcurrency:  true,
		}
	}

	return &Search{
		store:      store,
		similarity: simCtx,
	}
}

// FindSimilar finds similar embeddings based on a query embedding
func (s *Search) FindSimilar(ctx context.Context, queryEmbed []float32, opts SearchOptions) ([]SearchResult, error) {
	// Get all candidate keys
	keys, err := s.store.Keys(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get keys: %w", err)
	}

	// Process in parallel batches
	var results []SearchResult
	var mu sync.Mutex
	sem := make(chan struct{}, opts.MaxParallel)

	var g sync.WaitGroup
	for i := 0; i < len(keys); i += opts.SimilarityCtx.BatchSize {
		end := min(i+opts.SimilarityCtx.BatchSize, len(keys))
		batch := keys[i:end]

		g.Add(1)
		go func(batchKeys []string) {
			defer g.Done()
			sem <- struct{}{}        // Acquire semaphore
			defer func() { <-sem }() // Release semaphore

			// Get embeddings for batch
			embeddings, err := s.store.MGet(ctx, batchKeys)
			if err != nil {
				// Log error but continue processing
				fmt.Printf("Error getting embeddings for batch: %v\n", err)
				return
			}

			// Calculate similarities
			batchResults := make([]SearchResult, 0, len(batchKeys))
			for i, embed := range embeddings {
				metrics, err := similarity.Calculate(queryEmbed, embed, opts.SimilarityCtx)
				if err != nil {
					continue
				}

				// Apply filtering
				if metrics.Contextual < opts.MinScore ||
					metrics.ConfidenceScore < opts.MinConfidence ||
					metrics.Significance < opts.MinSignificance {
					continue
				}

				result := SearchResult{
					Key:           batchKeys[i],
					Score:         metrics.Contextual,
					Confidence:    metrics.ConfidenceScore,
					Significance:  metrics.Significance,
					RelativeScore: metrics.RelativeScore,
				}

				if opts.FilterFn != nil && !opts.FilterFn(result) {
					continue
				}

				if opts.IncludeMetadata {
					if metadata, err := s.store.GetMetadata(ctx, batchKeys[i]); err == nil {
						result.Metadata = metadata
					}
				}

				batchResults = append(batchResults, result)
			}

			// Add batch results to global results
			mu.Lock()
			results = append(results, batchResults...)
			mu.Unlock()
		}(batch)
	}

	g.Wait()

	// Sort results
	sort.Slice(results, func(i, j int) bool {
		switch opts.SortBy {
		case SortByConfidence:
			return results[i].Confidence > results[j].Confidence
		case SortBySignificance:
			return results[i].Significance > results[j].Significance
		case SortByRelative:
			return results[i].RelativeScore > results[j].RelativeScore
		default: // SortByScore
			return results[i].Score > results[j].Score
		}
	})

	// Limit results
	if len(results) > opts.MaxResults {
		results = results[:opts.MaxResults]
	}

	return results, nil
}

// UpdateSimilarityContext updates the similarity context configuration
func (s *Search) UpdateSimilarityContext(ctx *similarity.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.similarity = ctx
}

// GetSimilarityContext returns the current similarity context configuration
func (s *Search) GetSimilarityContext() *similarity.Context {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.similarity
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
