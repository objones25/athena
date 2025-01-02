package graph

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Edge represents a connection between two vectors with a similarity score
type Edge struct {
	From      string    // Key of the source vector
	To        string    // Key of the target vector
	Score     float64   // Similarity score
	Timestamp time.Time // When this edge was last updated
}

// Store manages a graph of vector similarities
type Store struct {
	mu        sync.RWMutex
	edges     map[string]map[string]*Edge // From -> To -> Edge
	maxEdges  int                         // Maximum edges per vector
	ttl       time.Duration               // Time-to-live for edges
	batchSize int                         // Size of batches for updates
}

// Config holds configuration for the graph store
type Config struct {
	MaxEdgesPerVector int           // Maximum number of edges to store per vector
	EdgeTTL           time.Duration // Time-to-live for edges
	BatchSize         int           // Size of batches for updates
}

// DefaultConfig returns default configuration values
func DefaultConfig() Config {
	return Config{
		MaxEdgesPerVector: 100,
		EdgeTTL:           24 * time.Hour,
		BatchSize:         1000,
	}
}

// New creates a new graph store
func New(cfg Config) *Store {
	return &Store{
		edges:     make(map[string]map[string]*Edge),
		maxEdges:  cfg.MaxEdgesPerVector,
		ttl:       cfg.EdgeTTL,
		batchSize: cfg.BatchSize,
	}
}

// AddEdge adds or updates an edge in the graph
func (s *Store) AddEdge(ctx context.Context, from, to string, score float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize maps if needed
	if _, exists := s.edges[from]; !exists {
		s.edges[from] = make(map[string]*Edge)
	}

	// Add or update edge
	now := time.Now()
	s.edges[from][to] = &Edge{
		From:      from,
		To:        to,
		Score:     score,
		Timestamp: now,
	}

	// Prune edges if we exceed maxEdges
	if len(s.edges[from]) > s.maxEdges {
		s.pruneEdges(from)
	}

	return nil
}

// AddEdgeBatch adds multiple edges in a batch
func (s *Store) AddEdgeBatch(ctx context.Context, edges []Edge) error {
	// Process in batches to avoid holding lock too long
	for i := 0; i < len(edges); i += s.batchSize {
		end := i + s.batchSize
		if end > len(edges) {
			end = len(edges)
		}

		s.mu.Lock()
		for _, edge := range edges[i:end] {
			if _, exists := s.edges[edge.From]; !exists {
				s.edges[edge.From] = make(map[string]*Edge)
			}
			s.edges[edge.From][edge.To] = &edge

			// Prune if needed
			if len(s.edges[edge.From]) > s.maxEdges {
				s.pruneEdges(edge.From)
			}
		}
		s.mu.Unlock()
	}

	return nil
}

// GetEdges returns all edges from a vector
func (s *Store) GetEdges(ctx context.Context, from string) ([]Edge, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fromEdges, exists := s.edges[from]
	if !exists {
		return nil, fmt.Errorf("no edges found for vector: %s", from)
	}

	now := time.Now()
	edges := make([]Edge, 0, len(fromEdges))
	for _, edge := range fromEdges {
		// Skip expired edges
		if now.Sub(edge.Timestamp) > s.ttl {
			continue
		}
		edges = append(edges, *edge)
	}

	return edges, nil
}

// GetScore returns the similarity score between two vectors
func (s *Store) GetScore(ctx context.Context, from, to string) (float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	fromEdges, exists := s.edges[from]
	if !exists {
		return 0, fmt.Errorf("no edges found for source vector: %s", from)
	}

	edge, exists := fromEdges[to]
	if !exists {
		return 0, fmt.Errorf("no edge found between vectors: %s -> %s", from, to)
	}

	// Check if edge has expired
	if time.Now().Sub(edge.Timestamp) > s.ttl {
		return 0, fmt.Errorf("edge has expired: %s -> %s", from, to)
	}

	return edge.Score, nil
}

// Cleanup removes expired edges
func (s *Store) Cleanup(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()
	for from, edges := range s.edges {
		for to, edge := range edges {
			if now.Sub(edge.Timestamp) > s.ttl {
				delete(edges, to)
			}
		}
		if len(edges) == 0 {
			delete(s.edges, from)
		}
	}

	return nil
}

// pruneEdges removes the oldest edges to maintain maxEdges limit
// Caller must hold write lock
func (s *Store) pruneEdges(from string) {
	edges := s.edges[from]
	if len(edges) <= s.maxEdges {
		return
	}

	// Find oldest edges to remove
	type edgeAge struct {
		to        string
		timestamp time.Time
	}
	ages := make([]edgeAge, 0, len(edges))
	for to, edge := range edges {
		ages = append(ages, edgeAge{to, edge.Timestamp})
	}

	// Sort by timestamp (oldest first)
	numToRemove := len(edges) - s.maxEdges
	for i := 0; i < numToRemove; i++ {
		oldest := ages[0]
		for _, age := range ages[1:] {
			if age.timestamp.Before(oldest.timestamp) {
				oldest = age
			}
		}
		delete(edges, oldest.to)
		// Remove from ages slice
		for j := range ages {
			if ages[j].to == oldest.to {
				ages = append(ages[:j], ages[j+1:]...)
				break
			}
		}
	}
}
