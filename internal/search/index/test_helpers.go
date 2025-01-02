package index

// GetVectorKeys returns all vector keys in the index (for testing)
func (idx *VectorIndex) GetVectorKeys() []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	keys := make([]string, 0, len(idx.vectors))
	for key := range idx.vectors {
		keys = append(keys, key)
	}
	return keys
}

// GetNeighbors returns the neighbors of a vector (for testing)
func (idx *VectorIndex) GetNeighbors(key string) []string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if neighbors, exists := idx.graph[key]; exists {
		result := make([]string, len(neighbors))
		copy(result, neighbors)
		return result
	}
	return nil
}

// GetScore returns the similarity score between two vectors (for testing)
func (idx *VectorIndex) GetScore(from, to string) float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if scores, exists := idx.scores[from]; exists {
		return scores[to]
	}
	return 0
}
