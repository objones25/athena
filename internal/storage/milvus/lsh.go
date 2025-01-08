package milvus

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

var randSource = rand.NewSource(time.Now().UnixNano())
var rng = rand.New(randSource)

// LSHConfig holds configuration for LSH
type LSHConfig struct {
	NumHashTables    int     // Number of hash tables
	NumHashFunctions int     // Number of hash functions per table
	BucketSize       int     // Maximum size of each bucket
	Threshold        float32 // Similarity threshold for considering vectors as neighbors
}

// DefaultLSHConfig returns default LSH configuration
func DefaultLSHConfig() LSHConfig {
	return LSHConfig{
		NumHashTables:    8,
		NumHashFunctions: 4,
		BucketSize:       100,
		Threshold:        0.8,
	}
}

// LSHIndex implements locality-sensitive hashing for vector indexing
// It provides thread-safe operations for concurrent access to the index.
type LSHIndex struct {
	config     LSHConfig
	dimension  int
	hashTables []hashTable
	planes     [][]float32 // Random hyperplanes for hashing
	// mu protects concurrent access to the index's internal structures
	// during initialization and updates to the hash tables
	mu    sync.RWMutex
	stats struct {
		totalQueries atomic.Int64
		totalHits    atomic.Int64
	}
}

// hashTable represents a single LSH hash table
type hashTable struct {
	buckets sync.Map // map[uint64][]vectorEntry
}

type vectorEntry struct {
	id     string
	vector []float32
}

// NewLSHIndex creates a new LSH index
func NewLSHIndex(config LSHConfig, dimension int) *LSHIndex {
	idx := &LSHIndex{
		config:     config,
		dimension:  dimension,
		hashTables: make([]hashTable, config.NumHashTables),
	}

	// Initialize random hyperplanes for hashing
	idx.initializeHashFunctions()
	return idx
}

// initializeHashFunctions generates random hyperplanes for LSH
func (idx *LSHIndex) initializeHashFunctions() {
	idx.planes = make([][]float32, idx.config.NumHashTables*idx.config.NumHashFunctions)

	for i := range idx.planes {
		// Generate random hyperplane
		plane := make([]float32, idx.dimension)
		for j := range plane {
			plane[j] = normFloat32()
		}
		// Normalize the plane
		normalize(plane)
		idx.planes[i] = plane
	}
}

// Insert adds a vector to the LSH index in a thread-safe manner
func (idx *LSHIndex) Insert(id string, vector []float32) {
	entry := vectorEntry{id: id, vector: vector}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Compute hashes and insert into all tables
	for i := 0; i < idx.config.NumHashTables; i++ {
		hash := idx.computeHash(vector, i)

		// Get or create bucket using sync.Map for thread safety
		bucket, _ := idx.hashTables[i].buckets.LoadOrStore(hash, &sync.Map{})
		bucketMap := bucket.(*sync.Map)

		// Add to bucket with atomic operation
		bucketMap.Store(id, entry)
	}
}

// Query finds approximate nearest neighbors in a thread-safe manner
func (idx *LSHIndex) Query(vector []float32, limit int) []vectorEntry {
	idx.stats.totalQueries.Add(1)
	candidates := make(map[string]vectorEntry)

	idx.mu.RLock()
	numTables := len(idx.hashTables)
	idx.mu.RUnlock()

	// Query all hash tables in parallel
	var wg sync.WaitGroup
	resultChan := make(chan []vectorEntry, numTables)

	for i := 0; i < numTables; i++ {
		wg.Add(1)
		go func(tableIdx int) {
			defer wg.Done()

			idx.mu.RLock()
			hash := idx.computeHash(vector, tableIdx)
			idx.mu.RUnlock()

			if bucket, ok := idx.hashTables[tableIdx].buckets.Load(hash); ok {
				bucketMap := bucket.(*sync.Map)
				var tableResults []vectorEntry

				bucketMap.Range(func(_, value interface{}) bool {
					entry := value.(vectorEntry)
					if cosineSimilarity(vector, entry.vector) >= idx.config.Threshold {
						tableResults = append(tableResults, entry)
					}
					return len(tableResults) < idx.config.BucketSize
				})

				resultChan <- tableResults
			}
		}(i)
	}

	// Collect results
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Merge results
	for results := range resultChan {
		for _, entry := range results {
			candidates[entry.id] = entry
		}
	}

	// Convert map to slice and sort by similarity
	results := make([]vectorEntry, 0, len(candidates))
	for _, entry := range candidates {
		results = append(results, entry)
	}

	// Sort by similarity to query vector
	sortByDistance(results, vector)

	// Update stats
	idx.stats.totalHits.Add(int64(len(results)))

	// Return top k results
	if len(results) > limit {
		results = results[:limit]
	}
	return results
}

// computeHash generates LSH hash for a vector
func (idx *LSHIndex) computeHash(vector []float32, tableIdx int) uint64 {
	var hash uint64
	start := tableIdx * idx.config.NumHashFunctions

	// Compute hash bits using random hyperplanes
	for i := 0; i < idx.config.NumHashFunctions; i++ {
		plane := idx.planes[start+i]
		// If vector is on positive side of hyperplane, set bit to 1
		if dotProduct(vector, plane) >= 0 {
			hash |= 1 << i
		}
	}
	return hash
}

// Helper functions

func normalize(vector []float32) {
	var norm float32
	for _, v := range vector {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm == 0 {
		return
	}
	for i := range vector {
		vector[i] /= norm
	}
}

func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func sortByDistance(entries []vectorEntry, query []float32) {
	// Sort entries by cosine similarity to query vector
	for i := 0; i < len(entries)-1; i++ {
		for j := i + 1; j < len(entries); j++ {
			simI := cosineSimilarity(query, entries[i].vector)
			simJ := cosineSimilarity(query, entries[j].vector)
			if simJ > simI {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}
}

// normFloat32 returns a random number from standard normal distribution
func normFloat32() float32 {
	// Box-Muller transform
	u1 := rng.Float64()
	u2 := rng.Float64()

	r := math.Sqrt(-2.0 * math.Log(u1))
	theta := 2.0 * math.Pi * u2

	return float32(r * math.Cos(theta))
}
