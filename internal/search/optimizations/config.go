package optimizations

// Config holds optimization configuration
type Config struct {
	// Dimensionality reduction
	TargetDimension int     // Target number of dimensions after reduction
	MinVariance     float64 // Minimum variance to preserve in dimensionality reduction

	// Vector quantization
	NumCentroids   int     // Number of centroids for vector quantization
	MaxIterations  int     // Maximum iterations for k-means
	ConvergenceEps float64 // Convergence threshold for k-means
	SampleSize     int     // Number of vectors to sample for centroid initialization
}

// DefaultConfig returns default optimization configuration
func DefaultConfig() Config {
	return Config{
		TargetDimension: 32,
		MinVariance:     0.95,
		NumCentroids:    1000,
		MaxIterations:   100,
		ConvergenceEps:  1e-6,
		SampleSize:      10000,
	}
}
