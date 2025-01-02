# Search Optimizations

This package provides optimizations for high-dimensional vector search, including dimensionality reduction and vector quantization.

## Features

### Dimensionality Reduction (PCA)

Principal Component Analysis (PCA) is used to reduce the dimensionality of vectors while preserving most of the variance in the data. This optimization:

- Reduces memory usage and computational cost
- Preserves important features in the data
- Can be configured to target a specific number of dimensions or minimum variance

Example usage:
```go
cfg := optimizations.Config{
    TargetDimension: 32, // Reduce to 32 dimensions
}
reduced, projection, err := optimizations.PCA(vectors, cfg)
if err != nil {
    // Handle error
}

// Project new vectors using the same projection
projected, err := optimizations.Project(newVector, projection)
```

### Vector Quantization

Vector quantization using k-means clustering reduces memory usage by representing similar vectors with their cluster centroids. This optimization:

- Significantly reduces memory usage
- Speeds up similarity calculations
- Can be configured for different trade-offs between compression and accuracy

Example usage:
```go
cfg := optimizations.Config{
    NumCentroids:   100,   // Use 100 centroids
    MaxIterations:  100,   // Maximum k-means iterations
    ConvergenceEps: 1e-6,  // Convergence threshold
}
centroids, assignments, err := optimizations.Quantize(vectors, cfg)
if err != nil {
    // Handle error
}

// Compress new vectors using the centroids
compressed, err := optimizations.Compress(newVector, centroids)
```

## Configuration

The `Config` struct provides configuration options for both optimizations:

```go
type Config struct {
    // Dimensionality reduction
    TargetDimension int     // Target number of dimensions after reduction
    MinVariance     float64 // Minimum variance to preserve in dimensionality reduction

    // Vector quantization
    NumCentroids    int     // Number of centroids for vector quantization
    MaxIterations   int     // Maximum iterations for k-means
    ConvergenceEps  float64 // Convergence threshold for k-means
    SampleSize      int     // Number of vectors to sample for centroid initialization
}
```

Default configuration values can be obtained using `DefaultConfig()`.

## Performance Considerations

### Dimensionality Reduction

- The time complexity of PCA is O(nd²) where n is the number of vectors and d is the original dimension
- Memory usage during PCA is O(d²) for the covariance matrix
- After reduction, memory usage and computation time are reduced by the ratio of dimensions

### Vector Quantization

- The time complexity of k-means is O(nkdi) where:
  - n is the number of vectors
  - k is the number of centroids
  - d is the dimension
  - i is the number of iterations
- Memory usage after quantization is reduced to O(kd + n) where:
  - k is the number of centroids
  - d is the dimension
  - n is the number of vectors (for assignments)

## Best Practices

1. **Dimensionality Reduction**
   - Start with a target dimension that preserves 95% of variance
   - Adjust based on performance vs accuracy trade-off
   - Consider your data's intrinsic dimensionality

2. **Vector Quantization**
   - Choose number of centroids based on desired compression ratio
   - Use more centroids for better accuracy
   - Monitor cluster sizes for balanced assignments

3. **Combined Usage**
   - Apply PCA before quantization for best results
   - Benchmark different configurations
   - Consider your specific use case requirements

## Examples

### Combined Optimization

```go
// Configure optimizations
cfg := optimizations.Config{
    TargetDimension: 32,
    NumCentroids:   100,
    MaxIterations:  100,
    ConvergenceEps: 1e-6,
}

// Apply PCA
reduced, projection, err := optimizations.PCA(vectors, cfg)
if err != nil {
    // Handle error
}

// Apply quantization to reduced vectors
centroids, assignments, err := optimizations.Quantize(reduced, cfg)
if err != nil {
    // Handle error
}

// Process new vectors
projected, err := optimizations.Project(newVector, projection)
if err != nil {
    // Handle error
}
compressed, err := optimizations.Compress(projected, centroids)
if err != nil {
    // Handle error
}
```

## Benchmarks

The package includes benchmarks for both optimizations:

```bash
go test -bench=. ./internal/search/optimizations/...
```

Example benchmark results:
```
BenchmarkPCA/128d_to_32d-8            100    15234521 ns/op
BenchmarkQuantization/100_centroids-8  50    30123456 ns/op
``` 