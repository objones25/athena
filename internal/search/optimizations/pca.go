package optimizations

import (
	"fmt"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// PCA performs principal component analysis on the input vectors
func PCA(vectors [][]float32, cfg Config) ([][]float32, [][]float32, error) {
	if len(vectors) == 0 {
		return nil, nil, fmt.Errorf("empty input vectors")
	}

	// Convert to float64 matrix
	rows := len(vectors)
	cols := len(vectors[0])
	data := make([]float64, rows*cols)
	for i, vec := range vectors {
		if len(vec) != cols {
			return nil, nil, fmt.Errorf("vector %d has inconsistent dimensions", i)
		}
		for j, val := range vec {
			data[i*cols+j] = float64(val)
		}
	}
	X := mat.NewDense(rows, cols, data)

	// Center the data
	means := make([]float64, cols)
	for j := 0; j < cols; j++ {
		col := mat.Col(nil, j, X)
		mean := 0.0
		for _, val := range col {
			mean += val
		}
		mean /= float64(rows)
		means[j] = mean
		for i := 0; i < rows; i++ {
			X.Set(i, j, X.At(i, j)-mean)
		}
	}

	// Compute covariance matrix
	var covDense mat.Dense
	covDense.Mul(X.T(), X)
	covDense.Scale(1/float64(rows-1), &covDense)

	// Convert to symmetric matrix
	cov := mat.NewSymDense(cols, nil)
	for i := 0; i < cols; i++ {
		for j := i; j < cols; j++ {
			cov.SetSym(i, j, covDense.At(i, j))
		}
	}

	// Compute eigenvalues and eigenvectors
	var eigen mat.EigenSym
	ok := eigen.Factorize(cov, true)
	if !ok {
		return nil, nil, fmt.Errorf("eigendecomposition failed")
	}

	// Get eigenvalues and eigenvectors
	eigenValues := eigen.Values(nil)
	var eigenVectors mat.Dense
	eigen.VectorsTo(&eigenVectors)

	// Sort eigenvalues and eigenvectors in descending order
	indices := make([]int, len(eigenValues))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return eigenValues[indices[i]] > eigenValues[indices[j]]
	})

	// Determine number of components to keep
	totalVariance := 0.0
	for _, val := range eigenValues {
		totalVariance += val
	}

	var numComponents int
	if cfg.TargetDimension > 0 {
		numComponents = cfg.TargetDimension
	} else {
		// Keep components that explain MinVariance of total variance
		explainedVariance := 0.0
		for i, idx := range indices {
			explainedVariance += eigenValues[idx] / totalVariance
			if explainedVariance >= cfg.MinVariance {
				numComponents = i + 1
				break
			}
		}
	}
	if numComponents > cols {
		numComponents = cols
	}

	// Create projection matrix
	projection := make([][]float32, numComponents)
	for i := 0; i < numComponents; i++ {
		projection[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			projection[i][j] = float32(eigenVectors.At(j, indices[i]))
		}
	}

	// Project data
	var reduced mat.Dense
	reducedEigenVectors := mat.NewDense(cols, numComponents, nil)
	for i := 0; i < numComponents; i++ {
		col := mat.Col(nil, indices[i], &eigenVectors)
		reducedEigenVectors.SetCol(i, col)
	}
	reduced.Mul(X, reducedEigenVectors)

	// Convert back to [][]float32
	result := make([][]float32, rows)
	for i := range result {
		result[i] = make([]float32, numComponents)
		for j := 0; j < numComponents; j++ {
			result[i][j] = float32(reduced.At(i, j))
		}
	}

	return result, projection, nil
}

// Project projects a vector using the PCA projection matrix
func Project(vector []float32, projection [][]float32) ([]float32, error) {
	if len(vector) != len(projection[0]) {
		return nil, fmt.Errorf("vector dimension mismatch: got %d, want %d",
			len(vector), len(projection[0]))
	}

	result := make([]float32, len(projection))
	for i, proj := range projection {
		var sum float32
		for j, val := range proj {
			sum += val * vector[j]
		}
		result[i] = sum
	}

	return result, nil
}

// Reconstruct reconstructs a vector from its PCA projection
func Reconstruct(projected []float32, projection [][]float32) ([]float32, error) {
	if len(projected) != len(projection) {
		return nil, fmt.Errorf("projection dimension mismatch: got %d, want %d",
			len(projected), len(projection))
	}

	originalDim := len(projection[0])
	result := make([]float32, originalDim)
	for i := 0; i < originalDim; i++ {
		var sum float32
		for j, proj := range projection {
			sum += proj[i] * projected[j]
		}
		result[i] = sum
	}

	return result, nil
}
