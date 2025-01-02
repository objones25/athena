package similarity

import "errors"

var (
	// ErrMismatchedVectors is returned when the input vector slices have different lengths
	ErrMismatchedVectors = errors.New("input vector slices must have the same length")

	// ErrInvalidVectors is returned when input vectors are nil or empty
	ErrInvalidVectors = errors.New("input vectors must not be nil or empty")

	// ErrDimensionMismatch is returned when vectors have different dimensions
	ErrDimensionMismatch = errors.New("input vectors must have the same dimension")
)
