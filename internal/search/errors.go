package search

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidQuery is returned when the query is invalid
	ErrInvalidQuery = errors.New("invalid query")

	// ErrNoResults is returned when no results are found
	ErrNoResults = errors.New("no results found")

	// ErrStoreUnavailable is returned when the embedding store is unavailable
	ErrStoreUnavailable = errors.New("embedding store unavailable")

	// ErrInvalidDimension is returned when embedding dimensions don't match
	ErrInvalidDimension = errors.New("invalid embedding dimension")
)

// SearchError represents a search-specific error with context
type SearchError struct {
	Op      string // Operation that failed
	Err     error  // Underlying error
	Context string // Additional context
}

func (e *SearchError) Error() string {
	if e.Context != "" {
		return fmt.Sprintf("%s: %v (%s)", e.Op, e.Err, e.Context)
	}
	return fmt.Sprintf("%s: %v", e.Op, e.Err)
}

func (e *SearchError) Unwrap() error {
	return e.Err
}

// NewSearchError creates a new SearchError
func NewSearchError(op string, err error, context string) error {
	return &SearchError{
		Op:      op,
		Err:     err,
		Context: context,
	}
}

// IsNoResults checks if an error is a "no results" error
func IsNoResults(err error) bool {
	return errors.Is(err, ErrNoResults)
}

// IsInvalidQuery checks if an error is an "invalid query" error
func IsInvalidQuery(err error) bool {
	return errors.Is(err, ErrInvalidQuery)
}

// IsStoreUnavailable checks if an error is a "store unavailable" error
func IsStoreUnavailable(err error) bool {
	return errors.Is(err, ErrStoreUnavailable)
}

// IsInvalidDimension checks if an error is an "invalid dimension" error
func IsInvalidDimension(err error) bool {
	return errors.Is(err, ErrInvalidDimension)
}
