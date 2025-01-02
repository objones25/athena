package cache

import "errors"

var (
	// ErrNotFound is returned when a requested key is not found in the cache
	ErrNotFound = errors.New("key not found in cache")

	// ErrInvalidKey is returned when a key is invalid
	ErrInvalidKey = errors.New("invalid cache key")

	// ErrInvalidValue is returned when a value is invalid
	ErrInvalidValue = errors.New("invalid cache value")
)
