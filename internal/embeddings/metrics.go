package embeddings

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// EmbeddingRequestsTotal tracks the total number of embedding requests
	EmbeddingRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "embedding_requests_total",
			Help: "The total number of embedding requests",
		},
		[]string{"model", "content_type"},
	)

	// EmbeddingRequestDuration tracks the duration of embedding requests
	EmbeddingRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "embedding_request_duration_seconds",
			Help:    "The duration of embedding requests in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // From 1ms to ~16s
		},
		[]string{"model", "content_type"},
	)

	// EmbeddingBatchSize tracks the size of batch embedding requests
	EmbeddingBatchSize = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "embedding_batch_size",
			Help:    "The size of batch embedding requests",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128},
		},
		[]string{"model"},
	)

	// EmbeddingErrors tracks the number of embedding errors
	EmbeddingErrors = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "embedding_errors_total",
			Help: "The total number of embedding errors",
		},
		[]string{"model", "error_type"},
	)

	// ModelLoadDuration tracks the duration of model loading
	ModelLoadDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "model_load_duration_seconds",
			Help:    "The duration of model loading in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10), // From 100ms to ~51s
		},
		[]string{"model"},
	)

	// GPUMemoryUsage tracks GPU memory usage per model
	GPUMemoryUsage = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_memory_bytes",
			Help: "Current GPU memory usage in bytes",
		},
		[]string{"model", "gpu_id"},
	)

	// TokenizationDuration tracks the duration of tokenization
	TokenizationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "tokenization_duration_seconds",
			Help:    "The duration of tokenization in seconds",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 10), // From 100µs to ~51ms
		},
		[]string{"model"},
	)

	// InputLength tracks the length of input text
	InputLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "input_length_chars",
			Help:    "The length of input text in characters",
			Buckets: prometheus.ExponentialBuckets(10, 2, 12), // From 10 to ~20k chars
		},
		[]string{"model", "content_type"},
	)

	// TokenLength tracks the number of tokens after tokenization
	TokenLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "token_length",
			Help:    "The number of tokens after tokenization",
			Buckets: []float64{16, 32, 64, 128, 256, 512},
		},
		[]string{"model"},
	)
)
