package monitor

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Cache metrics
	CacheOperations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "storage_cache_operations_total",
		Help: "Total number of cache operations",
	}, []string{"operation", "status"})

	CacheLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "storage_cache_latency_seconds",
		Help:    "Latency of cache operations",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
	}, []string{"operation"})

	CacheSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "storage_cache_size_bytes",
		Help: "Current size of cache in bytes",
	})

	CacheEvictions = promauto.NewCounter(prometheus.CounterOpts{
		Name: "storage_cache_evictions_total",
		Help: "Total number of cache evictions",
	})

	// Vector store metrics
	VectorOperations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "storage_vector_operations_total",
		Help: "Total number of vector store operations",
	}, []string{"operation", "status"})

	VectorLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "storage_vector_latency_seconds",
		Help:    "Latency of vector store operations",
		Buckets: []float64{.01, .05, .1, .25, .5, 1, 2.5, 5, 10},
	}, []string{"operation"})

	VectorSize = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "storage_vector_size_total",
		Help: "Total number of vectors stored",
	})

	// Consistency metrics
	ConsistencyErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "storage_consistency_errors_total",
		Help: "Total number of consistency errors detected",
	}, []string{"type"})

	ConsistencyCheckLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "storage_consistency_check_latency_seconds",
		Help:    "Latency of consistency checks",
		Buckets: []float64{.1, .5, 1, 2.5, 5, 10, 30},
	})

	// Connection pool metrics
	PoolUtilization = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "storage_pool_utilization_ratio",
		Help: "Utilization ratio of connection pools",
	}, []string{"store"})

	PoolWaitTime = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "storage_pool_wait_seconds",
		Help:    "Time spent waiting for connection pool",
		Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5},
	}, []string{"store"})

	// Error metrics
	ErrorsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "storage_errors_total",
		Help: "Total number of storage errors",
	}, []string{"store", "operation", "error_type"})

	// Circuit breaker metrics
	CircuitBreakerState = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "storage_circuit_breaker_state",
		Help: "Current state of circuit breakers (0=closed, 1=open)",
	}, []string{"store"})

	CircuitBreakerTrips = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "storage_circuit_breaker_trips_total",
		Help: "Total number of circuit breaker trips",
	}, []string{"store"})
)
