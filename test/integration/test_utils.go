package integration

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/yalue/onnxruntime_go"
)

// TestTimeout is the default timeout for integration tests
const TestTimeout = 5 * time.Minute

func init() {
	// Set ONNX runtime shared library path from environment variable
	libPath := os.Getenv("DYLD_LIBRARY_PATH")
	if libPath == "" {
		// Fallback to default path
		_, currentFile, _, ok := runtime.Caller(0)
		if !ok {
			panic("Failed to get current file path")
		}
		rootDir := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
		libPath = filepath.Join(rootDir, "onnxruntime-osx-arm64-1.14.0", "lib")
	}
	libPath = filepath.Join(libPath, "libonnxruntime.1.14.0.dylib")

	// Check if library exists
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		panic(fmt.Sprintf("ONNX Runtime library not found at %s", libPath))
	}

	// Set the library path
	onnxruntime_go.SetSharedLibraryPath(libPath)

	// Get ONNX runtime version before initialization
	version := onnxruntime_go.GetVersion()
	fmt.Printf("ONNX Runtime version: %s\n", version)

	// Initialize ONNX runtime environment
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		panic(fmt.Sprintf("Failed to initialize ONNX runtime environment: %v", err))
	}

	// Verify initialization
	if !onnxruntime_go.IsInitialized() {
		panic("ONNX Runtime not initialized after successful initialization call")
	}
}

// GetTestContext returns a context with timeout for integration tests
func GetTestContext(t *testing.T) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), TestTimeout)
}

// GetTestModelPath returns the path to the test ONNX model
func GetTestModelPath(t *testing.T) string {
	modelPath := os.Getenv("TEST_ONNX_MODEL_PATH")
	if modelPath == "" {
		modelPath = os.Getenv("ONNX_MODEL_PATH")
	}
	if modelPath == "" {
		t.Skip("Neither TEST_ONNX_MODEL_PATH nor ONNX_MODEL_PATH is set")
	}

	// If it's an absolute path, use it directly
	if filepath.IsAbs(modelPath) {
		_, err := os.Stat(modelPath)
		require.NoError(t, err, "Model file does not exist at %s", modelPath)
		return modelPath
	}

	// For relative paths, resolve from project root
	_, currentFile, _, ok := runtime.Caller(0)
	require.True(t, ok, "Failed to get current file path")

	// Get the project root directory (two levels up from test/integration)
	rootDir := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))

	// Resolve model path relative to project root
	absPath := filepath.Join(rootDir, modelPath)
	_, err := os.Stat(absPath)
	require.NoError(t, err, "Model file does not exist at %s", absPath)

	return absPath
}

// GetTestDataPath returns the path to the test data directory
func GetTestDataPath(t *testing.T) string {
	dataPath := os.Getenv("TEST_DATA_PATH")
	if dataPath == "" {
		// Get the absolute path of the current file
		_, currentFile, _, ok := runtime.Caller(0)
		require.True(t, ok, "Failed to get current file path")

		// Get the integration test directory
		integrationDir := filepath.Dir(currentFile)

		// Default to testdata in the integration test directory
		dataPath = filepath.Join(integrationDir, "testdata")
	} else if !filepath.IsAbs(dataPath) {
		// For relative paths, resolve from project root
		_, currentFile, _, ok := runtime.Caller(0)
		require.True(t, ok, "Failed to get current file path")

		// Get the project root directory (two levels up from test/integration)
		rootDir := filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))

		// Resolve data path relative to project root
		dataPath = filepath.Join(rootDir, dataPath)
	}

	// Check if directory exists
	_, err := os.Stat(dataPath)
	require.NoError(t, err, "Test data directory does not exist at %s", dataPath)

	return dataPath
}

// LoadTestFile loads a test file from the test data directory
func LoadTestFile(t *testing.T, filename string) string {
	path := filepath.Join(GetTestDataPath(t), filename)
	data, err := os.ReadFile(path)
	require.NoError(t, err, "Failed to read test file %s", path)
	return string(data)
}

// CreateTempDir creates a temporary directory for testing
func CreateTempDir(t *testing.T) string {
	dir, err := os.MkdirTemp("", "onnx-test-*")
	require.NoError(t, err, "Failed to create temp directory")
	t.Cleanup(func() {
		os.RemoveAll(dir)
	})
	return dir
}

// WaitForCondition waits for a condition to be true with timeout
func WaitForCondition(t *testing.T, condition func() bool, timeout time.Duration, message string) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("Timeout waiting for condition: %s", message)
}
