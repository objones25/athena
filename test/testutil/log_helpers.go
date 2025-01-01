package testutil

import (
	"os"
	"testing"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// SetLogLevel sets the global log level for testing
func SetLogLevel(level zerolog.Level) {
	zerolog.SetGlobalLevel(level)
}

// TestLogLevel is a helper to set log level for a specific test
func TestLogLevel(t *testing.T, level zerolog.Level) func() {
	prevLevel := zerolog.GlobalLevel()
	zerolog.SetGlobalLevel(level)
	return func() {
		zerolog.SetGlobalLevel(prevLevel)
	}
}

// InitTestLogger initializes a test-friendly logger
func InitTestLogger() {
	// Configure console writer for better test output
	output := zerolog.ConsoleWriter{Out: os.Stdout, TimeFormat: "15:04:05"}
	log.Logger = zerolog.New(output).With().Timestamp().Caller().Logger()
}

// ParseLogLevel parses log level from environment variable or returns default
func ParseLogLevel(defaultLevel zerolog.Level) zerolog.Level {
	levelStr := os.Getenv("LOG_LEVEL")
	if levelStr == "" {
		return defaultLevel
	}

	level, err := zerolog.ParseLevel(levelStr)
	if err != nil {
		return defaultLevel
	}
	return level
}
