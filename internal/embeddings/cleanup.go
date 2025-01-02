package embeddings

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/rs/zerolog/log"
)

// CleanupConfig holds configuration for cleanup operations
type CleanupConfig struct {
	// Maximum age of unused files before they are deleted
	MaxFileAge time.Duration
	// Directory paths to clean up
	Directories []string
	// File patterns to match for cleanup
	Patterns []string
	// Whether to perform a dry run (no actual deletions)
	DryRun bool
}

// DefaultCleanupConfig returns the default cleanup configuration
func DefaultCleanupConfig() CleanupConfig {
	return CleanupConfig{
		MaxFileAge: 7 * 24 * time.Hour, // 7 days
		Directories: []string{
			"models",
			"models/tokenizers",
			"models/configs",
		},
		Patterns: []string{
			"*.onnx",
			"*.json",
			"*.bin",
			"*.txt",
		},
		DryRun: false,
	}
}

// CleanupUnusedFiles removes old unused files from the specified directories
func CleanupUnusedFiles(cfg CleanupConfig) error {
	now := time.Now()
	var totalSize int64
	var deletedCount int

	// Get list of active model files
	activeFiles := make(map[string]bool)
	for _, modelCfg := range DefaultModelConfigs() {
		activeFiles[modelCfg.ModelPath] = true
		activeFiles[modelCfg.TokenizerConfigPath] = true
	}

	// Process each directory
	for _, dir := range cfg.Directories {
		err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// Skip directories
			if info.IsDir() {
				return nil
			}

			// Check if file matches any pattern
			matched := false
			for _, pattern := range cfg.Patterns {
				if match, _ := filepath.Match(pattern, filepath.Base(path)); match {
					matched = true
					break
				}
			}
			if !matched {
				return nil
			}

			// Skip active files
			if activeFiles[path] {
				log.Debug().
					Str("path", path).
					Msg("Skipping active file")
				return nil
			}

			// Check file age
			if now.Sub(info.ModTime()) > cfg.MaxFileAge {
				if cfg.DryRun {
					log.Info().
						Str("path", path).
						Time("mod_time", info.ModTime()).
						Int64("size", info.Size()).
						Msg("Would delete file (dry run)")
				} else {
					log.Info().
						Str("path", path).
						Time("mod_time", info.ModTime()).
						Int64("size", info.Size()).
						Msg("Deleting old file")

					if err := os.Remove(path); err != nil {
						return fmt.Errorf("failed to delete file %s: %w", path, err)
					}
				}

				totalSize += info.Size()
				deletedCount++
			}

			return nil
		})

		if err != nil {
			return fmt.Errorf("failed to clean up directory %s: %w", dir, err)
		}
	}

	log.Info().
		Int("deleted_count", deletedCount).
		Int64("total_size", totalSize).
		Bool("dry_run", cfg.DryRun).
		Msg("Cleanup completed")

	return nil
}

// CleanupOldModels removes old model files that are no longer in use
func CleanupOldModels(dryRun bool) error {
	cfg := DefaultCleanupConfig()
	cfg.DryRun = dryRun
	return CleanupUnusedFiles(cfg)
}
