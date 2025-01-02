package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	// KeyPrefix is the prefix for all embedding cache keys
	KeyPrefix = "emb"
)

// DefaultKeyGenerator implements KeyGenerator using SHA-256 hashing
type DefaultKeyGenerator struct {
	prefix string
}

// NewDefaultKeyGenerator creates a new default key generator
func NewDefaultKeyGenerator(prefix string) *DefaultKeyGenerator {
	if prefix == "" {
		prefix = KeyPrefix
	}
	return &DefaultKeyGenerator{prefix: prefix}
}

// GenerateKey implements KeyGenerator
func (g *DefaultKeyGenerator) GenerateKey(content string, model string) string {
	// Create a hash of the content
	hasher := sha256.New()
	hasher.Write([]byte(content))
	contentHash := hex.EncodeToString(hasher.Sum(nil))

	// Format: prefix:model:contenthash
	return fmt.Sprintf("%s:%s:%s", g.prefix, strings.ToLower(model), contentHash)
}

// GenerateKeys implements KeyGenerator
func (g *DefaultKeyGenerator) GenerateKeys(contents []string, model string) []string {
	keys := make([]string, len(contents))
	for i, content := range contents {
		keys[i] = g.GenerateKey(content, model)
	}
	return keys
}
