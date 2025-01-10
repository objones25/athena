package embedding

import (
	"strings"
	"unicode"
)

// TextPreprocessor handles text preprocessing while preserving semantic meaning
type TextPreprocessor struct {
	// Configuration options for preprocessing
	TrimSpace       bool
	NormalizeSpaces bool
	RemoveZeroWidth bool
}

// NewDefaultPreprocessor creates a TextPreprocessor with recommended settings
func NewDefaultPreprocessor() *TextPreprocessor {
	return &TextPreprocessor{
		TrimSpace:       true,
		NormalizeSpaces: true,
		RemoveZeroWidth: true,
	}
}

// Process applies the preprocessing steps to a single text
func (p *TextPreprocessor) Process(text string) string {
	if text == "" {
		return text
	}

	// Remove zero-width characters if configured
	if p.RemoveZeroWidth {
		text = removeZeroWidth(text)
	}

	// Normalize spaces if configured
	if p.NormalizeSpaces {
		text = normalizeSpaces(text)
	}

	// Trim spaces if configured
	if p.TrimSpace {
		text = strings.TrimSpace(text)
	}

	return text
}

// ProcessBatch applies preprocessing to a batch of texts
func (p *TextPreprocessor) ProcessBatch(texts []string) []string {
	if len(texts) == 0 {
		return texts
	}

	results := make([]string, len(texts))
	for i, text := range texts {
		results[i] = p.Process(text)
	}
	return results
}

// removeZeroWidth removes zero-width characters that don't contribute to meaning
func removeZeroWidth(text string) string {
	return strings.Map(func(r rune) rune {
		if unicode.Is(unicode.Cf, r) { // Format characters
			return -1
		}
		return r
	}, text)
}

// normalizeSpaces replaces multiple spaces with a single space
func normalizeSpaces(text string) string {
	return strings.Join(strings.Fields(text), " ")
}
