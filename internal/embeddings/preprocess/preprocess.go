package preprocess

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/jdkato/prose/v2"
)

var (
	// Common patterns for text preprocessing
	urlRE            = regexp.MustCompile(`https?://\S+`)
	emailRE          = regexp.MustCompile(`\S+@\S+\.\S+`)
	mentionRE        = regexp.MustCompile(`@\w+`)
	hashtagRE        = regexp.MustCompile(`#\w+`)
	punctuationRE    = regexp.MustCompile(`[^\w\s]`)
	numberRE         = regexp.MustCompile(`\d+`)
	textWhitespaceRE = regexp.MustCompile(`\s+`)
	specialCharsRE   = regexp.MustCompile(`[^a-zA-Z0-9\s]`)
)

// TextPreprocessOptions holds configuration for text preprocessing
type TextPreprocessOptions struct {
	LowerCase         bool
	StripAccents      bool
	CleanText         bool
	HandleChinese     bool
	HandleNumbers     bool
	MaxCharsPerDoc    int
	RemoveURLs        bool
	RemoveEmails      bool
	RemoveMentions    bool
	RemoveHashtags    bool
	RemovePunctuation bool
	Language          string
}

// DefaultTextPreprocessOptions returns default preprocessing options for BERT
func DefaultTextPreprocessOptions() TextPreprocessOptions {
	return TextPreprocessOptions{
		LowerCase:         true,
		StripAccents:      true,
		CleanText:         true,
		HandleChinese:     true,
		HandleNumbers:     true,
		MaxCharsPerDoc:    100000,
		RemoveURLs:        true,
		RemoveEmails:      true,
		RemoveMentions:    true,
		RemoveHashtags:    true,
		RemovePunctuation: false,
		Language:          "en",
	}
}

// TextPreprocessor handles text preprocessing for BERT-style models
type TextPreprocessor struct {
	options TextPreprocessOptions
	// Precompiled regexes
	controlCharsRegex *regexp.Regexp
	whitespaceRegex   *regexp.Regexp
	chineseCharsRegex *regexp.Regexp
	numberRegex       *regexp.Regexp
	doc               *prose.Document
}

// NewTextPreprocessor creates a new text preprocessor
func NewTextPreprocessor(options TextPreprocessOptions) (*TextPreprocessor, error) {
	p := &TextPreprocessor{
		options: options,
	}

	var err error
	p.controlCharsRegex, err = regexp.Compile(`[\x00-\x1f\x7f-\x9f]`)
	if err != nil {
		return nil, fmt.Errorf("failed to compile control chars regex: %w", err)
	}

	p.whitespaceRegex, err = regexp.Compile(`\s+`)
	if err != nil {
		return nil, fmt.Errorf("failed to compile whitespace regex: %w", err)
	}

	p.chineseCharsRegex, err = regexp.Compile(`[\p{Han}]`)
	if err != nil {
		return nil, fmt.Errorf("failed to compile Chinese chars regex: %w", err)
	}

	p.numberRegex, err = regexp.Compile(`\d+`)
	if err != nil {
		return nil, fmt.Errorf("failed to compile number regex: %w", err)
	}

	return p, nil
}

// Process preprocesses text according to BERT requirements
func (p *TextPreprocessor) Process(text string) (string, error) {
	if text == "" {
		return "", nil
	}

	// Initialize prose document for NLP tasks if needed
	var err error
	if p.doc == nil {
		p.doc, err = prose.NewDocument(text)
		if err != nil {
			return "", err
		}
	}

	// Truncate if too long
	if p.options.MaxCharsPerDoc > 0 && len(text) > p.options.MaxCharsPerDoc {
		text = text[:p.options.MaxCharsPerDoc]
	}

	// Remove unwanted elements
	if p.options.RemoveURLs {
		text = urlRE.ReplaceAllString(text, " ")
	}
	if p.options.RemoveEmails {
		text = emailRE.ReplaceAllString(text, " ")
	}
	if p.options.RemoveMentions {
		text = mentionRE.ReplaceAllString(text, " ")
	}
	if p.options.RemoveHashtags {
		text = hashtagRE.ReplaceAllString(text, " ")
	}
	if p.options.RemovePunctuation {
		text = punctuationRE.ReplaceAllString(text, " ")
	}

	// Clean text
	if p.options.CleanText {
		// Remove control characters
		text = p.controlCharsRegex.ReplaceAllString(text, " ")

		// Replace multiple whitespace with single space
		text = p.whitespaceRegex.ReplaceAllString(text, " ")
	}

	// Handle Chinese characters
	if p.options.HandleChinese {
		// Add spaces around Chinese characters
		text = p.chineseCharsRegex.ReplaceAllStringFunc(text, func(s string) string {
			return " " + s + " "
		})
	}

	// Handle numbers
	if p.options.HandleNumbers {
		// Add spaces around numbers
		text = p.numberRegex.ReplaceAllStringFunc(text, func(s string) string {
			return " " + s + " "
		})
	}

	// Convert to lowercase
	if p.options.LowerCase {
		text = strings.ToLower(text)
	}

	// Strip accents
	if p.options.StripAccents {
		text = stripAccents(text)
	}

	// Clean up whitespace
	text = strings.TrimSpace(text)
	text = p.whitespaceRegex.ReplaceAllString(text, " ")

	return text, nil
}

// GetTokens returns preprocessed tokens from the text
func (p *TextPreprocessor) GetTokens(text string) ([]string, error) {
	if text == "" {
		return nil, nil
	}

	// Initialize prose document for tokenization
	var err error
	p.doc, err = prose.NewDocument(text)
	if err != nil {
		return nil, err
	}

	// Get tokens and preprocess each one
	tokens := make([]string, 0, len(p.doc.Tokens()))
	for _, token := range p.doc.Tokens() {
		if token.Tag == "PUNCT" && p.options.RemovePunctuation {
			continue
		}
		if isNumber(token.Text) && !p.options.HandleNumbers {
			continue
		}

		text := token.Text
		if p.options.LowerCase {
			text = strings.ToLower(text)
		}

		if text != "" {
			tokens = append(tokens, text)
		}
	}

	return tokens, nil
}

// stripAccents removes diacritical marks from text
func stripAccents(text string) string {
	result := make([]rune, 0, len(text))
	for _, r := range text {
		if unicode.Is(unicode.Mn, r) {
			continue // Skip combining diacritical marks
		}
		result = append(result, r)
	}
	return string(result)
}

// isNumber checks if a string contains only digits
func isNumber(s string) bool {
	for _, r := range s {
		if !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}
