package preprocess

import (
	"bytes"
	"go/format"
	"go/parser"
	"go/token"
	"regexp"
	"strings"
)

var (
	// Common patterns for code preprocessing
	singleLineCommentRE = regexp.MustCompile(`//.*$`)
	multiLineCommentRE  = regexp.MustCompile(`/\*[\s\S]*?\*/`)
	whitespaceRE        = regexp.MustCompile(`\s+`)
	emptyLineRE         = regexp.MustCompile(`(?m)^\s*$`)
)

// CodePreprocessor handles code preprocessing for different languages
type CodePreprocessor struct {
	language string
	options  CodePreprocessOptions
}

// CodePreprocessOptions configures code preprocessing behavior
type CodePreprocessOptions struct {
	RemoveComments      bool
	NormalizeWhitespace bool
	RemoveEmptyLines    bool
	FormatCode          bool
	MaxLength           int
}

// DefaultCodePreprocessOptions returns default preprocessing options
func DefaultCodePreprocessOptions() CodePreprocessOptions {
	return CodePreprocessOptions{
		RemoveComments:      true,
		NormalizeWhitespace: true,
		RemoveEmptyLines:    true,
		FormatCode:          true,
		MaxLength:           512,
	}
}

// NewCodePreprocessor creates a new code preprocessor for the specified language
func NewCodePreprocessor(language string, options CodePreprocessOptions) *CodePreprocessor {
	return &CodePreprocessor{
		language: strings.ToLower(language),
		options:  options,
	}
}

// Process preprocesses code according to the configured options
func (p *CodePreprocessor) Process(code string) (string, error) {
	if code == "" {
		return "", nil
	}

	// Apply language-specific preprocessing
	var err error
	switch p.language {
	case "go":
		code, err = p.preprocessGo(code)
	case "python":
		code = p.preprocessPython(code)
	case "javascript", "typescript":
		code = p.preprocessJavaScript(code)
	default:
		code = p.preprocessGeneric(code)
	}

	if err != nil {
		return "", err
	}

	// Apply common preprocessing steps
	if p.options.RemoveComments {
		code = p.removeComments(code)
	}
	if p.options.NormalizeWhitespace {
		code = p.normalizeWhitespace(code)
	}
	if p.options.RemoveEmptyLines {
		code = p.removeEmptyLines(code)
	}

	// Truncate if necessary
	if p.options.MaxLength > 0 && len(code) > p.options.MaxLength {
		code = code[:p.options.MaxLength]
	}

	return code, nil
}

// preprocessGo handles Go-specific preprocessing
func (p *CodePreprocessor) preprocessGo(code string) (string, error) {
	if !p.options.FormatCode {
		return code, nil
	}

	// Parse and format Go code
	fset := token.NewFileSet()
	astFile, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return code, nil // Return original code if parsing fails
	}

	var buf bytes.Buffer
	if err := format.Node(&buf, fset, astFile); err != nil {
		return code, nil // Return original code if formatting fails
	}

	return buf.String(), nil
}

// preprocessPython handles Python-specific preprocessing
func (p *CodePreprocessor) preprocessPython(code string) string {
	// Remove Python docstrings
	code = regexp.MustCompile("'''[\\s\\S]*?'''").ReplaceAllString(code, "")
	code = regexp.MustCompile(`"""[\s\S]*?"""`).ReplaceAllString(code, "")

	// Normalize string quotes
	code = regexp.MustCompile(`'([^'\\]|\\.)*'`).ReplaceAllStringFunc(code, func(s string) string {
		return `"` + strings.Trim(s, "'") + `"`
	})

	return code
}

// preprocessJavaScript handles JavaScript/TypeScript preprocessing
func (p *CodePreprocessor) preprocessJavaScript(code string) string {
	// Remove template literals
	code = regexp.MustCompile("`[\\s\\S]*?`").ReplaceAllString(code, `""`)

	// Normalize string quotes
	code = regexp.MustCompile(`'([^'\\]|\\.)*'`).ReplaceAllStringFunc(code, func(s string) string {
		return `"` + strings.Trim(s, "'") + `"`
	})

	return code
}

// preprocessGeneric handles generic code preprocessing
func (p *CodePreprocessor) preprocessGeneric(code string) string {
	return code
}

// removeComments removes single-line and multi-line comments
func (p *CodePreprocessor) removeComments(code string) string {
	// Remove multi-line comments first to handle nested comments
	code = multiLineCommentRE.ReplaceAllString(code, "")
	// Remove single-line comments
	code = singleLineCommentRE.ReplaceAllString(code, "")
	return code
}

// normalizeWhitespace normalizes all whitespace to single spaces
func (p *CodePreprocessor) normalizeWhitespace(code string) string {
	return whitespaceRE.ReplaceAllString(strings.TrimSpace(code), " ")
}

// removeEmptyLines removes empty or whitespace-only lines
func (p *CodePreprocessor) removeEmptyLines(code string) string {
	return emptyLineRE.ReplaceAllString(code, "")
}
