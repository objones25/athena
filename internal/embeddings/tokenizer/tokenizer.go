package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/objones25/athena/internal/embeddings/preprocess"
)

const (
	defaultVocabURL = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
	defaultMaxLen   = 512
)

// TokenizerType represents the type of tokenizer
type TokenizerType string

const (
	TokenizerWordPiece TokenizerType = "WordPiece"
	TokenizerUnigram   TokenizerType = "Unigram"
	TokenizerBPE       TokenizerType = "BPE"
)

// Tokenizer handles text tokenization for different model types
type Tokenizer struct {
	mu            sync.RWMutex
	config        TokenizerConfig
	modelConfig   *TokenizerModelConfig
	vocab         map[string]int
	maxLength     int
	preprocessor  *preprocess.TextPreprocessor
	tokenizerType TokenizerType
	specialTokens map[string]string // Maps token type to actual token
}

// TokenizerConfig holds configuration for the tokenizer
type TokenizerConfig struct {
	MaxLength         int
	PreprocessOptions preprocess.TextPreprocessOptions
	ModelName         string // Name of the model to use for tokenization
	TokenizerType     TokenizerType
}

// TokenizerModelConfig holds the configuration loaded from tokenizer_config.json
type TokenizerModelConfig struct {
	BosToken                 string `json:"bos_token"`
	EosToken                 string `json:"eos_token"`
	UNKToken                 string `json:"unk_token"`
	SepToken                 string `json:"sep_token"`
	PadToken                 string `json:"pad_token"`
	ClsToken                 string `json:"cls_token"`
	MaskToken                string `json:"mask_token"`
	PadTokenID               int    `json:"pad_token_type_id"`
	ModelMaxLength           int    `json:"model_max_length"`
	DoLowerCase              bool   `json:"do_lower_case"`
	TokenizeChineseChars     bool   `json:"tokenize_chinese_chars"`
	StripAccents             bool   `json:"strip_accents"`
	CleanUpTokenizationSpace bool   `json:"clean_up_tokenization_spaces"`
	TokenizerClass           string `json:"tokenizer_class"`
}

// downloadVocabFile downloads the BERT vocabulary file if it doesn't exist
func downloadVocabFile(vocabPath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(vocabPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Check if file already exists
	if _, err := os.Stat(vocabPath); err == nil {
		return nil // File exists
	}

	// Download file
	resp, err := http.Get(defaultVocabURL)
	if err != nil {
		return fmt.Errorf("failed to download vocabulary: %w", err)
	}
	defer resp.Body.Close()

	// Create file
	file, err := os.Create(vocabPath)
	if err != nil {
		return fmt.Errorf("failed to create vocabulary file: %w", err)
	}
	defer file.Close()

	// Copy content
	if _, err := io.Copy(file, resp.Body); err != nil {
		return fmt.Errorf("failed to write vocabulary file: %w", err)
	}

	return nil
}

// loadVocabulary loads the BERT vocabulary from file
func loadVocabulary(vocabPath string) (map[string]int, error) {
	file, err := os.Open(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open vocabulary file: %w", err)
	}
	defer file.Close()

	vocab := make(map[string]int)
	scanner := bufio.NewScanner(file)
	index := 0
	for scanner.Scan() {
		token := scanner.Text()
		vocab[token] = index
		index++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read vocabulary file: %w", err)
	}

	return vocab, nil
}

// loadTokenizerConfig loads the configuration from tokenizer_config.json
func loadTokenizerConfig(configPath string) (*TokenizerModelConfig, error) {
	file, err := os.Open(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open tokenizer_config.json: %w", err)
	}
	defer file.Close()

	var config TokenizerModelConfig
	if err := json.NewDecoder(file).Decode(&config); err != nil {
		return nil, fmt.Errorf("failed to decode tokenizer_config.json: %w", err)
	}

	return &config, nil
}

// NewTokenizer creates a new tokenizer instance
func NewTokenizer(cfg TokenizerConfig) (*Tokenizer, error) {
	if cfg.MaxLength <= 0 {
		return nil, fmt.Errorf("max length must be positive")
	}

	if cfg.ModelName == "" {
		return nil, fmt.Errorf("model name is required")
	}

	preprocessor, err := preprocess.NewTextPreprocessor(cfg.PreprocessOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to create preprocessor: %w", err)
	}

	// Get model directory from environment variable
	modelsDir := os.Getenv("TEST_MODELS_DIR")
	if modelsDir == "" {
		return nil, fmt.Errorf("TEST_MODELS_DIR environment variable not set")
	}

	// Load tokenizer files from the model-specific tokenizer directory
	tokenizerDir := filepath.Join(modelsDir, cfg.ModelName)
	if _, err := os.Stat(tokenizerDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer directory not found at %s", tokenizerDir)
	}

	// Load tokenizer.json file
	tokenizerPath := filepath.Join(tokenizerDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("tokenizer.json not found at %s", tokenizerPath)
	}

	// Load tokenizer config
	configPath := filepath.Join(tokenizerDir, "tokenizer_config.json")
	modelConfig, err := loadTokenizerConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer config: %w", err)
	}

	// Determine tokenizer type from config
	tokenizerType := determineTokenizerType(modelConfig.TokenizerClass)
	cfg.TokenizerType = tokenizerType

	// Load vocab from tokenizer.json
	vocab, err := loadVocabularyFromTokenizerJSON(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary: %w", err)
	}

	// Initialize special tokens based on tokenizer type
	specialTokens := initializeSpecialTokens(modelConfig)

	// Create tokenizer
	t := &Tokenizer{
		config:        cfg,
		modelConfig:   modelConfig,
		vocab:         vocab,
		preprocessor:  preprocessor,
		maxLength:     modelConfig.ModelMaxLength,
		tokenizerType: tokenizerType,
		specialTokens: specialTokens,
	}

	return t, nil
}

// determineTokenizerType determines the tokenizer type from the tokenizer class
func determineTokenizerType(tokenizerClass string) TokenizerType {
	switch tokenizerClass {
	case "BertTokenizer":
		return TokenizerWordPiece
	case "XLMRobertaTokenizer", "RobertaTokenizer":
		return TokenizerBPE
	default:
		return TokenizerUnigram
	}
}

// initializeSpecialTokens initializes special tokens based on the model config
func initializeSpecialTokens(config *TokenizerModelConfig) map[string]string {
	return map[string]string{
		"bos_token":  config.BosToken,
		"eos_token":  config.EosToken,
		"unk_token":  config.UNKToken,
		"sep_token":  config.SepToken,
		"pad_token":  config.PadToken,
		"cls_token":  config.ClsToken,
		"mask_token": config.MaskToken,
	}
}

// tokenize splits text into tokens based on the tokenizer type
func (t *Tokenizer) tokenize(text string) []string {
	switch t.tokenizerType {
	case TokenizerWordPiece:
		return t.wordPieceTokenize(text)
	case TokenizerBPE:
		return t.bpeTokenize(text)
	case TokenizerUnigram:
		return t.unigramTokenize(text)
	default:
		return t.wordPieceTokenize(text)
	}
}

// wordPieceTokenize splits a word into WordPiece tokens
func (t *Tokenizer) wordPieceTokenize(text string) []string {
	if text == "" {
		return nil
	}

	words := strings.Fields(text)
	tokens := make([]string, 0, len(words)*2)

	for _, word := range words {
		if _, ok := t.vocab[word]; ok {
			tokens = append(tokens, word)
			continue
		}

		// Try to split the word into subwords
		start := 0
		subTokens := make([]string, 0)
		wordLen := len(word)

		for start < wordLen {
			end := wordLen
			curSubstr := ""
			for end > start {
				substr := word[start:end]
				if start > 0 {
					substr = "##" + substr
				}
				if _, ok := t.vocab[substr]; ok {
					curSubstr = substr
					break
				}
				end--
			}

			if curSubstr == "" {
				subTokens = []string{t.specialTokens["unk_token"]}
				break
			}

			subTokens = append(subTokens, curSubstr)
			start = end
		}

		tokens = append(tokens, subTokens...)
	}

	return tokens
}

// bpeTokenize implements byte-pair encoding tokenization
func (t *Tokenizer) bpeTokenize(text string) []string {
	// Add space prefix for RoBERTa/MPNet style tokenization
	text = " " + text
	words := strings.Fields(text)
	tokens := make([]string, 0, len(words)*2)

	for _, word := range words {
		if _, ok := t.vocab[word]; ok {
			tokens = append(tokens, word)
			continue
		}

		// Convert to bytes and encode
		bytes := []byte(word)
		byteTokens := make([]string, 0, len(bytes))
		for _, b := range bytes {
			byteTokens = append(byteTokens, string(b))
		}

		// Merge byte pairs according to vocab
		for {
			bestPair := ""
			bestScore := -1.0
			for i := 0; i < len(byteTokens)-1; i++ {
				pair := byteTokens[i] + byteTokens[i+1]
				if score, ok := t.vocab[pair]; ok && float64(score) > bestScore {
					bestPair = pair
					bestScore = float64(score)
				}
			}

			if bestPair == "" {
				break
			}

			// Merge the best pair
			newTokens := make([]string, 0, len(byteTokens))
			i := 0
			for i < len(byteTokens) {
				if i < len(byteTokens)-1 && byteTokens[i]+byteTokens[i+1] == bestPair {
					newTokens = append(newTokens, bestPair)
					i += 2
				} else {
					newTokens = append(newTokens, byteTokens[i])
					i++
				}
			}
			byteTokens = newTokens
		}

		tokens = append(tokens, byteTokens...)
	}

	return tokens
}

// unigramTokenize implements unigram tokenization
func (t *Tokenizer) unigramTokenize(text string) []string {
	// Add space prefix for multilingual models
	text = "▁" + strings.ReplaceAll(text, " ", "▁")

	// Split into potential tokens
	var tokens []string
	start := 0
	for start < len(text) {
		bestLen := 1
		bestScore := float64(0)

		// Try all possible substrings
		for end := start + 1; end <= len(text); end++ {
			substr := text[start:end]
			if score, ok := t.vocab[substr]; ok {
				if float64(score) > bestScore {
					bestScore = float64(score)
					bestLen = end - start
				}
			}
		}

		// Add the best token
		token := text[start : start+bestLen]
		if _, ok := t.vocab[token]; !ok {
			token = t.specialTokens["unk_token"]
		}
		tokens = append(tokens, token)
		start += bestLen
	}

	return tokens
}

// Tokenize converts text into token IDs with special tokens
func (t *Tokenizer) Tokenize(text string) ([]int, error) {
	if text == "" {
		return nil, fmt.Errorf("empty text")
	}

	// Preprocess text
	processed, err := t.preprocessor.Process(text)
	if err != nil {
		return nil, fmt.Errorf("failed to preprocess text: %w", err)
	}

	// Tokenize text
	tokens := t.tokenize(processed)

	// Convert tokens to IDs
	tokenIds := make([]int, 0, len(tokens)+2) // +2 for special tokens

	// Add BOS/CLS token
	if t.tokenizerType == TokenizerWordPiece {
		tokenIds = append(tokenIds, t.vocab[t.specialTokens["cls_token"]])
	} else {
		tokenIds = append(tokenIds, t.vocab[t.specialTokens["bos_token"]])
	}

	// Add content tokens
	for _, token := range tokens {
		id, ok := t.vocab[token]
		if !ok {
			// Handle unknown tokens
			id = t.vocab[t.specialTokens["unk_token"]]
		}
		tokenIds = append(tokenIds, id)
	}

	// Add EOS/SEP token
	if t.tokenizerType == TokenizerWordPiece {
		tokenIds = append(tokenIds, t.vocab[t.specialTokens["sep_token"]])
	} else {
		tokenIds = append(tokenIds, t.vocab[t.specialTokens["eos_token"]])
	}

	// Truncate if necessary
	if len(tokenIds) > t.maxLength {
		tokenIds = tokenIds[:t.maxLength]
	}

	return tokenIds, nil
}

// BatchTokenize tokenizes multiple texts
func (t *Tokenizer) BatchTokenize(texts []string) ([][]int, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	results := make([][]int, len(texts))
	for i, text := range texts {
		tokens, err := t.Tokenize(text)
		if err != nil {
			return nil, fmt.Errorf("failed to tokenize text at index %d: %w", i, err)
		}
		results[i] = tokens
	}

	return results, nil
}

// PadTokens pads a batch of token IDs to the same length
func (t *Tokenizer) PadTokens(tokenIds [][]int) [][]int {
	if len(tokenIds) == 0 {
		return nil
	}

	// Find max length in batch
	maxLen := 0
	for _, ids := range tokenIds {
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
	}
	if maxLen > t.maxLength {
		maxLen = t.maxLength
	}

	// Pad sequences
	padded := make([][]int, len(tokenIds))
	for i, ids := range tokenIds {
		padded[i] = make([]int, maxLen)
		copy(padded[i], ids)
		// Fill remaining positions with padding token
		for j := len(ids); j < maxLen; j++ {
			padded[i][j] = t.vocab["[PAD]"]
		}
	}

	return padded
}

// GetAttentionMask creates attention masks for padded sequences
func (t *Tokenizer) GetAttentionMask(tokenIds [][]int) [][]int {
	if len(tokenIds) == 0 {
		return nil
	}

	masks := make([][]int, len(tokenIds))
	for i, ids := range tokenIds {
		masks[i] = make([]int, len(ids))
		for j := range ids {
			if ids[j] == t.vocab["[PAD]"] {
				masks[i][j] = 0
			} else {
				masks[i][j] = 1
			}
		}
	}

	return masks
}

// Close cleans up any resources used by the tokenizer
func (t *Tokenizer) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Reset internal state
	t.config = TokenizerConfig{}
	t.vocab = nil
	t.preprocessor = nil

	return nil
}

// loadVocabularyFromTokenizerJSON loads the vocabulary from tokenizer.json or unigram.json
func loadVocabularyFromTokenizerJSON(tokenizerPath string) (map[string]int, error) {
	// First try to load unigram.json if it exists
	unigramPath := filepath.Join(filepath.Dir(tokenizerPath), "unigram.json")
	if _, err := os.Stat(unigramPath); err == nil {
		fmt.Printf("DEBUG: Found unigram.json, attempting to load vocabulary from it\n")
		file, err := os.Open(unigramPath)
		if err != nil {
			return nil, fmt.Errorf("failed to open unigram.json: %w", err)
		}
		defer file.Close()

		var unigramConfig struct {
			Type  string   `json:"type"`
			UnkID int      `json:"unk_id"`
			Vocab [][2]any `json:"vocab"` // [token, score] pairs where score can be float64 or int
		}

		if err := json.NewDecoder(file).Decode(&unigramConfig); err != nil {
			return nil, fmt.Errorf("failed to decode unigram.json: %w", err)
		}

		if unigramConfig.Type == "Unigram" {
			fmt.Printf("DEBUG: Successfully parsed Unigram vocabulary with %d tokens\n", len(unigramConfig.Vocab))
			vocab := make(map[string]int)
			for i, pair := range unigramConfig.Vocab {
				if token, ok := pair[0].(string); ok {
					vocab[token] = i
				}
			}
			return vocab, nil
		}
	}

	// If unigram.json doesn't exist or isn't valid, try tokenizer.json
	file, err := os.Open(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open tokenizer.json: %w", err)
	}
	defer file.Close()

	// Read the entire file content
	content, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json: %w", err)
	}

	fmt.Printf("DEBUG: Reading tokenizer file from: %s\n", tokenizerPath)
	fmt.Printf("DEBUG: First 200 bytes of content: %s\n", string(content[:200]))

	// Try to decode as a modern tokenizer first
	var modernConfig struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(content, &modernConfig); err == nil {
		fmt.Printf("DEBUG: Successfully parsed modern tokenizer format\n")
		vocab := make(map[string]int)

		// Add special tokens first
		for _, token := range modernConfig.AddedTokens {
			vocab[token.Content] = token.ID
			fmt.Printf("DEBUG: Added special token: %s with ID: %d\n", token.Content, token.ID)
		}

		// Add model vocabulary if present
		if len(modernConfig.Model.Vocab) > 0 {
			fmt.Printf("DEBUG: Found %d tokens in model vocabulary\n", len(modernConfig.Model.Vocab))
			for token, id := range modernConfig.Model.Vocab {
				vocab[token] = id
			}
		}

		if len(vocab) > 0 {
			return vocab, nil
		}
	} else {
		fmt.Printf("DEBUG: Failed to parse modern tokenizer format: %v\n", err)
	}

	// Try to decode as a BERT tokenizer
	var bertConfig struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}

	if err := json.Unmarshal(content, &bertConfig); err == nil && len(bertConfig.Model.Vocab) > 0 {
		fmt.Printf("DEBUG: Successfully parsed BERT tokenizer format with %d tokens\n", len(bertConfig.Model.Vocab))
		return bertConfig.Model.Vocab, nil
	} else {
		fmt.Printf("DEBUG: Failed to parse BERT tokenizer format: %v\n", err)
	}

	// Try to decode as a raw vocabulary map
	var rawVocab map[string]int
	if err := json.Unmarshal(content, &rawVocab); err == nil && len(rawVocab) > 0 {
		fmt.Printf("DEBUG: Successfully parsed raw vocabulary map with %d tokens\n", len(rawVocab))
		return rawVocab, nil
	} else {
		fmt.Printf("DEBUG: Failed to parse raw vocabulary map: %v\n", err)
	}

	// Try to decode as a token array
	var tokenArray []struct {
		Token string `json:"token"`
		Id    int    `json:"id"`
	}
	if err := json.Unmarshal(content, &tokenArray); err == nil && len(tokenArray) > 0 {
		fmt.Printf("DEBUG: Successfully parsed token array with %d tokens\n", len(tokenArray))
		vocab := make(map[string]int)
		for _, item := range tokenArray {
			vocab[item.Token] = item.Id
		}
		return vocab, nil
	} else {
		fmt.Printf("DEBUG: Failed to parse token array: %v\n", err)
	}

	return nil, fmt.Errorf("failed to parse token data: unsupported format")
}
