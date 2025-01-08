package compression

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
)

// Compressor handles data compression and decompression
type Compressor struct {
	// Threshold in bytes above which compression is applied
	Threshold int
}

// Compress compresses the input data using gzip
func (c *Compressor) Compress(data []byte) ([]byte, error) {
	if len(data) <= c.Threshold {
		return data, nil
	}

	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)

	_, err := writer.Write(data)
	if err != nil {
		return nil, fmt.Errorf("failed to write compressed data: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close gzip writer: %w", err)
	}

	return buf.Bytes(), nil
}

// Decompress decompresses gzipped data
func (c *Compressor) Decompress(data []byte) ([]byte, error) {
	// Check if data is compressed by looking for gzip magic number
	if len(data) < 2 || data[0] != 0x1f || data[1] != 0x8b {
		return data, nil
	}

	reader, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer reader.Close()

	decompressed, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read decompressed data: %w", err)
	}

	return decompressed, nil
}
