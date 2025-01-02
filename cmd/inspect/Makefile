.PHONY: run inspect-all inspect-minilm inspect-mpnet inspect-multilingual clean

# Default target to inspect all models
run: inspect-all

# Inspect all models
inspect-all:
	DYLD_LIBRARY_PATH="../../onnxruntime-osx-arm64-1.14.0/lib" \
	go run main.go

# Inspect specific models
inspect-minilm:
	DYLD_LIBRARY_PATH="../../onnxruntime-osx-arm64-1.14.0/lib" \
	MODEL_PATH="../../models/all-MiniLM-L6-v2.onnx" \
	go run main.go

inspect-mpnet:
	DYLD_LIBRARY_PATH="../../onnxruntime-osx-arm64-1.14.0/lib" \
	MODEL_PATH="../../models/all-mpnet-base-v2.onnx" \
	go run main.go

inspect-multilingual:
	DYLD_LIBRARY_PATH="../../onnxruntime-osx-arm64-1.14.0/lib" \
	MODEL_PATH="../../models/paraphrase-multilingual-MiniLM-L12-v2.onnx" \
	go run main.go

# Clean build artifacts
clean:
	rm -f inspect
