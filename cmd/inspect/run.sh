#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Set the ONNX Runtime library path
export DYLD_LIBRARY_PATH="$PROJECT_ROOT/onnxruntime-osx-arm64-1.14.0/lib:$DYLD_LIBRARY_PATH"

# Print environment for debugging
echo "Environment Information:"
echo "----------------------"
echo "DYLD_LIBRARY_PATH: $DYLD_LIBRARY_PATH"
echo "Working Directory: $(pwd)"
echo "Project Root: $PROJECT_ROOT"

# Print model paths
echo -e "\nModel Paths:"
echo "------------"
echo "MiniLM: $PROJECT_ROOT/models/all-MiniLM-L6-v2.onnx"
echo "MPNet: $PROJECT_ROOT/models/all-mpnet-base-v2.onnx"
echo "Multilingual: $PROJECT_ROOT/models/paraphrase-multilingual-MiniLM-L12-v2.onnx"

# Check if a specific model was requested
if [ -n "$1" ]; then
    case "$1" in
        "minilm")
            echo -e "\nInspecting MiniLM model..."
            make inspect-minilm
            ;;
        "mpnet")
            echo -e "\nInspecting MPNet model..."
            make inspect-mpnet
            ;;
        "multilingual")
            echo -e "\nInspecting Multilingual model..."
            make inspect-multilingual
            ;;
        *)
            echo "Unknown model: $1"
            echo "Available options: minilm, mpnet, multilingual"
            exit 1
            ;;
    esac
else
    # Run inspection for all models
    echo -e "\nInspecting all models..."
    make inspect-all
fi
