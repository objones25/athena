#!/bin/bash
set -e

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    brew install python
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 not found. Installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
fi

# Create models directory if it doesn't exist
mkdir -p models

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install --user -r scripts/requirements.txt

# Convert model to ONNX format
echo "Converting BERT model to ONNX format..."
python3 scripts/convert_model.py

echo "Model converted successfully to models/bert-base-embeddings.onnx"

# Download ONNX Runtime
echo "Downloading ONNX Runtime..."
ONNX_VERSION="1.14.0"
ONNX_RUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"

curl -L -o onnxruntime.tgz "$ONNX_RUNTIME_URL"
tar xzf onnxruntime.tgz
rm onnxruntime.tgz

echo "Setup completed successfully!" 