.PHONY: all build clean test coverage lint run install test-embeddings test-embeddings-debug

# Build variables
BINARY_NAME=athena
GO=go
GOFLAGS=-v
GOOS?=$(shell $(GO) env GOOS)
GOARCH?=$(shell $(GO) env GOARCH)

# Project variables
PROJECT_NAME=athena
PROJECT_ROOT=$(shell pwd)

# Test variables
TEST_ENV=TEST_MODELS_DIR=$(PROJECT_ROOT)/models TEST_DATA_PATH=$(PROJECT_ROOT)/test/integration/testdata DYLD_LIBRARY_PATH=$(PROJECT_ROOT)/onnxruntime-osx-arm64-1.14.0/lib
DEBUG_ENV=ZEROLOG_LEVEL=debug

all: lint test build

build:
	$(GO) build $(GOFLAGS) -o bin/$(BINARY_NAME) ./cmd/athena

clean:
	rm -rf bin/
	rm -f coverage.out

test:
	$(GO) test -v -race ./...

test-embeddings:
	$(TEST_ENV) $(GO) test -v ./test/unit -run TestONNXService
	$(TEST_ENV) $(GO) test -v ./test/integration -run TestONNXIntegration

test-embeddings-unit:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/unit -run TestONNXService

test-embeddings-integration:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/integration -run TestONNXIntegration

test-embeddings-debug: test-embeddings-unit test-embeddings-integration

coverage:
	$(GO) test -v -coverprofile=coverage.out ./...
	$(GO) tool cover -html=coverage.out

lint:
	golangci-lint run

run:
	$(GO) run ./cmd/athena

install:
	$(GO) mod download

generate:
	$(GO) generate ./...

# Development helpers
dev: install generate build run

# Docker commands (for future use)
docker-build:
	docker build -t $(PROJECT_NAME) .

docker-run:
	docker run -p 8080:8080 $(PROJECT_NAME)

# Embeddings test targets
.PHONY: test-embeddings test-embeddings-unit test-embeddings-integration test-embeddings-bench test-embeddings-all test-embeddings-similarity

test-embeddings-unit:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/unit -run TestONNXService

test-embeddings-integration:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/integration -run TestONNXClient

test-embeddings-bench:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -v ./test/integration -run=^$ -bench=BenchmarkONNXClient

test-embeddings-similarity:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/integration -run "TestONNXClient/.*Semantic_Similarity$$"

test-embeddings: test-embeddings-unit test-embeddings-integration

test-embeddings-all: test-embeddings test-embeddings-bench test-embeddings-similarity