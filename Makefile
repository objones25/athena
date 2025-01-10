.PHONY: all build clean test coverage lint run install test-storage test-storage-unit test-storage-integration test-storage-debug test-env-up test-env-down test-embedding test-embedding-unit

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
TEST_ENV=TEST_DATA_PATH=$(PROJECT_ROOT)/test/integration/testdata
DEBUG_ENV=ZEROLOG_LEVEL=debug
HUGGINGFACE_API_KEY ?= $(shell cat .env | grep HUGGINGFACE_API_KEY | cut -d '=' -f2)
EMBEDDING_TEST_ENV=HUGGINGFACE_API_KEY=$(HUGGINGFACE_API_KEY)

all: lint test build

build:
	$(GO) build $(GOFLAGS) -o bin/$(BINARY_NAME) ./cmd/athena

clean:
	rm -rf bin/
	rm -f coverage.out

test:
	$(GO) test -v -race ./...

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

# Docker commands
docker-build:
	docker build -t $(PROJECT_NAME) .

docker-run:
	docker run -p 8080:8080 $(PROJECT_NAME)

# Test environment commands
test-env-up:
	@if [ -z "$(HUGGINGFACE_API_KEY)" ]; then \
		echo "Error: HUGGINGFACE_API_KEY environment variable is not set"; \
		exit 1; \
	fi
	mkdir -p volumes/milvus
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services to be healthy..."
	@echo "Waiting for Redis..."
	@until docker-compose -f docker-compose.test.yml exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; do \
		sleep 1; \
	done
	@echo "Redis is ready"
	@echo "Waiting for Milvus..."
	@until curl -s http://localhost:9091/healthz 2>/dev/null | grep -q "OK"; do \
		sleep 1; \
	done
	@echo "Milvus is ready"
	@echo "Waiting for MinIO..."
	@until curl -s http://localhost:9000/minio/health/live 2>/dev/null | grep -q "ok"; do \
		sleep 1; \
	done
	@echo "MinIO is ready"
	@echo "Waiting for ETCD..."
	@until docker-compose -f docker-compose.test.yml exec -T etcd etcdctl endpoint health 2>/dev/null | grep -q "healthy"; do \
		sleep 1; \
	done
	@echo "ETCD is ready"
	@echo "All services are ready"

test-env-down:
	docker-compose -f docker-compose.test.yml down -v
	rm -rf volumes

# Storage test targets
test-storage: test-env-up
	$(TEST_ENV) $(GO) test -v ./test/unit -run "TestStorage|TestShardedCache|TestMilvusStore"
	$(TEST_ENV) $(GO) test -v ./test/integration -run "TestStorageIntegration"
	make test-env-down

test-storage-unit:
	$(TEST_ENV) $(DEBUG_ENV) MILVUS_HOST=localhost MILVUS_PORT=19530 REDIS_ADDR=localhost:6379 $(GO) test -count=1 -v ./test/unit -run "TestStorage|TestShardedCache|TestMilvusStore"

test-storage-integration: test-env-up
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/integration -run "TestStorageIntegration"
	make test-env-down

test-storage-debug: test-storage-unit test-storage-integration

# Embedding test targets
test-embedding: test-env-up
	@if [ -z "$(HUGGINGFACE_API_KEY)" ]; then \
		echo "Error: HUGGINGFACE_API_KEY environment variable is not set. Run with: make test-embedding HUGGINGFACE_API_KEY=your_api_key"; \
		exit 1; \
	fi
	$(TEST_ENV) $(EMBEDDING_TEST_ENV) $(GO) test -v ./test/unit -run "TestHuggingFaceService|TestMilvusStore_.*WithEmbeddings"
	make test-env-down

test-embedding-unit:
	@if [ -z "$(HUGGINGFACE_API_KEY)" ]; then \
		echo "Error: HUGGINGFACE_API_KEY environment variable is not set. Run with: make test-embedding-unit HUGGINGFACE_API_KEY=your_api_key"; \
		exit 1; \
	fi
	$(TEST_ENV) $(DEBUG_ENV) $(EMBEDDING_TEST_ENV) MILVUS_HOST=localhost MILVUS_PORT=19530 REDIS_ADDR=localhost:6379 $(GO) test -count=1 -v ./test/unit -run "TestHuggingFaceService|TestMilvusStore_.*WithEmbeddings"