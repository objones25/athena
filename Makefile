.PHONY: all build clean test coverage lint run install test-storage test-storage-unit test-storage-integration test-storage-debug test-env-up test-env-down

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
	mkdir -p volumes/milvus
	docker-compose -f docker-compose.test.yml up -d
	@echo "Waiting for services to be healthy..."
	@for i in $$(seq 1 60); do \
		if docker-compose -f docker-compose.test.yml ps | grep -q "(healthy)"; then \
			echo "Test environment is ready"; \
			break; \
		fi; \
		if [ $$i -eq 60 ]; then \
			echo "Timeout waiting for services to be healthy"; \
			exit 1; \
		fi; \
		sleep 1; \
	done

test-env-down:
	docker-compose -f docker-compose.test.yml down -v
	rm -rf volumes

# Storage test targets
test-storage: test-env-up
	$(TEST_ENV) $(GO) test -v ./test/unit -run "TestShardedCache"
	$(TEST_ENV) $(GO) test -v ./test/integration -run "TestStorageIntegration"
	make test-env-down

test-storage-unit:
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/unit -run "TestShardedCache"

test-storage-integration: test-env-up
	$(TEST_ENV) $(DEBUG_ENV) $(GO) test -count=1 -v ./test/integration -run "TestStorageIntegration"
	make test-env-down

test-storage-debug: test-storage-unit test-storage-integration