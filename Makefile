.PHONY: all build clean test coverage lint run install

# Build variables
BINARY_NAME=athena
GO=go
GOFLAGS=-v
GOOS?=$(shell $(GO) env GOOS)
GOARCH?=$(shell $(GO) env GOARCH)

# Project variables
PROJECT_NAME=athena
PROJECT_ROOT=$(shell pwd)

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

# Docker commands (for future use)
docker-build:
	docker build -t $(PROJECT_NAME) .

docker-run:
	docker run -p 8080:8080 $(PROJECT_NAME)