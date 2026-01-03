.PHONY: install dev test lint format clean docker-build docker-run help

# Default target
help:
	@echo "LLM Fine-Tuning Pipeline with LangChain"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup Targets:"
	@echo "  install      Install package"
	@echo "  dev          Install with dev dependencies"
	@echo "  install-llm  Install with LLM providers (Anthropic, OpenAI)"
	@echo ""
	@echo "Development Targets:"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Docker Targets:"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run Docker container"
	@echo ""
	@echo "Pipeline Targets:"
	@echo "  prepare-data CONFIG=path/to/config.yaml"
	@echo "  train CONFIG=path/to/config.yaml"
	@echo "  generate-qa INPUT=path/to/docs OUTPUT=output.jsonl"

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

install-llm:
	pip install -e ".[llm]"

install-all:
	pip install -e ".[dev,llm,flash-attn]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=finetune_project --cov-report=html

# Code quality
lint:
	ruff check src/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -t llm-finetune-langchain .

docker-run:
	docker run --gpus all -it -v $(PWD)/data:/workspace/data llm-finetune-langchain

# CLI shortcuts
prepare-data:
	python -m finetune_project prepare-data --config $(CONFIG)

prepare-data-qa:
	python -m finetune_project prepare-data --config $(CONFIG) --enable-qa

prepare-data-augment:
	python -m finetune_project prepare-data --config $(CONFIG) --augment

train:
	python -m finetune_project train --config $(CONFIG)

evaluate:
	python -m finetune_project evaluate --config $(CONFIG)

generate-qa:
	python -m finetune_project generate-qa --input $(INPUT) --output $(OUTPUT)

# Config initialization
init-config:
	python -m finetune_project init --output config.yaml

init-qa-config:
	python -m finetune_project init --output config.yaml --template qa_generation

validate-config:
	python -m finetune_project validate --config $(CONFIG)
