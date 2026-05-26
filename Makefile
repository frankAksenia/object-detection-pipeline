.PHONY: help install install-dev format lint test check api docker-build docker-up docker-down clean

PYTHON := python
PIP := pip

API_HOST := 0.0.0.0
API_PORT := 8000

DATASET_ROOT := /content/drive/MyDrive/mlops-visdrone/data/processed/visdrone-yolo
RUNS_DIR := /content/drive/MyDrive/mlops-visdrone/runs
EVAL_DIR := /content/drive/MyDrive/mlops-visdrone/evaluation
REPORTS_DIR := /content/drive/MyDrive/mlops-visdrone/reports

TRAIN_CONFIG ?= configs/training/yolo26n_img960_e100.yaml
EVAL_CONFIG ?= configs/training/yolo26n_img960_e100.yaml
MODEL_PATH ?= /content/drive/MyDrive/mlops-visdrone/runs/yolo26n_img960_e100/weights/best.pt
METRICS_OUTPUT ?= /content/drive/MyDrive/mlops-visdrone/reports/metrics/yolo26n_img960_e100.json

help:
	@echo "Available commands:"
	@echo "  make install        Install API/runtime dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make format         Format code with Black and Ruff"
	@echo "  make lint           Run Ruff and Black checks"
	@echo "  make test           Run tests"
	@echo "  make check          Run lint + tests"
	@echo "  make api            Run FastAPI locally"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Run Docker Compose"
	@echo "  make docker-down    Stop Docker Compose"
	@echo "  make train          Train YOLO using TRAIN_CONFIG"
	@echo "  make evaluate       Evaluate YOLO using EVAL_CONFIG and MODEL_PATH"
	@echo "  make dataset-stats  Compute dataset statistics"
	@echo "  make clean          Remove local caches"

install:
	$(PIP) install -r requirements-api.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

format:
	ruff check src tests --fix
	black src tests

lint:
	ruff check src tests
	black --check src tests

test:
	pytest tests -q

check: lint test

api:
	uvicorn src.serving.app:app --host $(API_HOST) --port $(API_PORT) --reload

docker-build:
	docker compose build --no-cache

docker-up:
	docker compose up

docker-down:
	docker compose down

train:
	$(PYTHON) src/training/train.py \
		--config $(TRAIN_CONFIG) \
		--exist-ok

evaluate:
	$(PYTHON) src/training/evaluate.py \
		--config $(EVAL_CONFIG) \
		--model $(MODEL_PATH) \
		--metrics-output $(METRICS_OUTPUT) \
		--exist-ok

dataset-stats:
	$(PYTHON) src/data/dataset_stats.py \
		--labels-dir $(DATASET_ROOT)/labels/train \
		--output $(REPORTS_DIR)/dataset_stats_train.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete