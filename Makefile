# rag-advanced-2023 Makefile, developer convenience commands
# Author: Malav Patel
# Usage: make <target>

.PHONY: help install dev test lint format clean docker-up docker-down docker-build worker

PYTHON := python3
PIP    := pip
PYTEST := pytest
APP    := src.api.main:app

# ── default ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "rag-advanced-2023, available make targets:"
	@echo ""
	@echo "  install       Install all dependencies"
	@echo "  dev           Start the FastAPI dev server (hot reload)"
	@echo "  worker        Start the Celery worker"
	@echo "  frontend      Start the Streamlit UI"
	@echo "  test          Run the full test suite with coverage"
	@echo "  lint          Run flake8 + isort check"
	@echo "  format        Auto-format with black + isort"
	@echo "  clean         Remove caches and build artifacts"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-up     Start all services with Docker Compose"
	@echo "  docker-down   Stop all Docker services and remove volumes"
	@echo ""

# ── local development ─────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

dev:
	uvicorn $(APP) --host 0.0.0.0 --port 8000 --reload

worker:
	celery -A src.worker.celery_app worker --loglevel=info

frontend:
	streamlit run frontend/streamlit_app.py

# ── quality ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=xml --tb=short

test-unit:
	$(PYTEST) tests/unit/ -v --tb=short

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short

lint:
	flake8 src/ frontend/ tests/ --max-line-length=100
	isort --check-only --diff src/ frontend/ tests/

format:
	black src/ frontend/ tests/ --line-length 100
	isort src/ frontend/ tests/

# ── cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage coverage.xml htmlcov/ .pytest_cache/ dist/ build/ *.egg-info

# ── docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t rag_naive:latest .

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker-compose -f docker/docker-compose.yml down -v

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f
