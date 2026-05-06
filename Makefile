.PHONY: help install lint test train serve docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

lint: ## Run linter
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

train: ## Train the model
	python -m src.models.train

serve: ## Start the API server
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build Docker images
	docker compose build

docker-up: ## Start all services with Docker
	docker compose up --build

docker-train: ## Train model in Docker
	docker compose run --rm trainer

docker-serve: ## Start API in Docker
	docker compose up api

clean: ## Clean generated files
	rm -rf models/*.joblib models/*.json
	rm -rf mlruns/
	rm -rf __pycache__ .pytest_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
