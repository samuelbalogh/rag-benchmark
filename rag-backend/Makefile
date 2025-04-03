.PHONY: test test-unit test-cov lint format clean help

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run only unit tests"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make lint       - Run linting checks (black, isort, mypy)"
	@echo "  make format     - Format code with black and isort"
	@echo "  make clean      - Clean up cache and build files"

test:
	poetry run pytest

test-unit:
	poetry run pytest tests/unit

test-cov:
	poetry run pytest --cov=. --cov-report=term --cov-report=html

lint:
	poetry run black --check .
	poetry run isort --check-only --profile black .
	poetry run mypy .

format:
	poetry run black .
	poetry run isort --profile black .

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf __pycache__
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} + 