.PHONY: help install install-dev test test-cov lint format format-check check clean build upload docs

help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install package in development mode with dev dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage (html, xml, term)"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  format       Format code (black, isort)"
	@echo "  format-check Check formatting without changing files"
	@echo "  check        Run all checks (lint, format-check)"
	@echo "  clean        Clean build artifacts, pycache, and coverage reports"
	@echo "  build        Build package for distribution"
	@echo "  upload       Upload package to PyPI"
	@echo "  docs         Build HTML documentation from docstrings"

install:
	pip install .

install-dev:
	@echo "Note: Ensure 'pdoc' is in your dev dependencies for the 'docs' command."
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=graphizy --cov-report=html --cov-report=xml --cov-report=term-missing

lint:
	flake8 src/graphizy tests/
	mypy src/graphizy

format:
	black .
	isort .

format-check:
	black --check .
	isort --check .

check: lint format-check

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf docs/api/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	@echo "Building documentation with pdoc..."
	pdoc --html src/graphizy -o docs/api
	@echo "Documentation built in docs/api/"