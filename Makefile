.PHONY: help install install-dev test test-cov lint format clean build upload docs

help:
	@echo "Available commands:"
	@echo "  install     Install package"
	@echo "  install-dev Install package in development mode with dev dependencies"
	@echo "  test        Run tests"
	@echo "  test-cov    Run tests with coverage"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  upload      Upload to PyPI"
	@echo "  docs        Build documentation"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=graphizy --cov-report=html --cov-report=term-missing

lint:
	flake8 src/graphizy tests/
	mypy src/graphizy

format:
	black src/graphizy tests/ examples/
	isort src/graphizy tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

docs:
	@echo "Documentation generation not yet implemented"