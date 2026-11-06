.PHONY: install test test-cov lint format run clean help

PYTHON := .venv/bin/python
UV := uv

help:
	@echo "install   Create .venv with Python 3.11 and install the project (editable) + dev tools"
	@echo "test      Run the test suite"
	@echo "test-cov  Run the test suite with coverage"
	@echo "lint      Run ruff"
	@echo "format    Run ruff format"
	@echo "run       Start the Streamlit app"

install:
	$(UV) venv --python 3.11 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=src --cov=config --cov=dashboard --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run:
	$(PYTHON) -m streamlit run app.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
