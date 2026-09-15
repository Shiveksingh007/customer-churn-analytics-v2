.PHONY: notebooks export-notebooks install install-dev lint test ci

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

notebooks: export-notebooks

export-notebooks:
	python scripts/export_notebooks.py

lint:
	ruff check src tests app scripts

test:
	pytest tests/

ci: lint test
