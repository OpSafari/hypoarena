PY := .venv/bin/python
SCOPE := src tests examples scripts

.PHONY: help build test test-all format format-check lint typecheck release clean demo check-package

help:
	@echo "build         - build a wheel with the local venv (no isolation)"
	@echo "test          - fast test run (excludes slow and model-marked tests)"
	@echo "test-all      - full test run including slow/model tests"
	@echo "format        - ruff format + import/rule autofix over $(SCOPE)"
	@echo "format-check  - ruff format --check + ruff check over $(SCOPE)"
	@echo "lint          - ruff check and format --check"
	@echo "typecheck     - mypy over src/hypoarena and scripts"
	@echo "release       - clean, build sdist+wheel, smoke-test the wheel"
	@echo "demo          - run the offline end-to-end demo"

build:
	.venv/bin/python -m build --wheel --no-isolation

test:
	.venv/bin/python -m pytest -q -m "not slow"

test-all:
	.venv/bin/python -m pytest -q

format:
	.venv/bin/ruff format $(SCOPE)
	.venv/bin/ruff check --fix $(SCOPE)

format-check:
	.venv/bin/ruff format --check src tests examples scripts && .venv/bin/ruff check src tests examples scripts

lint:
	.venv/bin/ruff check $(SCOPE)
	.venv/bin/ruff format --check $(SCOPE)

typecheck:
	.venv/bin/mypy

release: clean
	$(PY) -m build
	$(PY) scripts/check_package.py

check-package: build
	$(PY) scripts/check_package.py

demo:
	$(PY) -m hypoarena demo

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
