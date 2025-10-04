SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
NPM := npm
TypeScriptTargets := web/ui web/docs-site

.PHONY: help bootstrap setup install lint test format typecheck precommit run clean docker-build docker-up docker-down

help:
	@echo "Available targets:"
	@echo "  bootstrap    - Dry-run bootstrap plan"
	@echo "  setup        - Full bootstrap with --apply"
	@echo "  install      - Install Python/Rust/Node dependencies"
	@echo "  lint         - Run Python, Rust, and JS linters"
	@echo "  test         - Execute unit tests across stacks"
	@echo "  format       - Auto-format source files"
	@echo "  typecheck    - Static analysis for Python, TS, Rust"
	@echo "  precommit    - Run pre-commit hooks on all files"
	@echo "  run          - Launch CLI banner smoke test"
	@echo "  clean        - Remove build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up    - Start docker-compose stack"
	@echo "  docker-down  - Stop docker-compose stack"

bootstrap:
	bash scripts/bootstrap.sh

setup:
	bash scripts/bootstrap.sh --apply

$(VENV): requirements.txt requirements-dev.txt
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

install: $(VENV)
	@echo "Installing Rust deps"
	cargo fetch
	@echo "Installing Node deps"
	$(NPM) ci

lint: $(VENV)
	$(VENV)/bin/ruff check python tests
	$(VENV)/bin/black --check python tests
	$(VENV)/bin/isort --check-only python tests
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	$(NPM) run lint

format: $(VENV)
	$(VENV)/bin/ruff check python tests --fix --unsafe-fixes
	$(VENV)/bin/black python tests
	$(VENV)/bin/isort python tests
	cargo fmt --all
	$(NPM) run format

typecheck: $(VENV)
	$(VENV)/bin/mypy python
	@for pkg in $(TypeScriptTargets); do \
		$(NPM) run typecheck -w $$pkg || exit 1; \
	done
	cargo check --all

pytest:
	$(VENV)/bin/pytest -q

jstest:
	$(NPM) run test

rusttest:
	cargo test --all

test: $(VENV)
	make pytest
	make rusttest
	make jstest

precommit: $(VENV)
	$(VENV)/bin/pre-commit run --all-files

run:
	cargo run -p ledger-cli -- banner

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache target node_modules web/ui/dist web/docs-site/dist

DockerImage := ledger-engine

docker-build:
	docker build -t $(DockerImage) .

docker-up:
	docker compose up -d

docker-down:
	docker compose down
