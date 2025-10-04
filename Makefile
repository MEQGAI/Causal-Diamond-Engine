.PHONY: setup fmt lint test build_wheels run_server train eval docker_build

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ./model
	$(PYTHON) -m pip install -r requirements-dev.txt

fmt:
	$(PYTHON) -m ruff check --fix model tests
	$(PYTHON) -m black model tests
	cargo fmt

lint:
	$(PYTHON) -m ruff check model tests
	$(PYTHON) -m black --check model tests
	cargo fmt -- --check

test:
	$(PYTHON) -m pytest -q
	cargo test -p engine
	cargo test --manifest-path serving/rust/Cargo.toml

build_wheels:
	maturin build -m model/fm_bindings/Cargo.toml
	$(PYTHON) -m pip install -e ./model
	$(PYTHON) model/fm_kernels/setup.py build

run_server:
	uvicorn serving.python.src.app:app --reload

train:
	$(PYTHON) -m fm_train.trainer.run --config configs/train/default.yaml

eval:
	$(PYTHON) -m fm_eval.runner --config configs/eval/kill_numbers.yaml

docker_build:
	docker build -f Dockerfile -t foundation/workspace:latest .
