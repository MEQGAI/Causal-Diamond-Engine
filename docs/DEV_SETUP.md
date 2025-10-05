# Developer Setup

This guide describes how to prepare a workstation for Reality's Ledger.

## Prerequisites

- Linux (Ubuntu/Fedora/Arch), macOS, or Windows (MSYS2)
- Git ≥ 2.40, CMake ≥ 3.24, Python 3.11, Node.js ≥ 20, Rust toolchain 1.76
- Optional: CUDA 12.1 for accelerated kernels, Docker for container workflows

## Bootstrap workflow

1. Inspect the dry-run plan:

   ```bash
   make bootstrap
   ```

2. Apply the full setup (system packages, virtualenv, npm install, cargo fetch):

   ```bash
   make setup
   ```

3. Activate the Python environment and verify smoke checks:

   ```bash
   source .venv/bin/activate
   make install
   make lint
   make test
   ```

## Python environment

- Virtualenv lives at `.venv/`
- Runtime dependencies: `requirements.txt`
- Tooling dependencies: `requirements-dev.txt`
- Entry point: `python -m python.trainer.run --task=tool_reasoning --budget=2.0`
- Launch toy run + eval: `make train-toy`
- Full run (resume-aware): `make train`
- Scripted run: `./scripts/train_and_eval.sh --config configs/train/toy_local.yaml --steps 50 --thresholds configs/eval/thresholds.toy.json`
- Integration smoke tests:
  - Python path: `pytest tests/integration/test_training_flow.py`
  - Serving API: `pytest tests/integration/test_serving.py`
  - CLI (requires Rust): `pytest tests/integration/test_cli.py -m "slow"`

## Rust toolchain

- `rust-toolchain.toml` pin points to 1.76.0 with rustfmt/clippy
- Workspace crates:
  - `engine` — causal-diamond core
  - `apps/cli` — operator CLI
  - `apps/server` — HTTP server
  - `model/fm_bindings` — PyO3 bindings

## Node / TypeScript

- Root `package.json` orchestrates workspaces `web/ui` and `web/docs-site`
- Run `npm ci` after bootstrap to materialize dependencies
- Development scripts: `npm run lint`, `npm run typecheck`, `npm run build`

## Optional components

- CUDA kernels: build via `cmake -S cpp -B build && cmake --build build`
- Docker image: `make docker-build`
- Terraform/K8s manifests placeholder located in `infra/`

## Updating locks

- Python: update versions in `requirements*.txt`
- Rust: `cargo update` then commit `Cargo.lock`
- Node: `npm update` + regenerate `package-lock.json`

## Troubleshooting

- Ensure `scripts/detect_os.sh` resolves to a supported target
- Use `make clean` to wipe build artifacts if compilation fails
- Regenerate pre-commit hooks: `make precommit`
