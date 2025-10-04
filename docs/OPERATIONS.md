# Operations Guide

## Environment variables

Key variables (see `.env.example`):

- `DATA_ROOT` — location for raw/interim/processed datasets
- `LEDGER_CONFIG` — path to YAML config consumed by Rust server and trainer
- `CUDA_VISIBLE_DEVICES` — GPU selection when running CUDA kernels
- `LEDGER_LOG_LEVEL` — overrides tracing/structured logging level
- `WANDB_API_KEY` — optional experiment tracking credentials

Load a local override with:

```bash
cp .env.example .env
source .env
```

## Make targets

- `make bootstrap` — dry-run plan (no changes)
- `make setup` — full bootstrap (`--apply`)
- `make install` — ensure Python venv + npm ci + cargo fetch
- `make lint` — ruff/black/isort + cargo fmt/clippy + npm lint
- `make typecheck` — mypy, tsc (workspace), cargo check
- `make test` — pytest, cargo test, npm run test
- `make run` — prints engine banner via CLI
- `make precommit` — run pre-commit hooks locally
- `make docker-build` — builds multi-stage image with server + UI
- `make docker-up` / `make docker-down` — compose orchestration

## CI/CD pipeline

GitHub Actions workflow `.github/workflows/ci.yml` runs on push and PR:

1. Checkout with submodules
2. Install Rust 1.76 + clippy/rustfmt
3. Setup Python 3.10–3.12 matrix & cache virtualenv
4. Install Node 20 + npm cache
5. Run lint, typecheck, test jobs mirroring `make`

Artifacts are not persisted yet; extend `ops/ci/` for future pipelines (e.g., notebook regression, docker publish).

## Docker operations

- `docker-compose.yml` provides a single-service stack exposing port 8080
- Health checks rely on `curl http://localhost:8080/healthz`
- Bind-mount `configs/` and `data/` for reproducible state
- Image base: `python:3.11-slim` with Rust-built `ledger-server`

## Infra roadmap

- Populate `infra/terraform` for cloud deploys; include remote state backend
- Populate `infra/k8s` with Helm charts for server + vector memory + UI
- Add secrets management (e.g., SOPS or Vault) as dependencies grow

## Operational playbooks

- **Restart server**: `docker compose restart ledger`
- **Rotate configs**: update `configs/ledger.default.yaml`, restart service
- **Upgrade dependencies**: bump versions in manifests, regenerate locks, run CI
- **Audit modal ledger**: inspect `data/processed/runs/ledger.json`

Document service-level objectives and monitoring plans in `docs/design/` as features mature.
