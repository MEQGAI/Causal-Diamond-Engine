# Architecture Overview

Reality's Ledger implements a causal-diamond reasoning engine composed of three coupled channels:

- **Geometric channel** constrains compute capacity and structural priors
- **Entanglement channel** gathers evidence via retrieval and attention modules
- **Modal ledger** records uncertainty collapse under null-stability checks

## Rust core

The Rust workspace provides deterministic execution:

- `engine/` exposes `CausalDiamondEngine`, orchestrating entangle → extremize → stability gate
- `engine/src/diamond` models compute budgets, `geometry` validates budgets, `modal` emits updates
- `engine/src/ledger` stores accepted updates; `stability` enforces KL-bounded gates
- `apps/cli` wraps the engine in a low-latency CLI
- `apps/server` serves HTTP (`/healthz`) for operators and automation
- `engine/bindings/python` delivers a PyO3 shim (`ledger_python`) for research workflows

## Python research stack

- `python/trainer` offers a CLI-compatible trainer skeleton
- `python/losses` implements modal penalties + null-stability heuristics
- `python/datasets` catalogues curricula for experiments
- `tests/` contains pytest smoke checks across the Python API
- Virtualenv pinned at Python 3.11 with Torch, Transformers, FAISS, Hydra, Typer

## Tooling & memory

- `memory/` (vector, graph, views) — placeholders for FAISS/HNSW and knowledge graph backends
- `retrievers/` (text, code, multimodal) — retrieval strategies hooking into the entanglement channel
- `tools/` (planner, sandbox, connectors) — deterministic tool execution with auditable effects
- `cpp/kernels` — CUDA kernels (e.g., `null_stability.cu`) for high-throughput KL filtering

## Web & docs

- `web/ui` — React + Vite operator console; compiles to `/web/ui/dist`
- `web/docs-site` — TypeDoc-powered documentation builder
- `docs/` — conceptual, design, and API references; devops docs in `docs/DEV_SETUP.md` and `docs/OPERATIONS.md`

## Infra & ops

- `infra/` — placeholder for Terraform / Kubernetes manifests
- `ops/ci` & `.github/workflows/ci.yml` — GitHub Actions matrix (Ubuntu & macOS, Python 3.10–3.12)
- `docker-compose.yml` orchestrates the Rust server with mounted configs/data

## Data layout

```
data/
  raw/        # immutable inputs from external sources
  external/   # vendored corpora or embeddings
  interim/    # on-disk scratch space during training
  processed/  # committed ledgers & model artifacts
```

Git ignores the data tree by default; `.env.example` sets `DATA_ROOT` pointing to this location.

## Extension points

- `external/sources.yml` documents third-party repos used via packages or future submodules
- Pre-commit + CI ensure ruff/black/isort, cargo fmt/clippy, mypy, npm lint/test
- `configs/` centralises ledger defaults (`ledger.default.yaml`) and tool version snapshots

The bootstrap process ensures all layers are wired while remaining idempotent; `make setup` triggers the full chain and runs smoke checks.
