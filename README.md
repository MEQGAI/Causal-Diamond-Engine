# MEQG Diamond — A Reasoning‑First Foundation Stack

**Causal‑Diamond Engine + Modal Ledger + Null‑Stability Gate**

[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license) [![status: research-preview](https://img.shields.io/badge/status-research--preview-orange)](#status) [![docs](https://img.shields.io/badge/docs-theory%20&%20api-informational)](#documentation)

**MEQG Diamond** is a foundation‑model stack optimized for **long‑horizon, tool‑using, retrieval‑heavy reasoning**. It introduces:

* a **Causal‑Diamond optimization episode** (bounded time/FLOP/memory),
* a **Modal Ledger** that measures information discarded when committing to a working view,
* a **Null‑Stability Gate** that accepts only locally stable updates (trust‑region + Armijo + KL‑smoothness).

> **Design goal:** trade a bit of raw fluency for **stability, auditability, and sample‑efficiency**—with measurable reductions in cascade errors and better calibration under tight budgets.

---

## Table of contents

1. [Quickstart](#quickstart)
2. [Checkpoints & Sizes](#checkpoints--sizes)
3. [Architecture](#architecture)
4. [Objective (precise math)](#objective-precise-math)
5. [Null‑Stability (Acceptance Criteria)](#nullstability-acceptance-criteria)
6. [Memory, Retrieval & Views](#memory-retrieval--views)
7. [Tool Use](#tool-use)
8. [Evaluation](#evaluation)
9. [Performance & Hardware](#performance--hardware)
10. [Reproducibility](#reproducibility)
11. [API Reference](#api-reference)
12. [Safety, Limitations & Intended Use](#safety-limitations--intended-use)
13. [Project Layout](#project-layout)
14. [Roadmap & Milestones](#roadmap--milestones)
15. [Citations & Background](#citations--background)
16. [License](#license)

---

## Quickstart

> **Prereqs:** Python ≥ 3.10, CUDA‑enabled PyTorch (or CPU), `gcc`/`clang` for optional fused ops.

```bash
# 1) Install
git clone https://github.com/MEQGAI/meqg-diamond.git
cd meqg-diamond
pip install -e ./model  # dev install for all fm_* packages

# 2) (Optional) Bootstrap helpers
make setup  # installs dev dependencies + hooks

# 3) (Optional) Download a toy checkpoint
# See: assets/checkpoints/README.md for links
```

**Run an inference episode (a “diamond”) with a FLOP/time budget and a Python tool:**

```python
from diamond.runtime import DiamondEngine, Budget, Tools

engine = DiamondEngine.from_pretrained("meqg-diamond-tiny-220M")
budget = Budget(seconds=2.0, max_flops=3e11, max_kv_bytes=128*1024*1024)

resp = engine.run(
    prompt="What will the temperature be in Paris next weekend? Use the weather tool.",
    tools=[Tools.python_sandbox()],       # plug tools here
    budget=budget,
    telemetry=True                        # emits ledger & gate stats
)

print(resp.text)
print(resp.telemetry["modal_ledger"])     # KL ledger entries
```

**CLI (one‑liner):**

```bash
diamond run \
  --model meqg-diamond-tiny-220M \
  --budget.seconds 2.0 \
  --tools python \
  --prompt "Solve: (x+3)(x-2)=0. Show steps."
```

### Monorepo workflow

The new monorepo exposes consistent automation via `make`:

```bash
make setup         # pip install -e ./model + dev dependencies
make lint          # ruff + black + cargo fmt --check
make test          # pytest + cargo test (engine + serving)
make build_wheels  # maturin build + kernel stubs
make run_server    # uvicorn serving.python.src.app:app --reload
make train         # fm_train CLI w/ configs/train/default.yaml
make eval          # fm_eval kill-numbers gate
```

The CI workflows mirror these targets (`.github/workflows/python.yml`, `rust.yml`, `docker.yml`, `docs.yml`).

**Training (end‑to‑end sample command):**

```bash
diamond train \
  --model fresh-tiny-220M \
  --data.path ./data/multihop_synthetic \
  --optim.lr 2e-4 --train.batch_size 256 \
  --budget.seconds 1.0 --budget.max_flops 1e11 \
  --obj.lambda_mod 0.5 --obj.lambda_geo.kv_bytes 1e-9 \
  --null.alpha 0.1 --null.trust_radius 0.02 --null.l_mod 5.0
```

---

## Checkpoints & Sizes

> All checkpoints use the **same objective** and **gate**, differing only in backbone size and context window.

| Name                      | Params | Context | dtype | Notes                  |
| ------------------------- | ------ | ------- | ----: | ---------------------- |
| `meqg-diamond-tiny-220M`  | 0.22B  | 8k      |  bf16 | Toy / fast prototyping |
| `meqg-diamond-small-1.3B` | 1.3B   | 16k     |  bf16 | Research default       |
| `meqg-diamond-base-7B`    | 7.0B   | 32k     |  bf16 | Full evaluation        |

**Downloads:** see `assets/checkpoints/README.md`.
**Tokenizer:** SentencePiece (Unigram), 32k vocab, byte‑fallback.
**License:** Apache‑2.0 for code; model weights under `LICENSE-MODEL` (research use, see file).

---

## Architecture

A **diamond** is a bounded optimization episode with a stateful planner and a **view‑projected working memory**.

```
Input → Embed → Backbone (Transformer)
                 │
                 ├─ Planner Head → qθ(z|x)  (latent programs/tool plans)
                 │            │
                 │            └─ Project-to-View ΠA q
                 │
                 ├─ Retrieval Policy ↔ Vector Store / Reranker
                 │
                 ├─ Tools (Python, search, code-runner, …) → Observations
                 │
                 └─ Ledger & Gate: compute Δ; run optimizer step; accept/reject
Output ← Decoder ← Accepted step(s)
```

**Key components**

* **Backbone:** standard decoder‑only Transformer with KV‑budget accounting.
* **Planner head:** small module producing a compact **latent plan distribution** (q_\theta(z\mid x)) over action tokens (tool calls + reasoning sketches).
* **View ((\Pi_A)):** projection family implementing a **slot‑factorized memory** (retrieved chunks occupy slots; cross‑slot couplings masked).
* **Modal ledger:** measures *information discarded by the view* via a KL.
* **Null‑stability gate:** Armijo descent + trust region + KL‑smoothness bound.

---

## Objective (precise math)

We minimize a **single scalar** per episode:
[
\Delta ;=; S_{\text{geo}} ;+; S_{\text{ent}} ;+; \lambda_{\text{mod}}, S_{\text{mod}}
\qquad \text{with } \lambda_{\text{mod}} > 0.
]

### Terms

* **Geometry / capacity penalty** (S_{\text{geo}})
  Encodes budget usage and conditioning:
  [
  S_{\text{geo}} = \lambda_{\text{flop}}, \mathrm{FLOPs}

  * \lambda_{\text{kv}}, \text{KV_bytes}
  * \lambda_{\text{lat}}, \text{tool_latency}
  * \lambda_{\text{sparse}}, (1-\text{sparsity})
  * \lambda_{\kappa}, \widehat{\kappa}
    ]
    where (\widehat{\kappa}) is a curvature/condition proxy if second‑order preconditioning is used.

* **Evidence / data‑fit** (S_{\text{ent}})
  Negative log‑likelihood and constraint terms:
  [
  S_{\text{ent}} =

  * \mathbb{E}*{(x,y)}\big[\log p*\theta(y\mid x, \text{tools})\big]

  - \lambda_{\text{cons}} \cdot \text{constraint_violation}
    ]
    Includes reranker scores / tool observation likelihoods when available.

* **Modal penalty (ledger)** (S_{\text{mod}})
  **Information discarded by the view**:
  [
  S_{\text{mod}} = D_{\mathrm{KL}}!\left(q_\theta(z\mid x); \Big|; \Pi_A, q_\theta(z\mid x)\right).
  ]
  This is convex in the second argument and has unbiased gradient estimators via samples from (q_\theta).

> **Interpretation:** larger (S_{\text{mod}}) means a **riskier commitment** (more information lost by the working view). The penalty discourages destructive over‑collapse.

---

## Null‑Stability (Acceptance Criteria)

A candidate update ((\theta_{t}!\to!\theta_{t+1}, A_t!\to!A_{t+1})) is **accepted** iff all hold:

1. **Armijo monotone descent**
   [
   \Delta_{t+1} \le \Delta_{t} - \alpha ,|\nabla \Delta_t|_2^2
   ]
2. **Trust region**
   [
   |\theta_{t+1} - \theta_{t}|_2 \le r_t \quad (\text{adaptive } r_t)
   ]
3. **KL‑smoothness (ledger Lipschitz)**
   [
   \big|S_{\text{mod}}(t+1) - S_{\text{mod}}(t)\big|
   \le L_{\text{mod}} \cdot |\theta_{t+1}-\theta_t|_2.
   ]

If any check fails: backtrack line search OR **widen the view** (A \leftarrow A \cup \delta A) (e.g., add slots/unmask heads) and retry.

**Default knobs (good starting points)**

```yaml
null_stability:
  alpha: 0.1          # Armijo slope fraction
  trust_radius: 0.02  # L2 radius (per-layer adaptive)
  l_mod: 5.0          # KL Lipschitz bound
objective:
  lambda_mod: 0.5
  lambda_kv: 1.0e-9           # per-byte
  lambda_flop: 5.0e-13        # per-FLOP
  lambda_lat: 1.0e-3          # per-ms tool latency
  lambda_sparse: 0.05
```

---

## Memory, Retrieval & Views

* **View family ((\Pi_A))**: slot‑factorized distributions over retrieved chunks; attention is limited within slots; cross‑slot attention heads can be masked or sparsified.
* **Projection**: **I‑projection** (information projection) onto the view family; for exponential‑family parameterizations, moment matching yields (\Pi_A q).
* **Retrieval policy**: BM25 → encoder → reranker; the view determines how many chunks (slots) are maintained and how they are coupled.
* **Diagnostics**: log `view.size`, `collapse_pressure = S_mod / view.size`.

---

## Tool Use

Tools are explicit actions with typed observations.

```jsonc
{
  "action": "python.run",
  "args": {"code": "import math; print(math.sqrt(2))"},
  "observation_schema": {"stdout": "str", "stderr": "str", "time_ms": "int"}
}
```

* **Latency** contributes to (S_{\text{geo}}).
* **Observation likelihoods** (when available) contribute to (S_{\text{ent}}).
* **Safety**: tools execute in a sandbox (FS/network policies configurable).

---

## Evaluation

We focus on **reasoning stability** and **calibration** under fixed budgets.

### Core metrics

* **Cascade‑fail rate** (multi‑step tasks; lower is better).
* **Calibration** (Brier score / ECE per step).
* **Tool‑success rate** under latency budgets.
* **Retrieval quality** (nDCG@k, exact support coverage).
* **Acceptance rate** of the null‑stability gate and **backtrack count**.

### “Kill‑numbers” (pre‑registered thresholds)

* ≥ **10%** relative reduction in cascade‑fail vs. same backbone **without** ledger+gate (toy tasks, fixed FLOPs).
* Calibration within **±5%** ECE on held‑out synthetic multi‑step curricula.
* ≥ **15%** **tool‑success** improvement at tight budgets.

> See `eval/benchmarks/` for scripts and dataset notes.

---

## Performance & Hardware

* **Throughput** scales with accepted steps; the gate adds modest overhead (1–2 Hvps per accepted step if curvature checks are enabled).
* **Mixed precision** (bf16/FP8) supported; **KV‑budget accounting** included in (S_{\text{geo}}).
* **Latency knobs**: fewer step attempts per diamond; cheaper view families; anneal (\lambda_{\text{mod}}) with budget.

---

## Reproducibility

* Fixed seeds for data shuffling, sampler, and planner temperature.
* CI checks: monotone descent rate ≥ 0.9 on smoke tests; ledger variance within tolerance.
* Determinism notes: fused kernels can introduce non‑determinism—toggle with `TORCH_USE_DETERMINISTIC_ALGORITHMS=1`.

---

## API Reference

### Python

```python
from diamond.runtime import DiamondEngine, Budget

engine = DiamondEngine.from_pretrained("meqg-diamond-small-1.3B")
out = engine.run(
    prompt="Plan a 3-step troubleshooting process for a flaky unit test.",
    tools=[],
    budget=Budget(seconds=1.5, max_flops=2e12),
    config=dict(obj=dict(lambda_mod=0.4))
)
print(out.text)
print(out.telemetry["gate"])   # {"accepted": true, "retries": 1, ...}
```

### HTTP (gRPC/REST)

```
POST /v1/diamond/run
{
  "model": "meqg-diamond-small-1.3B",
  "prompt": "...",
  "budget": {"seconds": 1.5, "max_flops": 2e12, "max_kv_bytes": 134217728},
  "tools": ["python"],
  "config": {"obj": {"lambda_mod": 0.4}}
}
```

**Response (truncated)**

```json
{
  "text": "…",
  "telemetry": {
    "objective": {"Delta": 1.284, "S_geo": 0.192, "S_ent": 0.955, "S_mod": 0.275},
    "gate": {"accepted": true, "armijo": true, "trust": true, "kl_smooth": true,
             "retries": 1, "view_size": 6, "collapse_pressure": 0.046}
  }
}
```

---

## Safety, Limitations & Intended Use

**Intended use:** research on **reasoning stability**, **auditable uncertainty**, and **tool‑augmented retrieval**.
**Non‑goals:** chasing SOTA on raw next‑token perplexity or short single‑hop QA.

**Known failure modes**

* Over‑regularization when (\lambda_{\text{mod}}) is too high → under‑commitment and stalled plans.
* Ledger estimator noise for high‑dimensional (q_\theta).
* Latency spikes if curvature checks are enabled at every micro‑step.

**Mitigations**

* **Budget‑aware (\lambda_{\text{mod}})** (anneal up with wider views).
* Low‑variance KL estimators (control variates) and compact planner vocabularies.
* Adaptive gate frequency (e.g., check every k steps) and cheap Hvps.

---

## Project Layout

```
.
├── model/
│   ├── fm_core/       # architectures & tokenization
│   ├── fm_data/       # dataset builders & catalogs
│   ├── fm_train/      # trainer, schedulers, CLI adapters
│   ├── fm_eval/       # eval harness + kill-switch runner
│   ├── fm_serving/    # python-facing serving adapters
│   ├── fm_kernels/    # C++/CUDA extensions (torch cpp_extension)
│   ├── fm_bindings/   # PyO3 bindings (maturin package)
│   └── fm_rag/        # retrieval & memory adapters
├── serving/
│   ├── python/        # ASGI shim + legacy ledger-server crate
│   ├── rust/          # new Axum runtime placeholder
│   └── ui/            # (moved) front-end assets
├── configs/
│   ├── data/
│   ├── eval/
│   ├── models/
│   ├── serving/
│   └── train/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── golden/
├── experiments/notebooks/  # sandbox experiments
├── docs/ (MODEL_CARD, SAFETY_CARD, EVALUATION)
└── scripts/, engine/, external/, infra/, LICENSE*
```

See `docs/MODEL_CARD.md` for model specifics and `docs/EVALUATION.md` for the new gating workflow.

---

## Roadmap & Milestones

**Phase 0 — Minimal viable diamond (4–6 weeks)**

* 150–350M backbone + planner head; ledger with **slot‑factorized view**; full gate.
* **Target:** Hotpot‑style multi‑hop + toy tool tasks.
* **Kill‑number:** ≥10% cascade‑fail reduction vs. baseline at fixed FLOPs.

**Phase 1 — Memory, tools, curriculum (6–10 weeks)**

* Views: factorized → **structured** (dependency graphs / AST views).
* Tools: add web‑search stub + robust Python sandbox.
* Curriculum: synthetic long‑horizon; track per‑step ECE/Brier.
* **Kill‑number:** calibration within ±5%; ≥15% tool‑success improvement.

**Phase 2 — Scale & ablations (8–12 weeks)**

* Scale to 1–7B; mixed precision; KV accounting in (S_{\text{geo}}).
* Ablations: **no ledger**, **no gate**, **sign flip**.
* **Kill‑number:** monotone win curves on long‑horizon tasks.

**Phase 3 — Foundation release**

* Model card, API, **ledger audit dashboards**, and stability traces.

A more detailed plan lives in `docs/roadmap.md`.

---

## Citations & Background

* **Information Bottleneck / Free‑Energy**: compression vs. reward framing.
* **Trust‑Region & Line‑Search**: TRPO, Armijo/Goldstein conditions.
* **Tool‑augmented LLMs**: ReAct, ToT, planner‑executor patterns.

The **novelty here** is the **auditable, local modal penalty** with an explicit **accept/reject gate** that blocks brittle updates *before* they propagate. See `docs/theory/` for derivations and proofs sketch.

---

## Status

This repository is a **research preview**. Expect API churn around view parameterizations and telemetry schemas. Objective and gate semantics are stable:
[
\Delta = S_{\text{geo}} + S_{\text{ent}} + \lambda_{\text{mod}} S_{\text{mod}}, \quad \lambda_{\text{mod}}>0
]
with the **Null‑Stability Gate** defined exactly as in this README.

---

## License

* **Code:** Apache‑2.0 (`LICENSE`)
* **Weights:** Research use license (`LICENSE-MODEL`) with attribution. See file for terms.

---

### Appendix A — Pseudocode (reference)

```python
def diamond_step(state, budget, cfg):
    A = build_view(state)                 # choose subalgebra/features
    q = propose_latent(state)             # distribution over plans/hypotheses
    qA = project_to_view(q, A)            # information projection Π_A q

    S_geo = geometry_penalty(state, budget, cfg)
    S_ent = evidence_term(state, q, cfg)
    S_mod = kl_divergence(q, qA)          # modal ledger

    Delta = S_geo + S_ent + cfg.lambda_mod * S_mod

    theta_new, A_new, meta = optimizer_step(Delta, state.params, A, cfg)

    # ---- Null-stability checks ----
    monotone = meta["Delta_new"] <= meta["Delta"] - cfg.alpha * meta["grad_norm"]**2
    trust_ok = meta["step_norm"] <= meta["trust_radius"]
    kl_smooth = abs(meta["S_mod_new"] - S_mod) <= cfg.l_mod * meta["step_norm"]

    if monotone and trust_ok and kl_smooth:
        commit(theta_new, A_new)
        ledger.log(S_mod_new=meta["S_mod_new"],
                   view_size=A_new.size, budget=budget, extras=meta)
        return "ACCEPT"
    else:
        widen_view_or_backtrack(state, A, meta)
        return "RETRY"
```

### Appendix B — Telemetry (sample)

```json
{
  "ts": "2025-03-20T19:10:47Z",
  "run_id": "dmd-3f2a",
  "objective": {"Delta": 1.284, "S_geo": 0.192, "S_ent": 0.955, "S_mod": 0.275},
  "gate": {"accepted": true, "armijo": true, "trust": true, "kl_smooth": true,
           "retries": 1, "view_size": 6, "collapse_pressure": 0.046},
  "budget": {"seconds": 2.0, "max_flops": 3e11, "max_kv_bytes": 134217728}
}
```

---
