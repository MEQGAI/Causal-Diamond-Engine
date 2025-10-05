"""Python bindings for custom CUDA/C++ kernels used across the stack."""

from __future__ import annotations

from types import SimpleNamespace

import torch

try:  # pragma: no cover - compiled extension optional in CI
    from . import _ops as ops  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - we want to swallow any build-time errors
    ops = SimpleNamespace()  # allows attribute access without crashing during import


def categorical_kl(
    p_log_probs: torch.Tensor, q_log_probs: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence KL(p || q) for batched categorical distributions."""

    p = p_log_probs.exp().clamp_min(1e-9)
    return torch.sum(p * (p_log_probs - q_log_probs), dim=-1)


__all__ = ["ops", "categorical_kl"]
