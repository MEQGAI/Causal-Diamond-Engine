"""Python bindings for custom CUDA/C++ kernels used across the stack."""

from __future__ import annotations

from types import SimpleNamespace

try:  # pragma: no cover - compiled extension optional in CI
    from . import _ops as ops  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - we want to swallow any build-time errors
    ops = SimpleNamespace()  # allows attribute access without crashing during import

__all__ = ["ops"]
