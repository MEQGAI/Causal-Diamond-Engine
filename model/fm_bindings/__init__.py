"""Thin Python shim around the Rust PyO3 bindings."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_NATIVE_MODULE = "ledger_python"

try:  # pragma: no cover - optional native extension
    _binding: ModuleType | None = importlib.import_module(_NATIVE_MODULE)
except ModuleNotFoundError:  # pragma: no cover - extension not built yet
    _binding = None


def __getattr__(name: str) -> Any:
    if _binding is None:
        raise ImportError(
            "fm_bindings native module not built. Run `maturin develop` to compile the extension."
        )
    return getattr(_binding, name)


__all__ = ["_binding"]
