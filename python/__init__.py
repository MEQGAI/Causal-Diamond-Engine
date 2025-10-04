"""Compatibility shim for legacy `python.*` imports.

This adapter forwards module lookups to the new `fm_train` package so that
existing tooling keeps working during the refactor rollout.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable

import fm_train as _fm_train

# Allow `import fm_train.datasets` by pointing the loader at the relocated package.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "model" / "fm_train"
__path__: Iterable[str] = [str(_PACKAGE_ROOT)]


def __getattr__(name: str) -> Any:
    if hasattr(_fm_train, name):
        return getattr(_fm_train, name)
    return importlib.import_module(f"fm_train.{name}")


__all__ = getattr(_fm_train, "__all__", [])
