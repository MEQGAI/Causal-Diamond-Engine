"""Modal objective utilities."""

from __future__ import annotations

import math


def modal_penalty(task: str, epoch: int) -> float:
    """Return a dummy KL divergence shaped by task/epoch for smoke tests."""
    base = 0.5 if task == "tool_reasoning" else 0.7
    return float(abs(math.sin(epoch)) * base)
