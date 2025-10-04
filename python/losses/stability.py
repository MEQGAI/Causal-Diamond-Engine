"""Null-stability heuristics for smoke testing."""

from __future__ import annotations


def null_stability_metric(kl_divergence: float, budget: float) -> float:
    """Compute a surrogate stability metric; the lower the better."""
    return max(0.0, budget - kl_divergence)
