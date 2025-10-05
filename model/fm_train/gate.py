"""Null-stability gate implementing Armijo, trust-region, and KL smoothness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GateMetrics:
    delta_before: float
    delta_after: float
    grad_norm: float
    step_norm: float
    kl_before: Optional[float] = None
    kl_after: Optional[float] = None


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    trust_radius: float


def accept_update(
    delta_prev: float,
    delta_new: float,
    grad_norm: float,
    theta_delta_norm: float,
    *,
    alpha: float,
    trust_radius: float,
    kl_prev: Optional[float] = None,
    kl_new: Optional[float] = None,
    kl_smooth_max: float = 1.0,
) -> tuple[bool, str]:
    """Return whether the update should be accepted and the failing criterion."""

    monotone = delta_new <= delta_prev - alpha * (grad_norm**2)
    if not monotone:
        return False, "armijo"

    trust_ok = theta_delta_norm <= trust_radius
    if not trust_ok:
        return False, "trust_radius"

    if kl_prev is None or kl_new is None:
        return True, "accepted"

    kl_delta = abs(kl_new - kl_prev)
    limit = kl_smooth_max * max(theta_delta_norm, 1e-9)
    if kl_delta > limit:
        return False, "kl_smooth"
    return True, "accepted"


class NullStabilityGate:
    """Implements the composite acceptance tests with adaptive trust radius."""

    def __init__(self, alpha: float, trust_radius: float, kl_smooth_max: float) -> None:
        self.alpha = alpha
        self.trust_radius = trust_radius
        self.kl_smooth_max = kl_smooth_max

    def evaluate(self, metrics: GateMetrics) -> GateDecision:
        accepted, reason = accept_update(
            metrics.delta_before,
            metrics.delta_after,
            metrics.grad_norm,
            metrics.step_norm,
            alpha=self.alpha,
            trust_radius=self.trust_radius,
            kl_prev=metrics.kl_before,
            kl_new=metrics.kl_after,
            kl_smooth_max=self.kl_smooth_max,
        )

        if accepted:
            self.trust_radius *= 1.1
        else:
            self.trust_radius *= 0.9

        return GateDecision(accepted=accepted, reason=reason, trust_radius=self.trust_radius)


__all__ = ["NullStabilityGate", "GateMetrics", "GateDecision", "accept_update"]
