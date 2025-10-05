from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateMetrics:
    delta_before: float
    delta_after: float
    grad_norm: float
    step_norm: float
    kl_change: float


@dataclass
class GateDecision:
    accepted: bool
    reason: str
    trust_radius: float


class NullStabilityGate:
    """Implements Armijo + trust region + KL smoothness acceptance checks."""

    def __init__(self, alpha: float, trust_radius: float, kl_smooth_max: float) -> None:
        self.alpha = alpha
        self.trust_radius = trust_radius
        self.kl_smooth_max = kl_smooth_max

    def evaluate(self, metrics: GateMetrics) -> GateDecision:
        accepted = True
        reason = "accepted"

        if (
            metrics.delta_after
            > metrics.delta_before - self.alpha * metrics.grad_norm**2
        ):
            accepted = False
            reason = "armijo"
        elif metrics.step_norm > self.trust_radius:
            accepted = False
            reason = "trust_radius"
        elif metrics.kl_change > self.kl_smooth_max * metrics.step_norm:
            accepted = False
            reason = "kl_smooth"

        if accepted:
            self.trust_radius *= 1.1
        else:
            self.trust_radius *= 0.9

        return GateDecision(
            accepted=accepted, reason=reason, trust_radius=self.trust_radius
        )


__all__ = ["NullStabilityGate", "GateMetrics", "GateDecision"]
