from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class PlannerOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor

    def kl_against(self, other: "PlannerOutput") -> torch.Tensor:
        probs = torch.softmax(self.logits, dim=-1)
        return torch.sum(probs * (self.log_probs - other.log_probs), dim=-1)


class PlannerHead(nn.Module):
    """Latent program head producing q_θ(z|x) logits."""

    def __init__(self, hidden_size: int, vocab_size: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.temperature = temperature

    def forward(self, hidden: torch.Tensor) -> PlannerOutput:
        logits = self.linear(hidden) / self.temperature
        log_probs = torch.log_softmax(logits, dim=-1)
        return PlannerOutput(logits=logits, log_probs=log_probs)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


__all__ = ["PlannerHead", "PlannerOutput"]
