"""Stage scheduler for curriculum mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import logging

import torch
import torch.distributed as dist

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Stage:
    until_tokens: int
    mixture: Dict[str, float]


class TrainingScheduler:
    """Token-based stage scheduler.

    The scheduler tracks the cumulative number of tokens seen during
    training and advances through the configured stages when thresholds
    are exceeded. Stage definitions are read from a YAML file with the
    following structure::

        stages:
          - until_tokens: 5_000_000
            mixture:
              dataset_a: 0.7
              dataset_b: 0.3
    """

    def __init__(self, stages: List[Stage]) -> None:
        if not stages:
            raise ValueError("scheduler requires at least one stage")
        self._stages = stages
        self._stage_idx = 0
        self._tokens_seen = 0

    @classmethod
    def from_file(cls, path: Path | str) -> "TrainingScheduler":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        raw_stages = data.get("stages")
        if not raw_stages:
            raise ValueError("mixture file must define at least one stage")
        stages = [
            Stage(
                until_tokens=int(stage["until_tokens"]),
                mixture={k: float(v) for k, v in stage["mixture"].items()},
            )
            for stage in raw_stages
        ]
        return cls(stages)

    def state_dict(self) -> Dict[str, int]:
        return {"stage_idx": self._stage_idx, "tokens_seen": self._tokens_seen}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self._stage_idx = state.get("stage_idx", 0)
        self._tokens_seen = state.get("tokens_seen", 0)

    @property
    def current_stage(self) -> Stage:
        return self._stages[self._stage_idx]

    @property
    def stage_idx(self) -> int:
        return self._stage_idx

    def advance(self, tokens: int) -> None:
        self._tokens_seen += tokens
        tokens_global = self._tokens_seen
        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor([tokens_global], dtype=torch.long)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tokens_global = int(tensor.item())
        while (
            self._stage_idx < len(self._stages) - 1
            and tokens_global >= self._stages[self._stage_idx].until_tokens
        ):
            self._stage_idx += 1
            logger.info(
                "advancing to stage %s", self._stage_idx, extra={"tokens": tokens_global}
            )

    def mixture(self) -> Dict[str, float]:
        return self.current_stage.mixture

    def remaining_tokens(self) -> Optional[int]:
        current_limit = self.current_stage.until_tokens
        if current_limit <= 0:
            return None
        return max(0, current_limit - self._tokens_seen)


__all__ = ["TrainingScheduler", "Stage"]
