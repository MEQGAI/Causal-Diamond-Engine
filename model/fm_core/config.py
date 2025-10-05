from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import json


@dataclass
class RopeScaling:
    """Configuration for RoPE extrapolation scaling."""

    type: str = "linear"
    factor: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RopeScaling":
        if not data:
            return cls()
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "factor": self.factor}


@dataclass
class SpecialTokens:
    bos: str = "<BOS>"
    eos: str = "<EOS>"
    pad: str = "<PAD>"
    unk: str = "<UNK>"
    diamond_start: str = "<DIAMOND_START>"
    diamond_end: str = "<DIAMOND_END>"
    view_start: str = "<VIEW_START>"
    view_end: str = "<VIEW_END>"
    plan_start: str = "<PLAN_START>"
    plan_end: str = "<PLAN_END>"
    tool: str = "<TOOL>"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SpecialTokens":
        if not data:
            return cls()
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ModelConfig:
    """Configuration for the foundation transformer backbone."""

    arch_version: str
    vocab_size: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    mlp_hidden_dim: int
    rope_theta: float
    seq_len: int
    planner_vocab: int
    attn_impl: str = "flash2"
    rope_scaling: RopeScaling = field(default_factory=RopeScaling)
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    dropout: float = 0.0
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        rope_scaling = RopeScaling.from_dict(data.get("rope_scaling"))
        special_tokens = SpecialTokens.from_dict(data.get("special_tokens"))
        kwargs = dict(data)
        kwargs["rope_scaling"] = rope_scaling
        kwargs["special_tokens"] = special_tokens
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path | str) -> "ModelConfig":
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["rope_scaling"] = self.rope_scaling.to_dict()
        data["special_tokens"] = self.special_tokens.to_dict()
        return data

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def dropout_prob(self) -> float:
        return self.dropout

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if self.planner_vocab <= 0:
            raise ValueError("planner_vocab must be positive")


def load_config(path: Path | str) -> ModelConfig:
    cfg = ModelConfig.from_json(path)
    cfg.validate()
    return cfg


__all__ = [
    "ModelConfig",
    "RopeScaling",
    "SpecialTokens",
    "load_config",
]
