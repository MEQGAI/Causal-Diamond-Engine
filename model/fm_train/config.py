from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class OptimizerConfig:
    name: str
    betas: List[float]
    weight_decay: float
    lr: float
    warmup_steps: int
    schedule: str


@dataclass
class DistributedConfig:
    backend: str = "nccl"
    fsdp: bool = False
    fsdp_policy: str = "full_shard"
    grad_accum_steps: int = 1


@dataclass
class LoggingConfig:
    wandb: bool = False
    tensorboard: bool = False
    modal_ledger_interval: int = 10


@dataclass
class LedgerConfig:
    lambda_mod: float
    apply_on: List[str] = field(default_factory=lambda: ["planner"])
    span_tokens: List[str] = field(default_factory=list)
    view: Dict[str, int] = field(default_factory=dict)


@dataclass
class GateConfig:
    armijo_alpha: float
    trust_radius_init: float
    kl_smooth_max: float


@dataclass
class TrainConfig:
    model_cfg: Path
    data_catalog: Path
    mixture: str
    precision: str
    optimizer: OptimizerConfig
    distributed: DistributedConfig
    logging: LoggingConfig
    ledger: LedgerConfig
    gate: GateConfig


def load_train_config(path: Path | str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    optimizer = OptimizerConfig(**data["optimizer"])
    distributed = DistributedConfig(**data.get("distributed", {}))
    logging_cfg = LoggingConfig(**data.get("logging", {}))
    ledger = LedgerConfig(**data.get("ledger", {}))
    gate = GateConfig(**data.get("gate", {}))
    cfg = TrainConfig(
        model_cfg=Path(data["model_cfg"]),
        data_catalog=Path(data["data_catalog"]),
        mixture=data["mixture"],
        precision=data.get("precision", "bf16"),
        optimizer=optimizer,
        distributed=distributed,
        logging=logging_cfg,
        ledger=ledger,
        gate=gate,
    )
    return cfg


__all__ = ["TrainConfig", "load_train_config"]
