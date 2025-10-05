from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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
class ModalLossConfig:
    apply_on: List[str] = field(default_factory=lambda: ["planner"])
    lambda_mod: float = 0.25
    lambda_planner: float = 1.0
    lambda_token: float = 0.0
    tau_planner: float = 1.0
    tau_token: float = 1.0
    slot_len: int = 512
    view_window: int = 1
    slot_weights: List[float] = field(default_factory=lambda: [1.0, 0.5])
    stop_grad_projection: bool = True
    eps: float = 1e-6
    clip_kl: float = 10.0
    token_topk: int = 256


@dataclass
class LossConfig:
    modal: ModalLossConfig = field(default_factory=ModalLossConfig)
    lambda_geo: float = 0.0


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
    losses: LossConfig


def load_train_config(path: Path | str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    optimizer = OptimizerConfig(**data["optimizer"])
    distributed = DistributedConfig(**data.get("distributed", {}))
    logging_cfg = LoggingConfig(**data.get("logging", {}))
    ledger = LedgerConfig(**data.get("ledger", {}))
    gate = GateConfig(**data.get("gate", {}))
    losses_raw = data.get("losses", {})
    modal_raw = losses_raw.get("modal", {})
    losses = LossConfig(
        modal=ModalLossConfig(**modal_raw),
        lambda_geo=losses_raw.get("lambda_geo", 0.0),
    )
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
        losses=losses,
    )
    return cfg


__all__ = [
    "TrainConfig",
    "load_train_config",
    "LossConfig",
    "ModalLossConfig",
]
