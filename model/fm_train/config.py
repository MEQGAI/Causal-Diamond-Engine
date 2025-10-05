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
    ddp: bool = True
    fsdp: bool = False
    fsdp_policy: str = "full_shard"
    grad_accum_steps: int = 1
    gradient_clip: float = 1.0


@dataclass
class LoggingConfig:
    wandb: bool = False
    tensorboard: bool = False
    modal_ledger_interval: int = 10
    log_every: int = 20
    wandb_project: Optional[str] = None


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
    stability_slice_ratio: float = 0.125
    backtrack_factor: float = 0.5
    widen_view_on_reject: bool = False
    max_view_window: int = 3


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
class CheckpointConfig:
    output_dir: Path = Path("checkpoints")
    save_every_steps: int = 2000
    save_every_tokens: int = 50_000_000
    keep_last: int = 5
    resume: str = "auto"


@dataclass
class EngineEvalConfig:
    enabled: bool = False
    eval_interval: int = 2000
    batch_size: int = 8
    qfc_tolerance: float = 1.0e-8


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
    checkpoints: CheckpointConfig
    engine_eval: EngineEvalConfig


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
        lambda_geo=float(losses_raw.get("lambda_geo", 0.0)),
    )
    checkpoints_cfg = CheckpointConfig(
        **data.get(
            "checkpoints",
            {"output_dir": data.get("output_dir", "checkpoints")},
        )
    )
    if not isinstance(checkpoints_cfg.output_dir, Path):
        checkpoints_cfg.output_dir = Path(checkpoints_cfg.output_dir)
    engine_eval_cfg = EngineEvalConfig(**data.get("engine", {}))
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
        checkpoints=checkpoints_cfg,
        engine_eval=engine_eval_cfg,
    )
    return cfg


__all__ = [
    "TrainConfig",
    "load_train_config",
    "LossConfig",
    "ModalLossConfig",
    "CheckpointConfig",
    "EngineEvalConfig",
]
