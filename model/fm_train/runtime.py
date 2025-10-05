from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import IterableDataset, DataLoader

from fm_core.config import ModelConfig, load_config as load_model_config
from fm_core.projection import SlotLayout
from fm_core.transformer import FoundationModel
from fm_data.catalog import Catalog, load_catalog
from fm_data.packing import DiamondPacker, PackedSequence
from fm_data.webdataset_stream import WebDatasetStream

from .config import TrainConfig, load_train_config
from .gate import GateMetrics, NullStabilityGate
from .objectives import compute_total_loss

import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional Rust engine
    from engine import CausalDiamondEngine  # type: ignore
except Exception:  # pragma: no cover
    CausalDiamondEngine = None


@dataclass
class TrainerState:
    step: int = 0
    tokens_processed: int = 0
    delta_previous: float = float("inf")
    loss_mod_previous: float = 0.0


class MixtureIterableDataset(IterableDataset):
    def __init__(self, catalog: Catalog, mixture: str, packer: DiamondPacker) -> None:
        super().__init__()
        self.catalog = catalog
        self.mixture = mixture
        self.packer = packer
        self.schedule = catalog.mixtures[mixture]
        self.dataset_lookup = catalog.dataset_lookup()
        self.stage_idx = 0
        self.tokens_emitted = 0
        self.tokens_per_seq = catalog.seq_len

    def _advance_stage(self) -> None:
        while (
            self.stage_idx < len(self.schedule) - 1
            and self.tokens_emitted >= self.schedule[self.stage_idx].until_tokens
        ):
            self.stage_idx += 1
            logger.info(
                "advancing mixture stage",
                extra={"stage": self.stage_idx, "tokens": self.tokens_emitted},
            )

    def _sample_text(self, dataset_id: str) -> str:
        cfg = self.dataset_lookup[dataset_id]
        stream = WebDatasetStream(cfg, shuffle=True)
        for sample in stream:
            sources = [".txt", ".text", ".code", "txt", "text", "code"]
            for key in sources:
                if key in sample.data:
                    blob = sample.data[key]
                    if isinstance(blob, bytes):
                        text = blob.decode("utf-8", errors="ignore")
                    else:
                        text = str(blob)
                    if text.strip():
                        return text
            if "text" in sample.meta:
                text = str(sample.meta["text"])
                if text.strip():
                    return text
        raise RuntimeError(
            f"dataset {dataset_id} yielded no valid samples; ensure shards are accessible"
        )

    def __iter__(self) -> Iterator[PackedSequence]:
        self.stage_idx = 0
        self.tokens_emitted = 0
        while True:
            self._advance_stage()
            stage = self.schedule[self.stage_idx]
            dataset_ids = list(stage.weights.keys())
            weights = torch.tensor([stage.weights[ds] for ds in dataset_ids])
            weights = weights / weights.sum()
            choice = torch.multinomial(weights, 1).item()
            dataset_id = dataset_ids[choice]
            text = self._sample_text(dataset_id)
            packed = self.packer.pack_raw_text(text)
            self.tokens_emitted += int(packed.input_ids.numel())
            yield packed


def collate_packed(batch: List[PackedSequence]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item.input_ids for item in batch])
    planner_mask = torch.stack([item.planner_mask for item in batch])
    return {"input_ids": input_ids, "planner_mask": planner_mask}


@contextmanager
def autocast(precision: str):  # pragma: no cover - simple context helper
    if precision.lower() in {"bf16", "bfloat16"} and torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            yield
    elif precision.lower() == "fp16" and torch.cuda.is_available():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            yield
    else:
        yield


class Trainer:
    def __init__(self, cfg: TrainConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_cfg: ModelConfig = load_model_config(cfg.model_cfg)
        self.catalog: Catalog = load_catalog(cfg.data_catalog)
        self.model = FoundationModel(
            self.model_cfg, slot_window=cfg.ledger.view.get("window", 1)
        )
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.optimizer.warmup_steps
        )
        self.state = TrainerState()
        self.gate = NullStabilityGate(
            alpha=cfg.gate.armijo_alpha,
            trust_radius=cfg.gate.trust_radius_init,
            kl_smooth_max=cfg.gate.kl_smooth_max,
        )
        slot_layout = SlotLayout(
            slot_len=self.catalog.packing.slot_len,
            slots_per_seq=self.catalog.packing.slots_per_seq,
        )
        self.slot_layout = slot_layout
        try:
            self.packer = DiamondPacker(self.catalog, self.model_cfg.special_tokens)
        except (FileNotFoundError, ImportError) as exc:
            raise RuntimeError(
                f"Failed to initialise tokenizer from {self.catalog.tokenizer}. "
                "Ensure checkpoints include tokenizer.model and sentencepiece is installed."
            ) from exc
        self.dataset = MixtureIterableDataset(self.catalog, cfg.mixture, self.packer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.distributed.grad_accum_steps,
            collate_fn=collate_packed,
        )
        self.ledger_path = Path("ledger.jsonl")
        self.ledger_path.write_text("", encoding="utf-8")
        self.engine = CausalDiamondEngine() if CausalDiamondEngine else None

    def _log_ledger(self, payload: Dict[str, object]) -> None:
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _engine_report(self, metrics: Dict[str, float]) -> None:
        if not self.engine:
            return
        serialized = json.dumps(metrics)
        try:
            self.engine.step(serialized, 1.0)
        except Exception:  # pragma: no cover
            pass

    def fit(self, steps: int) -> None:
        accum = self.cfg.distributed.grad_accum_steps
        data_iter = iter(self.dataloader)
        # scaler not used with bf16/fp32 training in this loop
        for step in range(steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(self.device)
            planner_mask = batch["planner_mask"].to(self.device)
            with autocast(self.cfg.precision):
                outputs = self.model(input_ids, planner_mask=planner_mask)
                losses = compute_total_loss(
                    outputs,
                    batch={"input_ids": input_ids},
                    slot_layout=self.slot_layout,
                    lambda_mod=self.cfg.ledger.lambda_mod,
                )
                loss = losses["loss_total"] / accum
            loss.backward()

            if (step + 1) % accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                ).item()
                self.optimizer.step()
                self.scheduler.step()
                step_norm = sum(p.data.norm().item() for p in self.model.parameters())
                self.optimizer.zero_grad(set_to_none=True)

                metrics = GateMetrics(
                    delta_before=self.state.delta_previous,
                    delta_after=losses["loss_total"].item(),
                    grad_norm=grad_norm,
                    step_norm=step_norm,
                    kl_change=abs(
                        losses["loss_mod"].item() - self.state.loss_mod_previous
                    ),
                )
                decision = self.gate.evaluate(metrics)
                self.state.delta_previous = losses["loss_total"].item()
                self.state.loss_mod_previous = losses["loss_mod"].item()
                self.state.step += 1
                self.state.tokens_processed += input_ids.numel()

                ledger_payload = {
                    "step": self.state.step,
                    "delta": metrics.delta_after,
                    "loss_ent": losses["loss_ent"].item(),
                    "loss_mod": losses["loss_mod"].item(),
                    "accepted": decision.accepted,
                    "reason": decision.reason,
                    "trust_radius": decision.trust_radius,
                }
                self._log_ledger(ledger_payload)
                self._engine_report(ledger_payload)


def train_from_config(config_path: Path | str, steps: int) -> None:
    cfg = load_train_config(config_path)
    trainer = Trainer(cfg)
    trainer.fit(steps)


__all__ = ["Trainer", "train_from_config"]
