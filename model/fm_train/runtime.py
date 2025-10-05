from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import IterableDataset, DataLoader

from fm_core.config import ModelConfig, load_config as load_model_config
from fm_core.projection import SlotLayout, make_view_mask
from fm_core.transformer import FoundationModel
from fm_data.catalog import Catalog, load_catalog
from fm_data.packing import DiamondPacker, PackedSequence
from fm_train.datasets.catalog import load_source

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
    def __init__(
        self,
        catalog: Catalog,
        packer: DiamondPacker,
        initial_mixture: Mapping[str, float],
    ) -> None:
        super().__init__()
        self.catalog = catalog
        self.packer = packer
        self.dataset_lookup = catalog.dataset_lookup()
        self.tokens_per_seq = catalog.seq_len
        self.current_ids: List[str] = []
        self.weights = torch.tensor([])
        self._iterators: Dict[str, Iterator[Any]] = {}
        self._generator = torch.Generator()
        self.set_mixture(initial_mixture)

    def set_mixture(self, mixture: Mapping[str, float]) -> None:
        if not mixture:
            raise ValueError("mixture must contain at least one dataset")
        self.current_ids = list(mixture.keys())
        weights = torch.tensor([float(mixture[k]) for k in self.current_ids])
        weights = weights.clamp_min(0.0)
        total = weights.sum()
        if total <= 0:
            raise ValueError("mixture weights must sum to > 0")
        self.weights = weights / total

    def _dataset_entry(self, dataset_id: str) -> Dict[str, Any]:
        cfg = self.dataset_lookup[dataset_id]
        entry = asdict(cfg)
        entry.setdefault("type", entry.get("format", "webdataset"))
        return entry

    def _next_sample(self, dataset_id: str) -> str:
        iterator = self._iterators.get(dataset_id)
        if iterator is None:
            iterator = load_source(self._dataset_entry(dataset_id), split="train")
            self._iterators[dataset_id] = iterator
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = load_source(self._dataset_entry(dataset_id), split="train")
            self._iterators[dataset_id] = iterator
            sample = next(iterator)
        text = sample.text.strip()
        if not text:
            return self._next_sample(dataset_id)
        return text

    def __iter__(self) -> Iterator[PackedSequence]:
        while True:
            choice = torch.multinomial(self.weights, 1, replacement=True, generator=self._generator).item()
            dataset_id = self.current_ids[choice]
            text = self._next_sample(dataset_id)
            packed = self.packer.pack_raw_text(text)
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
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
        self.device = self._init_distributed(device)
        self.is_master = self.rank == 0

        self.model_cfg: ModelConfig = load_model_config(cfg.model_cfg)
        self.catalog: Catalog = load_catalog(cfg.data_catalog)

        self.loss_cfg = cfg.losses
        self.modal_cfg = cfg.losses.modal
        self.gate_cfg = cfg.gate
        self.current_view_window = self.modal_cfg.view_window

        self.model = FoundationModel(
            self.model_cfg, slot_window=self.current_view_window
        )
        self.model.to(self.device)
        self._wrap_model()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.optimizer.warmup_steps
        )
        self.grad_clip = cfg.distributed.gradient_clip

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
        self.special_ids = getattr(self.packer, "special_ids", None)

        mixture, scheduler = self._resolve_mixture(cfg.mixture)
        self.stage_scheduler = scheduler
        self.dataset = MixtureIterableDataset(self.catalog, self.packer, mixture)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.distributed.grad_accum_steps,
            collate_fn=collate_packed,
        )

        self.checkpoint_cfg = cfg.checkpoints
        self.output_dir = self.checkpoint_cfg.output_dir
        if self.is_master:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_checkpoints: deque[Path] = deque()
        self.tokens_since_last_checkpoint = 0

        self.logging_cfg = cfg.logging
        self.log_interval = self.logging_cfg.log_every
        self.tb_writer = self._maybe_create_summary_writer()
        self.wandb_run = self._maybe_init_wandb()

        self.engine_cfg = cfg.engine_eval
        self.engine = CausalDiamondEngine() if (CausalDiamondEngine and self.engine_cfg.enabled) else None

        self.ledger_path = Path("ledger.jsonl")
        if self.is_master:
            self.ledger_path.write_text("", encoding="utf-8")

        self.tokens_per_step = 0

        self._maybe_resume()

        if self.is_distributed:
            dist.barrier()

    def _log_ledger(self, payload: Dict[str, object]) -> None:
        if not self.is_master:
            return
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _engine_report(self, metrics: Dict[str, float]) -> None:
        if not self.engine or not self.engine_cfg.enabled:
            return
        serialized = json.dumps(metrics)
        try:
            self.engine.step(serialized, 1.0)
        except Exception:  # pragma: no cover
            pass

    def fit(self, steps: int) -> None:
        accum = self.cfg.distributed.grad_accum_steps
        data_iter = iter(self.dataloader)
        tokens_accum = 0
        for micro_step in range(steps):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(self.device)
            planner_tensor = batch["planner_mask"].to(self.device)
            if planner_tensor.dtype != torch.bool:
                planner_tensor = planner_tensor.bool()
            planner_mask = planner_tensor if planner_tensor.any() else None

            tokens_accum += int(input_ids.numel())

            with autocast(self.cfg.precision):
                outputs = self.model(input_ids, planner_mask=planner_mask)

                modal_inputs = None
                apply_modal = (
                    self.modal_cfg.lambda_mod > 0.0
                    and planner_mask is not None
                    and planner_tensor.any()
                )

                if apply_modal:
                    center_slots = self._compute_center_slots(input_ids)
                    view_mask = make_view_mask(
                        input_ids.size(0),
                        input_ids.size(1),
                        self.modal_cfg.slot_len,
                        center_slots,
                        self.current_view_window,
                        device=input_ids.device,
                        dtype=outputs["logits"].dtype,
                    )
                    outputs_view = self.model(
                        input_ids,
                        planner_mask=planner_mask,
                        attn_bias=view_mask,
                    )

                    apply_token = (
                        "token_spans" in self.modal_cfg.apply_on
                        and self.modal_cfg.lambda_token > 0.0
                    )
                    plan_span_mask = planner_tensor if apply_token else None
                    modal_inputs = {
                        "planner_logits": outputs["planner"].logits,
                        "planner_logits_view": outputs_view["planner"].logits,
                        "token_logits": outputs["logits"] if apply_token else None,
                        "token_logits_view": (
                            outputs_view["logits"] if apply_token else None
                        ),
                        "plan_pos_mask": planner_tensor,
                        "plan_span_mask": plan_span_mask,
                        "slot_len": self.modal_cfg.slot_len,
                        "center_slots": center_slots,
                        "view_window": self.current_view_window,
                        "slot_weights": tuple(
                            float(w) for w in self.modal_cfg.slot_weights
                        ),
                        "tau_planner": self.modal_cfg.tau_planner,
                        "tau_token": self.modal_cfg.tau_token,
                        "eps": self.modal_cfg.eps,
                        "clip_kl": self.modal_cfg.clip_kl,
                        "token_topk": self.modal_cfg.token_topk,
                        "stop_grad_projection": self.modal_cfg.stop_grad_projection,
                        "lambda_planner": self.modal_cfg.lambda_planner,
                        "lambda_token": self.modal_cfg.lambda_token,
                    }

                losses = compute_total_loss(
                    outputs,
                    batch={"input_ids": input_ids},
                    slot_layout=self.slot_layout,
                    modal_cfg=self.modal_cfg,
                    modal_inputs=modal_inputs,
                    lambda_geo=self.loss_cfg.lambda_geo,
                )
                loss = losses["loss_total"] / accum

            sync_context = nullcontext()
            if self._should_no_sync(micro_step, accum):
                sync_context = self.model.no_sync()  # type: ignore[attr-defined]

            with sync_context:
                loss.backward()

            if (micro_step + 1) % accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            ).item()
            step_norm = sum(p.data.norm().item() for p in self.model.parameters())

            metrics = GateMetrics(
                delta_before=self.state.delta_previous,
                delta_after=losses["loss_total"].item(),
                grad_norm=grad_norm,
                step_norm=step_norm,
                kl_before=self.state.loss_mod_previous,
                kl_after=losses["loss_mod"].item(),
            )
            decision = self.gate.evaluate(metrics)
            self.state.delta_previous = losses["loss_total"].item()
            self.state.loss_mod_previous = losses["loss_mod"].item()

            if not decision.accepted:
                self.optimizer.zero_grad(set_to_none=True)
                self._handle_rejection()
                self._log_step(losses, grad_norm, decision)
                self._advance_scheduler(tokens_accum)
                tokens_accum = 0
                continue

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.state.step += 1
            self.state.tokens_processed += tokens_accum

            ledger_payload = {
                "step": self.state.step,
                "delta": metrics.delta_after,
                "loss_ent": losses["loss_ent"].item(),
                "loss_mod": losses["loss_mod"].item(),
                "loss_mod_planner": losses["loss_mod_planner"].item(),
                "loss_mod_token": losses["loss_mod_token"].item(),
                "accepted": decision.accepted,
                "reason": decision.reason,
                "trust_radius": decision.trust_radius,
                "view": {
                    "window": self.current_view_window,
                },
            }
            self._log_ledger(ledger_payload)
            self._engine_report(ledger_payload)
            self._log_step(losses, grad_norm, decision)
            self._maybe_save_checkpoint(tokens_accum)
            self._advance_scheduler(tokens_accum)
            tokens_accum = 0

        if self.tb_writer:
            self.tb_writer.flush()

    def _handle_rejection(self) -> None:
        self._scale_lr(self.gate_cfg.backtrack_factor)
        if (
            self.gate_cfg.widen_view_on_reject
            and self.current_view_window < self.gate_cfg.max_view_window
        ):
            self.current_view_window += 1
            logger.info("widened view window to %s", self.current_view_window)


    def _compute_center_slots(self, input_ids: torch.Tensor) -> torch.LongTensor:
        batch = input_ids.size(0)
        device = input_ids.device
        default_center = self.slot_layout.slots_per_seq // 2
        centers = torch.full((batch,), default_center, device=device, dtype=torch.long)
        if self.special_ids is None:
            return centers
        view_start = getattr(self.special_ids, "view_start", None)
        view_end = getattr(self.special_ids, "view_end", None)
        slot_len = self.modal_cfg.slot_len
        for idx in range(batch):
            candidate = None
            if view_start is not None:
                pos = (input_ids[idx] == view_start).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    candidate = int(pos[0].item()) // slot_len
            if candidate is None and view_end is not None:
                pos = (input_ids[idx] == view_end).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    candidate = int(pos[0].item()) // slot_len
            if candidate is not None:
                centers[idx] = candidate
        centers.clamp_(0, self.slot_layout.slots_per_seq - 1)
        return centers


def train_from_config(config_path: Path | str, steps: int) -> None:
    cfg = load_train_config(config_path)
    trainer = Trainer(cfg)
    trainer.fit(steps)


__all__ = ["Trainer", "train_from_config"]
