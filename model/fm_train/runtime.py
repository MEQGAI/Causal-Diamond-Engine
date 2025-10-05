from __future__ import annotations

import json
import os
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # pragma: no cover - optional FSDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover
    FSDP = None

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
        self._generator.manual_seed(torch.initial_seed())
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
        self._iterators = {k: v for k, v in self._iterators.items() if k in self.current_ids}

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
                center_slots = self._compute_center_slots(input_ids)
                apply_modal = (
                    self.modal_cfg.lambda_mod > 0.0
                    and planner_mask is not None
                    and planner_tensor.any()
                )

                if apply_modal:
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

            step_id = self.state.step + 1

            ledger_payload = {
                "step": step_id,
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
                    "centers": center_slots.tolist(),
                },
            }
            self._log_ledger(ledger_payload)
            self._engine_report(ledger_payload)

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

            self.state.step = step_id
            self.state.tokens_processed += tokens_accum
            self._log_step(losses, grad_norm, decision)
            self._maybe_save_checkpoint(tokens_accum)
            self._advance_scheduler(tokens_accum)
            tokens_accum = 0

        if self.tb_writer:
            self.tb_writer.flush()
        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception:  # pragma: no cover
                pass

    def _handle_rejection(self) -> None:
        self._scale_lr(self.gate_cfg.backtrack_factor)
        if (
            self.gate_cfg.widen_view_on_reject
            and self.current_view_window < self.gate_cfg.max_view_window
        ):
            self.current_view_window += 1
            logger.info("widened view window to %s", self.current_view_window)

    def _scale_lr(self, factor: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] *= factor

    def _log_step(self, losses: Dict[str, torch.Tensor], grad_norm: float, decision: GateDecision) -> None:
        if not self.is_master:
            return
        if self.state.step % max(1, self.log_interval) != 0:
            return
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            "step=%s loss=%.4f mod=%.4f grad=%.3f lr=%.6f accepted=%s",
            self.state.step,
            losses["loss_total"].item(),
            losses["loss_mod"].item(),
            grad_norm,
            lr,
            decision.accepted,
        )
        if self.tb_writer:
            step = self.state.step
            self.tb_writer.add_scalar("loss/total", losses["loss_total"].item(), step)
            self.tb_writer.add_scalar("loss/modal", losses["loss_mod"].item(), step)
            self.tb_writer.add_scalar("optim/grad_norm", grad_norm, step)
            self.tb_writer.add_scalar("optim/lr", lr, step)
        if self.wandb_run:
            try:
                self.wandb_run.log(
                    {
                        "loss/total": losses["loss_total"].item(),
                        "loss/modal": losses["loss_mod"].item(),
                        "grad_norm": grad_norm,
                        "lr": lr,
                        "trust_radius": self.gate.trust_radius,
                    },
                    step=self.state.step,
                )
            except Exception:  # pragma: no cover
                pass

    def _advance_scheduler(self, tokens: int) -> None:
        if not self.stage_scheduler:
            return
        prev_idx = self.stage_scheduler.stage_idx
        self.stage_scheduler.advance(tokens)
        if self.stage_scheduler.stage_idx != prev_idx:
            mixture = self.stage_scheduler.mixture()
            self.dataset.set_mixture(mixture)
            if self.is_master:
                logger.info(
                    "advanced to scheduler stage %s", self.stage_scheduler.stage_idx
                )

    def _maybe_save_checkpoint(self, tokens: int) -> None:
        if not self.is_master:
            return
        self.tokens_since_last_checkpoint += tokens
        step_trigger = (
            self.checkpoint_cfg.save_every_steps > 0
            and self.state.step % self.checkpoint_cfg.save_every_steps == 0
        )
        token_trigger = (
            self.checkpoint_cfg.save_every_tokens > 0
            and self.tokens_since_last_checkpoint >= self.checkpoint_cfg.save_every_tokens
        )
        if not (step_trigger or token_trigger):
            return
        self._save_checkpoint()
        self.tokens_since_last_checkpoint = 0

    def _save_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / f"checkpoint_step_{self.state.step:07d}.pt"
        payload = {
            "model": self.model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "trainer_state": asdict(self.state),
            "scheduler_state": self.stage_scheduler.state_dict()
            if self.stage_scheduler
            else None,
            "view_window": self.current_view_window,
            "trust_radius": self.gate.trust_radius,
        }
        torch.save(payload, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        while len(self.saved_checkpoints) > self.checkpoint_cfg.keep_last:
            old = self.saved_checkpoints.popleft()
            if old.exists():
                old.unlink(missing_ok=True)
        latest = self.output_dir / "latest.ckpt"
        try:
            if latest.exists():
                latest.unlink()
            latest.symlink_to(checkpoint_path.name)
        except Exception:
            latest.write_text(checkpoint_path.name, encoding="utf-8")

    def _maybe_resume(self) -> None:
        mode = (self.checkpoint_cfg.resume or "auto").lower()
        if mode == "never":
            return
        latest = self._find_latest_checkpoint()
        if latest is None:
            return
        map_location = self.device
        checkpoint = torch.load(latest, map_location=map_location)
        self.model_to_save.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        trainer_state = checkpoint.get("trainer_state", {})
        self.state = TrainerState(**trainer_state)
        scheduler_state = checkpoint.get("scheduler_state")
        if self.stage_scheduler and scheduler_state:
            self.stage_scheduler.load_state_dict(scheduler_state)
            self.dataset.set_mixture(self.stage_scheduler.mixture())
        self.current_view_window = checkpoint.get("view_window", self.current_view_window)
        self.gate.trust_radius = checkpoint.get("trust_radius", self.gate.trust_radius)
        if self.is_master:
            logger.info("resumed training from %s", latest)

    def _find_latest_checkpoint(self) -> Optional[Path]:
        if not self.output_dir.exists():
            return None
        checkpoints = sorted(self.output_dir.glob("checkpoint_step_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _init_distributed(self, device_override: Optional[torch.device]) -> torch.device:
        device = device_override
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if world_size > 1:
            backend = self.cfg.distributed.backend
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_distributed = True
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
        return device or torch.device("cpu")

    def _wrap_model(self) -> None:
        self.use_fsdp = (
            self.cfg.distributed.fsdp
            and self.is_distributed
            and FSDP is not None
        )
        self.use_ddp = (
            self.cfg.distributed.ddp and self.is_distributed and not self.use_fsdp
        )
        if self.use_fsdp:
            self.model = FSDP(self.model)
        elif self.use_ddp:
            device_ids = [self.device.index] if self.device.type == "cuda" else None
            self.model = DDP(self.model, device_ids=device_ids)
        self.model_to_save = self.model
        if hasattr(self.model, "module"):
            self.model_to_save = self.model.module  # type: ignore[assignment]

    def _resolve_mixture(
        self, mixture_spec: str | Path
    ) -> tuple[Mapping[str, float], Optional[TrainingScheduler]]:
        path = Path(mixture_spec)
        if path.exists():
            scheduler = TrainingScheduler.from_file(path)
            return scheduler.mixture(), scheduler
        return self._mixture_from_catalog(str(mixture_spec)), None

    def _mixture_from_catalog(self, name: str) -> Mapping[str, float]:
        if name in self.catalog.mixtures:
            stage = self.catalog.mixtures[name][0]
            return dict(stage.weights)
        if self.catalog.datasets:
            weights = {cfg.id: float(cfg.weight) for cfg in self.catalog.datasets}
            total = sum(weights.values())
            if total > 0:
                return {k: v / total for k, v in weights.items()}
        raise ValueError(f"unknown mixture '{name}'")

    def _maybe_create_summary_writer(self):
        if not (self.is_master and self.logging_cfg.tensorboard):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:  # pragma: no cover
            logger.warning("TensorBoard not available; disabling writer")
            return None
        log_dir = self.output_dir / "tb"
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))

    def _maybe_init_wandb(self):
        if not (self.is_master and self.logging_cfg.wandb):
            return None
        try:
            import wandb
        except Exception:  # pragma: no cover
            logger.warning("wandb not available; disabling logging")
            return None
        project = self.logging_cfg.wandb_project or "causal-diamond"
        run = wandb.init(project=project, config={"model": str(self.cfg.model_cfg)}, reinit=True)
        return run

    def _should_no_sync(self, micro_step: int, accum: int) -> bool:
        return self.use_ddp and (micro_step + 1) % accum != 0

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
