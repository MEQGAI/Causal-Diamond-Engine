from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from fm_core.projection import SlotLayout
from fm_kernels import categorical_kl


def language_model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return loss


def planner_kl(
    planner_log_probs: torch.Tensor, slot_layout: SlotLayout, window: int = 1
) -> torch.Tensor:
    batch, seq, vocab = planner_log_probs.shape
    device = planner_log_probs.device
    slots = torch.arange(seq, device=device) // slot_layout.slot_len
    projected = torch.zeros_like(planner_log_probs)
    valid = torch.zeros((batch, seq), device=device, dtype=torch.bool)
    # probs variable not needed beyond debug; avoid unused variable

    for slot_idx in range(slot_layout.slots_per_seq):
        mask = (slots >= slot_idx - window) & (slots <= slot_idx + window)
        if mask.sum() == 0:
            continue
        log_prob = planner_log_probs[:, mask, :]
        slot_probs = log_prob.exp().clamp_min(1e-9)
        mean_prob = slot_probs.mean(dim=1).clamp_min(1e-9)
        mean_log_prob = torch.log(mean_prob)
        projected[:, mask, :] = mean_log_prob.unsqueeze(1).expand(-1, mask.sum(), -1)
        valid[:, mask] = True

    # Default to original distribution for tokens outside any slot (should be none).
    projected = torch.where(valid.unsqueeze(-1), projected, planner_log_probs)
    kl = categorical_kl(planner_log_probs, projected)
    return kl.mean()


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    slot_layout: SlotLayout,
    lambda_mod: float,
    lambda_geo: float = 0.0,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    planner: torch.Tensor = outputs["planner"].log_probs
    labels = batch["input_ids"]

    losses: Dict[str, torch.Tensor] = {}
    losses["loss_ent"] = language_model_loss(logits, labels)
    losses["loss_mod"] = planner_kl(planner, slot_layout)
    losses["loss_geo"] = torch.tensor(lambda_geo, device=logits.device)
    losses["loss_total"] = (
        losses["loss_ent"]
        + lambda_mod * losses["loss_mod"]
        + lambda_geo * losses["loss_geo"]
    )
    return losses


__all__ = ["compute_total_loss"]
