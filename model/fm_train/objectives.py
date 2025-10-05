from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fm_core.projection import SlotLayout


def language_model_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    return loss


def _compute_slot_weights(
    seq_len: int,
    slot_len: int,
    center_slots: torch.LongTensor,
    slot_weights: Tuple[float, ...],
    device: torch.device,
) -> torch.Tensor:
    slots = torch.arange(seq_len, device=device) // slot_len
    weights = torch.zeros((center_slots.size(0), seq_len), device=device)
    for offset, value in enumerate(slot_weights):
        if value == 0:
            continue
        mask = (slots.unsqueeze(0) - center_slots.view(-1, 1)).abs() == offset
        weights = torch.where(mask, torch.tensor(value, device=device), weights)
    return weights


def compute_modal_penalty(
    planner_logits: torch.Tensor,
    planner_logits_view: torch.Tensor,
    token_logits: Optional[torch.Tensor],
    token_logits_view: Optional[torch.Tensor],
    plan_pos_mask: torch.BoolTensor,
    plan_span_mask: Optional[torch.BoolTensor],
    slot_len: int,
    center_slots: torch.LongTensor,
    view_window: int = 1,
    slot_weights: Tuple[float, ...] = (1.0, 0.5),
    tau_planner: float = 1.0,
    tau_token: float = 1.0,
    eps: float = 1e-6,
    clip_kl: float = 10.0,
    token_topk: int = 256,
    stop_grad_projection: bool = True,
    lambda_planner: float = 1.0,
    lambda_token: float = 0.0,
) -> Dict[str, torch.Tensor]:
    device = planner_logits.device
    dtype = planner_logits.dtype
    batch, seq_len, _ = planner_logits.shape

    if stop_grad_projection:
        planner_logits_view = planner_logits_view.detach()
        if token_logits_view is not None:
            token_logits_view = token_logits_view.detach()

    plan_mask = plan_pos_mask.to(device=device)
    if plan_mask.dtype != torch.bool:
        plan_mask = plan_mask.bool()

    weights = _compute_slot_weights(
        seq_len,
        slot_len,
        center_slots.to(device=device),
        slot_weights,
        device,
    )
    weights = weights * plan_mask.float()
    weight_sum_raw = weights.sum()
    total_weight = weight_sum_raw.clamp_min(eps)

    # Planner KL
    planner_scaled = planner_logits / tau_planner
    planner_view_scaled = planner_logits_view / tau_planner
    p = torch.softmax(planner_scaled, dim=-1).clamp_min(eps)
    q = torch.softmax(planner_view_scaled, dim=-1).clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl_planner = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)
    kl_planner = kl_planner.clamp_min(0.0).clamp_max(clip_kl)

    planner_loss = torch.tensor(0.0, device=device, dtype=dtype)
    planner_mean = torch.tensor(0.0, device=device, dtype=dtype)

    if weight_sum_raw.item() > 0:
        planner_loss = (weights * kl_planner).sum() / total_weight
        planner_mean = (
            kl_planner * plan_mask.float()
        ).sum() / plan_mask.float().sum().clamp_min(1.0)

    planner_loss = planner_loss * lambda_planner

    # Token KL (optional)
    token_loss = torch.tensor(0.0, device=device, dtype=dtype)
    token_mean = torch.tensor(0.0, device=device, dtype=dtype)
    if (
        lambda_token > 0.0
        and token_logits is not None
        and token_logits_view is not None
        and plan_span_mask is not None
        and plan_span_mask.any()
    ):
        span_mask = plan_span_mask.to(device=device)
        selected_logits = token_logits[span_mask]
        selected_view = token_logits_view[span_mask]
        vocab = selected_logits.size(-1)
        topk = min(token_topk, vocab)
        vals1, idx1 = torch.topk(selected_logits, topk, dim=-1)
        vals2, idx2 = torch.topk(selected_view, topk, dim=-1)
        gather_mask = torch.zeros_like(selected_logits, dtype=torch.bool)
        gather_mask.scatter_(1, idx1, True)
        gather_mask.scatter_(1, idx2, True)

        neg_inf = torch.finfo(selected_logits.dtype).min
        logits_union = torch.where(gather_mask, selected_logits, neg_inf)
        view_union = torch.where(gather_mask, selected_view, neg_inf)

        p_tok = torch.softmax(logits_union / tau_token, dim=-1).clamp_min(eps)
        q_tok = torch.softmax(view_union / tau_token, dim=-1).clamp_min(eps)
        p_tok = p_tok / p_tok.sum(dim=-1, keepdim=True)
        q_tok = q_tok / q_tok.sum(dim=-1, keepdim=True)
        kl_token = (p_tok * (torch.log(p_tok) - torch.log(q_tok))).sum(dim=-1)
        kl_token = kl_token.clamp_min(0.0).clamp_max(clip_kl)

        token_loss = kl_token.mean() * lambda_token
        token_mean = kl_token.mean()

    total = planner_loss + token_loss

    return {
        "l_mod_total": total,
        "l_mod_planner": planner_loss,
        "l_mod_token": token_loss,
        "planner_kl_mean": planner_mean,
        "token_kl_mean": token_mean,
    }


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    slot_layout: SlotLayout,
    modal_cfg,
    modal_inputs: Optional[Dict[str, torch.Tensor]] = None,
    lambda_geo: float = 0.0,
) -> Dict[str, torch.Tensor]:
    logits = outputs["logits"]
    labels = batch["input_ids"]

    losses: Dict[str, torch.Tensor] = {}
    losses["loss_ent"] = language_model_loss(logits, labels)
    losses["loss_geo"] = torch.tensor(
        lambda_geo, device=logits.device, dtype=losses["loss_ent"].dtype
    )

    modal_zero = torch.zeros(1, device=logits.device, dtype=losses["loss_ent"].dtype)[0]
    if modal_inputs is not None:
        modal_terms = compute_modal_penalty(**modal_inputs)
        losses["loss_mod_planner"] = modal_terms["l_mod_planner"]
        losses["loss_mod_token"] = modal_terms["l_mod_token"]
        losses["planner_kl_mean"] = modal_terms["planner_kl_mean"]
        losses["token_kl_mean"] = modal_terms["token_kl_mean"]
        losses["loss_mod"] = modal_terms["l_mod_total"]
    else:
        losses["loss_mod_planner"] = modal_zero
        losses["loss_mod_token"] = modal_zero
        losses["planner_kl_mean"] = modal_zero
        losses["token_kl_mean"] = modal_zero
        losses["loss_mod"] = modal_zero

    losses["loss_total"] = (
        losses["loss_ent"]
        + modal_cfg.lambda_mod * losses["loss_mod"]
        + lambda_geo * losses["loss_geo"]
    )
    return losses


__all__ = ["compute_total_loss", "compute_modal_penalty"]
