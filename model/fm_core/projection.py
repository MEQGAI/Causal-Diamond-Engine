from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SlotLayout:
    slot_len: int
    slots_per_seq: int

    @property
    def seq_len(self) -> int:
        return self.slot_len * self.slots_per_seq


def compute_slot_indices(seq_len: int, slot_len: int) -> torch.Tensor:
    """Return slot index per position (shape [seq_len])."""

    slots = torch.arange(seq_len) // slot_len
    return slots


def build_slot_mask(
    seq_len: int,
    slot_len: int,
    window: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct an attention mask implementing Π_A.

    Mask is True where attention is allowed.
    """

    slots = compute_slot_indices(seq_len, slot_len).to(device=device)
    slot_matrix = slots.unsqueeze(0) - slots.unsqueeze(1)
    mask = slot_matrix.abs() <= window
    return mask


def mask_attention(attn_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply slot mask to attention logits (in-place masking using -inf)."""

    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    return attn_scores


def extract_view(
    hidden: torch.Tensor, center_slot: int, layout: SlotLayout
) -> torch.Tensor:
    """Slice the hidden state corresponding to the actualised view."""

    start = max(0, (center_slot - 1) * layout.slot_len)
    end = min(layout.seq_len, (center_slot + 2) * layout.slot_len)
    return hidden[..., start:end, :]


__all__ = [
    "SlotLayout",
    "build_slot_mask",
    "mask_attention",
    "compute_slot_indices",
    "extract_view",
]
