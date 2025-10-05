from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def make_view_mask(
    batch_size: int,
    seq_len: int,
    slot_len: int,
    center_slots: torch.LongTensor,
    window: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Construct an additive attention bias enforcing the Π_A view window.

    Returns a tensor of shape ``[B, 1, T, T]`` with zeros on permitted
    positions and ``-inf`` outside the allowed slot window (and for
    strictly causal violations j > i).
    """

    if dtype is None:
        dtype = torch.float32
    slots = torch.arange(seq_len, device=device) // slot_len
    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    for idx in range(batch_size):
        center = int(center_slots[idx].item())
        allowed = (slots >= center - window) & (slots <= center + window)
        mask[idx, 0, :, allowed] = 0.0

    # Apply causal structure: no attention to future positions.
    causal = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), 1)
    mask[:, :, causal] = float("-inf")
    return mask


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
    "make_view_mask",
]
