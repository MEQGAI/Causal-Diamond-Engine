from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - optional dependency
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn
except Exception:  # pragma: no cover - CPU fallback path
    _flash_attn = None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Wrapper that uses FlashAttention if available, otherwise SDPA fallback."""

    if _flash_attn is None or q.dtype not in (torch.float16, torch.bfloat16):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    if attn_mask is not None:
        raise NotImplementedError("FlashAttention path does not support custom attn_mask yet")

    return _flash_attn(q, k, v, dropout_p=dropout_p, softmax_scale=None, causal=is_causal)


__all__ = ["flash_attention"]
