from __future__ import annotations

import math
from typing import Tuple

import torch


def build_rope_cache(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling: Tuple[str, float] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create precomputed RoPE cos/sin caches.

    Parameters
    ----------
    dim: int
        Head dimension (must be even).
    max_seq_len: int
        Longest sequence that will be encoded.
    base: float
        Base theta used for rotary embeddings.
    scaling: Tuple[str, float] | None
        Optional scaling directive (type, factor) for extrapolation.
    device: torch.device | None
        Where to place the cache.
    dtype: torch.dtype | None
        Tensor dtype (defaults to torch.get_default_dtype()).
    """

    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")

    if scaling:
        scale_type, factor = scaling
        if scale_type == "linear":
            base = base * factor
        elif scale_type == "dynamic":
            base = base * math.log(max_seq_len) / math.log(factor)
        else:
            raise ValueError(f"unsupported rope scaling type: {scale_type}")

    theta = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    seq = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(-1)
    angles = seq * theta
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors.

    Assumes shape (..., seq_len, head_dim).
    """

    seq_len = x.size(-2)
    cos = cos[:seq_len]
    sin = sin[:seq_len]

    # Reshape cos/sin so they broadcast across any leading batch dims.
    broadcast_shape = (1,) * (x.dim() - 2) + (seq_len, cos.size(-1))
    cos = cos.view(broadcast_shape)
    sin = sin.view(broadcast_shape)
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)

    return (x * cos) + (rotated * sin)


__all__ = ["build_rope_cache", "apply_rope"]
