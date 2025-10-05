from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .config import ModelConfig
from .flash_attn_wrapper import flash_attention
from .planner_head import PlannerHead, PlannerOutput
from .projection import SlotLayout, build_slot_mask
from .rope import apply_rope, build_rope_cache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
        rope_theta: float,
        rope_scaling: tuple[str, float],
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * n_kv_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_seq_len = max_seq_len
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _maybe_init_rope(self, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.rope_cos is not None
            and self.rope_cos.device == device
            and self.rope_cos.dtype == dtype
        ):
            return
        cos, sin = build_rope_cache(
            self.head_dim,
            self.max_seq_len,
            base=self.rope_theta,
            scaling=self.rope_scaling,
            device=device,
            dtype=dtype,
        )
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        x: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        self._maybe_init_rope(device, dtype)

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        expand_factor = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(expand_factor, dim=1)
        v = v.repeat_interleave(expand_factor, dim=1)

        if slot_mask is not None:
            # scaled_dot_product_attention expects attn_mask with True=mask
            mask = (~slot_mask.bool()).unsqueeze(0).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )
        else:
            out = flash_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            )

        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(
            dim=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            dropout=cfg.dropout_prob,
            rope_theta=cfg.rope_theta,
            rope_scaling=(cfg.rope_scaling.type, cfg.rope_scaling.factor),
            max_seq_len=cfg.seq_len,
        )
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        slot_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.attn_norm(x), slot_mask=slot_mask)
        x = x + self.dropout(attn_out)
        mlp_out = self.mlp(self.mlp_norm(x))
        return x + self.dropout(mlp_out)


class FoundationModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        slot_layout: Optional[SlotLayout] = None,
        slot_window: int = 1,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.planner_head = PlannerHead(cfg.d_model, cfg.planner_vocab)
        self.slot_layout = slot_layout or SlotLayout(
            slot_len=cfg.seq_len, slots_per_seq=1
        )
        self.slot_window = slot_window
        self.register_buffer(
            "slot_mask",
            build_slot_mask(
                self.slot_layout.seq_len,
                self.slot_layout.slot_len,
                window=self.slot_window,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        planner_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | PlannerOutput]:
        if input_ids.size(1) > self.slot_layout.seq_len:
            raise ValueError("input length exceeds configured slot layout")
        x = self.embeddings(input_ids)
        slot_mask = self.slot_mask[: input_ids.size(1), : input_ids.size(1)]
        for block in self.blocks:
            x = block(x, slot_mask=slot_mask)
        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        planner = self.planner_head(hidden)
        if planner_mask is not None:
            # mask planner logits outside plan spans by setting to -inf
            masked_logits = planner.logits.masked_fill(
                ~planner_mask.bool().unsqueeze(-1), float("-inf")
            )
            planner = PlannerOutput(
                logits=masked_logits, log_probs=torch.log_softmax(masked_logits, dim=-1)
            )
        return {
            "hidden_states": hidden,
            "logits": logits,
            "planner": planner,
        }


__all__ = ["FoundationModel", "TransformerBlock", "RMSNorm", "SwiGLU", "SlotLayout"]
