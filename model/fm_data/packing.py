from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .catalog import Catalog
from fm_core.config import SpecialTokens

try:  # pragma: no cover - optional dependency
    import sentencepiece as spm  # type: ignore
except ImportError:  # pragma: no cover
    spm = None


@dataclass
class SpecialTokenIds:
    bos: int
    eos: int
    pad: int
    unk: int
    diamond_start: int
    diamond_end: int
    view_start: int
    view_end: int
    plan_start: int
    plan_end: int
    tool: int

    @classmethod
    def from_processor(cls, processor: spm.SentencePieceProcessor, specials: SpecialTokens) -> "SpecialTokenIds":
        ids = {
            "bos": processor.piece_to_id(specials.bos),
            "eos": processor.piece_to_id(specials.eos),
            "pad": processor.piece_to_id(specials.pad),
            "unk": processor.piece_to_id(specials.unk),
            "diamond_start": processor.piece_to_id(specials.diamond_start),
            "diamond_end": processor.piece_to_id(specials.diamond_end),
            "view_start": processor.piece_to_id(specials.view_start),
            "view_end": processor.piece_to_id(specials.view_end),
            "plan_start": processor.piece_to_id(specials.plan_start),
            "plan_end": processor.piece_to_id(specials.plan_end),
            "tool": processor.piece_to_id(specials.tool),
        }
        missing = {name: token for name, token in ids.items() if token < 0}
        if missing:
            raise ValueError(f"special tokens missing from tokenizer: {missing}")
        return cls(**ids)


class _FallbackProcessor:
    def __init__(self, vocab_size: int = 32768) -> None:
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}

    def encode(self, text: str) -> List[int]:
        ids = []
        for ch in text:
            if ch not in self.token_to_id:
                self.token_to_id[ch] = len(self.token_to_id) % self.vocab_size
            ids.append(self.token_to_id[ch])
        return ids

    def piece_to_id(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id) % self.vocab_size
        return self.token_to_id[token]


class SentencePieceTokenizer:
    def __init__(self, model_path: str | None, allow_fallback: bool = False) -> None:
        path = Path(model_path) if model_path else None
        if spm is None:
            if allow_fallback:
                self.processor = _FallbackProcessor()
            else:
                raise ImportError("sentencepiece is required; install sentencepiece or set allow_fallback=True")
        elif path is None or not path.exists():
            if allow_fallback:
                self.processor = _FallbackProcessor()
            else:
                raise FileNotFoundError(f"tokenizer model not found at {model_path}")
        else:
            self.processor = spm.SentencePieceProcessor()
            self.processor.load(str(path))

    def encode(self, text: str) -> List[int]:
        if hasattr(self.processor, "encode"):
            try:
                return list(self.processor.encode(text, out_type=int))  # type: ignore[arg-type]
            except TypeError:
                return list(self.processor.encode(text))  # type: ignore[call-arg]
        return list(self.processor.encode(text))

    def piece_to_id(self, token: str) -> int:
        if hasattr(self.processor, "piece_to_id"):
            return int(self.processor.piece_to_id(token))  # type: ignore[call-arg]
        return int(self.processor.piece_to_id(token))


@dataclass
class PackedSequence:
    input_ids: torch.LongTensor
    planner_mask: torch.BoolTensor


class DiamondPacker:
    def __init__(
        self,
        catalog: Catalog,
        special_tokens: SpecialTokens,
        plan_probability: float = 0.3,
        seed: Optional[int] = None,
        allow_tokenizer_fallback: bool = False,
    ) -> None:
        self.catalog = catalog
        self.tokenizer = SentencePieceTokenizer(str(catalog.tokenizer), allow_fallback=allow_tokenizer_fallback)
        self.special_ids = SpecialTokenIds.from_processor(self.tokenizer.processor, special_tokens)
        self.seq_len = catalog.seq_len
        self.slot_len = catalog.packing.slot_len
        self.slots_per_seq = catalog.packing.slots_per_seq
        self.plan_probability = plan_probability
        self.rng = random.Random(seed)

    def _insert_specials(self, tokens: List[int]) -> Tuple[List[int], List[bool]]:
        tokens = tokens[: max(0, self.seq_len - 4)]
        seq = [self.special_ids.bos, self.special_ids.diamond_start]
        seq.extend(tokens)
        seq.append(self.special_ids.diamond_end)
        seq.append(self.special_ids.eos)

        planner_mask = [False] * len(seq)

        # Stamp view tokens around central slot.
        slot_center = self.slots_per_seq // 2
        view_start_idx = max(2, slot_center * self.slot_len)
        view_end_idx = min(len(seq) - 1, (slot_center + 1) * self.slot_len - 1)
        if view_start_idx < len(seq):
            seq[view_start_idx] = self.special_ids.view_start
        if view_end_idx < len(seq):
            seq[view_end_idx] = self.special_ids.view_end

        # Optionally insert plan span markers.
        if self.rng.random() < self.plan_probability and view_end_idx - view_start_idx > 4:
            span_start = self.rng.randint(view_start_idx + 1, view_end_idx - 2)
            span_end = min(len(seq) - 2, span_start + self.rng.randint(4, self.slot_len // 4))
            seq[span_start] = self.special_ids.plan_start
            seq[span_end] = self.special_ids.plan_end
            for idx in range(span_start, span_end + 1):
                planner_mask[idx] = True

        if len(seq) < self.seq_len:
            pad_length = self.seq_len - len(seq)
            seq.extend([self.special_ids.pad] * pad_length)
            planner_mask.extend([False] * pad_length)
        else:
            seq = seq[: self.seq_len]
            planner_mask = planner_mask[: self.seq_len]

        return seq, planner_mask

    def pack_raw_text(self, text: str) -> PackedSequence:
        tokens = self.tokenizer.encode(text)
        if not tokens:
            raise ValueError("tokenizer produced empty sequence; confirm tokenizer model and input text")
        seq, planner_mask = self._insert_specials(tokens)
        input_ids = torch.tensor(seq, dtype=torch.long)
        planner = torch.tensor(planner_mask, dtype=torch.bool)
        return PackedSequence(input_ids=input_ids, planner_mask=planner)


__all__ = ["DiamondPacker", "PackedSequence", "SentencePieceTokenizer", "SpecialTokenIds"]
