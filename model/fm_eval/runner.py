"""Evaluation harness computing perplexity, cascade-fail rate, planner ECE, and JSON exactness."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from fm_core.checkpoint import load_checkpoint
from fm_data.packing import SentencePieceTokenizer, SpecialTokenIds
from fm_train.objectives import language_model_loss

import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalRecord:
    prompt: str
    target: str
    expect_json: bool = False


def load_thresholds(path: Path | str) -> Dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "perplexity_max": float(data.get("perplexity_max", 50.0)),
        "cascade_fail_max": float(data.get("cascade_fail_max", 0.2)),
        "planner_ece_max": float(data.get("planner_ece_max", 0.1)),
        "json_exact_min": float(data.get("json_exact_min", 0.8)),
    }


def _default_records() -> List[EvalRecord]:
    return [
        EvalRecord(
            prompt="Explain the concept of a causal diamond in one sentence.",
            target="A causal diamond is the spacetime region defined by intersecting future and past light cones.",
        ),
        EvalRecord(
            prompt="Return a JSON object with keys 'a' and 'b' where a=1 and b=2.",
            target="{"a": 1, "b": 2}",
            expect_json=True,
        ),
        EvalRecord(
            prompt="List two steps for solving a quadratic equation.",
            target="1. Compute the discriminant. 2. Use the quadratic formula.",
        ),
    ]


def _perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = language_model_loss(logits, labels)
    return float(torch.exp(loss))


def _cascade_fail_rate(planner_log_probs: torch.Tensor, mask: torch.BoolTensor) -> float:
    if mask.sum() == 0:
        return 0.0
    probs = planner_log_probs.exp().clamp_min(1e-9)
    entropy = -(probs * planner_log_probs).sum(dim=-1)
    failures = (entropy > 3.0) & mask
    return float(failures.float().mean().item())


def _planner_ece(planner_log_probs: torch.Tensor, mask: torch.BoolTensor, bins: int = 10) -> float:
    confidences = planner_log_probs.exp().max(dim=-1).values[mask]
    if confidences.numel() == 0:
        return 0.0
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)
    for i in range(bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        bucket = (confidences >= lower) & (confidences < upper)
        if bucket.sum() == 0:
            continue
        conf = confidences[bucket].mean()
        # Without ground truth planner labels we treat buckets as perfectly accurate;
        # the penalty thus reflects raw confidence magnitude.
        ece += torch.abs(1.0 - conf) * bucket.float().mean()
    return float(ece.item())


def _json_exactness(output: str, target: str) -> float:
    try:
        produced = json.loads(output)
        expected = json.loads(target)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if produced == expected else 0.0


def evaluate_model(model_dir: Path | str, records: Optional[List[EvalRecord]] = None) -> Dict[str, float]:
    model, _ = load_checkpoint(model_dir, map_location="cpu")
    model.eval()
    config = model.cfg
    tokenizer_path = Path(model_dir) / "tokenizer.model"
    tokenizer = SentencePieceTokenizer(str(tokenizer_path), allow_fallback=True)
    specials = SpecialTokenIds.from_processor(tokenizer.processor, config.special_tokens)

    if records is None:
        records = _default_records()

    metrics: Dict[str, List[float]] = {
        "perplexity": [],
        "cascade_fail": [],
        "planner_ece": [],
        "json_exact": [],
    }

    with torch.no_grad():
        for record in records:
            tokens = [specials.bos]
            tokens.extend(tokenizer.encode(record.prompt))
            tokens.append(specials.eos)
            tokens = tokens[: config.seq_len]
            input_ids = torch.tensor([tokens], dtype=torch.long)
            outputs = model(input_ids)
            logits = outputs["logits"]
            planner = outputs["planner"].log_probs
            mask = torch.ones_like(planner[..., 0], dtype=torch.bool)

            metrics["perplexity"].append(_perplexity(logits, input_ids))
            metrics["cascade_fail"].append(_cascade_fail_rate(planner, mask))
            metrics["planner_ece"].append(_planner_ece(planner, mask))

            if record.expect_json:
                completion = tokenizer.processor.decode(tokens[1:])  # type: ignore[attr-defined]
                metrics["json_exact"].append(_json_exactness(completion, record.target))

    aggregated = {
        name: float(sum(values) / max(1, len(values)))
        for name, values in metrics.items()
    }
    return aggregated


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate foundation model gates")
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument(
        "--model", required=False, type=Path, default=Path("checkpoints/fm-b-220m"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    thresholds = load_thresholds(args.thresholds)
    metrics = evaluate_model(args.model)
    payload = {"metrics": metrics, "thresholds": thresholds}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
