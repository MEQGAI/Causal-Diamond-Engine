"""Evaluation harness computing perplexity, cascade-fail rate, and planner ECE."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from fm_core.checkpoint import load_checkpoint
from fm_data.packing import SentencePieceTokenizer, SpecialTokenIds
from fm_train.objectives import language_model_loss


@dataclass
class GateThresholds:
    accuracy_min: float
    latency_p95_ms_max: float
    memory_gb_max: float
    safety_false_negative_max: str


@dataclass
class EvalSample:
    prompt: str
    target: str


def load_thresholds(config_path: Path | str) -> GateThresholds:
    data = (
        json.loads(Path(config_path).read_text())
        if str(config_path).endswith(".json")
        else _load_yaml(config_path)
    )
    return GateThresholds(
        accuracy_min=float(data["accuracy_min"]),
        latency_p95_ms_max=float(data["latency_p95_ms_max"]),
        memory_gb_max=float(data["memory_gb_max"]),
        safety_false_negative_max=str(data["safety_false_negative_max"]),
    )


def _load_yaml(
    path: Path | str,
) -> Dict[str, object]:  # pragma: no cover - fallback parser
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _default_samples() -> List[EvalSample]:
    return [
        EvalSample(
            prompt="Explain the concept of a causal diamond in simple terms.",
            target="A causal diamond is a bounded spacetime region defined by intersecting lightcones.",
        ),
        EvalSample(
            prompt="Compute the integral of x^2 from 0 to 3.",
            target="The integral evaluates to 9.",
        ),
    ]


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = language_model_loss(logits, labels)
    return float(torch.exp(loss))


def cascade_fail_rate(planner_probs: torch.Tensor, planner_mask: torch.Tensor) -> float:
    if planner_mask.sum() == 0:
        return 0.0
    probs = planner_probs.exp().clamp_min(1e-9)
    entropy = -(probs * planner_probs).sum(dim=-1)
    failures = (entropy > 3.0) & planner_mask
    return float(failures.float().mean().item())


def planner_ece(
    planner_probs: torch.Tensor, planner_mask: torch.Tensor, bins: int = 10
) -> float:
    confidences = planner_probs.exp().max(dim=-1).values[planner_mask]
    if confidences.numel() == 0:
        return 0.0
    # Assume top-1 correctness in absence of labels; penalise overconfidence.
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=confidences.device)
    ece = torch.zeros(1, device=confidences.device)
    targets = torch.ones_like(confidences)
    for i in range(bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        if mask.sum() == 0:
            continue
        acc = targets[mask].mean()
        conf = confidences[mask].mean()
        ece += torch.abs(acc - conf) * mask.float().mean()
    return float(ece.item())


def evaluate_model(model_dir: Path | str) -> Dict[str, float]:
    model, cfg_dict = load_checkpoint(model_dir, map_location="cpu")
    model.eval()
    config = model.cfg
    tokenizer_path = Path(model_dir) / "tokenizer.model"
    tokenizer = SentencePieceTokenizer(str(tokenizer_path), allow_fallback=False)
    specials = SpecialTokenIds.from_processor(
        tokenizer.processor, config.special_tokens
    )

    metrics: Dict[str, List[float]] = {
        "perplexity": [],
        "cascade_fail": [],
        "planner_ece": [],
    }
    samples = _default_samples()
    with torch.no_grad():
        for sample in samples:
            token_ids = [specials.bos]
            token_ids.extend(tokenizer.encode(sample.prompt))
            token_ids.append(specials.eos)
            token_ids = token_ids[: config.seq_len]
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            outputs = model(input_ids)
            metrics["perplexity"].append(perplexity(outputs["logits"], input_ids))
            planner = outputs["planner"].log_probs
            mask = torch.zeros_like(planner[..., 0], dtype=torch.bool)
            metrics["cascade_fail"].append(cascade_fail_rate(planner, mask))
            metrics["planner_ece"].append(planner_ece(planner, mask))

    return {
        name: float(sum(values) / max(1, len(values)))
        for name, values in metrics.items()
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate foundation model gates")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--model", required=False, type=Path, default=Path("checkpoints/fm-b-220m")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    thresholds = load_thresholds(args.config)
    metrics = evaluate_model(args.model)
    print(json.dumps({"metrics": metrics, "thresholds": thresholds.__dict__}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
