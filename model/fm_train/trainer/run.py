"""Training entrypoint for large-scale runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from fm_eval import runner as eval_runner
from fm_train.config import TrainConfig, load_train_config
from fm_train.runtime import train_from_config

import logging

logger = logging.getLogger(__name__)


DEFAULT_STEPS = 1000


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    checkpoints = sorted(output_dir.glob("checkpoint_step_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    return None


def _load_trainer_config(path: Path) -> TrainConfig:
    cfg = load_train_config(path)
    root = path.resolve().parents[2]
    cfg.model_cfg = (root / cfg.model_cfg).resolve()
    cfg.data_catalog = (root / cfg.data_catalog).resolve()
    cfg.checkpoints.output_dir = (root / cfg.checkpoints.output_dir).resolve()
    if Path(cfg.mixture).exists():
        cfg.mixture = (root / cfg.mixture).resolve().as_posix()
    return cfg


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal-Diamond Trainer")
    parser.add_argument("--config", type=Path, required=True, help="Training YAML config")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of optimizer steps")
    parser.add_argument(
        "--resume", choices=["auto", "never"], default="auto", help="Resume policy"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation harness after training completes",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Optional thresholds JSON for evaluation gating",
    )
    parser.add_argument(
        "--eval-model-dir",
        type=Path,
        help="Explicit model directory for evaluation",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_evaluation(model_dir: Path, thresholds_path: Optional[Path]) -> None:
    if thresholds_path:
        thresholds = eval_runner.load_thresholds(thresholds_path)
    else:
        thresholds = {
            "perplexity_max": 50.0,
            "cascade_fail_max": 0.2,
            "planner_ece_max": 0.1,
            "json_exact_min": 0.8,
        }
    metrics = eval_runner.evaluate_model(model_dir)
    payload = {"metrics": metrics, "thresholds": thresholds}
    print(json.dumps(payload, indent=2))

    fail_reasons = []
    if metrics["perplexity"] > thresholds["perplexity_max"]:
        fail_reasons.append("perplexity")
    if metrics["cascade_fail"] > thresholds["cascade_fail_max"]:
        fail_reasons.append("cascade_fail")
    if metrics["planner_ece"] > thresholds["planner_ece_max"]:
        fail_reasons.append("planner_ece")
    if metrics.get("json_exact", 1.0) < thresholds["json_exact_min"]:
        fail_reasons.append("json_exact")
    if fail_reasons:
        logger.warning("Evaluation thresholds not met: %s", ", ".join(fail_reasons))
    else:
        logger.info("Evaluation thresholds satisfied")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_trainer_config(args.config)
    cfg.checkpoints.resume = args.resume
    trainer = train_from_config(cfg, steps=args.steps)

    if args.evaluate and cfg.checkpoints.output_dir.exists():
        model_dir = args.eval_model_dir or cfg.checkpoints.output_dir
        latest = _find_latest_checkpoint(cfg.checkpoints.output_dir)
        if latest is None:
            logger.warning("No checkpoint found for evaluation; skipping")
        else:
            _run_evaluation(model_dir, args.thresholds)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
