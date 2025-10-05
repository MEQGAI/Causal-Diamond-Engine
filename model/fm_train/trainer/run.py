"""Training entrypoint for the Reality's Ledger research stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fm_train.trainer.scheduler import TrainingScheduler


@dataclass
class TrainerConfig:
    task: str
    budget: float
    epochs: int
    output_dir: Path


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="Reality's Ledger trainer")
    parser.add_argument(
        "--task", default="tool_reasoning", help="Curriculum identifier"
    )
    parser.add_argument(
        "--budget", type=float, default=2.0, help="Causal-diamond budget"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/processed/runs"),
        help="Where to store training artifacts",
    )
    args = parser.parse_args()
    return TrainerConfig(args.task, args.budget, args.epochs, args.output_dir)


def main() -> None:
    config = parse_args()
    scheduler = TrainingScheduler(config.task, config.budget, config.epochs)
    scheduler.run(config.output_dir)


if __name__ == "__main__":
    main()
