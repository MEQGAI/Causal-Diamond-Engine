"""Lightweight training scheduler placeholder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from fm_train.losses.modal import modal_penalty
from fm_train.losses.stability import null_stability_metric


@dataclass
class TrainingScheduler:
    task: str
    budget: float
    epochs: int

    def run(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger = []
        for epoch in range(self.epochs):
            kl = modal_penalty(self.task, epoch)
            stability = null_stability_metric(kl, self.budget)
            ledger.append({"epoch": epoch, "kl": kl, "stability": stability})
        self._write_ledger(output_dir, ledger)

    def _write_ledger(self, output_dir: Path, ledger: Iterable[dict]) -> None:
        target = output_dir / "ledger.json"
        import json

        with target.open("w", encoding="utf-8") as fp:
            json.dump(list(ledger), fp, indent=2)
