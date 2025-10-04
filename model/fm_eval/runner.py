"""Evaluation entrypoint used by CI kill-switch gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def simulate_metrics(config: Dict[str, Any]) -> Dict[str, float]:
    return {
        "accuracy": float(config["accuracy_min"]) + 0.05,
        "latency_p95_ms": float(config["latency_p95_ms_max"]) - 5,
        "memory_gb": float(config["memory_gb_max"]) - 1,
        "safety_false_negative": 0.01,
    }


def check_gates(config: Dict[str, Any], metrics: Dict[str, float]) -> list[str]:
    failures: list[str] = []
    if metrics["accuracy"] < float(config["accuracy_min"]):
        failures.append("accuracy")
    if metrics["latency_p95_ms"] > float(config["latency_p95_ms_max"]):
        failures.append("latency_p95_ms")
    if metrics["memory_gb"] > float(config["memory_gb_max"]):
        failures.append("memory_gb")
    safety_threshold_raw = str(config["safety_false_negative_max"]).rstrip("%")
    safety_threshold = float(safety_threshold_raw) / 100.0
    if metrics["safety_false_negative"] > safety_threshold:
        failures.append("safety_false_negative")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation kill switches")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/kill_numbers.yaml"),
        help="Path to the kill numbers configuration.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = simulate_metrics(config)
    failures = check_gates(config, metrics)

    payload = {
        "config": config,
        "metrics": metrics,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
