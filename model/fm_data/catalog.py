from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml


@dataclass
class DatasetConfig:
    id: str
    kind: str
    shards: str
    format: str
    weight: float
    license: Optional[str] = None
    filters: List[Mapping[str, object]] = field(default_factory=list)


@dataclass
class MixtureWeight:
    until_tokens: float
    weights: Mapping[str, float]


@dataclass
class PackingConfig:
    slot_len: int
    slots_per_seq: int


@dataclass
class Catalog:
    version: int
    tokenizer: Path
    seq_len: int
    packing: PackingConfig
    datasets: List[DatasetConfig]
    mixtures: Mapping[str, List[MixtureWeight]]

    def dataset_lookup(self) -> Dict[str, DatasetConfig]:
        return {dataset.id: dataset for dataset in self.datasets}


def _validate_weights(weights: Mapping[str, float], dataset_ids: Iterable[str]) -> None:
    missing = set(weights.keys()) - set(dataset_ids)
    if missing:
        raise ValueError(f"mixture references unknown datasets: {sorted(missing)}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"mixture weights must sum to 1 (got {total})")


def load_catalog(path: Path | str) -> Catalog:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    packing = PackingConfig(**data["packing"])
    datasets = [DatasetConfig(**entry) for entry in data.get("datasets", [])]
    mixtures = {
        block["name"]: [
            MixtureWeight(until_tokens=stage["until_tokens"], weights=stage["weights"])
            for stage in block.get("schedule", [])
        ]
        for block in data.get("mixtures", [])
    }

    catalog = Catalog(
        version=int(data["version"]),
        tokenizer=Path(data["tokenizer"]),
        seq_len=int(data["seq_len"]),
        packing=packing,
        datasets=datasets,
        mixtures=mixtures,
    )

    dataset_ids = [d.id for d in datasets]
    for name, schedule in mixtures.items():
        if not schedule:
            raise ValueError(f"mixture '{name}' has empty schedule")
        for stage in schedule:
            _validate_weights(stage.weights, dataset_ids)

    return catalog


__all__ = [
    "Catalog",
    "DatasetConfig",
    "MixtureWeight",
    "PackingConfig",
    "load_catalog",
]
