from __future__ import annotations

import io
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import torch

from .catalog import DatasetConfig


@dataclass
class Sample:
    data: Mapping[str, bytes]
    meta: Mapping[str, object]


def _expand_shards(pattern: str) -> List[str]:
    if "{" not in pattern:
        return [pattern]
    prefix, rest = pattern.split("{", 1)
    span, suffix = rest.split("}", 1)
    start, end = span.split("..")
    start_int = int(start)
    end_int = int(end)
    width = len(start)
    return [f"{prefix}{idx:0{width}d}{suffix}" for idx in range(start_int, end_int + 1)]


class WebDatasetStream:
    """Minimal WebDataset-like stream for tar shards."""

    def __init__(
        self,
        cfg: DatasetConfig,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.shards = _expand_shards(cfg.shards)

    def __iter__(self) -> Iterator[Sample]:
        shards = self.shards.copy()
        if self.shuffle:
            self.rng.shuffle(shards)
        for shard in shards:
            path = Path(shard)
            if not path.exists():
                continue
            with tarfile.open(path, "r:*") as tar:
                members = [m for m in tar.getmembers() if m.isfile()]
                if self.shuffle:
                    self.rng.shuffle(members)
                grouped: Dict[str, Dict[str, bytes]] = {}
                for member in members:
                    base, ext = Path(member.name).stem, Path(member.name).suffix
                    payload = tar.extractfile(member)
                    if payload is None:
                        continue
                    grouped.setdefault(base, {})[ext] = payload.read()
                for blobs in grouped.values():
                    meta = {}
                    if ".json" in blobs:
                        import json

                        meta = json.loads(blobs[".json"].decode("utf-8"))
                    yield Sample(data=blobs, meta=meta)


__all__ = ["WebDatasetStream", "Sample"]
