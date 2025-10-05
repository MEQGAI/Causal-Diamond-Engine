from __future__ import annotations

import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional

import fsspec

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
    """Minimal WebDataset-like stream for tar shards with fsspec support."""

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
        if not self.shards:
            raise ValueError(f"dataset {cfg.id} has no shard paths")

    def __iter__(self) -> Iterator[Sample]:
        shards = self.shards.copy()
        if self.shuffle:
            self.rng.shuffle(shards)
        for shard in shards:
            with fsspec.open(shard, "rb") as stream:
                try:
                    tar = tarfile.open(fileobj=stream, mode="r|*")
                except tarfile.ReadError as err:
                    raise RuntimeError(f"failed to open shard {shard}: {err}") from err

                grouped: Dict[str, Dict[str, bytes]] = {}
                for member in tar:
                    if not member.isfile():
                        continue
                    payload = tar.extractfile(member)
                    if payload is None:
                        continue
                    base = Path(member.name).stem
                    ext = Path(member.name).suffix
                    grouped.setdefault(base, {})[ext] = payload.read()
                keys = list(grouped.keys())
                if self.shuffle:
                    self.rng.shuffle(keys)
                for key in keys:
                    blobs = grouped[key]
                    meta: Mapping[str, object] = {}
                    if ".json" in blobs:
                        import json

                        meta = json.loads(blobs[".json"].decode("utf-8"))
                    yield Sample(data=blobs, meta=meta)


__all__ = ["WebDatasetStream", "Sample"]
