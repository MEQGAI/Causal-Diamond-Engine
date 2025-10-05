"""Dataset source loaders used by the training pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import fsspec

try:  # Optional dependency: Hugging Face datasets
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - datasets not available in minimal envs
    load_dataset = None  # type: ignore

from fm_data.catalog import DatasetConfig
from fm_data.webdataset_stream import WebDatasetStream

from fm_train.datasets.filters import build_filter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSample:
    text: str
    meta: Dict[str, Any]


def load_source(
    entry: Mapping[str, Any],
    split: str,
    *,
    streaming: bool = True,
) -> Iterator[SourceSample]:
    """Yield samples from a catalog entry.

    Parameters
    ----------
    entry:
        Dictionary describing the dataset source (typically loaded from YAML).
    split:
        Logical split name (e.g., ``"train"`` or ``"validation"``).
    streaming:
        If ``True`` the loader will prefer streaming modes where possible.
    """

    dataset_type = (entry.get("type") or entry.get("format") or "hf").lower()
    filters_cfg = entry.get("filters", [])
    allow_filter = build_filter(filters_cfg)

    if dataset_type in {"hf", "huggingface"}:
        if load_dataset is None:
            raise RuntimeError(
                "datasets is not installed; required to load Hugging Face sources"
            )
        hf_id = entry.get("hf_id") or entry.get("name")
        if not hf_id:
            raise ValueError("hf sources require an 'hf_id' key")
        config = entry.get("config")
        text_field = entry.get("text_key", "text")
        dataset = load_dataset(
            hf_id,
            config,
            split=split,
            streaming=streaming,
        )
        for sample in dataset:
            text = sample.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            if not allow_filter(text, sample):
                continue
            yield SourceSample(text=text, meta=dict(sample))
        return

    if dataset_type in {"webdataset", "wds"}:
        cfg = _dataset_config_from_entry(entry)
        stream = WebDatasetStream(cfg, shuffle=not streaming)
        for sample in stream:
            text = _extract_text(sample.data)
            if not text:
                continue
            meta = dict(sample.meta)
            if not allow_filter(text, meta):
                continue
            yield SourceSample(text=text, meta=meta)
        return

    if dataset_type in {"http", "text", "jsonl"}:
        path = entry.get("path") or entry.get("uri")
        if not path:
            raise ValueError("http/text sources require a 'path' or 'uri'")
        opener = fsspec.open(path, mode="rt")
        with opener as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload: Dict[str, Any]
                if dataset_type == "jsonl":
                    try:
                        payload = json.loads(line)
                        text = payload.get("text") or payload.get("content")
                    except json.JSONDecodeError:
                        logger.debug("skipping invalid json line")
                        continue
                else:
                    payload = {}
                    text = line
                if not isinstance(text, str) or not text.strip():
                    continue
                if not allow_filter(text, payload):
                    continue
                yield SourceSample(text=text, meta=payload)
        return

    raise ValueError(f"unsupported dataset type '{dataset_type}'")


def _dataset_config_from_entry(entry: Mapping[str, Any]) -> DatasetConfig:
    cfg = DatasetConfig(
        id=entry.get("id", "unnamed"),
        kind=entry.get("kind", "text"),
        shards=entry["shards"],
        format="webdataset",
        weight=float(entry.get("weight", 1.0)),
        license=entry.get("license"),
        filters=list(entry.get("filters", [])),
    )
    return cfg


def _extract_text(blobs: Mapping[str, Any]) -> Optional[str]:
    preferred = (
        ".text",
        ".txt",
        ".code",
        "text",
        "txt",
        "code",
        ".json",
    )
    for key in preferred:
        if key in blobs:
            value = blobs[key]
            if isinstance(value, bytes):
                try:
                    text = value.decode("utf-8", errors="ignore")
                except Exception:  # pragma: no cover - defensive
                    continue
            else:
                text = str(value)
            if text.strip():
                return text
    return None


__all__ = ["SourceSample", "load_source"]
