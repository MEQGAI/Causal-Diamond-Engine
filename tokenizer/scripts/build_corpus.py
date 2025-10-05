#!/usr/bin/env python3
"""Construct tokenizer corpora from Hugging Face datasets.

Reads `corpus_manifest.yaml` and produces gzipped text files in the target directory.
Supports `hf` (Hugging Face datasets) and `glob` (local filesystem) sources.
"""
from __future__ import annotations

import gzip
import itertools
import math
import unicodedata
from pathlib import Path
from typing import Iterator, List, Mapping, Sequence

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "corpus_manifest.yaml"
SEED_PATH = ROOT / "seeds" / "seed_sentences.txt"
REQUIRED_CHARS_PATH = ROOT / "seeds" / "required_chars.txt"


def load_manifest() -> Mapping[str, object]:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def sanitize_text(text: str) -> str:
    """Remove control characters except TAB and LF."""
    return "".join(
        ch
        for ch in text
        if (ord(ch) >= 32 and not unicodedata.category(ch).startswith("C"))
        or ch in {"\t", "\n"}
    )


def normalize_text(text: str) -> str:
    text = sanitize_text(text)

    def transform_line(line: str) -> str:
        prefix_len = 0
        tokens: List[str] = []
        while prefix_len < len(line) and line[prefix_len] in {" ", "\t"}:
            if line[prefix_len] == "\t":
                tokens.append("<|tab|>")
                prefix_len += 1
            else:
                start = prefix_len
                while prefix_len < len(line) and line[prefix_len] == " ":
                    prefix_len += 1
                spaces = prefix_len - start
                tokens.extend(["<|indent_4|>"] * (spaces // 4))
                remainder = spaces % 4
                if remainder:
                    tokens.append(f"<|indent_{remainder}|>")
        return "".join(tokens) + line[prefix_len:]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in text.split("\n") if line]
    return "\n".join(transform_line(line) for line in lines if line)


def resolve_nested(sample: Mapping[str, object], key: str) -> str:
    current = sample
    for part in key.split('.'):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return ""
        if current is None:
            return ""
    if isinstance(current, str):
        return current
    if isinstance(current, Mapping):
        return "\n".join(str(v) for v in current.values())
    return str(current)


def iter_hf_source(spec: Mapping[str, object], limit: int) -> Iterator[str]:
    dataset_id = spec["dataset"]
    split = spec.get("split", "train")
    config = spec.get("config")
    text_key = spec.get("text_key")
    secondary_key = spec.get("secondary_text_key")
    template = spec.get("template", "{primary}\n{secondary}") if secondary_key else "{primary}"

    ds = load_dataset(dataset_id, config, split=split, streaming=True)
    for example in itertools.islice(ds, limit):
        primary = resolve_nested(example, text_key) if text_key else ""
        if not primary:
            continue
        if secondary_key:
            secondary = resolve_nested(example, secondary_key)
            text = template.format(primary=primary, secondary=secondary)
        else:
            text = primary
        text = text.strip()
        if text:
            yield text


def iter_glob_source(spec: Mapping[str, object], limit: int) -> Iterator[str]:
    import gzip as gz
    import bz2
    import lzma

    pattern = spec["pattern"]
    count = 0
    for path in sorted(Path().glob(pattern)):
        if count >= limit:
            break
        opener = open
        if path.suffix == ".gz":
            opener = gz.open
        elif path.suffix == ".bz2":
            opener = bz2.open
        elif path.suffix in {".xz", ".lzma"}:
            opener = lzma.open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if count >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                yield line
                count += 1


def build_split(name: str, spec: Mapping[str, object], out_dir: Path) -> None:
    target_lines = int(spec.get("target_lines", 0))
    if target_lines <= 0:
        raise ValueError(f"split '{name}' must define a positive target_lines")

    sources: Sequence[Mapping[str, object]] = spec.get("sources", [])
    if not sources:
        raise ValueError(f"split '{name}' has no sources configured")

    total_weight = sum(float(src.get("weight", 1.0)) for src in sources)
    if total_weight <= 0:
        raise ValueError(f"split '{name}' has non-positive total weight")

    dest = out_dir / f"{name}.txt.gz"
    seen = set()
    emitted = 0

    with gzip.open(dest, "wt", encoding="utf-8") as writer:
        for src in sources:
            weight = float(src.get("weight", 1.0))
            limit = max(1, math.floor(target_lines * weight / total_weight))
            if emitted + limit > target_lines:
                limit = max(1, target_lines - emitted)

            src_type = src.get("type", "hf")
            if src_type == "hf":
                iterator = iter_hf_source(src, limit * 5)
            elif src_type == "glob":
                iterator = iter_glob_source(src, limit * 5)
            else:
                raise ValueError(f"unsupported source type '{src_type}'")

            for text in iterator:
                normalized = normalize_text(text)
                if not normalized or normalized in seen:
                    continue
                writer.write(normalized)
                writer.write("\n")
                seen.add(normalized)
                emitted += 1
                if emitted % 1000 == 0:
                    print(f"[{name}] emitted {emitted} lines", flush=True)
                if emitted >= target_lines:
                    break
            if emitted >= target_lines:
                break

        if emitted < target_lines:
            print(f"[{name}] emitted {emitted} < target {target_lines}; continuing with seeds", flush=True)

        seeds = [line.strip() for line in SEED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        for seed in seeds:
            writer.write(seed)
            writer.write("\n")

    print(f"[OK] wrote {dest} with {emitted} lines (+seeds)")


def main() -> None:
    manifest = load_manifest()
    out_dir = Path(manifest["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, spec in manifest.get("splits", {}).items():
        build_split(split_name, spec, out_dir)

    print("Seeds appended. Required chars file at", REQUIRED_CHARS_PATH)


if __name__ == "__main__":
    main()
