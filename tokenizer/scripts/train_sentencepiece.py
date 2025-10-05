#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus_manifest.yaml"
CTRL = ROOT / "symbols" / "control_symbols.txt"
UDS = ROOT / "symbols" / "user_defined_symbols.txt"
REQUIRED = ROOT / "seeds" / "required_chars.txt"

config = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
out_dir = Path(config["output_dir"]).resolve()

control_symbols = ",".join(line.strip() for line in CTRL.read_text(encoding="utf-8").splitlines() if line.strip())
user_symbols = ",".join(line.strip() for line in UDS.read_text(encoding="utf-8").splitlines() if line.strip())
required_chars = REQUIRED.read_text(encoding="utf-8").replace("\n", "")

spm.SentencePieceTrainer.train(
    input=str(out_dir / "train.txt.gz"),
    model_prefix=str(out_dir / "tokenizer"),
    model_type="bpe",
    vocab_size=131072,
    character_coverage=1.0,
    byte_fallback=True,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    add_dummy_prefix=False,
    split_by_unicode_script=False,
    unk_id=0,
    bos_id=-1,
    eos_id=-1,
    pad_id=-1,
    control_symbols=control_symbols,
    user_defined_symbols=user_symbols,
    required_chars=required_chars,
    input_sentence_size=2000000,
    shuffle_input_sentence=True,
)

print("[OK] Trained SentencePiece model at", out_dir / "tokenizer.model")
