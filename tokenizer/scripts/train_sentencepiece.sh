#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" "$ROOT/scripts/train_sentencepiece.py"
"$PYTHON_BIN" "$ROOT/scripts/lock_and_prune.py" \
  --in_model "$(yq -r '.output_dir' "$ROOT/corpus_manifest.yaml")/tokenizer.model" \
  --out_model "$(yq -r '.output_dir' "$ROOT/corpus_manifest.yaml")/tokenizer.model" \
  --control_list "$ROOT/symbols/control_symbols.txt" \
  --user_list "$ROOT/symbols/user_defined_symbols.txt" \
  --enforce_contiguous_bands \
  --prune_regex '^(\u2581){3,}$|^[\pZ\pC]+$' \
  --verbose
