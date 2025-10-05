# Universal Composite Tokenizer Package

This directory packages everything required to reproduce the `uct-v1.0.0` tokenizer.

## Layout

- `corpus_manifest.yaml` — declarative manifest describing the weighted corpus mix.
- `symbols/` — enumerated control and user-defined symbol lists (ordering defines ID bands).
- `seeds/` — short seed sentences and required characters to guarantee coverage.
- `scripts/` — helpers for corpus construction, normalization, SentencePiece training, and post-processing.

## One-shot workflow

```bash
bash tokenizer/scripts/make_corpus.sh
bash tokenizer/scripts/train_sentencepiece.sh
```

Outputs land in `data/sp_text/`:

- `train.txt.gz` / `dev.txt.gz` — normalized corpora used for tokenizer training.
- `tokenizer.model` / `tokenizer.vocab` — SentencePiece assets aligned with reserved ID bands.

## Notes

- All scripts assume GNU userland (bash, find, coreutils) plus `yq`, `jq`, `python3`, and `sentencepiece` in `PATH`.
- The training command fixes vocabulary size at **131072** with byte fallback enabled; control IDs follow the order declared in `symbols/control_symbols.txt`.
- `lock_and_prune.py` enforces contiguous control/user-defined bands and removes pathological artifacts via regex pruning.
- Indentation handling converts leading whitespace into explicit `<|indent_n|>` and `<|tab|>` markers via `preprocess_indents.py`.

Adjust the manifest if your corpora reside in different locations; the rest of the pipeline remains unchanged.
