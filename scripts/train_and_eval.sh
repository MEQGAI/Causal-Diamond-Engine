#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") --config CONFIG --steps STEPS [--thresholds FILE] [--resume {auto|never}]

Runs training followed by evaluation using fm_train.trainer.run.
USAGE
}

CONFIG=""
STEPS=""
THRESHOLDS=""
RESUME="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --thresholds)
      THRESHOLDS="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" || -z "$STEPS" ]]; then
  echo "--config and --steps are required" >&2
  usage
  exit 1
fi

run_args=("--config" "$CONFIG" "--steps" "$STEPS" "--resume" "$RESUME" "--evaluate")
if [[ -n "$THRESHOLDS" ]]; then
  run_args+=("--thresholds" "$THRESHOLDS")
fi

python -m fm_train.trainer.run "${run_args[@]}"
