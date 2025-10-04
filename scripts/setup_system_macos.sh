#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

echo "[system] Target: macOS (apply=$APPLY)"
PACKAGES=(
  git
  git-lfs
  cmake
  python@3.11
  node@20
  rustup-init
  docker
)

if [[ $APPLY -eq 1 ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required. Install from https://brew.sh" >&2
    exit 1
  fi
  brew update
  brew install "${PACKAGES[@]}"
else
  echo "Dry run. Would install via brew: ${PACKAGES[*]}"
fi
