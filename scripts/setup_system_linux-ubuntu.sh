#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

echo "[system] Target: linux-ubuntu (apply=$APPLY)"
PACKAGES=(
  build-essential
  pkg-config
  curl
  git
  git-lfs
  cmake
  python3.11
  python3.11-venv
  python3-pip
  nodejs
  npm
  docker.io
)

if [[ $APPLY -eq 1 ]]; then
  sudo apt-get update
  sudo apt-get install -y "${PACKAGES[@]}"
  sudo systemctl enable docker || true
else
  echo "Dry run. Would install: ${PACKAGES[*]}"
fi
