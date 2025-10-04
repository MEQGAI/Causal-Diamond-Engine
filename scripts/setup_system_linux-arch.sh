#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

echo "[system] Target: linux-arch (apply=$APPLY)"
PACKAGES=(
  base-devel
  git
  git-lfs
  cmake
  python
  python-pip
  nodejs
  npm
  docker
)

if [[ $APPLY -eq 1 ]]; then
  sudo pacman -Syu --noconfirm
  sudo pacman -S --noconfirm "${PACKAGES[@]}"
  sudo systemctl enable docker || true
else
  echo "Dry run. Would install: ${PACKAGES[*]}"
fi
