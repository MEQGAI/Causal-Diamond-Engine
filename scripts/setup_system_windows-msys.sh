#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

echo "[system] Target: Windows (MSYS2) (apply=$APPLY)"
PACKAGES=(
  mingw-w64-x86_64-toolchain
  git
  git-lfs
  cmake
  python
  python-pip
  nodejs
  npm
)

if [[ $APPLY -eq 1 ]]; then
  pacman -Syu --noconfirm
  pacman -S --noconfirm "${PACKAGES[@]}"
else
  echo "Dry run. Would install via pacman: ${PACKAGES[*]}"
fi
