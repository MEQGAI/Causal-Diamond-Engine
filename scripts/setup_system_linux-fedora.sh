#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
fi

echo "[system] Target: linux-fedora (apply=$APPLY)"
PACKAGES=(
  gcc
  gcc-c++
  make
  pkgconf
  git
  git-lfs
  cmake
  python3.11
  python3.11-devel
  python3-pip
  nodejs
  npm
  docker
)

if [[ $APPLY -eq 1 ]]; then
  sudo dnf install -y "${PACKAGES[@]}"
  sudo systemctl enable --now docker || true
else
  echo "Dry run. Would install: ${PACKAGES[*]}"
fi
