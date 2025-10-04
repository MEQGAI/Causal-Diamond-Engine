#!/usr/bin/env bash
set -euo pipefail

uname_out="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${uname_out}" in
  linux*)
    if [ -f "/etc/os-release" ]; then
      . /etc/os-release
      case "${ID:-}" in
        ubuntu|debian)
          echo "linux-ubuntu"
          exit 0
          ;;
        fedora)
          echo "linux-fedora"
          exit 0
          ;;
        arch)
          echo "linux-arch"
          exit 0
          ;;
      esac
    fi
    echo "linux-unknown"
    ;;
  darwin*)
    echo "macos"
    ;;
  msys*|mingw*)
    echo "windows-msys"
    ;;
  *)
    echo "unknown"
    ;;
esac
