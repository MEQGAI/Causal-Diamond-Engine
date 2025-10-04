#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error on line $LINENO" >&2' ERR

DRY_RUN=1
APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  DRY_RUN=0
  APPLY=1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

say() {
  echo "==> $*"
}

run_or_print() {
  if [[ $APPLY -eq 1 ]]; then
    eval "$1"
  else
    echo "[dry-run] $1"
  fi
}

say "Detecting host OS"
os="$(scripts/detect_os.sh)"
say "Planning setup for: $os (dry-run=$DRY_RUN)"

setup_script="scripts/setup_system_${os}.sh"
if [[ ! -x "$setup_script" ]]; then
  echo "Unsupported OS detected: $os" >&2
  exit 1
fi

if [[ $APPLY -eq 1 ]]; then
  "$setup_script" --apply
else
  "$setup_script"
fi

say "Preparing Python environment"
if [[ -f requirements.txt ]]; then
  if [[ $APPLY -eq 1 ]]; then
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install -r requirements-dev.txt
  else
    echo "[dry-run] python3 -m venv .venv"
    echo "[dry-run] pip install -r requirements.txt -r requirements-dev.txt"
  fi
elif [[ -f pyproject.toml ]]; then
  say "Found pyproject.toml; please manage dependencies via preferred backend."
fi

say "Installing Node workspace"
if [[ -f package.json ]]; then
  if [[ $APPLY -eq 1 ]]; then
    npm ci
  else
    echo "[dry-run] npm ci"
  fi
fi

say "Synchronising Rust toolchain"
if [[ $APPLY -eq 1 ]]; then
  rustup show >/dev/null 2>&1 || curl https://sh.rustup.rs -sSf | sh -s -- -y
  cargo fetch
else
  echo "[dry-run] rustup show"
  echo "[dry-run] cargo fetch"
fi

say "Syncing git submodules"
if [[ -f .gitmodules ]]; then
  if [[ $APPLY -eq 1 ]]; then
    git submodule update --init --recursive
  else
    echo "[dry-run] git submodule update --init --recursive"
  fi
else
  echo "No submodules declared."
fi

say "Installing pre-commit hooks"
if [[ $APPLY -eq 1 ]]; then
  .venv/bin/pre-commit install -t pre-commit -t commit-msg || pre-commit install -t pre-commit -t commit-msg || true
else
  echo "[dry-run] pre-commit install -t pre-commit -t commit-msg"
fi

say "Running smoke checks"
if [[ $APPLY -eq 1 ]]; then
  make lint || true
  make test || true
else
  echo "[dry-run] make lint"
  echo "[dry-run] make test"
fi

say "Bootstrap complete (apply=${APPLY})"
