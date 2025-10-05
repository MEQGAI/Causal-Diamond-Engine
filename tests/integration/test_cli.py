import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_cli_banner_runs():
    result = subprocess.run(
        ["cargo", "run", "-p", "ledger-cli", "--quiet", "--", "banner"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Causal-Diamond Runtime" in result.stdout


@pytest.mark.slow
def test_cli_step_roundtrip():
    result = subprocess.run(
        [
            "cargo",
            "run",
            "-p",
            "ledger-cli",
            "--quiet",
            "--",
            "step",
            "hello world",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "diamonds=" in result.stdout or result.stdout.strip() == "ack"
