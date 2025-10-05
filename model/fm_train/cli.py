from __future__ import annotations

from pathlib import Path

import typer

from .runtime import train_from_config

app = typer.Typer(add_completion=False)


@app.command()
def fit(config: Path, steps: int = typer.Option(100, help="Number of optimizer steps")) -> None:
    """Run pre-training using the provided YAML configuration."""

    train_from_config(config, steps)


def entrypoint() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    entrypoint()
