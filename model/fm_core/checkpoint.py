from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .config import ModelConfig, load_config
from .transformer import FoundationModel


def load_checkpoint(
    model_dir: Path | str,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> Tuple[FoundationModel, Dict[str, Any]]:
    """Load a checkpoint directory containing config.json + model.safetensors/pt."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json at {config_path}")

    cfg = load_config(config_path)
    model = FoundationModel(cfg)

    weight_path = model_dir / "model.safetensors"
    if weight_path.exists():
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "safetensors package required to load .safetensors weights"
            ) from exc
        state_dict = load_file(weight_path, device=map_location or "cpu")
    else:
        weight_path = model_dir / "pytorch_model.bin"
        if not weight_path.exists():
            raise FileNotFoundError(
                "no weights file found (model.safetensors/pytorch_model.bin)"
            )
        state_dict = torch.load(weight_path, map_location=map_location)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )

    return model.to(map_location or "cpu"), cfg.to_dict()


def save_checkpoint(
    model: FoundationModel, cfg: ModelConfig, out_dir: Path | str
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out_dir / "config.json")
    state_dict = model.state_dict()
    torch.save(state_dict, out_dir / "pytorch_model.bin")


__all__ = ["load_checkpoint", "save_checkpoint"]
