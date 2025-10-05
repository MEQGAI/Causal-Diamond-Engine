"""Training loops, curriculum builders, and CLI entrypoints."""

from .runtime import Trainer, train_from_config

__all__ = ["Trainer", "train_from_config"]
