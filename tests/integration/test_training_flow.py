import json
from pathlib import Path

import pytest
import torch

from fm_train.config import load_train_config
from fm_train.runtime import Trainer


def test_toy_training_emits_ledger(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg_path = root / "configs/train/toy_local.yaml"
    cfg = load_train_config(cfg_path)

    # Force absolute paths so training can resolve assets after we move the ledger.
    cfg.model_cfg = (root / cfg.model_cfg).resolve()
    cfg.data_catalog = (root / cfg.data_catalog).resolve()

    torch.set_num_threads(1)
    trainer = Trainer(cfg, device=torch.device("cpu"))
    trainer.ledger_path = tmp_path / "ledger.jsonl"

    trainer.fit(steps=2)

    ledger_file = trainer.ledger_path
    assert ledger_file.exists(), "ledger not created"

    lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "ledger file empty"
    payload = json.loads(lines[-1])
    assert payload["step"] == 2
    assert "loss_ent" in payload


def test_py_engine_binding_available():
    module = pytest.importorskip(
        "engine_py",
        reason="PyO3 bindings not built; run maturin develop to enable",
    )
    engine = module.PyCausalDiamondEngine()
    result = engine.step('{"step": 1, "delta": 0.1}', 1.0)
    assert result == "ack"
