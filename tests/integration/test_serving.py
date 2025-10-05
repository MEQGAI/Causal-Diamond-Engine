from __future__ import annotations

import sys
from types import SimpleNamespace

from pathlib import Path

import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

serving_module = importlib.import_module("serving.python.src.app")
ModelBundle = serving_module.ModelBundle
fastapi_app = serving_module.app


class _DummyModel:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(seq_len=16)

    def __call__(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        vocab = 8
        logits = torch.zeros(batch, seq_len, vocab)
        planner_logits = torch.zeros(batch, seq_len, 4)
        planner_log_probs = torch.log_softmax(planner_logits, dim=-1)
        planner = SimpleNamespace(log_probs=planner_log_probs)
        return {"logits": logits, "planner": planner}


class _DummyTokenizer:
    def __init__(self) -> None:
        class _Processor:
            def decode(self, _tokens):
                return "decoded"

        self.processor = _Processor()

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]


class _DummySpecials(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            bos=0,
            eos=1,
            pad=2,
            unk=3,
            diamond_start=4,
            diamond_end=5,
            view_start=6,
            view_end=7,
            plan_start=8,
            plan_end=9,
            tool=10,
        )


def _fake_loader(_path):
    return ModelBundle(
        model=_DummyModel(),
        tokenizer=_DummyTokenizer(),
        specials=_DummySpecials(),
    )


def _make_client(monkeypatch):
    serving_module.MODEL_CACHE.clear()
    serving_module.LEDGER.clear()
    monkeypatch.setattr(serving_module, "_load_model", _fake_loader)
    return TestClient(fastapi_app)


def test_health_endpoint(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_run_diamond(monkeypatch):
    client = _make_client(monkeypatch)
    payload = {
        "prompt": "hello",
        "budget": 1.0,
        "tools": [],
        "model_dir": "ignored",
    }
    response = client.post("/v1/diamond/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    run_id = data["ledger"]["id"]

    trace = client.get(f"/v1/ledger/trace/{run_id}")
    assert trace.status_code == 200
    assert trace.json()["prompt"] == "hello"
