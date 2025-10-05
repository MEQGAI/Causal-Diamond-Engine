from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from fm_core.checkpoint import load_checkpoint

app = FastAPI(title="Foundation Serving", version="0.0.1")

MODEL_CACHE: Dict[str, object] = {}
LEDGER: Dict[str, Dict[str, object]] = {}


class DiamondRunRequest(BaseModel):
    prompt: str
    budget: float = Field(2.0, ge=0.0, le=10.0)
    tools: List[str] = Field(default_factory=list)
    model_dir: str = Field("checkpoints/fm-b-220m")


class DiamondRunResponse(BaseModel):
    text: str
    ledger: Dict[str, object]


def _load_model(model_dir: Path) -> object:
    key = str(model_dir)
    if key not in MODEL_CACHE:
        model, _ = load_checkpoint(model_dir, map_location="cpu")
        MODEL_CACHE[key] = model
    return MODEL_CACHE[key]


@app.post("/v1/diamond/run", response_model=DiamondRunResponse)
def run_diamond(request: DiamondRunRequest) -> DiamondRunResponse:
    model = _load_model(Path(request.model_dir))
    input_ids = torch.tensor([[ord(c) % model.cfg.vocab_size for c in request.prompt[: model.cfg.seq_len]]], dtype=torch.long)
    outputs = model(input_ids)
    logits = outputs["logits"]
    # Greedy decode a short completion for demonstration.
    next_token = logits[:, -1, :].argmax(dim=-1)
    completion = request.prompt + " " + str(int(next_token.item()))
    ledger_entry = {
        "prompt": request.prompt,
        "budget": request.budget,
        "loss_mod": float(outputs["planner"].log_probs.exp().mean().item()),
        "accepted": True,
    }
    run_id = f"run-{len(LEDGER) + 1}"
    LEDGER[run_id] = ledger_entry
    return DiamondRunResponse(text=completion, ledger={"id": run_id, **ledger_entry})


@app.get("/v1/ledger/trace/{run_id}")
def ledger_trace(run_id: str) -> Dict[str, object]:
    if run_id not in LEDGER:
        raise HTTPException(status_code=404, detail="run not found")
    return LEDGER[run_id]


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}
