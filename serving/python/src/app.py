from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from fm_core.checkpoint import load_checkpoint
from fm_data.packing import SentencePieceTokenizer, SpecialTokenIds

app = FastAPI(title="Foundation Serving", version="0.0.1")

@dataclass
class ModelBundle:
    model: object
    tokenizer: SentencePieceTokenizer
    specials: SpecialTokenIds


MODEL_CACHE: Dict[str, ModelBundle] = {}
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
        tokenizer_path = model_dir / "tokenizer.model"
        if not tokenizer_path.exists():
            raise HTTPException(status_code=500, detail=f"tokenizer.model missing in {model_dir}")
        model, _ = load_checkpoint(model_dir, map_location="cpu")
        tokenizer = SentencePieceTokenizer(str(tokenizer_path), allow_fallback=False)
        specials = SpecialTokenIds.from_processor(tokenizer.processor, model.cfg.special_tokens)
        MODEL_CACHE[key] = ModelBundle(model=model, tokenizer=tokenizer, specials=specials)
    return MODEL_CACHE[key]


@app.post("/v1/diamond/run", response_model=DiamondRunResponse)
def run_diamond(request: DiamondRunRequest) -> DiamondRunResponse:
    bundle = _load_model(Path(request.model_dir))
    model = bundle.model
    tokenizer = bundle.tokenizer
    specials = bundle.specials
    tokens = [specials.bos]
    tokens.extend(tokenizer.encode(request.prompt))
    input_ids = torch.tensor([tokens[-model.cfg.seq_len :]], dtype=torch.long)
    outputs = model(input_ids)
    logits = outputs["logits"]
    next_token = logits[:, -1, :].argmax(dim=-1)
    completion_tokens = tokens + [int(next_token.item())]
    try:
        completion = tokenizer.processor.decode(completion_tokens)  # type: ignore[attr-defined]
    except AttributeError:
        completion = request.prompt
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
