from pathlib import Path

from python.trainer.scheduler import TrainingScheduler


def test_scheduler_writes_ledger(tmp_path: Path) -> None:
    scheduler = TrainingScheduler(task="tool_reasoning", budget=2.0, epochs=2)
    scheduler.run(tmp_path)
    ledger = (tmp_path / "ledger.json").read_text(encoding="utf-8")
    assert "epoch" in ledger
    assert "kl" in ledger
    assert "stability" in ledger
