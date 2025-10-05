from pathlib import Path

import yaml

from fm_train.trainer.scheduler import TrainingScheduler


def test_scheduler_writes_ledger(tmp_path: Path) -> None:
    stages = {
        "stages": [
            {"until_tokens": 10, "mixture": {"a": 0.7, "b": 0.3}},
            {"until_tokens": 20, "mixture": {"c": 1.0}},
        ]
    }
    config_path = tmp_path / "mix.yaml"
    config_path.write_text(yaml.safe_dump(stages), encoding="utf-8")

    scheduler = TrainingScheduler.from_file(config_path)
    assert scheduler.mixture() == {"a": 0.7, "b": 0.3}
    scheduler.advance(12)
    assert scheduler.mixture() == {"c": 1.0}
    state = scheduler.state_dict()
    assert state["stage_idx"] == 1
