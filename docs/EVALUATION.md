# Evaluation Playbook

## Overview
The evaluation stack is coordinated via the `fm_eval` package. It provides config-driven gates that
must pass before merging or promoting checkpoints.

## Prereg Gates
- `configs/eval/kill_numbers.yaml` tracks the kill-switch thresholds for accuracy, latency, memory,
  and safety metrics. Update these values in consultation with the safety board.

## Running Evaluations
```bash
python -m fm_eval.runner --config configs/eval/kill_numbers.yaml
```

The runner prints gate metrics as JSON and returns a non-zero exit code for failures so that CI can
block the change.
