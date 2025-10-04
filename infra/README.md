# Infrastructure Layout

This directory hosts deployment automation assets. Populate the subdirectories as the project matures:

- `terraform/` — IaC for cloud resources
- `k8s/` — Kubernetes manifests / Helm charts
- `ansible/` — configuration management for bare-metal clusters

Each environment should expose idempotent `make` targets and integrate with the CI pipeline defined in `ops/ci/`.
