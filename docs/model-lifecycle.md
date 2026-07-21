# Model lifecycle and quality gate

This project treats a checkpoint as an immutable, traceable release artifact.
The local implementation has no infrastructure dependencies, while the
`ModelRegistry` protocol provides a stable boundary for later S3 or MLflow
adapters.

## Reproducible model registration

Start training from a version-controlled JSON configuration, for example
`configs/training/default.json`. After evaluation, register the checkpoint:

```bash
python scripts/model_register.py checkpoints/best.npz \
  --name dqn-v3 --version 2026.07.21-1 --seed 42 \
  --config configs/training/default.json \
  --metrics candidate-metrics.json \
  --experiment-id exp-20260721-001 \
  --registry model-registry
```

The registry creates
`model-registry/<name>/<version>/{artifact,manifest.json}`. A version is
immutable: registering it twice fails. The manifest records:

- Git commit and experiment ID;
- random seed and complete training configuration;
- installed dependency versions and Python version;
- platform, CPU count and detected CPU/CUDA/MPS accelerator;
- artifact filename, size and SHA-256 checksum;
- evaluation metrics and an RFC 3339 creation timestamp.

Loading through `FileSystemModelRegistry.resolve()` verifies the checksum and
fails if the artifact has been changed. In production, implement the same
`ModelRegistry` protocol for S3/MinIO or MLflow. The adapter should retain the
manifest unchanged, use server-side encryption and an immutable bucket policy,
and verify the checksum after downloading to a local cache.

## Quality gate

Stable and candidate evaluation files may either be a metric object or contain
one under `metrics`:

```json
{
  "model_version": "v2",
  "metrics": {
    "mean_score": 1050,
    "p90_score": 1800,
    "win_rate": 0.72,
    "invalid_action_rate": 0.0,
    "evaluation_games": 500
  }
}
```

Run the default policy with:

```bash
python scripts/model_quality_gate.py \
  --stable stable-metrics.json \
  --candidate candidate-metrics.json \
  --policy configs/quality-gate.json \
  --output quality-gate-result.json
```

Exit code `0` means every check passed; exit code `2` blocks the release. A
missing/non-numeric configured metric fails closed. Rules support metrics that
should be maximized or minimized, relative/absolute regression limits, hard
minimums/maximums, and a minimum evaluation sample size.

The gate is deliberately separate from online latency/error-rate gates. A
healthy service can still host a worse model, so both quality and operational
gates must pass before promotion.
