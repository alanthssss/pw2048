# Demo: model-quality regression is blocked

This demo proves that a healthy container is not enough to release a model.
The candidate below is fast and deployable, but its decision quality regresses.

## Run the gate

```bash
python scripts/model_quality_gate.py \
  --stable docs/examples/stable-metrics.json \
  --candidate docs/examples/candidate-regression-metrics.json \
  --policy configs/quality-gate.json \
  --output /tmp/rejected-quality-gate.json

echo $?
```

Expected exit code: `2`.

Expected failed checks include:

- mean score regression exceeds 3%;
- P90 score regression exceeds 5%;
- win-rate absolute regression exceeds 0.01;
- invalid-action rate increased when no regression is allowed.

The complete expected result is stored in
[`examples/quality-gate-rejected.json`](examples/quality-gate-rejected.json).

## Why this matters

Infrastructure metrics answer “is the service operating?” Model metrics answer
“is the service still useful?” Combining them into one gate hides which failure
occurred. pw2048 keeps them separate and requires both to pass.
