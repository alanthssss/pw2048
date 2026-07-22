# Demo: model registration and release preparation

This demonstration takes a candidate artifact from a local checkpoint to an
immutable, traceable registry entry. It takes about three minutes and does not
require Kubernetes.

## 1. Create a demonstration artifact

```bash
mkdir -p /tmp/pw2048-demo
printf 'candidate-model-weights-v2\n' > /tmp/pw2048-demo/best_checkpoint.npz
```

## 2. Register the candidate

```bash
python scripts/model_register.py /tmp/pw2048-demo/best_checkpoint.npz \
  --name dqn-v3 \
  --version interview-v2 \
  --seed 42 \
  --config configs/training/default.json \
  --metrics docs/examples/candidate-good-metrics.json \
  --experiment-id interview-demo-001 \
  --registry /tmp/pw2048-demo/registry
```

Expected result:

```text
/tmp/pw2048-demo/registry/dqn-v3/interview-v2/
├── best_checkpoint.npz
└── manifest.json
```

The manifest links the artifact to its Git commit, experiment ID, random seed,
training configuration, dependency versions, hardware metadata and SHA-256.

## 3. Verify immutability

Run the same registration command again. It must fail because model name and
version identify an immutable release candidate. Then change the registered
artifact and call `FileSystemModelRegistry.resolve()`; checksum verification
must reject the corrupted artifact.

## 4. Run the offline quality gate

```bash
python scripts/model_quality_gate.py \
  --stable docs/examples/stable-metrics.json \
  --candidate docs/examples/candidate-good-metrics.json \
  --policy configs/quality-gate.json \
  --output /tmp/pw2048-demo/quality-gate-result.json
```

Exit code `0` is permission to continue to image construction—not permission to
skip test deployment, load testing or canary release.

## What to explain during the demo

- A checkpoint alone is not a reproducible model release.
- Model identity and service image identity are related but separate.
- Offline model quality and online service health are independent gates.
- An immutable artifact makes rollback deterministic and auditable.
