# Load testing and capacity planning

The capacity workflow measures one inference replica at a known request rate,
checks the service SLO, discounts the observed throughput, and adds failure
reserve. It deliberately refuses to recommend capacity from a test point that
already violates latency or error-rate objectives.

## Run locally

Start the API, then run k6:

```bash
python -m src.api_server
TARGET_URL=http://localhost:8080 TARGET_RPS=50 \
  k6 run loadtest/inference.js
```

`handleSummary` writes `loadtest-summary.json`. Convert it into both a
machine-readable plan and a reviewable report:

```bash
python scripts/capacity_plan.py loadtest-summary.json \
  --peak-rps 250 --safety-factor 0.70 --n-plus 1 \
  --json-output capacity-plan.json \
  --markdown-output capacity-plan.md
```

For example, an SLO-compliant 100 req/s test yields 70 safe req/s after a 70%
safety factor. A 250 req/s peak needs four workload replicas; N+1 reserve makes
the recommendation five replicas.

## Test method

1. Run against a single replica with production-equivalent CPU/memory limits.
2. Warm the model before measurement.
3. Increase `TARGET_RPS` until P95, P99, errors, or saturation show a knee.
4. Use the highest point that still meets the SLO as the input summary.
5. Repeat after model, runtime, instance type, or resource-limit changes.

The default SLO is P95 <= 250 ms and error rate <= 1%. The planning defaults are
a 70% safety factor and N+1 reserve. These values are explicit CLI inputs so the
generated JSON is auditable. For streaming ASR/TTS, extend this method with
active streams, session duration, first-token/audio latency, and RTF; QPS alone
is not sufficient.

## GitLab CI controls

The pipeline runs unit tests and a deterministic model quality gate before
building. Registry scanning uses GitLab container scanning. Test deployment and
load testing require `TEST_BASE_URL` and a configured runner/kube context.
Canary deployment requires `CANARY_IMAGE`; production promotion is manual and
is only available on the default branch. Protected environment variables should
hold kubeconfig/token material—never commit credentials.
