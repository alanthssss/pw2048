# Incident and fault-injection runbook

## Service controls

The inference service pre-warms its selected model at startup, then applies
bounded concurrency, optional rate limiting, a short queue admission window,
and an inference deadline. `MAX_CONCURRENCY` defaults to 8,
`MAX_REQUESTS_PER_SECOND` to 0 (disabled), `QUEUE_TIMEOUT_MS` to 50, and
`INFERENCE_TIMEOUT_MS` to 1000. Saturated or limited requests receive HTTP 429;
over-deadline inference receives 504. Every request
accepts or creates `X-Request-ID`, which is returned to the caller and included
in structured logs.

Fault injection is disabled by default. A dedicated test deployment may set
`FAULT_DELAY_MS` or `FAULT_ERROR_RATE`; never enable these in the stable
overlay. Generate controlled traffic with:

```bash
python scripts/chaos_smoke.py https://pw2048.example --requests 100
```

## Triage order

1. Confirm user impact: error rate, P95/P99 latency, and affected release.
2. Compare stable and canary dashboards by `release` and `model` labels.
3. Use `X-Request-ID` to correlate gateway and application logs.
4. Check active/rejected requests, pod CPU/memory, restarts, and endpoints.
5. Stop canary traffic before investigating a candidate-specific regression.

## Recovery

```bash
./scripts/canary.sh rollback
kubectl -n pw2048 rollout status deployment/pw2048-stable
python scripts/chaos_smoke.py https://pw2048.example \
  --requests 100 --expected-release stable
```

Rollback first sets canary weight to zero and then scales candidate pods down.
If stable is unhealthy, use `kubectl rollout undo deployment/pw2048-stable` and
verify health, traffic, and model-quality metrics before closing the incident.

## Exercise scenarios

- candidate returns 100% failures (`FAULT_ERROR_RATE=1`)
- candidate adds latency (`FAULT_DELAY_MS=500`)
- candidate pods are deleted during traffic
- stable deployment is rolled back to its previous ReplicaSet
- monitoring is unavailable while direct health probes remain operational

Record detection time, rollback decision time, recovery time, user impact,
root cause, and follow-up actions. Target RTO is five minutes for a bad canary.
