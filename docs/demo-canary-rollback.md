# Demo: canary failure and automatic rollback

This demo shows that a technically bad candidate cannot continue receiving
traffic. It requires Kubernetes, NGINX Ingress and the manifests described in
[Production Kubernetes](kubernetes-production.md).

## Scenario

Stable serves the known-good model. Canary uses the same API image with fault
injection enabled:

```text
FAULT_ERROR_RATE=1.0
```

All canary inference calls return HTTP 500. The release script observes the
candidate error ratio, sets canary traffic to zero and scales the candidate to
zero replicas.

## 1. Establish the baseline

For the lightweight manifest:

```bash
./scripts/canary.sh bootstrap REGISTRY/pw2048:KNOWN_GOOD_SHA greedy v1
```

For the production Kustomize overlay, set the resource names documented in
[Production Kubernetes](kubernetes-production.md).

## 2. Deploy a deliberately failing candidate

Set fault injection only on canary:

```bash
kubectl -n pw2048 set env deployment/pw2048-canary FAULT_ERROR_RATE=1
```

Start progressive delivery with short demonstration thresholds:

```bash
OBSERVE_SECONDS=20 MIN_REQUESTS=10 MAX_ERROR_RATE=0.01 \
  ./scripts/canary.sh deploy REGISTRY/pw2048:CANDIDATE_SHA heuristic v2
```

Generate traffic in a second terminal:

```bash
python scripts/chaos_smoke.py http://pw2048.local \
  --requests 100 --max-error-rate 1
```

## 3. Observe the failure path

Expected release output:

```text
canary weight -> 5%
gate: requests=... error_rate=1 avg_latency=...
canary weight -> 0%
rollback complete: all traffic remains on stable
```

Verify recovery:

```bash
./scripts/canary.sh status
python scripts/chaos_smoke.py http://pw2048.local \
  --requests 100 --expected-release stable
```

## Design point

Rollback changes routing before scaling the candidate down. Reversing that
order risks sending requests to terminating or unavailable candidate pods.
Technical rollback is automatic; model-quality promotion remains a separate,
auditable decision.
