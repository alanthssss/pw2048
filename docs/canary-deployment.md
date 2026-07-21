# Canary deployment

This deployment runs two independently versioned inference services behind
NGINX Ingress. Stable receives normal traffic; canary receives a controlled
percentage. A cookie named `pw2048_canary` can pin test users to canary.

## Prerequisites

- Kubernetes with NGINX Ingress Controller
- `kubectl`
- a registry accessible by the cluster
- Prometheus may scrape the pod annotations, but is not required by the script

## API

Run locally:

```bash
MODEL_NAME=greedy MODEL_VERSION=v1 RELEASE_TRACK=local python -m src.api_server
curl -s localhost:8080/version
curl -s localhost:8080/v1/move -H 'content-type: application/json' \
  -d '{"board":[[2,2,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]}'
```

Endpoints are `/healthz`, `/readyz`, `/version`, `/metrics`, and `/v1/move`.
Every response includes `X-PW2048-Release` so traffic distribution is visible.

## Release

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t REGISTRY/pw2048:SHA .
docker push REGISTRY/pw2048:SHA

# One-time installation: establish the known-good stable baseline.
./scripts/canary.sh bootstrap REGISTRY/pw2048:KNOWN_GOOD_SHA greedy v1

# Deploy candidate and automatically advance through 5%, 20%, and 50%.
# Each stage checks request count, error rate, and average inference latency.
OBSERVE_SECONDS=300 MIN_REQUESTS=100 ./scripts/canary.sh deploy \
  REGISTRY/pw2048:SHA heuristic v2

# Promote only after model/algorithm business metrics are approved.
./scripts/canary.sh promote

# Immediately remove canary traffic and scale it to zero.
./scripts/canary.sh rollback
```

Defaults: error rate <= 1%, average inference latency <= 250 ms. Override with
`MAX_ERROR_RATE`, `MAX_AVG_LATENCY`, `MIN_REQUESTS`, and `OBSERVE_SECONDS`.

## Why technical and model gates are separate

HTTP success and low latency do not prove that a new model is better. The
script deliberately stops at 50%; promotion is a separate command after an
offline benchmark or online business-quality gate approves the candidate.
For ASR this gate would compare CER/WER and RTF; for TTS it would compare first
audio latency, RTF, and quality scores. Here it compares pw2048 benchmark score
and win rate outside the request path.

## Rollback semantics

Rollback sets canary weight to zero first, then scales down candidate pods.
Promotion copies the immutable candidate image and model metadata into stable,
waits for a successful rolling update, then removes canary traffic. Images
should use digest or commit-SHA tags in real environments, never mutable tags.
