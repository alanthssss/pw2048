# Production Kubernetes deployment

Kustomize separates reusable workload policy from environment configuration:

```text
deploy/k8s/base                 shared workloads and guardrails
deploy/k8s/components/namespace namespace, quota and default limits
deploy/k8s/overlays/dev         pw2048-dev / dev.pw2048.local
deploy/k8s/overlays/test        pw2048-test / test.pw2048.local
deploy/k8s/overlays/demo        pw2048-demo / demo.pw2048.local
deploy/k8s/overlays/prod        pw2048-prod / pw2048.example.com
```

Render before applying, and pin image digests in a real release:

```bash
kubectl kustomize deploy/k8s/overlays/prod > /tmp/pw2048-prod.yaml
kubectl apply --server-side --dry-run=server -f /tmp/pw2048-prod.yaml
kubectl apply -k deploy/k8s/overlays/prod
```

`deploy/k8s/all.yaml` remains the lightweight, backwards-compatible manifest
used by the existing canary script. New environment deployments should use the
overlays.

Kustomize suffixes resource names as well as namespaces. Run canary automation
against production with:

```bash
NAMESPACE=pw2048-prod DEPLOYMENT=pw2048-canary-prod \
STABLE_DEPLOYMENT=pw2048-stable-prod INGRESS=pw2048-canary-prod \
CANARY_SERVICE=pw2048-canary-prod ./scripts/canary.sh deploy IMAGE MODEL VERSION
```

## Production safeguards

- Dedicated ServiceAccount with API token automount disabled.
- Non-root containers, runtime-default seccomp, read-only root filesystem, no
  privilege escalation and all Linux capabilities dropped.
- Startup, readiness and liveness probes plus graceful termination.
- Rolling updates with zero unavailable stable replicas.
- HPA with conservative scale-down; production runs 3–20 stable replicas. The
  canary is deliberately excluded so rollback can reliably scale it to zero.
- PDB keeps two stable replicas available in production.
- Topology spreading reduces single-node impact.
- Default-deny NetworkPolicy permits ingress only from ingress and monitoring
  namespaces, and DNS egress only.
- ResourceQuota and LimitRange bound namespace consumption.

The network policies require the ingress controller namespace to carry the
standard `kubernetes.io/metadata.name=ingress-nginx` label. Add explicit egress
destinations if a future model registry, object store or telemetry collector is
used. CPU-based HPA is adequate for the current CPU service; a GPU inference
variant should scale from queue depth, active streams or latency rather than GPU
utilization alone.

## Environment and release practice

Build once and promote the same immutable image digest through dev, test, demo
and production. Store secrets outside Git (External Secrets, Sealed Secrets or a
cloud secret manager). Keep model version, image digest, configuration version
and evaluation record together in the release audit entry.

Production rollout order is stable baseline, canary at 5%, 20% and 50%, then
promotion. At every stage compare stable/canary technical SLOs and validate the
offline model-quality gate. Rollback sets canary weight to zero first, then
restores the last known-good immutable image/model.

## Disaster and disruption checks

Before launch, verify pod deletion, voluntary node drain, failed readiness,
resource exhaustion and canary rollback. Confirm the PDB protects voluntary
disruption, topology spread is effective on a multi-node cluster, HPA metrics are
available, and ingress stops routing before the pod's grace period expires.
