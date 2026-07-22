# Example capacity plan

**Decision:** `approved`

| Measurement | Result | Limit |
|---|---:|---:|
| Requests | 6,000 | — |
| Measured throughput | 100 req/s | — |
| P95 latency | 82 ms | 250 ms |
| P99 latency | 140 ms | — |
| Error rate | 0.20% | 1.00% |

## Recommendation

- Planning peak: **250 req/s**
- Safety factor: **70%**
- Safe throughput per replica: **70 req/s**
- Workload replicas: **4** (`ceil(250 / 70)`)
- Failure reserve: **N+1**
- Recommended production replicas: **5**

The safety factor reserves headroom for traffic variance, runtime noise and
measurement uncertainty. N+1 preserves the planned workload when one replica is
unavailable. This example is generated from
[`k6-summary.json`](k6-summary.json) with:

```bash
python scripts/capacity_plan.py docs/examples/k6-summary.json \
  --peak-rps 250 \
  --json-output /tmp/capacity-plan.json \
  --markdown-output /tmp/capacity-plan.md
```
