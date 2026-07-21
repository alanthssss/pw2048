#!/usr/bin/env python3
"""Turn a k6 JSON summary into a reproducible deployment capacity plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _metric(summary: dict[str, Any], name: str, field: str) -> float:
    try:
        return float(summary["metrics"][name]["values"][field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing numeric k6 metric: {name}.values.{field}") from exc


def build_plan(
    summary: dict[str, Any], *, peak_rps: float, safety_factor: float = 0.70,
    max_p95_ms: float = 250.0, max_error_rate: float = 0.01, n_plus: int = 1,
) -> dict[str, Any]:
    if peak_rps <= 0:
        raise ValueError("peak_rps must be positive")
    if not 0 < safety_factor <= 1:
        raise ValueError("safety_factor must be in (0, 1]")
    if n_plus < 0:
        raise ValueError("n_plus cannot be negative")

    measured_rps = _metric(summary, "http_reqs", "rate")
    requests = int(_metric(summary, "http_reqs", "count"))
    p95_ms = _metric(summary, "http_req_duration", "p(95)")
    p99_ms = _metric(summary, "http_req_duration", "p(99)")
    error_rate = _metric(summary, "http_req_failed", "rate")
    slo_passed = p95_ms <= max_p95_ms and error_rate <= max_error_rate
    safe_rps = measured_rps * safety_factor if slo_passed else 0.0
    workload_replicas = math.ceil(peak_rps / safe_rps) if safe_rps > 0 else None
    total_replicas = workload_replicas + n_plus if workload_replicas is not None else None

    return {
        "status": "approved" if slo_passed else "rejected",
        "observed": {
            "requests": requests, "throughput_rps": round(measured_rps, 3),
            "p95_ms": round(p95_ms, 3), "p99_ms": round(p99_ms, 3),
            "error_rate": round(error_rate, 6),
        },
        "slo": {"max_p95_ms": max_p95_ms, "max_error_rate": max_error_rate},
        "capacity": {
            "peak_rps": peak_rps, "safety_factor": safety_factor,
            "safe_rps_per_replica": round(safe_rps, 3),
            "workload_replicas": workload_replicas, "n_plus": n_plus,
            "recommended_replicas": total_replicas,
        },
    }


def render_markdown(plan: dict[str, Any]) -> str:
    o, c, slo = plan["observed"], plan["capacity"], plan["slo"]
    replicas = c["recommended_replicas"] if c["recommended_replicas"] is not None else "N/A"
    return f"""# pw2048 Capacity Plan

**Decision:** `{plan['status']}`

| Measurement | Result | Limit |
|---|---:|---:|
| Requests | {o['requests']} | — |
| Measured throughput | {o['throughput_rps']} req/s | — |
| P95 latency | {o['p95_ms']} ms | {slo['max_p95_ms']} ms |
| P99 latency | {o['p99_ms']} ms | — |
| Error rate | {o['error_rate']:.4%} | {slo['max_error_rate']:.4%} |

## Recommendation

- Planning peak: **{c['peak_rps']} req/s**
- Safety factor: **{c['safety_factor']:.0%}**
- Safe throughput per replica: **{c['safe_rps_per_replica']} req/s**
- Workload replicas: **{c['workload_replicas'] if c['workload_replicas'] is not None else 'N/A'}**
- Failure reserve: **N+{c['n_plus']}**
- Recommended production replicas: **{replicas}**

The result is rejected when the measured test point violates its SLO; no capacity
recommendation should be promoted from an unhealthy test.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--peak-rps", type=float, required=True)
    parser.add_argument("--safety-factor", type=float, default=0.70)
    parser.add_argument("--max-p95-ms", type=float, default=250.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--n-plus", type=int, default=1)
    parser.add_argument("--json-output", type=Path, default=Path("capacity-plan.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("capacity-plan.md"))
    args = parser.parse_args()

    plan = build_plan(json.loads(args.summary.read_text()), peak_rps=args.peak_rps,
                      safety_factor=args.safety_factor, max_p95_ms=args.max_p95_ms,
                      max_error_rate=args.max_error_rate, n_plus=args.n_plus)
    args.json_output.write_text(json.dumps(plan, indent=2) + "\n")
    args.markdown_output.write_text(render_markdown(plan))
    print(f"capacity decision: {plan['status']}; output: {args.markdown_output}")
    return 0 if plan["status"] == "approved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
