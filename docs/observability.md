# Observability and SLO

pw2048 exposes Prometheus metrics at `/metrics`. The resources in `observability/`
assume the Prometheus Operator (for example kube-prometheus-stack) is installed in
the `monitoring` namespace. Apply them with:

```bash
kubectl apply -k observability
```

The ServiceMonitor discovers stable and canary Services in all four application
namespaces. The Grafana sidecar imports the dashboard ConfigMap when it watches
the `grafana_dashboard=1` label.

## Service objectives

The production targets are measured over a rolling 30-day window:

| SLI | SLO | Measurement |
| --- | --- | --- |
| Availability | 99.9% successful inference requests | `1 - errors / requests` |
| Latency | 95% of inference requests below 100 ms | Prometheus histogram P95 |
| Rejections | below 0.1% | rejected requests / all requests |
| Model quality | candidate passes the offline quality policy | versioned evaluation report |

HTTP failures consume the availability error budget. Client input errors should
be separated from server errors before using the metric for a contractual SLA.
The online SLO and the offline model-quality gate are independent: a fast model
that produces worse decisions must not be promoted.

## Dashboard and alerts

The dashboard compares request rate, error ratio and latency percentiles by
`release`, making stable/canary regression visible during rollout. Alerts:

- `Pw2048HighErrorRate`: error ratio over 1% for ten minutes.
- `Pw2048HighP95Latency`: P95 above 100 ms for ten minutes.
- `Pw2048NoReadyReplicas`: no available replica for five minutes.
- `Pw2048CanaryRegression`: canary exceeds stable error ratio by one percentage
  point; the release controller/operator should roll back the canary.

Alerts deliberately notify rather than directly changing the cluster. Production
automation should route `action=rollback` to an authenticated deployment system,
which records the decision and runs `scripts/canary.sh rollback`.

## Triage runbook

1. Identify affected namespace, release track, model version and first failing
   time from the alert and `pw2048_build_info`.
2. Compare stable and canary request rate, errors and P95/P99 latency.
3. Inspect Deployment events, ready replicas, restarts and resource saturation.
4. Correlate request IDs with OpenTelemetry traces exported over OTLP/HTTP to
   the collector configured by `OTEL_EXPORTER_OTLP_ENDPOINT`.
5. Set canary weight to zero immediately if only canary regressed; otherwise
   restore the last known-good image/model and preserve evidence.
6. Record timeline, impact, root cause, recovery and preventive actions.

The API extracts W3C trace context and emits server spans with release and model
attributes. The container includes the OTLP/HTTP exporter; unset
`OTEL_EXPORTER_OTLP_ENDPOINT` to run without a collector.
