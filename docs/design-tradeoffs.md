# Design trade-offs

The project deliberately records why each mechanism exists and what it costs.

| Decision | Why it was chosen | Cost or limitation | When to change it |
|---|---|---|---|
| 2048 as the environment | Small, inspectable and cheap enough for repeated experiments | Does not reproduce speech, GPU or patient-data complexity | Keep lifecycle pattern; replace domain metrics and serving protocol for ASR/TTS |
| Separate Env/Train/Eval/Play | Prevents browser automation and exploration from contaminating evaluation | More interfaces and test cases | Preserve separation as the system grows |
| Frozen policy and matched seeds | Makes Stable/Candidate comparison defensible | Does not eliminate every source of statistical variance | Add confidence intervals and larger evaluation sets for costly releases |
| Immutable model versions | Makes provenance and rollback deterministic | Requires retention and cleanup policies | Add lifecycle retention in S3/MLflow, never overwrite versions |
| Filesystem registry first | Runs locally and in CI without infrastructure | Not multi-node or highly available | Use the same registry contract with MinIO/S3 or MLflow in production |
| Separate model and service versions | A model may be served by several runtime images and vice versa | Release metadata must link two identities | Preserve the relationship in a release record |
| Offline and online gates | A stable service can host a worse model; a good model can have a bad runtime | Promotion needs multiple sources of evidence | Automate evidence collection, not the business decision itself |
| NGINX weighted canary | Simple, visible and widely understood | Less sophisticated than a dedicated rollout controller | Adopt Argo Rollouts/Flagger for larger fleets and richer analysis windows |
| 5% → 20% → 50% | Limits blast radius while collecting progressively stronger evidence | Slow for low-traffic services | Use minimum request counts and time windows, not percentages alone |
| Rollback before scale-down | Removes traffic before terminating candidate pods | Keeps failed pods briefly for evidence | Preserve logs/traces before cleanup |
| Average latency in shell gate | Dependency-free demonstration | Average hides tail latency | Query Prometheus P95/P99 in a production release controller |
| CPU HPA | Matches the current lightweight inference workload | GPU/streaming pressure is not represented by CPU | Scale ASR/TTS from queue depth, active streams, RTF and GPU memory |
| Standard-library HTTP API | Small image and easy auditability | Missing mature middleware and generated API contracts | Move to FastAPI/gRPC when streaming, auth and ecosystem integrations justify it |
| Optional OpenTelemetry exporter | Local execution remains possible without a collector | Missing collector means traces are unavailable | Make collector availability part of production readiness |
| Default-deny NetworkPolicy | Minimizes unintended connectivity | Every new dependency needs an explicit rule | Add registry/storage/collector destinations deliberately |
| Manual final promotion | Keeps responsibility visible for a model-affecting decision | Slower than full automation | Automate only after thresholds, ownership and rollback confidence mature |

## The central principle

The project prefers explicit evidence and reversible changes over maximum
automation. Automation removes repetitive work; it must not conceal model
quality, uncertainty or ownership.
