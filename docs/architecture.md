# System architecture

pw2048 is an AI engineering reference system built around a deliberately small
problem domain. The goal is to make every transition—from experiment to
production—observable, reproducible and reversible.

## Complete architecture

```mermaid
flowchart TB
    subgraph Development["Development and experiment layer"]
        Code["Git code and training config"]
        Env["Game2048Env"]
        Train["DQN / PPO trainer"]
        Eval["Frozen-policy evaluation"]
        Report["Benchmark report"]
        Code --> Train
        Env --> Train
        Train --> Eval
        Eval --> Report
    end

    subgraph Governance["Model governance layer"]
        Manifest["Model manifest<br/>Git SHA · seed · dependencies · hardware"]
        Registry["Immutable model registry"]
        Gate["Stable vs candidate quality gate"]
        Eval --> Manifest
        Manifest --> Registry
        Report --> Gate
        Registry --> Gate
    end

    subgraph Delivery["Delivery layer"]
        CI["GitLab CI / GitHub Actions"]
        Image["Immutable container image"]
        Test["Test deployment"]
        Load["k6 load test and capacity plan"]
        Canary["5% → 20% → 50% canary"]
        Promote["Production promotion"]
        Gate -->|pass| CI
        Gate -->|fail| Block["Release blocked"]
        CI --> Image --> Test --> Load --> Canary --> Promote
    end

    subgraph Runtime["Runtime platform"]
        Ingress["NGINX Ingress"]
        Stable["Stable Deployment"]
        Candidate["Canary Deployment"]
        API["Versioned inference API<br/>pre-warm · limit · backpressure · timeout"]
        Ingress --> Stable --> API
        Ingress --> Candidate --> API
    end

    subgraph Operations["Observability and recovery"]
        Metrics["Prometheus metrics and SLOs"]
        Dashboard["Grafana stable/canary comparison"]
        Traces["OpenTelemetry traces"]
        Alerts["Alert rules"]
        Rollback["Automatic rollback and runbook"]
        API --> Metrics --> Dashboard
        API --> Traces
        Metrics --> Alerts --> Rollback
        Rollback --> Stable
    end

    Canary --> Ingress
    Promote --> Stable
```

## Responsibility boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Training | Learning loop, checkpoint and seed | Production traffic |
| Evaluation | Frozen policy, matched seeds and quality metrics | Service availability |
| Registry | Artifact integrity and provenance | Deciding whether quality is acceptable |
| CI/CD | Repeatable gates and immutable delivery | Inventing missing business thresholds |
| Runtime | Admission, inference and version reporting | Retraining the model |
| Observability | Metrics, traces, alerts and evidence | Silently changing production state |

## Mapping to ASR/TTS

The engineering structure is reusable, while domain metrics change:

| pw2048 | ASR/TTS equivalent |
|---|---|
| Mean/P90 score and win rate | CER/WER, MOS or task-success quality |
| Single-board inference | Streaming audio session or synthesis request |
| CPU concurrency | GPU memory, batch size and active streams |
| HTTP request latency | First-token/first-audio latency, tail latency and RTF |
| Algorithm version | Acoustic/language/vocoder or personalized model version |

The project therefore demonstrates lifecycle engineering, not ASR/TTS algorithm
expertise.
