# Project presentation scripts

These scripts are speaking structures, not text that must be memorized word for
word. Replace example scales with verified project facts only.

## Three-minute version

> pw2048 is an AI engineering project that uses the 2048 game as a controlled
> environment. My goal was not simply to build a game-playing algorithm, but to
> answer a broader engineering question: how does a model move from an
> experiment to a service that can be reproduced, evaluated, released,
> observed and rolled back?
>
> I first separated the project into Env, Train, Eval and Play layers. The
> environment runs in pure Python for efficient training, while Playwright is
> retained for browser demonstration. DQN, PPO, Expectimax and other algorithms
> use a common interface. Evaluation freezes learning and exploration and uses
> matched seeds, because evaluating a model while it is still changing produces
> misleading results.
>
> I then added model governance. Every candidate has an immutable manifest that
> records its Git commit, seed, configuration, dependencies, hardware and
> SHA-256. Stable and candidate metrics pass through a fail-closed quality gate.
> A container that is healthy but has a lower score or win rate is still blocked.
>
> On the delivery side, CI performs tests, image scanning, test deployment, k6
> load testing and capacity planning. Production uses Stable and Canary
> deployments with 5%, 20% and 50% traffic stages. Error and latency regression
> triggers rollback; successful technical gates still require model-quality
> approval before promotion.
>
> The service exposes Prometheus metrics and OpenTelemetry traces, with Grafana
> dashboards, alerts and four isolated Kubernetes environments. It also has
> backpressure, concurrency limits, deadlines and an incident runbook.
>
> This is not an ASR/TTS project, but its lifecycle maps directly: scores become
> CER/WER, MOS and RTF; HTTP concurrency becomes streaming sessions, GPU memory
> and batching. My strongest contribution is turning model work into a reliable
> engineering system, while ASR/TTS model details are the domain knowledge I
> would deepen in the role.

## Ten-minute version

### 0:00–1:00 — Problem and positioning

- Many AI demos prove only that an algorithm ran once.
- pw2048 treats reproducibility, comparison, delivery and recovery as one system.
- 2048 was selected because behavior is inspectable and experiments are cheap.

### 1:00–2:30 — Experiment architecture

- Explain Env/Train/Eval/Play and why the browser is not in the training loop.
- Describe the common algorithm interface and honest Expectimax baseline.
- Explain frozen-policy evaluation, matched seeds and checkpoint selection.

### 2:30–4:00 — Model governance

- Show the manifest: Git SHA, seed, config, dependencies, hardware and checksum.
- Explain immutable versions and registry integrity.
- Run or describe the Stable/Candidate regression gate.
- Emphasize that infrastructure health and model quality are different questions.

### 4:00–5:30 — CI/CD and release

- Walk through test, scan, model gate, test deployment and load test.
- Explain why the same immutable image is promoted between environments.
- Show Stable/Canary at 5%, 20% and 50% with minimum request counts.
- Explain why final promotion is manual and rollback is automatic.

### 5:30–7:00 — Runtime and observability

- Describe pre-warm, rate limiting, bounded concurrency, backpressure and timeout.
- Show Prometheus request/error/latency/rejection metrics and Grafana comparison.
- Explain request IDs, W3C context and OpenTelemetry spans.
- Connect alerts to the incident runbook without letting alerts mutate production directly.

### 7:00–8:00 — Capacity and reliability

- Show the sample load-test report: 100 req/s measured, 70 req/s safe capacity.
- For a 250 req/s peak, calculate four workload replicas plus one failure reserve.
- Describe pod deletion, injected latency/error and rollback exercises.

### 8:00–9:00 — Design trade-offs

- Filesystem registry was chosen for a dependency-free demo; production uses S3/MLflow.
- Weighted NGINX canary is understandable; Argo Rollouts fits larger fleets.
- CPU HPA fits this workload; GPU ASR/TTS should scale on streams, queue and RTF.
- The project optimizes for explainability and reversibility rather than tool count.

### 9:00–10:00 — Transfer to the target role

- Map score/win rate to CER/WER/MOS/RTF.
- Map HTTP requests to streaming sessions and GPU-backed inference.
- State the boundary honestly: no claim of large ASR/TTS production experience.
- Close with the value proposition: strong platform engineering plus an explicit
  method for learning the speech domain.

## Likely follow-up questions

1. Why not evaluate during training?
2. Why maintain separate model and image versions?
3. What prevents a healthy but worse model from being promoted?
4. Why use request counts as well as observation time in canary gates?
5. How would the design change for streaming ASR?
6. What happens if Prometheus or the trace collector is unavailable?
7. Which part is a working implementation and which is a production extension point?
8. What evidence would be required before automating final promotion?
