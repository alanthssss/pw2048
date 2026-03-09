# Efficient Training Playbook

> **Goal:** get the best DQN / PPO model as fast as possible while using as
> little device time as possible.

This guide covers three complementary acceleration strategies — **GPU**,
**parallel workers**, and **early stopping** — plus the built-in tools for
**monitoring your model's status** at any point.

---

## 1. GPU acceleration (Apple Silicon / CUDA)

DQN-v3 and PPO-v3 automatically detect and use a GPU when
[PyTorch](https://pytorch.org/) is installed.

### Install

```bash
# CPU-only (works everywhere, still fast for shallow networks)
pip install torch

# Apple Silicon (M1 / M2 / M3) — Metal Performance Shaders
pip install torch          # MPS is bundled in modern macOS torch builds

# NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### How it works

Device selection is automatic:

```
Apple MPS  →  CUDA  →  CPU (torch)  →  NumPy fallback
```

No extra CLI flags are needed — just install `torch` and the GPU is used.
To verify which backend is active, add `--inspect-checkpoint` after training
(see §4) or print it in Python:

```python
from src.algorithms.dqn_algo import DQNAlgorithmV3, _detect_device
print(_detect_device())   # "mps", "cuda", "cpu", or None (NumPy)
```

### When GPU helps (and when it doesn't)

| Scenario | Speed-up |
|---|---|
| Large network (`hidden_size ≥ 512`) | ✅ 2–5× |
| Default 256-unit MLP | ⚠ marginal (overhead may cancel gain) |
| Batch training with large replay buffer | ✅ 2–4× |
| NumPy fallback (no torch) | baseline |

> **Tip:** the default 256-unit network is small enough that MPS/CUDA overhead
> can exceed the compute savings.  Set `hidden_size=512` or `1024` to make GPU
> acceleration worthwhile:

```bash
# In main.py ALGORITHMS dict (one-time edit):
"dqn": lambda: DQNAlgorithmV3(hidden_size=512),
```

### Force a specific device

```bash
# Force NumPy (useful for debugging or tiny networks):
# Edit ALGORITHMS in main.py:
"dqn": lambda: DQNAlgorithmV3(device="numpy"),

# Force CPU torch (avoids MPS/CUDA, but keeps PyTorch autograd):
"dqn": lambda: DQNAlgorithmV3(device="cpu"),
```

---

## 2. Parallel training workers (`--train-workers N`)

`--train-workers N` spawns **N independent training processes** simultaneously.
Each worker starts from the same checkpoint with a different random seed.
After all workers finish, the best-performing one's checkpoint is kept.

Wall-clock time reduces roughly N-fold on multi-core machines.

```bash
# Use 4 parallel workers — finishes in ≈ 1/4 of the time
python main.py --algorithm dqn \
               --train-games 5000 \
               --train-workers 4 \
               --checkpoint-dir checkpoints \
               --games 0

# Auto-select worker count = all physical cores
python main.py --algorithm dqn \
               --train-games 5000 \
               --train-workers $(nproc) \
               --checkpoint-dir checkpoints \
               --games 0
```

### Combining GPU + parallel workers

On a multi-core CPU machine with a GPU, GPU workers train the neural net
while the CPU handles environment simulation:

```bash
# 4 workers, each using GPU — fastest configuration on a workstation
python main.py --algorithm dqn \
               --train-games 10000 \
               --train-workers 4 \
               --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs \
               --games 0
```

> **Note:** each worker uses the same detected device.  If you have one GPU
> and many workers, workers share the GPU which may reduce individual GPU
> utilisation.  On Apple Silicon, 2–4 workers is usually optimal.

### Expected speed table

| Setup | Games/min (approx) |
|---|---|
| NumPy, 1 worker | ~300 |
| CPU torch, 1 worker | ~350 |
| MPS (M1/M2), 1 worker | ~400–600 |
| CUDA (RTX 3080), 1 worker | ~600–1000 |
| MPS + 4 workers | ~1500–2000 |
| CUDA + 4 workers | ~2000–4000 |

---

## 3. Early stopping — train until stable

> **Use this when you don't know how many games are needed.**
> Training stops automatically when performance plateaus.

```bash
# Auto-mode: no fixed game count — stops when score plateaus for 500 games.
python main.py --algorithm dqn \
               --early-stopping-patience 10 \
               --eval-freq 50 \
               --checkpoint-dir checkpoints \
               --games 0

# With an explicit upper bound (50 000 games max, or plateau, whichever first):
python main.py --algorithm dqn \
               --train-games 50000 \
               --early-stopping-patience 10 \
               --eval-freq 50 \
               --checkpoint-dir checkpoints \
               --games 0
```

The patience window = `patience × eval_freq` games.
`patience=10, eval_freq=50` → stop after **500 games** without improvement.

See [rl-training.md §Early stopping](rl-training.md#early-stopping--auto-training-until-stable)
for full details.

---

## 4. Combined recipe — maximum speed

```bash
# ① Install torch (GPU support)
pip install torch

# ② Run with GPU + 4 parallel workers + early stopping
python main.py --algorithm dqn \
               --train-workers 4 \
               --early-stopping-patience 15 \
               --early-stopping-min-delta 50 \
               --eval-freq 100 \
               --n-eval-games 30 \
               --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs \
               --games 0

# ③ After training: benchmark the saved best checkpoint
python main.py --algorithm dqn \
               --games 50 \
               --checkpoint-dir checkpoints \
               --report

# ④ Inspect what was learned
python main.py --inspect-checkpoint checkpoints/DQN-v3/best_checkpoint.npz
python main.py --training-status    tb_logs/DQN-v3
```

---

## 5. Monitoring model status

### 5a. Inspect a checkpoint file

```bash
python main.py --inspect-checkpoint checkpoints/DQN-v3/checkpoint.npz
python main.py --inspect-checkpoint checkpoints/DQN-v3/best_checkpoint.npz
```

**Example output:**

```
────────────────────────────────────────────────────────
  Checkpoint: checkpoints/DQN-v3/best_checkpoint.npz
────────────────────────────────────────────────────────
  Algorithm   : DQN-v3
  File size   : 2078.6 KB
  Parameters  : 530,448
  Global step : 24,631
  ε (epsilon) : 7.12%  [late / converging]
  Adam steps  : 4,892
  Weight norms: mean=5.16  min=0.01  max=23.14
────────────────────────────────────────────────────────
  q_W1          22.86   q_W2          23.14   q_W3           2.99
  ...
────────────────────────────────────────────────────────
```

**Reading the output:**

| Field | What it tells you |
|---|---|
| `ε (epsilon)` | Exploration rate. >20% = early training; <5% = converging/late |
| `Global step` | Total environment steps — bigger = more experience |
| `Adam steps` | Gradient updates applied; proxy for how much the network has changed |
| `Weight norms` | `mean ≈ 3–30` is typical; very low (<0.1) or very high (>100) suggests problems |

### 5b. Training convergence status

```bash
python main.py --training-status tb_logs/DQN-v3
```

**Example output:**

```
────────────────────────────────────────────────────────
  Training status: tb_logs/DQN-v3
────────────────────────────────────────────────────────
  Train games    : 5,000
  Eval rounds    : 100
  Best eval score: 8,432
  Recent eval mean (10 rounds): 8,109
  Recent train mean (last 50 games): 7,821
  Current ε      : 3.21%
  Score trend    : +42.3 pts/eval  [↑ improving]
  Convergence    : 🟡 IMPROVING  (still learning — keep training)
  Score sparkline: ▄▄▅▅▅▆▆▇▇█
  Score P25/50/75: 6,124 / 7,432 / 8,109
────────────────────────────────────────────────────────
```

**Reading the output:**

| Field | What it tells you |
|---|---|
| `Score trend` | Positive → still learning; negative → declining (try lower lr) |
| `Convergence` | 🟢 STABLE = plateau; 🟡 IMPROVING = keep training |
| `Score sparkline` | Visual shape of last N eval scores; flat ▄▄▄▄ = plateau |
| `P25/50/75` | Score distribution — wide spread = high variance (more eval games help) |

### 5c. TensorBoard (live charts)

```bash
pip install tensorboard
tensorboard --logdir tb_logs
# → open http://localhost:6006
```

Available tags:

| Tag | Description |
|---|---|
| `train/score` | Per-game score during training |
| `train/epsilon` | Exploration rate over time |
| `eval/mean_score` | Mean score per evaluation round |
| `eval/max_score` | Best score per evaluation round |
| `eval/max_tile` | Highest tile reached per evaluation round |

### 5d. Python API — inspect programmatically

```python
from src.training_status import inspect_checkpoint, print_training_status

# Checkpoint stats
info = inspect_checkpoint("checkpoints/DQN-v3/best_checkpoint.npz")
print(f"ε = {info['epsilon']:.1%}")
print(f"Steps = {info['step']:,}")

# Training log convergence
status = print_training_status("tb_logs/DQN-v3")
print(f"Best eval score: {status['best_eval_score']:.0f}")
print(f"Trend: {status['trend']:+.1f} pts/eval")
print(f"Stable: {status['stable']}")   # True = plateau reached
```

---

## 6. Decision guide

```
Start here
    │
    ├─ Don't know how long to train?
    │       → Use --early-stopping-patience (§3)
    │
    ├─ Want faster training?
    │   ├─ Have GPU?  → Install torch (§1)
    │   ├─ Have multiple cores?  → --train-workers N (§2)
    │   └─ Both?  → §4 combined recipe
    │
    ├─ Training complete — is it good enough?
    │       → python main.py --training-status tb_logs/DQN-v3 (§5b)
    │
    └─ Want to inspect what was saved?
            → python main.py --inspect-checkpoint checkpoints/DQN-v3/best_checkpoint.npz (§5a)
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Score not improving after 1000 games | Still in warm-up / high ε | Keep training; ε < 10% is needed for stable play |
| Sparkline flat (🟢 STABLE) too early | `min_delta` too low | Increase `--early-stopping-min-delta` |
| Score declining | Learning rate too high | Set `lr=1e-4` in ALGORITHMS dict |
| GPU not used despite torch install | PyTorch/MPS not available | Check `_detect_device()` output |
| Multiple workers slower than 1 | Overhead > gain (small network) | Use 1 worker or increase `hidden_size` |
