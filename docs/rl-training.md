# RL Training Guide — Getting High Scores with Learning Algorithms

> **Short answer:** the `dqn` and `ppo` algorithms ship with
> **behavioural-cloning (BC) pre-training** that bootstraps the network to
> heuristic level before the first game even begins.  Use `--train-games` for
> fast in-process training, and `--checkpoint-dir` to persist weights across
> sessions.

---

## Why learning algorithms score below heuristics on first run

Deep reinforcement learning needs **millions** of training steps to converge.
A short run (e.g. `--mode dev`, 100 games × ~200 moves ≈ 20 000 steps) is not
enough for a pure-numpy MLP starting from random weights.  All past DQN/PPO
versions (v1, v2) also used an inverted reward — merging two 64-tiles produced
reward `−5`, actively teaching the agent to *avoid* merging.

## What DQN-v3 / PPO-v3 fix

| Problem | Old behaviour | v3 fix |
|---|---|---|
| Inverted reward | `Δ(Σ log₂ tiles)` penalises good merges | `log₂(merge_score+1) + 0.1·empty` — always ≥ 0 |
| Cold-start | Random weights → random play for whole run | **BC pre-training**: network imitates Heuristic on 50 games before RL begins |
| Slow optimiser | Vanilla SGD | Adam (faster, more stable on noisy RL loss) |
| Weak encoding | 16-dim log₂ vector | 256-dim one-hot (16 cells × 16 tile levels) |
| Cross-game corruption | `_prev_board` bleeds across games | `on_game_start()` hook flushes state before each new game |

---

## Step-by-step: getting the highest possible score

### Step 1 — Quick sanity check (~2 min)

```bash
# 100 games, BC pre-training fires automatically at startup (~4 s overhead)
python main.py --algorithm dqn --mode dev --report
```

Expected: DQN-v3 should score noticeably above Random (~1 100) from game 1
because the network already knows the Heuristic's strategy.

### Step 2 — Fast in-process training (⚡ recommended)

Use `--train-games` to run thousands of games **without a browser** (10–50×
faster than Playwright mode):

```bash
# Train 5 000 games in ~5–10 min with periodic evaluation + checkpoints
python main.py --algorithm dqn \
               --train-games 5000 \
               --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs \
               --eval-freq 50 --n-eval-games 20

# After training: benchmark in the browser using the best checkpoint
python main.py --algorithm dqn --games 50 --checkpoint-dir checkpoints --report

# Training-only (skip browser benchmark)
python main.py --algorithm dqn --train-games 10000 \
               --checkpoint-dir checkpoints --games 0
```

### Step 3 — Persist weights across sessions (`--checkpoint-dir`)

> **This is the most important flag for achieving high scores.**  Without it
> every `python main.py` invocation starts from scratch (only BC pre-training).

```bash
# Session 1 — 500 browser games.  Weights saved after the run.
python main.py --algorithm dqn --games 500 --checkpoint-dir checkpoints

# Session 2 — loads from checkpoints/DQN-v3/checkpoint.npz, continues RL.
python main.py --algorithm dqn --games 500 --checkpoint-dir checkpoints

# Combine with --mode benchmark:
python main.py --algorithm dqn --mode benchmark --checkpoint-dir checkpoints --report
python main.py --algorithm ppo --mode benchmark --checkpoint-dir checkpoints --report
```

### Step 4 — Full benchmark (~20 min with parallelism)

```bash
# 500 games × 5 runs, all CPU cores, HTML leaderboard
python main.py --algorithm dqn --mode benchmark --parallel $(nproc) --report
python main.py --algorithm ppo --mode benchmark --parallel $(nproc) --report
```

### Step 5 — Compare against baselines

```bash
for algo in random greedy heuristic expectimax mcts dqn ppo; do
    python main.py --algorithm $algo --mode dev --report
done
```

### Step 6 — Tune hyperparameters (advanced)

Both `DQNAlgorithmV3` and `PPOAlgorithmV3` accept keyword arguments.
Edit `main.py`'s `ALGORITHMS` dict to pass custom values:

```python
"dqn": lambda: DQNAlgorithmV3(n_pretrain_games=200, lr=3e-4),
"ppo": lambda: PPOAlgorithmV3(n_pretrain_games=200, lr=1e-4),
```

| Parameter | DQN-v3 default | Effect of increasing |
|---|---|---|
| `n_pretrain_games` | 50 | Stronger heuristic start; +4 s per extra 50 games |
| `lr` | 5e-4 | Faster but less stable updates |
| `hidden_size` | 256 | Larger network; slower per step |
| `epsilon_decay` | 0.9998 | Slower exploration → exploitation trade-off |
| `buffer_size` | 50 000 | More diverse replay data |

---

## Checkpoint contents

The checkpoint is a single `.npz` file (~2 MB for the default 256-unit network).
It is overwritten after every run, so only the latest weights are kept.
`best_checkpoint.npz` is saved whenever `EvalCallback` finds a new best eval score.

| Component | DQN-v3 | PPO-v3 |
|---|---|---|
| Network weights | Q-net + target-net (W1,b1,W2,b2,W3,b3 × 2) | Actor-critic (W1,b1,W2,b2,W_a,b_a,W_v,b_v) |
| Optimizer state | Adam step counter, m/v moments | Adam step counter, m/v moments |
| Training progress | ε, global step | — |
| Replay / rollout buffer | ✗ (intentionally omitted) | ✗ (on-policy, stale data invalid) |

**Expected learning curve with persistence:**

| Cumulative games | Expected avg score |
|---:|---:|
| 500 | ~1 500 – 3 000 |
| 2 000 | ~3 000 – 6 000 |
| 10 000 | ~6 000 – 12 000 |
| 50,000+ | Approaching Heuristic level (~16,000) |

Scores improve unevenly — RL often plateaus then jumps.  Patience is the main ingredient.

---

## 4-layer Env / Train / Eval / Play architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Env   src/rl_env.py → Game2048Env                                  │
│  Pure-Python gym-style env: reset(), step(), valid_actions()         │
│  No Playwright.  10–50× faster than browser.                        │
├─────────────────────────────────────────────────────────────────────┤
│  Train src/rl_trainer.py → RLTrainer                                │
│  Drives episodes via Env. Calls algo.choose_move() so DQN/PPO       │
│  handle their own experience collection + gradient updates.         │
├─────────────────────────────────────────────────────────────────────┤
│  Eval  src/rl_trainer.py → EvalCallback                             │
│  Greedy eval every N games (uses predict()). Saves best_checkpoint. │
├─────────────────────────────────────────────────────────────────────┤
│  Play  src/runner.py (unchanged)                                    │
│  Browser benchmark/demo. Loads trained weights via --checkpoint-dir.│
└─────────────────────────────────────────────────────────────────────┘
```

### TensorBoard metrics

The `TrainingLogger` always writes `training_log.csv` in `--tensorboard-dir`.
With `pip install tensorboard` it also writes `.tfevents` files:

```bash
pip install tensorboard
tensorboard --logdir tb_logs
```

| Tag | Description |
|---|---|
| `train/score` | Game score each training episode |
| `train/max_tile` | Maximum tile each training episode |
| `train/n_steps` | Valid moves each training episode |
| `train/epsilon` | Current ε (DQN only) |
| `eval/mean_score` | Mean score over eval games |
| `eval/max_score` | Best score over eval games |
| `eval/max_tile` | Highest tile reached in eval |
| `summary/mean_score` | Final training mean score |

### Python API

```python
from src.rl_env import Game2048Env
from src.rl_trainer import make_trainer
from src.algorithms.dqn_algo import DQNAlgorithmV3

algo = DQNAlgorithmV3(n_pretrain_games=50)
trainer = make_trainer(
    algo,
    checkpoint_dir="checkpoints/DQN-v3",
    tensorboard_dir="tb_logs/DQN-v3",
    eval_freq=50,
    n_eval_games=20,
)
summary = trainer.train(total_games=5000)
print(f"Best eval score: {summary['best_eval_score']:.0f}")
```

### Why learning *can* beat Expectimax

Expectimax and Heuristic use hand-crafted heuristics that top out at ~33 000
avg score.  A trained neural network can discover board patterns those heuristics
miss.  The current shallow MLP needs many games to get there — use
`--train-games` + `--checkpoint-dir` to accumulate tens of thousands in-process.
