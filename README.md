# pw2048 — Play 2048 with Algorithms

Automate the [2048 game](https://play2048.co/) using [Playwright](https://playwright.dev/python/) and collect fancy data visualizations of each algorithm's performance.

## Screenshots

| 2048 Game | Web Launcher |
|:---------:|:------------:|
| ![2048 game launch page](https://github.com/user-attachments/assets/971b6d73-36f0-4d47-b9da-0c27a9c3b5f2) | ![pw2048 Web Launcher](https://github.com/user-attachments/assets/08d9b84a-8180-475b-ae71-63b753c0d55a) |

## At a Glance

| Field | Value |
|---|---|
| **Current best algorithm** | Expectimax |
| **Highest avg score** | ~33 000 (Expectimax, 73% win rate) |
| **Best learning algorithm** | DQN-v3 / PPO-v3 (behavioural-cloning pre-training) |
| **Benchmark protocol** | 5 runs × 500 games (auto-parallel) |

## Current Leaderboard

> Run `python main.py --mode benchmark --report` to populate this table.

| Rank | Algorithm | Version | Avg Score | P90 | Max Score | Best Tile | Win Rate |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Expectimax | v1 | 33 030 | 60 266 | 132 412 | 8192 | 73.4% |
| 2 | Heuristic | v1 | 16 061 | 28 185 | 61 064 | 4096 | 21.6% |
| 3 | MCTS | v2 | 7 821 | 12 649 | 15 416 | 1024 | 0.0% |
| 4 | Greedy | v1 | 3 050 | 5 416 | 13 820 | 1024 | 0.0% |
| 5 | Random | v1 | 1 102 | 1 720 | 3 324 | 256 | 0.0% |
| 6 | DQN-v3* | v3 | — | — | — | — | — |
| 7 | PPO-v3* | v3 | — | — | — | — | — |

\* DQN-v3 / PPO-v3 include behavioural-cloning pre-training.
See **[RL Training Guide →](docs/rl-training.md)** to push their scores higher.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

# Run 20 games with the random algorithm (default)
python main.py

# Run with the TUI wizard (arrow-key menus)
python main.py --tui

# Run with the web UI wizard (browser form)
python main.py --web

# Full benchmark with HTML leaderboard
python main.py --mode benchmark --report

# Fast in-process RL training (no browser, 10–50× faster)
python main.py --algorithm dqn \
               --train-games 5000 \
               --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs
```

## 4-layer RL Training Pipeline

The system exposes a structured Env / Train / Eval / Play stack for sustained
high-score improvement:

| Layer | Module | Description |
|---|---|---|
| **Env** | `src/rl_env.py` → `Game2048Env` | Pure-Python gym-style env, no browser, 10–50× faster |
| **Train** | `src/rl_trainer.py` → `RLTrainer` | In-process training loop; drives `choose_move()` |
| **Eval** | `src/rl_trainer.py` → `EvalCallback` | Greedy eval every N games; saves `best_checkpoint.npz` |
| **Play** | `src/runner.py` | Playwright browser benchmark / demo |

```bash
# Train, then benchmark
python main.py --algorithm dqn \
               --train-games 5000 --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs --eval-freq 50 --n-eval-games 20

python main.py --algorithm dqn --games 50 --checkpoint-dir checkpoints --report

tensorboard --logdir tb_logs   # view training curves
```

→ **[Full RL Training Guide](docs/rl-training.md)**

## Roadmap

### Baselines
- [x] **Random** — pick a random direction each turn
- [x] **Greedy** — pick the move that maximises immediate score gain
- [x] **Heuristic** — hand-crafted heuristics (corner strategy, monotonicity, empty tiles, merge potential)

### Search Algorithms
- [x] **Expectimax** — game-tree search with chance nodes for tile spawns
- [x] **MCTS** — Monte Carlo Tree Search (v1: random rollout, v2: greedy rollout)

### Learning Algorithms
- [x] **DQN v1/v2/v3** — v3 adds BC pre-training, Adam, one-hot encoding, score reward
- [x] **PPO v1/v2/v3** — v3 adds BC pre-training, Adam, one-hot encoding, score reward
- [x] **4-layer Env/Train/Eval/Play** — in-process training, EvalCallback, TensorBoard

## Project structure

```
pw2048/
├── game.html                  # Self-contained 2048 game (served locally)
├── main.py                    # CLI entry-point
├── requirements.txt
├── docs/                      # Detailed documentation
│   ├── rl-training.md         # RL guide: checkpoints, 4-layer pipeline, TensorBoard
│   ├── cli-reference.md       # All CLI flags, modes, result layout
│   └── ui-wizards.md          # TUI / GUI / Web UI usage
├── src/
│   ├── game.py                # Playwright wrapper
│   ├── runner.py              # Play layer: browser runner (sequential / parallel)
│   ├── rl_env.py              # Env layer: Game2048Env
│   ├── rl_trainer.py          # Train + Eval layers: RLTrainer, EvalCallback, TrainingLogger
│   ├── visualize.py           # Matplotlib charts
│   ├── report.py              # HTML dashboard generator
│   ├── storage.py             # S3 upload / prune helpers
│   ├── tui.py                 # TUI wizard (questionary + rich)
│   ├── gui.py                 # Desktop GUI wizard (tkinter)
│   ├── webui.py               # Web UI launcher (http.server)
│   └── algorithms/
│       ├── base.py
│       ├── random_algo.py
│       ├── greedy_algo.py
│       ├── heuristic_algo.py
│       ├── expectimax_algo.py
│       ├── mcts_algo.py
│       ├── dqn_algo.py        # DQN v1/v2/v3
│       └── ppo_algo.py        # PPO v1/v2/v3
└── tests/
    ├── test_game_and_algorithms.py
    ├── test_rl_env_and_trainer.py
    ├── test_storage_and_report.py
    ├── test_tui.py
    ├── test_gui.py
    └── test_webui.py
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Further reading

| Topic | Doc |
|---|---|
| Getting high scores with RL, checkpoints, TensorBoard | **[docs/rl-training.md](docs/rl-training.md)** |
| All CLI flags, parallel mode, result layout | **[docs/cli-reference.md](docs/cli-reference.md)** |
| TUI / GUI / Web UI wizards | **[docs/ui-wizards.md](docs/ui-wizards.md)** |

