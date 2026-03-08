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

\* DQN-v3 / PPO-v3 include behavioural-cloning pre-training — see [Getting high scores with learning algorithms](#getting-high-scores-with-learning-algorithms) for how to push their scores above the baseline.

The HTML dashboard (`--report`) keeps this table live and sorted by **Avg Score**.

## Benchmark Protocol

| Mode | Runs | Games / Run | Parallel |
|---|---|---|---|
| `dev` | 1 | 100 | auto (`os.cpu_count()`) |
| `release` | 1 | 1 000 | auto |
| `benchmark` | 5 | 500 | auto |

```bash
# Quick dev scratch (100 games)
python main.py --mode dev

# Release quality (1 000 games)
python main.py --mode release --report

# Full benchmark (5 × 500 games, HTML leaderboard)
python main.py --mode benchmark --report
```

## Getting high scores with learning algorithms

> **Short answer:** the `dqn` and `ppo` algorithms now ship with
> **behavioural-cloning (BC) pre-training** that bootstraps the network to
> heuristic level before the first game even begins.  Just run more games.

### Why learning algorithms score below heuristics on first run

Deep reinforcement learning needs **millions** of training steps to converge.
A short run (e.g. `--mode dev`, 100 games × ~200 moves ≈ 20 000 steps) is not
enough for a pure-numpy MLP starting from random weights.  All past DQN/PPO
versions (v1, v2) also used an inverted reward — merging two 64-tiles produced
reward `−5`, actively teaching the agent to *avoid* merging.

### What DQN-v3 / PPO-v3 fix

| Problem | Old behaviour | v3 fix |
|---|---|---|
| Inverted reward | `Δ(Σ log₂ tiles)` penalises good merges | `log₂(merge_score+1) + 0.1·empty` — always ≥ 0 |
| Cold-start | Random weights → random play for whole run | **BC pre-training**: network imitates Heuristic on 50 games before RL begins |
| Slow optimiser | Vanilla SGD | Adam (faster, more stable on noisy RL loss) |
| Weak encoding | 16-dim log₂ vector | 256-dim one-hot (16 cells × 16 tile levels) |
| Cross-game corruption | `_prev_board` bleeds across games | `on_game_start()` hook flushes state before each new game |

### Step-by-step: how to get the highest possible score

**Step 1 — Quick sanity check** (~2 min)

```bash
# 100 games, BC pre-training fires automatically at startup (~4 s overhead)
python main.py --algorithm dqn --mode dev --report
```

Expected: DQN-v3 should score noticeably above Random (~1 100) from game 1
because the network already knows the Heuristic's strategy.

**Step 2 — Full benchmark** (~20 min with parallelism)

```bash
# 500 games × 5 runs, all CPU cores, HTML leaderboard
python main.py --algorithm dqn --mode benchmark --parallel $(nproc) --report
python main.py --algorithm ppo --mode benchmark --parallel $(nproc) --report
```

As the run progresses the RL policy improves, so later games in each run
score higher than early ones.

**Step 3 — Compare against baselines**

```bash
# Run all algorithms and generate a single comparison report
for algo in random greedy heuristic expectimax mcts dqn ppo; do
    python main.py --algorithm $algo --mode dev --report
done
```

**Step 4 — Tune hyperparameters (advanced)**

Both `DQNAlgorithmV3` and `PPOAlgorithmV3` accept keyword arguments.
Edit `main.py`'s `ALGORITHMS` dict to pass custom values:

```python
# Example: more BC pre-training games → stronger starting policy
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

### Step 5 — Persist weights across sessions with `--checkpoint-dir`

> **This is the most important flag for achieving high scores.**  Without it
> every `python main.py` invocation starts from scratch (only BC pre-training).
> With it, RL training accumulates indefinitely across however many sessions
> you run.

```bash
# Session 1 — 500 games.  Weights saved after the run.
python main.py --algorithm dqn --games 500 --checkpoint-dir checkpoints

# Session 2 — loads from checkpoints/DQN-v3/checkpoint.npz, continues RL.
# No BC overhead, epsilon stays at the value it reached in session 1.
python main.py --algorithm dqn --games 500 --checkpoint-dir checkpoints

# Repeat until satisfied.  Combine with --mode benchmark:
python main.py --algorithm dqn --mode benchmark --checkpoint-dir checkpoints --report
python main.py --algorithm ppo --mode benchmark --checkpoint-dir checkpoints --report
```

**What is saved in the checkpoint?**

| Component | DQN-v3 | PPO-v3 |
|---|---|---|
| Network weights | Q-net + target-net (W1,b1,W2,b2,W3,b3 × 2) | Actor-critic (W1,b1,W2,b2,W_a,b_a,W_v,b_v) |
| Optimizer state | Adam step counter, m/v moments | Adam step counter, m/v moments |
| Training progress | ε, global step | — |
| Replay / rollout buffer | ✗ (intentionally omitted) | ✗ (on-policy, stale data invalid) |

The checkpoint is a single `.npz` file (~2 MB for the default 256-unit network).
It is overwritten after every run, so only the latest weights are kept.

**Expected learning curve with persistence:**

| Cumulative games | Expected avg score |
|---:|---:|
| 500 | ~1 500 – 3 000 |
| 2 000 | ~3 000 – 6 000 |
| 10 000 | ~6 000 – 12 000 |
| 50 000+ | Approaching Heuristic level (~16 000) |

Scores improve unevenly — RL often plateaus then jumps.  Patience (and more
games) is the main ingredient.

---

## 4-layer Env / Train / Eval / Play pipeline

For sustained high-score improvement, the system exposes a structured
four-layer RL training stack alongside the existing browser-based benchmark:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Env layer        Game2048Env (src/rl_env.py)                       │
│  Pure-Python gym-style environment — reset(), step(), valid_actions()│
│  No Playwright, no browser.  10–50× faster than the browser runner. │
├─────────────────────────────────────────────────────────────────────┤
│  Train layer      RLTrainer (src/rl_trainer.py)                     │
│  Drives training episodes via the Env.  Calls choose_move() so DQN/ │
│  PPO handle their own experience collection and gradient updates.   │
│  Logs scalar metrics to CSV (always) and TensorBoard (if installed).│
├─────────────────────────────────────────────────────────────────────┤
│  Eval layer       EvalCallback (src/rl_trainer.py)                  │
│  Runs deterministic (greedy) evaluation episodes every N games.     │
│  Saves the best-scoring checkpoint to best_checkpoint.npz.          │
├─────────────────────────────────────────────────────────────────────┤
│  Play layer       runner.py + Playwright browser                    │
│  Existing benchmark / demo mode.  Loads the trained weights.        │
└─────────────────────────────────────────────────────────────────────┘
```

### `--train-games` — fast in-process training

The `--train-games N` flag runs the full Env / Train / Eval pipeline
**without opening a browser**.  Use it to accumulate many more games of
RL experience per minute:

```bash
# Train 5000 games in-process (~5–10 min), saving weights and TB logs.
python main.py --algorithm dqn \
               --train-games 5000 \
               --checkpoint-dir checkpoints \
               --tensorboard-dir tb_logs \
               --eval-freq 50 \
               --n-eval-games 20

# After training: benchmark in the browser using the best checkpoint.
python main.py --algorithm dqn \
               --games 50 \
               --checkpoint-dir checkpoints \
               --report

# Training only — skip the browser benchmark entirely.
python main.py --algorithm dqn \
               --train-games 10000 \
               --checkpoint-dir checkpoints \
               --games 0
```

| Flag | Default | Description |
|---|---|---|
| `--train-games N` | — | Fast in-process training games (DQN / PPO only) |
| `--eval-freq N` | 50 | Run EvalCallback every N training games |
| `--n-eval-games N` | 20 | Number of in-process eval games per round |
| `--tensorboard-dir DIR` | — | CSV + optional TensorBoard log directory |

### TensorBoard visualisation

The `TrainingLogger` always writes `training_log.csv` inside the given
`--tensorboard-dir`.  If the `tensorboard` package is installed it also
writes `.tfevents` files:

```bash
pip install tensorboard          # one-time
tensorboard --logdir tb_logs     # launch the dashboard (port 6006)
```

Metrics logged:

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

You can also drive training directly from Python:

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

### Why learning *can* beat Expectimax given enough experience

Expectimax and Heuristic use hard-coded heuristics that top out at ~33 000
avg score because the hand-crafted weights are fixed.  A trained neural
network can in principle discover *board patterns the hand-crafted heuristics
miss*, and with access to deep tree search (not currently implemented) RL
approaches regularly reach 2048 tiles in research settings.

The current implementation uses a **shallow 2-layer MLP and 100–500 games** —
that is enough to show that BC pre-training works, but not enough to
surpass Expectimax.  For higher scores consider:

- Using `--train-games` + `--checkpoint-dir` to **train tens of thousands of
  games** in-process (much faster than the browser mode)
- Using `--checkpoint-dir` to **persist weights across many sessions**
- Running **longer** (`--mode release` or `--mode benchmark`)
- Increasing `n_pretrain_games` to 500–1000
- Switching to a deeper network (`hidden_size=512`)

## Roadmap

### Baselines
- [x] **Random** — pick a random direction each turn
- [x] **Greedy** — pick the move that maximises immediate score gain
- [x] **Heuristic** — hand-crafted heuristics (corner strategy, monotonicity, empty tiles, merge potential)

### Search Algorithms
- [x] **Expectimax** — game-tree search with chance nodes for tile spawns
- [x] **MCTS** — Monte Carlo Tree Search (v1: random rollout, v2: greedy rollout)

### Learning Algorithms
- [x] **DQN v1** — standard Deep Q-Network
- [x] **DQN v2** — Double DQN
- [x] **DQN v3** — BC pre-training + Adam + one-hot encoding + score-based reward *(current default)*
- [x] **PPO v1** — Proximal Policy Optimization (raw rewards)
- [x] **PPO v2** — PPO with EMA reward normalisation
- [x] **PPO v3** — BC pre-training + Adam + one-hot encoding + score-based reward *(current default)*

## Project structure

```
pw2048/
├── game.html                  # Self-contained 2048 game (served locally)
├── main.py                    # CLI entry-point
├── requirements.txt
├── src/
│   ├── game.py                # Playwright wrapper (board read, move execution)
│   ├── runner.py              # Play layer: run N games (browser, sequential or parallel)
│   ├── rl_env.py              # Env layer: Game2048Env — pure-Python gym-style environment
│   ├── rl_trainer.py          # Train + Eval layers: RLTrainer, EvalCallback, TrainingLogger
│   ├── visualize.py           # Matplotlib charts from results
│   ├── report.py              # Self-contained HTML dashboard generator
│   ├── storage.py             # S3 upload / prune helpers (lazy boto3 import)
│   ├── tui.py                 # Interactive TUI wizard (questionary + rich)
│   ├── gui.py                 # Desktop GUI wizard (tkinter – stdlib)
│   ├── webui.py               # Web UI launcher (http.server – stdlib)
│   └── algorithms/
│       ├── base.py            # Abstract BaseAlgorithm class
│       ├── random_algo.py     # Random algorithm
│       ├── greedy_algo.py     # Greedy (maximise immediate score gain)
│       ├── heuristic_algo.py  # Heuristic (empty tiles, monotonicity, corner, merge)
│       ├── expectimax_algo.py # Expectimax (game-tree search with chance nodes)
│       ├── mcts_algo.py       # MCTS v1 (random rollout) and v2 (greedy rollout)
│       ├── dqn_algo.py        # DQN v1/v2/v3 — v3 adds BC pre-training + Adam + one-hot
│       └── ppo_algo.py        # PPO v1/v2/v3 — v3 adds BC pre-training + Adam + one-hot
└── tests/
    ├── test_game_and_algorithms.py
    ├── test_rl_env_and_trainer.py   # Env / Train / Eval layer tests
    ├── test_storage_and_report.py
    ├── test_tui.py
    ├── test_gui.py
    └── test_webui.py
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

# Launch the interactive TUI wizard
python main.py --tui

# Launch the desktop GUI wizard (tkinter)
python main.py --gui

# Launch the web UI wizard in your browser
python main.py --web

# Run 20 games with the random algorithm (default)
python main.py

# Run 50 games and save charts to a custom directory
python main.py --games 50 --output my_results

# Show the browser window while playing
python main.py --games 5 --show
```

Charts and a CSV with raw game data are saved under `results/<AlgorithmName>/run_<YYYYMMDD_HHMMSS>/` (e.g. `results/Random/run_20260307_120000/`).
Each run directory contains `results.csv`, `chart.png`, and `metrics.json`.
All runs for the same algorithm are grouped together, making it easy to compare them side-by-side.
Use `--output` to change the root directory.

## Interactive TUI wizard

Not a fan of memorising flags?  Run the wizard:

```bash
python main.py --tui
```

The wizard walks you through every option step-by-step with arrow-key
selection menus and validated text prompts:

```
╭───────────────────────────────────────────────╮
│  pw2048 – Interactive Launcher                │
│  Use arrow keys to select, Enter to confirm.  │
╰───────────────────────────────────────────────╯

? Algorithm:        greedy
? Run mode:         custom – set games / runs / parallel manually
? Number of games per run:  50
? Number of runs:           2
? Parallel browser workers: 2
? Output directory:         results
? Show browser window while playing?  No
? Keep N most-recent runs per algorithm (0 = keep all):  10
? Generate HTML report?   Yes
? Upload results to S3?   No

   Configuration Summary
┌──────────────┬──────────┐
│ Algorithm    │ greedy   │
│ Games / run  │ 50       │
│ Runs         │ 2        │
│ Workers      │ 2        │
│ Output dir   │ results/ │
│ Show browser │ no       │
│ Keep N runs  │ 10       │
│ HTML report  │ yes      │
│ S3 bucket    │ –        │
└──────────────┴──────────┘

? Proceed? Yes
```

Press <kbd>Ctrl-C</kbd> at any prompt, or answer **No** at the final
confirmation, to abort without running any games.

The wizard covers all parameters available via CLI flags:

| Wizard step | Equivalent flag(s) |
|---|---|
| Algorithm | `--algorithm` |
| Run mode (preset) | `--mode` |
| Games / runs / workers (custom) | `--games`, `--runs`, `--parallel` |
| Output directory | `--output` |
| Checkpoint directory | `--checkpoint-dir` |
| Show browser | `--show` |
| Keep N runs | `--keep` |
| HTML report | `--report` |
| S3 upload | `--s3-bucket`, `--s3-prefix`, `--s3-public` |

## Desktop GUI wizard

Prefer a point-and-click interface?  Launch the native tkinter window:

```bash
python main.py --gui
```

The window exposes the same options as the TUI — algorithm, mode, games/runs/workers,
output directory, show-browser, keep-N, HTML report, and optional S3 upload — all
in a standard form layout with dropdown menus, checkboxes, and text fields.

**Prerequisites:** tkinter ships with Python on Windows and macOS.  On
Debian/Ubuntu it requires one extra package:

```bash
sudo apt-get install python3-tk
```

## Web UI wizard

Prefer a browser-based form?  Start the local web launcher:

```bash
python main.py --web
```

pw2048 starts an HTTP server on a random free port, prints the URL, and opens it
in your default browser automatically:

```
  Web UI → http://127.0.0.1:54321/
  (fill in the form and click Launch — check your terminal for progress)
```

The form stays open until you click **Launch ▶**, at which point it returns a
confirmation page, shuts the server down, and starts the run in your terminal.

![pw2048 Web Launcher](https://github.com/user-attachments/assets/08d9b84a-8180-475b-ae71-63b753c0d55a)

The web UI requires **no third-party packages** — it uses only the Python
standard library (`http.server`, `threading`, `webbrowser`).

## Shell autocompletion

pw2048 supports tab-completion for all CLI flags and their values via
[argcomplete](https://kislyuk.github.io/argcomplete/).

> **Note — `python main.py` vs `./main.py`**
>
> The standard `register-python-argcomplete` command only registers completion
> when the script is called **directly** (e.g. `./main.py`).  Because most
> users type `python main.py`, the instructions below use
> `activate-global-python-argcomplete`, which installs a global Python
> completion hook that works for any `python <script>` invocation whose first
> line contains the marker `# PYTHON_ARGCOMPLETE_OK`.

### One-time setup

**bash** — add to `~/.bashrc`:

```bash
eval "$(activate-global-python-argcomplete --dest -)"
```

**zsh** — add to `~/.zshrc`:

```zsh
eval "$(activate-global-python-argcomplete --dest -)"
```

Reload your shell (or run the command in your current session), then `cd` to
the repo directory before pressing <kbd>Tab</kbd>.

**fish** / other shells — see the [argcomplete docs](https://kislyuk.github.io/argcomplete/#activating-global-completion).

### Usage

After activation, press <kbd>Tab</kbd> after `python main.py` to see available
flags, and again after flags like `--algorithm` or `--mode` to complete their
values:

```
$ python main.py --algorithm <TAB>
expectimax    greedy    heuristic    mcts    random    dqn    ppo
dqn-v1    dqn-v2    dqn-v3    ppo-v1    ppo-v2    ppo-v3    mcts-v1    mcts-v2

$ python main.py --mode <TAB>
benchmark    dev    release

$ python main.py --<TAB>
--algorithm  --checkpoint-dir  --eval-freq  --games  --gui  --keep  --mode
--n-eval-games  --output  --parallel  --report  --runs  --show
--s3-bucket  --s3-prefix  --s3-public  --tensorboard-dir  --train-games
--tui  --web
```

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode MODE` | — | Preset: `dev` (100 games, 1 run), `release` (1 000 games, 1 run), `benchmark` (500 games, 5 runs). Explicit `--games`/`--runs`/`--parallel` override the preset. |
| `--games N` | `20` | Number of browser games to play per run (pass `0` to skip the browser benchmark after `--train-games`) |
| `--runs N` | `1` | Number of times to repeat the full set of games; each run gets its own `run_<timestamp>/` directory |
| `--algorithm NAME` | `random` | Algorithm to use (`random`, `greedy`, `heuristic`, `expectimax`, `mcts`, `dqn`, `ppo`). Versioned aliases: `mcts-v1`/`mcts-v2`, `dqn-v1`/`dqn-v2`/`dqn-v3`, `ppo-v1`/`ppo-v2`/`ppo-v3`. `dqn` and `ppo` point to the latest (v3). |
| `--output DIR` | `results` | Root directory for run artifacts |
| `--show` | off | Open a visible browser window while playing |
| `--keep N` | `10` | Keep only the N most-recent runs per algorithm; pass `0` to disable pruning |
| `--parallel N` | `1` | Number of parallel browser workers (see [Parallel execution](#parallel-execution)) |
| `--report` | off | Generate a self-contained HTML results dashboard (`index.html`) |
| `--s3-bucket BUCKET` | — | S3 bucket to upload artifacts and the report to (requires `boto3`) |
| `--s3-prefix PREFIX` | `results` | Key prefix inside the S3 bucket |
| `--s3-public` | off | Apply a public-read ACL to uploaded S3 objects |
| `--tui` | off | Launch the interactive TUI wizard to configure all parameters step-by-step |
| `--gui` | off | Launch the desktop GUI wizard (tkinter) to configure and start a run |
| `--web` | off | Open the web UI launcher in the system browser to configure and start a run |
| `--checkpoint-dir DIR` | — | Directory for persisting learning-algorithm model weights across runs. The checkpoint is loaded at startup (skipping BC pre-training) and saved after every run. Only applies to `dqn` / `ppo` (v3). |
| `--train-games N` | — | Run N in-process training games via the 4-layer Env/Train/Eval/Play pipeline (no browser, 10–50× faster). DQN / PPO only. |
| `--eval-freq N` | `50` | EvalCallback frequency during `--train-games` (every N games) |
| `--n-eval-games N` | `20` | Number of in-process eval games per EvalCallback round |
| `--tensorboard-dir DIR` | — | Directory for `training_log.csv` and optional TensorBoard events (requires `pip install tensorboard`) |

## Parallel execution

Use `--parallel N` to launch **N independent browser workers** simultaneously.
Each worker runs its share of the games in a separate Chromium instance, which
can cut wall-clock time by up to N×:

```bash
# Play 40 games using 4 workers — roughly 4× faster
python main.py --games 40 --parallel 4

# Combine with the HTML report
python main.py --games 40 --parallel 4 --report
```

> **Note:** `--show` is automatically silenced when `--parallel` > 1 because
> driving multiple headed windows simultaneously is not practical.

## HTML results dashboard (`--report`)

Pass `--report` to generate a self-contained `index.html` in the output
directory after each run:

```bash
python main.py --games 20 --report
# → results/index.html
```

The dashboard is fully self-contained (charts are embedded as base64 data
URIs) and works both locally (`file://`) and when hosted on S3.

**Dashboard features:**

- **Sticky navigation bar** — jump-links to each algorithm's section
- **Aggregate stats cards** — Total Games, Best Score, Avg Score, Best Tile,
  Win Rate (computed across all retained runs)
- **Run History accordion** — every stored run is a collapsible
  `<details>`/`<summary>` element; the most recent run is pre-expanded
- Each expanded run shows its **results chart** and a **per-game table**
  (game index, score, best tile, moves, duration, win/loss)
- Summary chips on every collapsed run: `latest`, `N games`, `avg score`,
  `best tile`, `win%`

## Example output

After running `python main.py --games 30`:

```
Running 30 games with the 'Random' algorithm…

  Game   1/30  score=  2424  max_tile= 256  moves= 210  won=False
  ...
  Game  30/30  score=   580  max_tile=  64  moves=  80  won=False

Raw data saved → results/Random/run_20260307_120000/results.csv
```

After running `python main.py --games 30 --parallel 4 --report`:

```
Running 30 games with the 'Random' algorithm (4 parallel workers)…

  [parallel] 30 games across 4 worker(s)…
  [worker offset=0] Game  1  score=  1820  …
  …

Raw data saved → results/Random/run_20260307_120000/results.csv
Report saved  → results/index.html
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Result Layout

Each run is saved in a timestamped subdirectory:

```
results/
└── <AlgorithmName>/
    └── run_<YYYYMMDD_HHMMSS>/
        ├── results.csv      # per-game data (score, best_tile, moves, duration, won)
        ├── chart.png        # visualisation chart
        └── metrics.json     # run metadata (mode, games, workers, git_commit, …)
```

The HTML dashboard (`--report`) is written to `results/index.html` and contains:
- **Hero cards** — best avg score, highest best tile, most stable & fastest algorithm
- **Main Leaderboard** — all algorithms sorted by avg_score with full stats
- **Stability Board** — mean/std of avg_score across multiple runs
- **Efficiency Board** — games/second throughput per algorithm
- **Comparison Charts** — avg/median/P90 grouped bars, score histogram, best-tile distribution, run stability
- **Per-algorithm sections** — aggregate stats, inline charts, and run-history accordion (each run shows a metadata box with `algorithm_version`, `mode`, `games`, `parallel_workers`, `timestamp`, `git_commit`)

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py` with a class that extends `BaseAlgorithm`
   and implements `choose_move(board)`.
2. Register it in `main.py`'s `ALGORITHMS` dict.

```python
from src.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    name = "MyAlgo"

    def choose_move(self, board):
        # board is a 4×4 list of ints (0 = empty)
        return "left"   # one of "up", "down", "left", "right"
```
