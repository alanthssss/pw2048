# pw2048 — Play 2048 with Algorithms

Automate the [2048 game](https://play2048.co/) using [Playwright](https://playwright.dev/python/) and collect fancy data visualizations of each algorithm's performance.

## At a Glance

| Field | Value |
|---|---|
| **Current best algorithm** | Heuristic |
| **Highest best tile** | — (run benchmarks to populate) |
| **Benchmark protocol** | 5 runs × 500 games (auto-parallel) |

## Current Leaderboard

> Run `python main.py --mode benchmark --report` to populate this table.

| Rank | Algorithm | Stage | Runs | Avg Score | Median | P90 | Best Tile | Win Rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — | — |

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

## Roadmap

### Baselines
- [x] **Random** — pick a random direction each turn
- [x] **Greedy** — pick the move that maximises immediate score gain
- [x] **Heuristic** — hand-crafted heuristics (e.g. corner strategy, monotonicity)

### Search Algorithms
- [ ] **Expectimax** — game-tree search with chance nodes for tile spawns
- [ ] **MCTS** — Monte Carlo Tree Search

### Learning Algorithms
- [ ] **DQN** — Deep Q-Network (reinforcement learning)
- [ ] **PPO** — Proximal Policy Optimization (reinforcement learning)

## Project structure

```
pw2048/
├── game.html                  # Self-contained 2048 game (served locally)
├── main.py                    # CLI entry-point
├── requirements.txt
├── src/
│   ├── game.py                # Playwright wrapper (board read, move execution)
│   ├── runner.py              # Run N games (sequential or parallel), collect results
│   ├── visualize.py           # Matplotlib charts from results
│   ├── report.py              # Self-contained HTML dashboard generator
│   ├── storage.py             # S3 upload / prune helpers (lazy boto3 import)
│   └── algorithms/
│       ├── base.py            # Abstract BaseAlgorithm class
│       ├── random_algo.py     # Random algorithm
│       ├── greedy_algo.py     # Greedy (maximise immediate score gain)
│       └── heuristic_algo.py  # Heuristic (empty tiles, monotonicity, corner, merge)
└── tests/
    ├── test_game_and_algorithms.py
    └── test_storage_and_report.py
```

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

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

## Shell autocompletion

pw2048 supports tab-completion for all CLI flags and their values via
[argcomplete](https://kislyuk.github.io/argcomplete/).

### One-time setup

**bash**

```bash
# Activate completion for this script only (add to ~/.bashrc to persist)
eval "$(register-python-argcomplete main.py)"
```

**zsh**

```zsh
# Add to ~/.zshrc
autoload -U bashcompinit && bashcompinit
eval "$(register-python-argcomplete main.py)"
```

**fish** / other shells — see the [argcomplete docs](https://kislyuk.github.io/argcomplete/#activating-global-completion).

### Usage

After activation, press <kbd>Tab</kbd> after `python main.py` to see available flags,
and again after flags like `--algorithm` or `--mode` to complete their values:

```
$ python main.py --algorithm <TAB>
greedy    heuristic    random

$ python main.py --mode <TAB>
benchmark    dev    release
```

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode MODE` | — | Preset: `dev` (100 games, 1 run), `release` (1 000 games, 1 run), `benchmark` (500 games, 5 runs). Explicit `--games`/`--runs`/`--parallel` override the preset. |
| `--games N` | `20` | Number of games to play per run |
| `--runs N` | `1` | Number of times to repeat the full set of games; each run gets its own `run_<timestamp>/` directory |
| `--algorithm NAME` | `random` | Algorithm to use (`random`, `greedy`, `heuristic`) |
| `--output DIR` | `results` | Root directory for run artifacts |
| `--show` | off | Open a visible browser window while playing |
| `--keep N` | `10` | Keep only the N most-recent runs per algorithm; pass `0` to disable pruning |
| `--parallel N` | `1` | Number of parallel browser workers (see [Parallel execution](#parallel-execution)) |
| `--report` | off | Generate a self-contained HTML results dashboard (`index.html`) |
| `--s3-bucket BUCKET` | — | S3 bucket to upload artifacts and the report to (requires `boto3`) |
| `--s3-prefix PREFIX` | `results` | Key prefix inside the S3 bucket |
| `--s3-public` | off | Apply a public-read ACL to uploaded S3 objects |

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
