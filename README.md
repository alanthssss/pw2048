# pw2048 — Play 2048 with Algorithms

Automate the [2048 game](https://play2048.co/) using [Playwright](https://playwright.dev/python/) and collect fancy data visualizations of each algorithm's performance.

## Screenshots

| 2048 Game | Web Launcher |
|:---------:|:------------:|
| ![2048 game launch page](https://github.com/user-attachments/assets/971b6d73-36f0-4d47-b9da-0c27a9c3b5f2) | ![pw2048 Web Launcher](https://github.com/user-attachments/assets/08d9b84a-8180-475b-ae71-63b753c0d55a) |

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
- [x] **Expectimax** — game-tree search with chance nodes for tile spawns
- [x] **MCTS** — Monte Carlo Tree Search

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
│   ├── tui.py                 # Interactive TUI wizard (questionary + rich)
│   ├── gui.py                 # Desktop GUI wizard (tkinter – stdlib)
│   ├── webui.py               # Web UI launcher (http.server – stdlib)
│   └── algorithms/
│       ├── base.py            # Abstract BaseAlgorithm class
│       ├── random_algo.py     # Random algorithm
│       ├── greedy_algo.py     # Greedy (maximise immediate score gain)
│       ├── heuristic_algo.py  # Heuristic (empty tiles, monotonicity, corner, merge)
│       ├── expectimax_algo.py # Expectimax (game-tree search with chance nodes)
│       └── mcts_algo.py       # MCTS (Monte Carlo Tree Search)
└── tests/
    ├── test_game_and_algorithms.py
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
expectimax    greedy    heuristic    mcts    random

$ python main.py --mode <TAB>
benchmark    dev    release

$ python main.py --<TAB>
--algorithm  --games  --gui  --keep  --mode  --output  --parallel
--report     --runs   --show  --s3-bucket  --s3-prefix  --s3-public
--tui        --web
```

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode MODE` | — | Preset: `dev` (100 games, 1 run), `release` (1 000 games, 1 run), `benchmark` (500 games, 5 runs). Explicit `--games`/`--runs`/`--parallel` override the preset. |
| `--games N` | `20` | Number of games to play per run |
| `--runs N` | `1` | Number of times to repeat the full set of games; each run gets its own `run_<timestamp>/` directory |
| `--algorithm NAME` | `random` | Algorithm to use (`random`, `greedy`, `heuristic`, `expectimax`, `mcts`) |
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
