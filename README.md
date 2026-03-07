# pw2048 — Play 2048 with Algorithms

Automate the [2048 game](https://play2048.co/) using [Playwright](https://playwright.dev/python/) and collect fancy data visualizations of each algorithm's performance.

## Roadmap

- [x] **Simple Random** — pick a random direction each turn (baseline)
- [ ] More algorithms coming soon…

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
│       └── random_algo.py     # Simple random algorithm
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

Charts and a CSV with raw game data are saved under `results/<algorithm>/<timestamp>.{png,csv}`
(e.g. `results/Random/20260307_120000.png`).
All runs for the same algorithm are grouped together, making it easy to compare them side-by-side.
Use `--output` to change the root directory.

## All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--games N` | `20` | Number of games to play |
| `--algorithm NAME` | `random` | Algorithm to use |
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

Raw data saved → results/Random/20260307_120000.csv
```

After running `python main.py --games 30 --parallel 4 --report`:

```
Running 30 games with the 'Random' algorithm (4 parallel workers)…

  [parallel] 30 games across 4 worker(s)…
  [worker offset=0] Game  1  score=  1820  …
  …

Raw data saved → results/Random/20260307_120000.csv
Report saved  → results/index.html
```

## Running tests

```bash
python -m pytest tests/ -v
```

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
