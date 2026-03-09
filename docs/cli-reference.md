# CLI Reference

Full reference for all `python main.py` flags, modes, and output structure.

---

## All flags

| Flag | Default | Description |
|---|---|---|
| `--mode MODE` | — | Preset: `dev` (100 games, 1 run), `release` (1 000 games, 1 run), `benchmark` (500 games, 5 runs). Explicit `--games`/`--runs`/`--parallel` override the preset. |
| `--games N` | `20` | Number of browser games per run (pass `0` to skip the browser benchmark after `--train-games`) |
| `--runs N` | `1` | Number of times to repeat the full set of games; each run gets its own `run_<timestamp>/` directory |
| `--algorithm NAME` | `random` | Algorithm to use (`random`, `greedy`, `heuristic`, `expectimax`, `mcts`, `dqn`, `ppo`). Versioned aliases: `mcts-v1`/`mcts-v2`, `dqn-v1`/`dqn-v2`/`dqn-v3`, `ppo-v1`/`ppo-v2`/`ppo-v3`. `dqn` and `ppo` point to the latest (v3). |
| `--output DIR` | `results` | Root directory for run artifacts |
| `--show` | off | Open a visible browser window while playing |
| `--keep N` | `10` | Keep only the N most-recent runs per algorithm; pass `0` to disable pruning |
| `--parallel N` | `1` | Number of parallel browser workers |
| `--report` | off | Generate a self-contained HTML results dashboard (`index.html`) |
| `--s3-bucket BUCKET` | — | S3 bucket to upload artifacts and the report to (requires `boto3`) |
| `--s3-prefix PREFIX` | `results` | Key prefix inside the S3 bucket |
| `--s3-public` | off | Apply a public-read ACL to uploaded S3 objects |
| `--tui` | off | Launch the interactive TUI wizard |
| `--gui` | off | Launch the desktop GUI wizard (tkinter) |
| `--web` | off | Open the web UI launcher in your browser |
| `--checkpoint-dir DIR` | — | Directory for persisting learning-algorithm model weights. Loaded at startup, saved after every run. DQN / PPO (v3) only. |
| `--train-games N` | — | Fast in-process training games via the 4-layer Env/Train/Eval/Play pipeline (no browser, 10–50× faster). DQN / PPO only. Omit when using `--early-stopping-patience` for auto-training. |
| `--eval-freq N` | `50` | EvalCallback frequency during `--train-games` (every N games) |
| `--n-eval-games N` | `20` | Number of in-process eval games per EvalCallback round |
| `--tensorboard-dir DIR` | — | Directory for `training_log.csv` and optional TensorBoard `.tfevents` files |
| `--train-workers N` | `1` | Parallel independent training workers (N workers each train for `--train-games` games, best one selected) |
| `--early-stopping-patience N` | `0` | **Auto-stop training** when eval mean score has not improved for N consecutive eval rounds. `0` = disabled. Combine with a large `--train-games` or omit `--train-games` entirely for auto-mode. |
| `--early-stopping-min-delta D` | `1.0` | Minimum absolute improvement in eval mean score needed to reset the patience counter. |
| `--algo-version TAG` | — | Override the algorithm version tag written to `metrics.json` |

---

## Run modes

| Mode | Games / run | Runs | Parallel |
|---|---|---|---|
| `dev` | 100 | 1 | `os.cpu_count()` |
| `release` | 1 000 | 1 | `os.cpu_count()` |
| `benchmark` | 500 | 5 | `os.cpu_count()` |

```bash
python main.py --mode dev
python main.py --mode release --report
python main.py --mode benchmark --report
```

---

## Parallel execution

Use `--parallel N` to launch **N independent browser workers** simultaneously:

```bash
python main.py --games 40 --parallel 4
python main.py --games 40 --parallel 4 --report
```

> **Note:** `--show` is automatically silenced when `--parallel > 1`.

---

## Shell autocompletion

pw2048 supports tab-completion via [argcomplete](https://kislyuk.github.io/argcomplete/).

**bash** — add to `~/.bashrc`:

```bash
eval "$(activate-global-python-argcomplete --dest -)"
```

**zsh** — add to `~/.zshrc`:

```zsh
eval "$(activate-global-python-argcomplete --dest -)"
```

After activation, press <kbd>Tab</kbd>:

```
$ python main.py --algorithm <TAB>
expectimax  greedy  heuristic  mcts  random  dqn  ppo  dqn-v1 …

$ python main.py --mode <TAB>
benchmark  dev  release
```

---

## Result layout

```
results/
└── <AlgorithmName>/
    └── run_<YYYYMMDD_HHMMSS>/
        ├── results.csv      # per-game data (score, best_tile, moves, duration, won)
        ├── chart.png        # visualisation chart
        └── metrics.json     # run metadata (mode, games, workers, git_commit, …)
```

---

## HTML results dashboard (`--report`)

Pass `--report` to generate `results/index.html`:

```bash
python main.py --games 20 --report
```

**Dashboard features:**

- **Hero cards** — best avg score, highest best tile, most stable & fastest algorithm
- **Main Leaderboard** — all algorithms sorted by avg_score with full stats
- **Stability Board** — mean/std of avg_score across multiple runs
- **Efficiency Board** — games/second throughput per algorithm
- **Comparison Charts** — avg/median/P90 grouped bars, score histogram, best-tile distribution, run stability
- **Per-algorithm sections** — aggregate stats, inline charts, and run-history accordion (metadata: algorithm_version, mode, games, parallel_workers, timestamp, git_commit)

The dashboard is fully self-contained (charts embedded as base64 data URIs) and
works both locally (`file://`) and when hosted on S3.

---

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py` extending `BaseAlgorithm`:

```python
from src.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    name = "MyAlgo"

    def choose_move(self, board):
        # board is a 4×4 list of ints (0 = empty)
        return "left"   # one of "up", "down", "left", "right"
```

2. Register it in `main.py`'s `ALGORITHMS` dict.

---

## Example output

```
Running 30 games with the 'Random' algorithm…

  Game   1/30  score=  2424  max_tile= 256  moves= 210  won=False
  ...
  Game  30/30  score=   580  max_tile=  64  moves=  80  won=False

Raw data saved → results/Random/run_20260307_120000/results.csv
```

With `--train-games`:

```
Fast in-process training: 5000 game(s) with 'DQN-v3'…
  TensorBoard / CSV logs → tb_logs/DQN-v3
  Checkpoint dir → checkpoints/DQN-v3  (eval every 50 games, 20 eval games)

  [train] game     1/5000  score=  1024  max_tile= 128  steps=  92
  [eval @ game    50]  mean_score=   812  max_score=  1320  max_tile= 128 ★ new best
  ...
  Training complete — 5000 games in 487.3s  mean_score=3241  max_score=18432
  Best eval mean_score = 2904
```
