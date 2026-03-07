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
│   ├── runner.py              # Run N games, collect results as DataFrame
│   ├── visualize.py           # Matplotlib charts from results
│   └── algorithms/
│       ├── base.py            # Abstract BaseAlgorithm class
│       └── random_algo.py     # Simple random algorithm
└── tests/
    └── test_game_and_algorithms.py
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

Charts and a CSV with raw game data are saved under `results/<algorithm>/<timestamp>.{png,csv}` (e.g. `results/Random/20260307_120000.png`).  
All runs for the same algorithm are grouped together, making it easy to compare them side-by-side.  
Use `--output` to change the root directory.

## Example output

After running `python main.py --games 30`:

```
Running 30 games with the 'Random' algorithm…

  Game   1/30  score=  2424  max_tile= 256  moves= 210  won=False
  ...
  Game  30/30  score=   580  max_tile=  64  moves=  80  won=False

==================================================
  Summary — Random Algorithm  (n=30 games)
==================================================
  Score   :  min=   492  mean=   1,173  max= 2,484
  Max tile:  min=    64  mean=   113.1  max=   256
  Moves   :  min=    70  mean=   125.9  max=   217
  Win rate: 0.0%
==================================================
```

## Running tests

```bash
python -m pytest tests/ -v
```

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py` with a class that extends `BaseAlgorithm` and implements `choose_move(board)`.
2. Register it in `main.py`'s `ALGORITHMS` dict.

```python
from src.algorithms.base import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    name = "MyAlgo"

    def choose_move(self, board):
        # board is a 4×4 list of ints (0 = empty)
        return "left"   # one of "up", "down", "left", "right"
```
