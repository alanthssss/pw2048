"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--games N] [--algorithm random] [--output results/]

    Results are saved to <output>/<timestamp>/<algorithm>/, e.g.::

        results/20260307_120000/Random/results_random.csv
        results/20260307_120000/Random/results_random.png

Examples
--------
    # Run 20 games with the random algorithm (default)
    python main.py --games 20

    # Show the browser window while playing
    python main.py --games 5 --show
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.algorithms.random_algo import RandomAlgorithm
from src.runner import run_games
from src.visualize import plot_results

ALGORITHMS = {
    "random": RandomAlgorithm,
}


def build_output_dir(base: str | Path, algorithm_name: str, timestamp: str | None = None) -> Path:
    """Return a timestamped, algorithm-scoped output directory.

    The resulting path has the form::

        <base>/<YYYYMMDD_HHMMSS>/<AlgorithmName>/

    Parameters
    ----------
    base:
        Root directory for all results (e.g. ``"results"``).
    algorithm_name:
        Human-readable algorithm name (used verbatim as a subdirectory).
    timestamp:
        Optional pre-formatted timestamp string (``YYYYMMDD_HHMMSS``).
        Defaults to the current local time when *None*.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / timestamp / algorithm_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 2048 with algorithms via Playwright")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument(
        "--algorithm",
        choices=list(ALGORITHMS.keys()),
        default="random",
        help="Algorithm to use (default: random)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Base directory for result charts (default: results/). "
             "Results are saved under <output>/<timestamp>/<algorithm>/",
    )
    parser.add_argument("--show", action="store_true", help="Show the browser window while playing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    algo_cls = ALGORITHMS[args.algorithm]
    algorithm = algo_cls()

    output_dir = build_output_dir(args.output, algorithm.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.games} games with the '{algorithm.name}' algorithm…\n")
    df = run_games(algorithm, n_games=args.games, headless=not args.show)

    plot_results(df, output_dir=output_dir)

    # Save raw data
    csv_path = output_dir / f"results_{algorithm.name.lower()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved → {csv_path}")


if __name__ == "__main__":
    main()
