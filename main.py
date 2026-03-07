"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--games N] [--algorithm random] [--output results/]

    Results are saved to <output>/<AlgorithmName>/<timestamp>.{csv,png}, e.g.::

        results/Random/20260307_120000.csv
        results/Random/20260307_120000.png

    Grouping by algorithm makes it easy to compare multiple runs side-by-side.

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


def build_output_dir(base: str | Path, algorithm_name: str) -> Path:
    """Return the algorithm-scoped output directory.

    The resulting path has the form::

        <base>/<AlgorithmName>/

    All runs for the same algorithm are grouped under this directory.
    Individual run files are named by their timestamp (see :func:`main`).

    Parameters
    ----------
    base:
        Root directory for all results (e.g. ``"results"``).
    algorithm_name:
        Human-readable algorithm name (used verbatim as a subdirectory).
    """
    return Path(base) / algorithm_name


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
             "Results are saved under <output>/<algorithm>/<timestamp>.{csv,png}",
    )
    parser.add_argument("--show", action="store_true", help="Show the browser window while playing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    algo_cls = ALGORITHMS[args.algorithm]
    algorithm = algo_cls()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = build_output_dir(args.output, algorithm.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {args.games} games with the '{algorithm.name}' algorithm…\n")
    df = run_games(algorithm, n_games=args.games, headless=not args.show)

    plot_results(df, output_dir=output_dir, output_stem=timestamp)

    # Save raw data
    csv_path = output_dir / f"{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw data saved → {csv_path}")


if __name__ == "__main__":
    main()
