"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--mode dev|release|benchmark]
                   [--games N] [--runs N] [--algorithm random] [--output results/]
                   [--keep N] [--report] [--parallel N]
                   [--s3-bucket BUCKET] [--s3-prefix PREFIX] [--s3-public]

    Results are saved to <output>/<AlgorithmName>/<timestamp>.{csv,png}, e.g.::

        results/Random/20260307_120000.csv
        results/Random/20260307_120000.png

    Grouping by algorithm makes it easy to compare multiple runs side-by-side.

    ``--mode``       sets games/runs/parallel automatically:
                     dev: 100 games, 1 run, parallel=auto
                     release: 1000 games, 1 run, parallel=auto
                     benchmark: 500 games, 5 runs, parallel=auto

    ``--keep N``     keeps only the *N* most-recent runs per algorithm (locally
    and, when ``--s3-bucket`` is given, on S3).  Defaults to 10.

    ``--report``     generates a self-contained HTML dashboard (``index.html``)
    in the output directory.  When ``--s3-bucket`` is given the report is also
    uploaded to ``<s3-prefix>/index.html``.

    ``--parallel N`` launches *N* browser workers simultaneously, each playing
    its share of the games.  Defaults to 1 (sequential).

Examples
--------
    # Run 20 games with the random algorithm (default)
    python main.py --games 20

    # Quick dev scratch run (100 games, auto-parallel)
    python main.py --mode dev

    # Benchmark run (500 games × 5 runs, auto-parallel)
    python main.py --mode benchmark --report

    # Show the browser window while playing
    python main.py --games 5 --show

    # Keep the 5 most-recent runs and generate an HTML report
    python main.py --games 10 --keep 5 --report

    # Run 40 games using 4 parallel browser workers
    python main.py --games 40 --parallel 4 --report

    # Upload results to S3
    python main.py --games 10 --s3-bucket my-bucket --report
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from src.algorithms.greedy_algo import GreedyAlgorithm
from src.algorithms.random_algo import RandomAlgorithm
from src.runner import run_games
from src.visualize import plot_results

ALGORITHMS = {
    "random": RandomAlgorithm,
    "greedy": GreedyAlgorithm,
}

_DEFAULT_KEEP = 10
_DEFAULT_S3_PREFIX = "results"

# Mode presets: games, runs, parallel ("auto" → os.cpu_count())
_MODE_PRESETS: dict[str, dict] = {
    "dev":       {"games": 100,  "runs": 1, "parallel": os.cpu_count() or 1},
    "release":   {"games": 1000, "runs": 1, "parallel": os.cpu_count() or 1},
    "benchmark": {"games": 500,  "runs": 5, "parallel": os.cpu_count() or 1},
}
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1


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


def prune_local_results(output_dir: Path, keep_n: int) -> list[Path]:
    """Delete old local result files, keeping only the latest *keep_n* runs.

    Each "run" is identified by a timestamped stem that corresponds to both a
    ``.csv`` and a ``.png`` file.  Runs are sorted lexicographically (which
    matches chronological order because stems are ``YYYYMMDD_HHMMSS``).

    Parameters
    ----------
    output_dir:
        Algorithm-scoped results directory (e.g. ``results/Random/``).
    keep_n:
        Number of most-recent runs to retain.  Pass ``0`` to keep all runs.

    Returns
    -------
    list[Path]
        Paths of the deleted files.
    """
    if keep_n <= 0:
        return []
    csv_files = sorted(output_dir.glob("*.csv"))
    old_stems = [f.stem for f in csv_files[:-keep_n]] if len(csv_files) > keep_n else []

    deleted: list[Path] = []
    for stem in old_stems:
        for ext in ("csv", "png"):
            path = output_dir / f"{stem}.{ext}"
            if path.exists():
                path.unlink()
                deleted.append(path)
    if deleted:
        print(f"  Pruned {len(deleted)} old local file(s) in '{output_dir}'")
    return deleted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 2048 with algorithms via Playwright")
    parser.add_argument(
        "--mode",
        choices=["dev", "release", "benchmark"],
        default=None,
        help=(
            "Run mode preset — sets games/runs/parallel automatically. "
            "dev: 100 games, 1 run; release: 1000 games, 1 run; "
            "benchmark: 500 games, 5 runs. "
            "Explicit --games/--runs/--parallel flags override the preset."
        ),
    )
    parser.add_argument("--games", type=int, default=None, help="Number of games per run (default: 20)")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        metavar="N",
        help="Number of times to repeat the full set of games (default: 1). "
             "Each run produces its own CSV/PNG with a unique run_id.",
    )
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
    parser.add_argument(
        "--keep",
        type=int,
        default=_DEFAULT_KEEP,
        metavar="N",
        help=f"Keep only the N most-recent runs per algorithm (default: {_DEFAULT_KEEP}). "
             "Pass 0 to disable pruning.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel browser workers (default: 1 = sequential). "
             "Each worker runs its share of games in a separate browser instance.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a self-contained HTML results dashboard (index.html) in the output directory.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        metavar="BUCKET",
        help="S3 bucket to upload results and the HTML report to.  "
             "Requires boto3 and valid AWS credentials.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=_DEFAULT_S3_PREFIX,
        metavar="PREFIX",
        help=f"Key prefix inside the S3 bucket (default: '{_DEFAULT_S3_PREFIX}').",
    )
    parser.add_argument(
        "--s3-public",
        action="store_true",
        help="Apply public-read ACL to uploaded S3 objects.",
    )

    args = parser.parse_args(argv)

    # Apply mode presets for any value not explicitly provided on the CLI.
    if args.mode is not None:
        preset = _MODE_PRESETS[args.mode]
        if args.games is None:
            args.games = preset["games"]
        if args.runs is None:
            args.runs = preset["runs"]
        if args.parallel is None:
            args.parallel = preset["parallel"]

    # Fall back to plain defaults when no mode was given.
    if args.games is None:
        args.games = _DEFAULT_GAMES
    if args.runs is None:
        args.runs = _DEFAULT_RUNS
    if args.parallel is None:
        args.parallel = _DEFAULT_PARALLEL

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    algo_cls = ALGORITHMS[args.algorithm]
    algorithm = algo_cls()

    output_dir = build_output_dir(args.output, algorithm.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, args.parallel)
    n_runs = max(1, args.runs)
    parallel_note = f" ({n_workers} parallel workers)" if n_workers > 1 else ""
    runs_note = f" × {n_runs} run(s)" if n_runs > 1 else ""
    print(
        f"\nRunning {args.games} games{runs_note} with the '{algorithm.name}' algorithm{parallel_note}…\n"
    )

    last_timestamp = None
    for run_num in range(1, n_runs + 1):
        if n_runs > 1:
            print(f"\n─── Run {run_num}/{n_runs} ───")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        last_timestamp = timestamp

        df = run_games(
            algorithm,
            n_games=args.games,
            headless=not args.show,
            n_workers=n_workers,
            run_id=timestamp,
        )

        plot_results(df, output_dir=output_dir, output_stem=timestamp)

        # Save raw data
        csv_path = output_dir / f"{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Raw data saved → {csv_path}")

    # ------------------------------------------------------------------
    # Prune old local results
    # ------------------------------------------------------------------
    prune_local_results(output_dir, keep_n=args.keep)

    # ------------------------------------------------------------------
    # Generate HTML report
    # ------------------------------------------------------------------
    if args.report or args.s3_bucket:
        from src.report import generate_html_report

        report_path = Path(args.output) / "index.html"
        generate_html_report(results_dir=args.output, output_path=report_path)
        print(f"Report saved  → {report_path}")

    # ------------------------------------------------------------------
    # Upload to S3
    # ------------------------------------------------------------------
    if args.s3_bucket:
        from src.storage import sync_run_to_s3, upload_report

        print(f"\nUploading results to s3://{args.s3_bucket}/{args.s3_prefix}/…")
        sync_run_to_s3(
            output_dir=output_dir,
            timestamp=last_timestamp,
            bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            algorithm_name=algorithm.name,
            keep_n=args.keep,
            public_read=args.s3_public,
        )

        report_path = Path(args.output) / "index.html"
        if report_path.exists():
            upload_report(
                report_path=report_path,
                bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                public_read=args.s3_public,
            )


if __name__ == "__main__":
    main()
