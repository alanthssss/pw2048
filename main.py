"""
pw2048 – Play 2048 with different algorithms and visualise the results.

Usage
-----
    python main.py [--games N] [--algorithm random] [--output results/]
                   [--keep N] [--report] [--s3-bucket BUCKET]
                   [--s3-prefix PREFIX] [--s3-public]

    Results are saved to <output>/<AlgorithmName>/<timestamp>.{csv,png}, e.g.::

        results/Random/20260307_120000.csv
        results/Random/20260307_120000.png

    Grouping by algorithm makes it easy to compare multiple runs side-by-side.

    ``--keep N``  keeps only the *N* most-recent runs per algorithm (locally
    and, when ``--s3-bucket`` is given, on S3).  Defaults to 10.

    ``--report``  generates a self-contained HTML dashboard (``index.html``)
    in the output directory.  When ``--s3-bucket`` is given the report is also
    uploaded to ``<s3-prefix>/index.html``.

Examples
--------
    # Run 20 games with the random algorithm (default)
    python main.py --games 20

    # Show the browser window while playing
    python main.py --games 5 --show

    # Keep the 5 most-recent runs and generate an HTML report
    python main.py --games 10 --keep 5 --report

    # Upload results to S3
    python main.py --games 10 --s3-bucket my-bucket --report
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

_DEFAULT_KEEP = 10
_DEFAULT_S3_PREFIX = "results"


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
    parser.add_argument(
        "--keep",
        type=int,
        default=_DEFAULT_KEEP,
        metavar="N",
        help=f"Keep only the N most-recent runs per algorithm (default: {_DEFAULT_KEEP}). "
             "Pass 0 to disable pruning.",
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
            timestamp=timestamp,
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
