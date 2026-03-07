"""Tests for src/storage.py (local helpers) and src/report.py."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from main import prune_local_results, parse_args
from src.report import generate_html_report


# ---------------------------------------------------------------------------
# prune_local_results
# ---------------------------------------------------------------------------


class TestPruneLocalResults:
    """Unit tests for the local result-pruning helper."""

    def _make_run(self, tmp_path: pathlib.Path, stem: str) -> None:
        """Create a dummy CSV + PNG pair for *stem* in *tmp_path*."""
        (tmp_path / f"{stem}.csv").write_text("game_index,score\n1,100")
        (tmp_path / f"{stem}.png").write_bytes(b"\x89PNG")

    def test_no_pruning_when_under_limit(self, tmp_path):
        self._make_run(tmp_path, "20260101_000000")
        self._make_run(tmp_path, "20260102_000000")
        deleted = prune_local_results(tmp_path, keep_n=5)
        assert deleted == []
        assert len(list(tmp_path.glob("*.csv"))) == 2

    def test_keeps_latest_n(self, tmp_path):
        for day in range(1, 6):  # 5 runs
            self._make_run(tmp_path, f"2026030{day}_120000")
        deleted = prune_local_results(tmp_path, keep_n=3)
        remaining_csvs = sorted(tmp_path.glob("*.csv"))
        assert len(remaining_csvs) == 3
        # The oldest two should be gone
        assert not (tmp_path / "20260301_120000.csv").exists()
        assert not (tmp_path / "20260302_120000.csv").exists()
        # The newest three should remain
        assert (tmp_path / "20260303_120000.csv").exists()
        assert (tmp_path / "20260304_120000.csv").exists()
        assert (tmp_path / "20260305_120000.csv").exists()

    def test_deletes_both_csv_and_png(self, tmp_path):
        for day in range(1, 4):
            self._make_run(tmp_path, f"2026{day:02d}01_000000")
        deleted = prune_local_results(tmp_path, keep_n=1)
        deleted_names = {p.name for p in deleted}
        assert "20260101_000000.csv" in deleted_names
        assert "20260101_000000.png" in deleted_names
        assert "20260201_000000.csv" in deleted_names
        assert "20260201_000000.png" in deleted_names

    def test_keep_zero_disables_pruning(self, tmp_path):
        for day in range(1, 6):
            self._make_run(tmp_path, f"20260{day:02d}01_000000")
        deleted = prune_local_results(tmp_path, keep_n=0)
        assert deleted == []
        assert len(list(tmp_path.glob("*.csv"))) == 5

    def test_exactly_keep_n_files_unchanged(self, tmp_path):
        for day in range(1, 4):
            self._make_run(tmp_path, f"20260{day:02d}01_000000")
        deleted = prune_local_results(tmp_path, keep_n=3)
        assert deleted == []
        assert len(list(tmp_path.glob("*.csv"))) == 3

    def test_returns_list_of_deleted_paths(self, tmp_path):
        for day in range(1, 4):
            self._make_run(tmp_path, f"2026{day:02d}01_000000")
        deleted = prune_local_results(tmp_path, keep_n=1)
        assert all(isinstance(p, pathlib.Path) for p in deleted)
        assert len(deleted) == 4  # 2 stems × 2 files each


# ---------------------------------------------------------------------------
# generate_html_report  (dashboard tests)
# ---------------------------------------------------------------------------


class TestGenerateHtmlReport:
    """Unit tests for the HTML report generator."""

    def _make_csv(self, directory: pathlib.Path, stem: str, n: int = 3) -> None:
        rows = ["run_id,game_index,algorithm,score,best_tile,moves,duration,won,timestamp"]
        for i in range(1, n + 1):
            rows.append(
                f"{stem},{i},Random,{i * 100},{2 ** (i + 2)},{50 * i},{i * 0.5},False,"
                f"2026-03-07T12:00:0{i}Z"
            )
        (directory / f"{stem}.csv").write_text("\n".join(rows))

    def test_creates_html_file(self, tmp_path):
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        assert report.exists()
        assert report.suffix == ".html"

    def test_html_contains_algorithm_name(self, tmp_path):
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Random" in content

    def test_html_is_valid_structure(self, tmp_path):
        results_dir = tmp_path / "results"
        (results_dir / "Random").mkdir(parents=True)
        self._make_csv(results_dir / "Random", "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    def test_empty_results_dir_produces_placeholder(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "No results found" in content

    def test_missing_results_dir_produces_placeholder(self, tmp_path):
        report = generate_html_report(
            tmp_path / "nonexistent", tmp_path / "index.html"
        )
        content = report.read_text(encoding="utf-8")
        assert "No results found" in content

    def test_multiple_algorithms_shown(self, tmp_path):
        results_dir = tmp_path / "results"
        for algo in ("AlgoA", "AlgoB"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "AlgoA" in content
        assert "AlgoB" in content

    def test_report_embeds_chart_when_png_present(self, tmp_path):
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_120000")
        # Write a tiny fake PNG (just needs to be non-empty bytes)
        (algo_dir / "20260307_120000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_output_path_parent_created(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        nested_output = tmp_path / "deep" / "dir" / "report.html"

        generate_html_report(results_dir, nested_output)
        assert nested_output.exists()

    # ------------------------------------------------------------------
    # Run-history accordion tests
    # ------------------------------------------------------------------

    def test_run_history_shows_all_runs(self, tmp_path):
        """Every stored run should appear as an accordion item."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        stems = ["20260307_100000", "20260307_110000", "20260307_120000"]
        for stem in stems:
            self._make_csv(algo_dir, stem)

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        for stem in stems:
            assert stem in content, f"Expected run {stem} in the report"

    def test_latest_run_is_pre_opened(self, tmp_path):
        """The most recent run's <details> must have the `open` attribute."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_110000")
        self._make_csv(algo_dir, "20260307_120000")  # latest

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        # The latest run's anchor id should appear in an open <details>
        assert 'id="run-20260307_120000"' in content
        # The details element for the latest run should carry `open`
        assert 'id="run-20260307_120000" open' in content

    def test_older_runs_are_collapsed(self, tmp_path):
        """Older runs must NOT have the `open` attribute on their <details>."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_110000")
        self._make_csv(algo_dir, "20260307_120000")  # latest

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert 'id="run-20260307_110000" open' not in content

    def test_latest_chip_present(self, tmp_path):
        """The most recent run should carry a 'latest' chip in its summary."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_110000")
        self._make_csv(algo_dir, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "chip-latest" in content

    def test_algo_nav_bar_rendered(self, tmp_path):
        """A navigation bar with links to each algorithm should be present."""
        results_dir = tmp_path / "results"
        for algo in ("AlgoA", "AlgoB"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert 'class="algo-nav"' in content
        assert "#algo-algoa" in content
        assert "#algo-algob" in content

    def test_run_history_heading_present(self, tmp_path):
        """Each algorithm section should include a 'Run History' heading."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Run History" in content

    # ------------------------------------------------------------------
    # Algorithm comparison section tests
    # ------------------------------------------------------------------

    def test_comparison_section_present_with_multiple_algos(self, tmp_path):
        """Comparison section must appear when two or more algorithms have results."""
        results_dir = tmp_path / "results"
        for algo in ("Random", "Greedy"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Algorithm Comparison" in content
        assert 'class="cmp-section"' in content

    def test_comparison_section_shows_all_algorithm_names(self, tmp_path):
        """Every algorithm name should appear inside the comparison table."""
        results_dir = tmp_path / "results"
        for algo in ("Random", "Greedy"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Random" in content
        assert "Greedy" in content

    def test_comparison_section_absent_for_single_algo(self, tmp_path):
        """Comparison section must NOT be rendered when only one algorithm exists."""
        results_dir = tmp_path / "results"
        algo_dir = results_dir / "Random"
        algo_dir.mkdir(parents=True)
        self._make_csv(algo_dir, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert 'class="cmp-section"' not in content

    def test_comparison_highlights_best_values(self, tmp_path):
        """Cells with the best metric value should carry the cmp-best CSS class."""
        results_dir = tmp_path / "results"
        # AlgoB has superior metrics, so its cells should be highlighted
        algo_a_dir = results_dir / "AlgoA"
        algo_a_dir.mkdir(parents=True)
        rows_a = ["run_id,game_index,algorithm,score,best_tile,moves,duration,won,timestamp"]
        rows_a.append("run1,1,AlgoA,100,4,10,0.5,False,2026-03-07T12:00:00Z")
        (algo_a_dir / "20260307_120000.csv").write_text("\n".join(rows_a))

        algo_b_dir = results_dir / "AlgoB"
        algo_b_dir.mkdir(parents=True)
        rows_b = ["run_id,game_index,algorithm,score,best_tile,moves,duration,won,timestamp"]
        rows_b.append("run1,1,AlgoB,9999,2048,500,5.0,True,2026-03-07T12:00:01Z")
        (algo_b_dir / "20260307_120000.csv").write_text("\n".join(rows_b))

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert 'class="cmp-best"' in content


# ---------------------------------------------------------------------------
# --parallel argument parsing
# ---------------------------------------------------------------------------


class TestParallelArgParsing:
    """Unit tests for the new --parallel CLI argument."""

    def test_default_parallel_is_one(self):
        args = parse_args([])
        assert args.parallel == 1

    def test_parallel_flag_sets_value(self):
        args = parse_args(["--parallel", "4"])
        assert args.parallel == 4

    def test_parallel_two_workers(self):
        args = parse_args(["--games", "10", "--parallel", "2"])
        assert args.parallel == 2
        assert args.games == 10


# ---------------------------------------------------------------------------
# S3 helpers (mocked – no real AWS calls)
# ---------------------------------------------------------------------------


class TestStorageHelpers:
    """Lightweight tests for S3 helpers using mocked boto3."""

    def test_upload_file_returns_url(self, tmp_path):
        from src.storage import upload_file

        dummy_file = tmp_path / "test.csv"
        dummy_file.write_text("a,b\n1,2")

        mock_client = MagicMock()
        mock_client.get_bucket_location.return_value = {"LocationConstraint": "us-east-1"}

        with patch("src.storage.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            # Patch _HAS_BOTO3 so the guard passes
            with patch("src.storage._HAS_BOTO3", True):
                url = upload_file(dummy_file, "my-bucket", "results/test.csv")

        assert "my-bucket" in url
        assert "results/test.csv" in url

    def test_prune_s3_results_deletes_old_stems(self, tmp_path):
        from src.storage import prune_s3_results

        stems = [f"2026010{i}_000000" for i in range(1, 6)]  # 5 stems

        mock_client = MagicMock()
        # list_objects_v2 returns 10 objects (csv + png per stem)
        mock_client.get_paginator.return_value.paginate.return_value = [
            {
                "Contents": [
                    {"Key": f"results/Random/{stem}.csv"}
                    for stem in stems
                ]
                + [
                    {"Key": f"results/Random/{stem}.png"}
                    for stem in stems
                ]
            }
        ]

        with patch("src.storage.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            with patch("src.storage._HAS_BOTO3", True):
                deleted = prune_s3_results("bucket", "results", "Random", keep_n=3)

        # Should delete the 2 oldest stems × 2 files = 4 keys
        assert len(deleted) == 4
        # The 2 oldest stems should be in the deleted list
        for ext in ("csv", "png"):
            assert f"results/Random/20260101_000000.{ext}" in deleted
            assert f"results/Random/20260102_000000.{ext}" in deleted

    def test_require_boto3_raises_when_missing(self):
        from src.storage import _require_boto3

        with patch("src.storage._HAS_BOTO3", False):
            with pytest.raises(ImportError, match="boto3"):
                _require_boto3()



# ---------------------------------------------------------------------------
# --mode argument parsing
# ---------------------------------------------------------------------------


class TestModeArgParsing:
    """Unit tests for the --mode CLI argument."""

    def test_default_mode_is_none(self):
        args = parse_args([])
        assert args.mode is None

    def test_mode_dev_sets_presets(self):
        args = parse_args(["--mode", "dev"])
        assert args.games == 100
        assert args.runs == 1

    def test_mode_release_sets_presets(self):
        args = parse_args(["--mode", "release"])
        assert args.games == 1000
        assert args.runs == 1

    def test_mode_benchmark_sets_presets(self):
        args = parse_args(["--mode", "benchmark"])
        assert args.games == 500
        assert args.runs == 5

    def test_explicit_games_overrides_mode(self):
        args = parse_args(["--mode", "dev", "--games", "42"])
        assert args.games == 42

    def test_explicit_runs_overrides_mode(self):
        args = parse_args(["--mode", "benchmark", "--runs", "2"])
        assert args.runs == 2

    def test_explicit_parallel_overrides_mode(self):
        args = parse_args(["--mode", "dev", "--parallel", "3"])
        assert args.parallel == 3

    def test_default_games_without_mode(self):
        args = parse_args([])
        assert args.games == 20

    def test_default_runs_without_mode(self):
        args = parse_args([])
        assert args.runs == 1

    def test_invalid_mode_raises(self):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "invalid"])


# ---------------------------------------------------------------------------
# Visualize metrics / distribution functions
# ---------------------------------------------------------------------------


class TestComputeRunMetrics:
    """Unit tests for compute_run_metrics() in visualize.py."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "run_id": ["r1"] * 4,
            "game_index": [1, 2, 3, 4],
            "algorithm": ["Random"] * 4,
            "score": [100, 200, 300, 400],
            "best_tile": [64, 128, 256, 512],
            "moves": [50, 100, 150, 200],
            "duration": [1.0, 2.0, 3.0, 4.0],
            "won": [False, False, False, True],
            "timestamp": ["2026-03-07T12:00:00Z"] * 4,
        })

    def test_avg_score(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["avg_score"] == pytest.approx(250.0)

    def test_median_score(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["median_score"] == pytest.approx(250.0)

    def test_p90_score(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["p90_score"] == pytest.approx(370.0)

    def test_max_score(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["max_score"] == 400

    def test_avg_moves(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["avg_moves"] == pytest.approx(125.0)

    def test_avg_duration(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["avg_duration"] == pytest.approx(2.5)

    def test_avg_best_tile(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["avg_best_tile"] == pytest.approx(240.0)

    def test_win_rate(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        assert m["win_rate"] == pytest.approx(25.0)

    def test_games_per_second(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        # 4 games / 10.0 total seconds = 0.4
        assert m["games_per_second"] == pytest.approx(0.4)

    def test_all_keys_present(self):
        from src.visualize import compute_run_metrics
        m = compute_run_metrics(self._make_df())
        expected_keys = {
            "avg_score", "median_score", "p90_score", "max_score",
            "avg_moves", "avg_duration", "avg_best_tile", "win_rate",
            "games_per_second",
        }
        assert set(m.keys()) == expected_keys


class TestDistributions:
    """Unit tests for score/moves/tile distribution functions."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "run_id": ["r1"] * 5,
            "game_index": [1, 2, 3, 4, 5],
            "algorithm": ["Random"] * 5,
            "score": [100, 200, 300, 400, 500],
            "best_tile": [64, 128, 128, 256, 512],
            "moves": [50, 100, 150, 200, 250],
            "duration": [1.0, 2.0, 3.0, 4.0, 5.0],
            "won": [False, False, False, False, True],
            "timestamp": ["2026-03-07T12:00:00Z"] * 5,
        })

    def test_tile_distribution_returns_dict(self):
        from src.visualize import tile_distribution
        d = tile_distribution(self._make_df())
        assert isinstance(d, dict)

    def test_tile_distribution_sums_to_n_games(self):
        from src.visualize import tile_distribution
        d = tile_distribution(self._make_df())
        assert sum(d.values()) == 5

    def test_tile_distribution_counts_correct(self):
        from src.visualize import tile_distribution
        d = tile_distribution(self._make_df())
        assert d[64] == 1
        assert d[128] == 2
        assert d[256] == 1
        assert d[512] == 1

    def test_score_distribution_returns_dict(self):
        from src.visualize import score_distribution
        d = score_distribution(self._make_df())
        assert isinstance(d, dict)

    def test_score_distribution_sums_to_n_games(self):
        from src.visualize import score_distribution
        d = score_distribution(self._make_df(), bins=5)
        assert sum(d.values()) == 5

    def test_moves_distribution_returns_dict(self):
        from src.visualize import moves_distribution
        d = moves_distribution(self._make_df())
        assert isinstance(d, dict)

    def test_moves_distribution_sums_to_n_games(self):
        from src.visualize import moves_distribution
        d = moves_distribution(self._make_df(), bins=5)
        assert sum(d.values()) == 5


# ---------------------------------------------------------------------------
# Dashboard comparison table (new Median/P90/Max columns)
# ---------------------------------------------------------------------------


class TestComparisonTableColumns:
    """Verify the new comparison table headers are rendered."""

    def _make_csv(self, directory: pathlib.Path, stem: str) -> None:
        rows = ["run_id,game_index,algorithm,score,best_tile,moves,duration,won,timestamp"]
        rows.append(f"{stem},1,AlgoA,500,128,100,2.0,False,2026-03-07T12:00:00Z")
        (directory / f"{stem}.csv").write_text("\n".join(rows))

    def test_comparison_table_has_median_column(self, tmp_path):
        results_dir = tmp_path / "results"
        for algo in ("AlgoA", "AlgoB"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Median" in content

    def test_comparison_table_has_p90_column(self, tmp_path):
        results_dir = tmp_path / "results"
        for algo in ("AlgoA", "AlgoB"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "P90" in content

    def test_comparison_table_has_max_column(self, tmp_path):
        results_dir = tmp_path / "results"
        for algo in ("AlgoA", "AlgoB"):
            d = results_dir / algo
            d.mkdir(parents=True)
            self._make_csv(d, "20260307_120000")

        report = generate_html_report(results_dir, tmp_path / "index.html")
        content = report.read_text(encoding="utf-8")
        assert "Max" in content
