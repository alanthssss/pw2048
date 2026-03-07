"""Generate a self-contained HTML results dashboard for pw2048.

The produced page is fully self-contained: chart images are embedded as
base64 data URIs so the file works both locally (file://) and when hosted
on S3.

Layout
------
  ┌─ Header ─────────────────────────────────────────────────────────────┐
  │  🎮 pw2048 Results Dashboard              Generated: …  N algo(s)   │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Algo Nav (sticky) ──────────────────────────────────────────────────┐
  │  [● Random]  [● AlgoB]  …                                            │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Algorithm Section ──────────────────────────────────────────────────┐
  │  ┌── Summary stats cards (aggregated across all stored runs) ──────┐ │
  │  └──────────────────────────────────────────────────────────────────┘ │
  │  ┌── Run History ──────────────────────────────────────────────────┐ │
  │  │  ▼ 20260307 130000  20 games  avg 2,399  best 512  10% wins  ◀latest
  │  │    [Chart] [Per-game table]                                      │ │
  │  │  ▶ 20260307 120000  20 games  avg 2,250  best 256   5% wins     │ │
  │  │  ▶ …                                                             │ │
  │  └──────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# ---------------------------------------------------------------------------
# 2048-inspired colour palette
# ---------------------------------------------------------------------------
_PALETTE = {
    "bg": "#faf8ef",
    "header_bg": "#bbada0",
    "header_fg": "#f9f6f2",
    "nav_bg": "#cdc1b4",
    "nav_fg": "#f9f6f2",
    "nav_active": "#f59563",
    "card_bg": "#eee4da",
    "card_accent": "#f59563",
    "text_dark": "#776e65",
    "text_light": "#f9f6f2",
    "table_even": "#f3ede4",
    "win_badge": "#8ec07c",
    "lose_badge": "#cc241d",
    "run_summary_bg": "#f3ede4",
    "run_summary_hover": "#e8dfd3",
    "run_latest_bg": "#fcefd8",
    "details_border": "#d3c4b4",
    "section_border": "#eee4da",
}

_CSS = """\
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: "Clear Sans", "Helvetica Neue", Arial, sans-serif;
    background: {bg};
    color: {text_dark};
    line-height: 1.5;
}}

/* ── Header ────────────────────────────────────────────────────────── */
header {{
    background: {header_bg};
    color: {header_fg};
    padding: 24px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}}
header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }}
header .meta {{ font-size: 0.85rem; opacity: 0.85; text-align: right; }}

/* ── Sticky algo nav bar ────────────────────────────────────────────── */
.algo-nav {{
    background: {nav_bg};
    position: sticky;
    top: 0;
    z-index: 100;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    border-bottom: 2px solid rgba(0,0,0,.08);
}}
.algo-nav .nav-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: {nav_fg};
    opacity: .7;
    margin-right: 4px;
    flex-shrink: 0;
}}
.algo-nav a {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,.18);
    color: {nav_fg};
    text-decoration: none;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    transition: background .15s;
}}
.algo-nav a:hover {{ background: rgba(255,255,255,.32); }}
.algo-nav a .nav-dot {{
    width: 8px; height: 8px;
    background: {nav_active};
    border-radius: 50%;
    display: inline-block;
}}

/* ── Main layout ────────────────────────────────────────────────────── */
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }}
.algo-section {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 48px;
    overflow: hidden;
    scroll-margin-top: 56px;  /* account for sticky nav */
}}
.algo-section-header {{
    background: {header_bg};
    color: {header_fg};
    padding: 14px 24px;
    font-size: 1.3rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.algo-section-body {{ padding: 24px; }}

/* ── Aggregate stats cards ─────────────────────────────────────────── */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
}}
.stat-card {{
    background: {card_bg};
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
}}
.stat-card .stat-label {{
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    opacity: .7;
    margin-bottom: 4px;
}}
.stat-card .stat-value {{
    font-size: 1.55rem;
    font-weight: 700;
    color: {card_accent};
}}

/* ── Run history section title ──────────────────────────────────────── */
.runs-heading {{
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: {text_dark};
    opacity: .65;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.runs-heading::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: {section_border};
}}

/* ── Run accordion (details/summary) ───────────────────────────────── */
.run-item {{
    border: 1px solid {details_border};
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}}
.run-item[open] {{
    border-color: {card_accent};
    box-shadow: 0 0 0 2px rgba(245,149,99,.2);
}}
.run-summary {{
    list-style: none;
    padding: 12px 16px;
    background: {run_summary_bg};
    cursor: pointer;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    user-select: none;
}}
.run-item[open] > .run-summary {{
    background: {run_latest_bg};
    border-bottom: 1px solid {details_border};
}}
.run-summary:hover {{
    background: {run_summary_hover};
}}
.run-summary::-webkit-details-marker {{ display: none; }}
.run-arrow {{
    font-size: 0.75rem;
    transition: transform .2s;
    flex-shrink: 0;
    width: 16px;
    text-align: center;
}}
.run-item[open] .run-arrow {{ transform: rotate(90deg); }}
.run-ts {{
    font-size: 0.92rem;
    font-weight: 700;
    color: {text_dark};
    flex-shrink: 0;
}}
.run-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-left: auto;
}}
.chip {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(119,110,101,.12);
    color: {text_dark};
}}
.chip-latest {{
    background: {card_accent};
    color: white;
}}

/* ── Expanded run body ──────────────────────────────────────────────── */
.run-body {{ padding: 20px 20px 16px; }}
.chart-wrap {{ text-align: center; margin-bottom: 20px; }}
.chart-wrap img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,.12);
}}

/* ── Per-game results table ─────────────────────────────────────────── */
.results-table {{ overflow-x: auto; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
}}
thead th {{
    background: {header_bg};
    color: {header_fg};
    padding: 9px 13px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}}
tbody tr:nth-child(even) {{ background: {table_even}; }}
tbody td {{ padding: 8px 13px; }}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    color: white;
}}
.badge-win  {{ background: {win_badge};  }}
.badge-lose {{ background: {lose_badge}; }}
.tile-chip {{
    display: inline-block;
    min-width: 46px;
    padding: 2px 7px;
    border-radius: 4px;
    text-align: center;
    font-weight: 700;
    font-size: 0.83rem;
}}

/* ── Algorithm comparison table ─────────────────────────────────── */
.cmp-section {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 48px;
    overflow: hidden;
}}
.cmp-header {{
    background: {header_bg};
    color: {header_fg};
    padding: 14px 24px;
    font-size: 1.3rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.cmp-body {{ padding: 24px; }}
.cmp-algo {{
    font-weight: 700;
    color: {text_dark};
}}
.cmp-best {{
    font-weight: 700;
    color: {card_accent};
}}

/* ── Inline summary charts ──────────────────────────────────────── */
.summary-charts {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.summary-chart-card {{
    background: {card_bg};
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}}
.summary-chart-card .chart-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    opacity: .7;
    margin-bottom: 8px;
}}
.summary-chart-card img {{
    max-width: 100%;
    border-radius: 4px;
}}

/* ── Footer ────────────────────────────────────────────────────────── */
footer {{
    text-align: center;
    font-size: 0.8rem;
    color: {text_dark};
    opacity: 0.55;
    padding: 16px;
    border-top: 1px solid {section_border};
    margin-top: 8px;
}}

/* ── Hero summary cards ──────────────────────────────────────────── */
.hero-section {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 40px;
}}
.hero-card {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    padding: 20px 24px;
    text-align: center;
    border-top: 4px solid {card_accent};
}}
.hero-card .hero-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    opacity: .65;
    margin-bottom: 8px;
}}
.hero-card .hero-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {card_accent};
}}
.hero-card .hero-sub {{
    font-size: 0.8rem;
    opacity: .7;
    margin-top: 4px;
}}

/* ── Board sections (leaderboard / stability / efficiency) ─────────── */
.board-section {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 48px;
    overflow: hidden;
}}
.board-header {{
    background: {header_bg};
    color: {header_fg};
    padding: 14px 24px;
    font-size: 1.3rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.board-body {{ padding: 24px; }}

/* ── Stage badge ─────────────────────────────────────────────────── */
.stage-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
}}
.stage-benchmark {{ background: {card_accent}; color: white; }}
.stage-release   {{ background: {win_badge}; color: white; }}
.stage-dev       {{ background: {nav_bg}; color: {nav_fg}; }}
.stage-custom    {{ background: {card_bg}; color: {text_dark}; }}

/* ── Rank cells ──────────────────────────────────────────────────── */
.rank-1 {{ font-weight: 700; color: {card_accent}; }}
.rank-2 {{ font-weight: 700; color: {header_bg}; }}
.rank-3 {{ font-weight: 600; color: {text_dark}; }}

/* ── Run metadata box ────────────────────────────────────────────── */
.run-meta-box {{
    background: {card_bg};
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 8px 16px;
    font-size: 0.82rem;
}}
.run-meta-item .meta-key {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    opacity: .6;
    margin-bottom: 2px;
}}
.run-meta-item .meta-val {{
    font-weight: 600;
    color: {text_dark};
    word-break: break-all;
}}

/* ── Global comparison charts grid ──────────────────────────────── */
.global-charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
}}
.global-chart-card {{
    background: {card_bg};
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}}
.global-chart-card .chart-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    opacity: .7;
    margin-bottom: 8px;
}}
.global-chart-card img {{
    max-width: 100%;
    border-radius: 4px;
}}

/* ── Responsive ────────────────────────────────────────────────────── */
@media (max-width: 600px) {{
    header h1 {{ font-size: 1.35rem; }}
    .stat-card .stat-value {{ font-size: 1.2rem; }}
    .run-chips {{ display: none; }}
}}
""".format(**_PALETTE)


# Map tile values to background colours (matching the game's palette).
_TILE_COLORS: dict[int, tuple[str, str]] = {
    2:    ("#eee4da", "#776e65"),
    4:    ("#ede0c8", "#776e65"),
    8:    ("#f2b179", "#f9f6f2"),
    16:   ("#f59563", "#f9f6f2"),
    32:   ("#f67c5f", "#f9f6f2"),
    64:   ("#f65e3b", "#f9f6f2"),
    128:  ("#edcf72", "#f9f6f2"),
    256:  ("#edcc61", "#f9f6f2"),
    512:  ("#edc850", "#f9f6f2"),
    1024: ("#edc53f", "#f9f6f2"),
    2048: ("#edc22e", "#f9f6f2"),
}


def _tile_chip(value: int) -> str:
    bg, fg = _TILE_COLORS.get(value, ("#cdc1b4", "#776e65"))
    return (
        f'<span class="tile-chip" style="background:{bg};color:{fg}">'
        f"{value}</span>"
    )


def _embed_image(path: Path) -> str | None:
    """Return a ``data:image/png;base64,...`` URI for *path*, or ``None`` if missing."""
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def _fig_to_data_uri(fig: plt.Figure) -> str:
    """Render *fig* to a PNG and return a ``data:image/png;base64,...`` URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


def _tile_dist_chart_src(df_all: pd.DataFrame) -> str | None:
    """Generate an inline tile-distribution bar chart and return a data URI.

    Shows best-tile value vs number of games, aggregated across all runs.
    Returns ``None`` if the DataFrame is empty.
    """
    if df_all.empty:
        return None
    try:
        tile_counts = df_all["best_tile"].value_counts().sort_index()
        labels = [str(t) for t in tile_counts.index]
        values = tile_counts.values

        # Assign tile colours from the game palette
        bar_colors = [
            _TILE_COLORS.get(int(t), ("#cdc1b4", "#776e65"))[0]
            for t in tile_counts.index
        ]

        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_title("Tile Distribution", fontsize=10, color="#776e65")
        ax.set_xlabel("Best Tile", fontsize=8, color="#776e65")
        ax.set_ylabel("Games", fontsize=8, color="#776e65")
        ax.tick_params(colors="#776e65", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                str(int(val)),
                ha="center", va="bottom", fontsize=7, color="#776e65",
            )
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _run_stability_chart_src(run_dirs: list[Path]) -> str | None:
    """Generate an inline run-stability chart (run_id vs avg_score) and return a data URI.

    Only produced when there are 2 or more runs.  Returns ``None`` otherwise.
    """
    if len(run_dirs) < 2:
        return None
    try:
        labels: list[str] = []
        avgs: list[float] = []
        for d in run_dirs:
            csv_path = d / "results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or "score" not in df.columns:
                continue
            # Use the timestamp part of the dir name (strip "run_" prefix) for labels
            labels.append(d.name.removeprefix("run_"))
            avgs.append(float(df["score"].mean()))

        if len(avgs) < 2:
            return None

        fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.9 + 1), 3))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")
        ax.plot(range(len(labels)), avgs, marker="o", color="#f59563",
                linewidth=2, markersize=5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(
            [lb.replace("_", "\n") for lb in labels],
            fontsize=6, color="#776e65",
        )
        ax.set_title("Run Stability (Avg Score per Run)", fontsize=10, color="#776e65")
        ax.set_xlabel("Run ID", fontsize=8, color="#776e65")
        ax.set_ylabel("Avg Score", fontsize=8, color="#776e65")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.tick_params(axis="y", colors="#776e65", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Leaderboard data collection
# ---------------------------------------------------------------------------

def _collect_leaderboard_data(algo_dirs: list[Path]) -> list[dict]:
    """Collect aggregate metrics for every algorithm directory.

    Returns a list of dicts (one per algorithm) sorted by avg_score descending.
    Each dict contains all fields needed for the leaderboard, stability, and
    efficiency sections.
    """
    rows: list[dict] = []
    for d in algo_dirs:
        run_dirs = sorted(
            rd for rd in d.iterdir()
            if rd.is_dir() and rd.name.startswith("run_")
        ) if d.exists() else []
        if not run_dirs:
            continue

        frames: list[pd.DataFrame] = []
        run_avgs: list[float] = []
        run_medians: list[float] = []
        run_p90s: list[float] = []

        for rd in run_dirs:
            csv_path = rd / "results.csv"
            try:
                df_run = pd.read_csv(csv_path)
                frames.append(df_run)
                if not df_run.empty and "score" in df_run.columns:
                    run_avgs.append(float(df_run["score"].mean()))
                    run_medians.append(float(df_run["score"].median()))
                    run_p90s.append(float(df_run["score"].quantile(0.9)))
            except Exception:
                continue

        if not frames:
            continue

        df_all = pd.concat(frames, ignore_index=True)

        # Derive stage from the most-recent run's metrics.json
        stage = "custom"
        for rd in reversed(run_dirs):
            meta_path = rd / "metrics.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    stage = meta.get("mode") or "custom"
                    break
                except Exception:
                    pass

        avg_duration = (
            float(df_all["duration"].mean()) if "duration" in df_all.columns else 0.0
        )

        rows.append(
            {
                "name": d.name,
                "stage": stage,
                "runs": len(run_dirs),
                "total_games": len(df_all),
                "avg_score": float(df_all["score"].mean()),
                "median_score": float(df_all["score"].median()),
                "p90_score": float(df_all["score"].quantile(0.9)),
                "max_score": float(df_all["score"].max()),
                "best_tile": int(df_all["best_tile"].max()),
                "win_rate": float(df_all["won"].mean()) * 100,
                "avg_moves": (
                    float(df_all["moves"].mean()) if "moves" in df_all.columns else 0.0
                ),
                "avg_duration": avg_duration,
                "games_per_second": (1.0 / avg_duration) if avg_duration > 0 else 0.0,
                # Per-run stability arrays
                "run_avgs": run_avgs,
                "run_medians": run_medians,
                "run_p90s": run_p90s,
                "mean_avg_score": (
                    float(pd.Series(run_avgs).mean()) if run_avgs else 0.0
                ),
                "std_avg_score": (
                    float(pd.Series(run_avgs).std()) if len(run_avgs) > 1 else 0.0
                ),
                "mean_median_score": (
                    float(pd.Series(run_medians).mean()) if run_medians else 0.0
                ),
                "mean_p90_score": (
                    float(pd.Series(run_p90s).mean()) if run_p90s else 0.0
                ),
            }
        )

    rows.sort(key=lambda r: r["avg_score"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Hero, Leaderboard, Stability, Efficiency sections
# ---------------------------------------------------------------------------

def _stage_badge(stage: str) -> str:
    """Return an HTML stage badge element."""
    cls = f"stage-{html.escape(stage.lower())}"
    return f'<span class="stage-badge {cls}">{html.escape(stage)}</span>'


def _hero_section(rows_data: list[dict]) -> str:
    """Build the top hero summary cards (best avg, best tile, most stable, fastest)."""
    if not rows_data:
        return ""

    best_avg_row = max(rows_data, key=lambda r: r["avg_score"])
    best_tile_row = max(rows_data, key=lambda r: r["best_tile"])

    stable_rows = [r for r in rows_data if len(r["run_avgs"]) >= 2]
    most_stable = (
        min(stable_rows, key=lambda r: r["std_avg_score"]) if stable_rows else None
    )
    fastest = (
        max(rows_data, key=lambda r: r["games_per_second"]) if rows_data else None
    )

    def _card(label: str, value: str, sub: str) -> str:
        return (
            f'<div class="hero-card">'
            f'<div class="hero-label">{label}</div>'
            f'<div class="hero-value">{value}</div>'
            f'<div class="hero-sub">{html.escape(sub)}</div>'
            f"</div>"
        )

    cards = [
        _card("Best Avg Score", f"{best_avg_row['avg_score']:,.0f}", best_avg_row["name"]),
        _card("Highest Best Tile", str(best_tile_row["best_tile"]), best_tile_row["name"]),
    ]
    if most_stable:
        cards.append(
            _card(
                "Most Stable",
                html.escape(most_stable["name"]),
                f"\u03c3={most_stable['std_avg_score']:,.0f}",
            )
        )
    if fastest and fastest["games_per_second"] > 0:
        cards.append(
            _card(
                "Fastest",
                html.escape(fastest["name"]),
                f"{fastest['games_per_second']:.2f} games/s",
            )
        )

    return f'<div class="hero-section">{"".join(cards)}</div>'


def _leaderboard_section(rows_data: list[dict]) -> str:
    """Build the main leaderboard table sorted by avg_score."""
    if not rows_data:
        return ""

    rows_html: list[str] = []
    for i, r in enumerate(rows_data, 1):
        rank_cls = f' class="rank-{i}"' if i <= 3 else ""
        rows_html.append(
            "<tr>"
            f"<td{rank_cls}>{i}</td>"
            f"<td><strong>{html.escape(r['name'])}</strong></td>"
            f"<td>{_stage_badge(r['stage'])}</td>"
            f"<td>{r['runs']}</td>"
            f"<td>{r['total_games']:,}</td>"
            f"<td><strong>{r['avg_score']:,.0f}</strong></td>"
            f"<td>{r['median_score']:,.0f}</td>"
            f"<td>{r['p90_score']:,.0f}</td>"
            f"<td>{r['max_score']:,.0f}</td>"
            f"<td>{_tile_chip(r['best_tile'])}</td>"
            f"<td>{r['win_rate']:.1f}%</td>"
            f"<td>{r['avg_moves']:.0f}</td>"
            f"<td>{r['avg_duration']:.1f}s</td>"
            "</tr>"
        )

    rows_html_str = "\n".join(rows_html)
    return f"""\
<section class="board-section" id="leaderboard">
  <div class="board-header">🏆 Main Leaderboard</div>
  <div class="board-body">
    <div class="results-table">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Algorithm</th>
            <th>Stage</th>
            <th>Runs</th>
            <th>Total Games</th>
            <th>Avg Score</th>
            <th>Median</th>
            <th>P90</th>
            <th>Max Score</th>
            <th>Best Tile</th>
            <th>Win Rate</th>
            <th>Avg Moves</th>
            <th>Avg Duration</th>
          </tr>
        </thead>
        <tbody>
{rows_html_str}
        </tbody>
      </table>
    </div>
  </div>
</section>"""


def _stability_section(rows_data: list[dict]) -> str:
    """Build the stability board table."""
    multi_run_rows = [r for r in rows_data if r["runs"] >= 2]
    if not multi_run_rows:
        return ""

    rows_html: list[str] = []
    for r in multi_run_rows:
        rows_html.append(
            "<tr>"
            f"<td><strong>{html.escape(r['name'])}</strong></td>"
            f"<td>{r['runs']}</td>"
            f"<td>{r['mean_avg_score']:,.0f}</td>"
            f"<td>{r['std_avg_score']:,.0f}</td>"
            f"<td>{r['mean_median_score']:,.0f}</td>"
            f"<td>{r['mean_p90_score']:,.0f}</td>"
            "</tr>"
        )

    rows_html_str = "\n".join(rows_html)
    return f"""\
<section class="board-section" id="stability">
  <div class="board-header">📈 Stability Board</div>
  <div class="board-body">
    <div class="results-table">
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Runs</th>
            <th>Mean Avg Score</th>
            <th>Std Avg Score</th>
            <th>Mean Median</th>
            <th>Mean P90</th>
          </tr>
        </thead>
        <tbody>
{rows_html_str}
        </tbody>
      </table>
    </div>
  </div>
</section>"""


def _efficiency_section(rows_data: list[dict]) -> str:
    """Build the efficiency board table."""
    if not rows_data:
        return ""

    rows_html: list[str] = []
    for r in rows_data:
        gps = f"{r['games_per_second']:.2f}" if r["games_per_second"] > 0 else "—"
        rows_html.append(
            "<tr>"
            f"<td><strong>{html.escape(r['name'])}</strong></td>"
            f"<td>{r['avg_score']:,.0f}</td>"
            f"<td>{r['avg_duration']:.2f}s</td>"
            f"<td>{gps}</td>"
            "</tr>"
        )

    rows_html_str = "\n".join(rows_html)
    return f"""\
<section class="board-section" id="efficiency">
  <div class="board-header">⚡ Efficiency Board</div>
  <div class="board-body">
    <div class="results-table">
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Avg Score</th>
            <th>Avg Duration</th>
            <th>Games / Second</th>
          </tr>
        </thead>
        <tbody>
{rows_html_str}
        </tbody>
      </table>
    </div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Global comparison chart generators
# ---------------------------------------------------------------------------

_CHART_COLORS = ["#f59563", "#edc22e", "#8ec07c", "#83a598", "#d3869b"]


def _avg_median_p90_chart_src(rows_data: list[dict]) -> str | None:
    """Grouped bar chart: avg, median, P90 per algorithm."""
    if not rows_data:
        return None
    try:
        import numpy as np  # already available via pandas

        names = [r["name"] for r in rows_data]
        avgs = [r["avg_score"] for r in rows_data]
        medians = [r["median_score"] for r in rows_data]
        p90s = [r["p90_score"] for r in rows_data]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.8 + 1), 4))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")

        ax.bar(x - width, avgs, width, label="Avg", color="#f59563", edgecolor="white")
        ax.bar(x, medians, width, label="Median", color="#edc22e", edgecolor="white")
        ax.bar(x + width, p90s, width, label="P90", color="#8ec07c", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, color="#776e65")
        ax.set_title("Score Comparison (Avg / Median / P90)", fontsize=10, color="#776e65")
        ax.set_ylabel("Score", fontsize=8, color="#776e65")
        ax.tick_params(colors="#776e65", labelsize=7)
        ax.legend(fontsize=7, framealpha=0.6)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _global_score_dist_chart_src(algo_dirs: list[Path]) -> str | None:
    """Overlaid score-distribution histograms for all algorithms."""
    algo_scores: dict[str, list[float]] = {}
    for d in algo_dirs:
        run_dirs = sorted(
            rd for rd in d.iterdir()
            if rd.is_dir() and rd.name.startswith("run_")
        ) if d.exists() else []
        frames: list[pd.DataFrame] = []
        for rd in run_dirs:
            try:
                frames.append(pd.read_csv(rd / "results.csv"))
            except Exception:
                continue
        if not frames:
            continue
        df_all = pd.concat(frames, ignore_index=True)
        if not df_all.empty and "score" in df_all.columns:
            algo_scores[d.name] = df_all["score"].tolist()

    if not algo_scores:
        return None
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")

        for i, (name, scores) in enumerate(algo_scores.items()):
            ax.hist(
                scores, bins=20, alpha=0.6,
                label=name, color=_CHART_COLORS[i % len(_CHART_COLORS)],
                edgecolor="white",
            )

        ax.set_title("Score Distribution", fontsize=10, color="#776e65")
        ax.set_xlabel("Score", fontsize=8, color="#776e65")
        ax.set_ylabel("Games", fontsize=8, color="#776e65")
        ax.tick_params(colors="#776e65", labelsize=7)
        if len(algo_scores) > 1:
            ax.legend(fontsize=7, framealpha=0.6)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _global_best_tile_chart_src(rows_data: list[dict]) -> str | None:
    """Best-tile bar chart comparing all algorithms."""
    if not rows_data:
        return None
    try:
        names = [r["name"] for r in rows_data]
        best_tiles = [r["best_tile"] for r in rows_data]
        bar_colors = [
            _TILE_COLORS.get(t, ("#cdc1b4", "#776e65"))[0] for t in best_tiles
        ]

        fig, ax = plt.subplots(figsize=(max(4, len(names) * 0.9 + 1), 3))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")
        bars = ax.bar(names, best_tiles, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_title("Best Tile by Algorithm", fontsize=10, color="#776e65")
        ax.set_xlabel("Algorithm", fontsize=8, color="#776e65")
        ax.set_ylabel("Best Tile", fontsize=8, color="#776e65")
        ax.tick_params(colors="#776e65", labelsize=7)
        top = max(best_tiles)
        for bar, val in zip(bars, best_tiles):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.01,
                str(val), ha="center", va="bottom", fontsize=8, color="#776e65",
            )
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _global_run_stability_chart_src(algo_dirs: list[Path]) -> str | None:
    """Run-stability line chart with one line per algorithm (needs ≥ 2 runs each)."""
    algo_run_avgs: dict[str, list[float]] = {}
    for d in algo_dirs:
        run_dirs = sorted(
            rd for rd in d.iterdir()
            if rd.is_dir() and rd.name.startswith("run_")
        ) if d.exists() else []
        avgs: list[float] = []
        for rd in run_dirs:
            try:
                df = pd.read_csv(rd / "results.csv")
                if not df.empty and "score" in df.columns:
                    avgs.append(float(df["score"].mean()))
            except Exception:
                continue
        if len(avgs) >= 2:
            algo_run_avgs[d.name] = avgs

    if not algo_run_avgs:
        return None
    try:
        max_runs = max(len(v) for v in algo_run_avgs.values())
        fig, ax = plt.subplots(figsize=(max(4, max_runs * 0.9 + 1), 3))
        fig.patch.set_facecolor("#faf8ef")
        ax.set_facecolor("#faf8ef")

        for i, (name, avgs) in enumerate(algo_run_avgs.items()):
            ax.plot(
                range(1, len(avgs) + 1), avgs,
                marker="o", color=_CHART_COLORS[i % len(_CHART_COLORS)],
                linewidth=2, markersize=5, label=name,
            )

        ax.set_title("Run Stability (Avg Score per Run)", fontsize=10, color="#776e65")
        ax.set_xlabel("Run #", fontsize=8, color="#776e65")
        ax.set_ylabel("Avg Score", fontsize=8, color="#776e65")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(colors="#776e65", labelsize=7)
        ax.legend(fontsize=7, framealpha=0.6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#d3c4b4")
        plt.tight_layout(pad=0.5)
        return _fig_to_data_uri(fig)
    except Exception:
        return None


def _global_comparison_charts(rows_data: list[dict], algo_dirs: list[Path]) -> str:
    """Build the global comparison charts section."""
    charts: list[str] = []

    src = _avg_median_p90_chart_src(rows_data)
    if src:
        charts.append(
            f'<div class="global-chart-card">'
            f'<div class="chart-title">Avg / Median / P90</div>'
            f'<img src="{src}" alt="Avg/Median/P90 comparison chart">'
            f"</div>"
        )

    src = _global_score_dist_chart_src(algo_dirs)
    if src:
        charts.append(
            f'<div class="global-chart-card">'
            f'<div class="chart-title">Score Distribution</div>'
            f'<img src="{src}" alt="Score distribution histogram">'
            f"</div>"
        )

    src = _global_best_tile_chart_src(rows_data)
    if src:
        charts.append(
            f'<div class="global-chart-card">'
            f'<div class="chart-title">Best Tile by Algorithm</div>'
            f'<img src="{src}" alt="Best tile distribution chart">'
            f"</div>"
        )

    src = _global_run_stability_chart_src(algo_dirs)
    if src:
        charts.append(
            f'<div class="global-chart-card">'
            f'<div class="chart-title">Run Stability</div>'
            f'<img src="{src}" alt="Global run stability chart">'
            f"</div>"
        )

    if not charts:
        return ""

    charts_html = "\n".join(charts)
    return f"""\
<section class="board-section" id="comparison-charts">
  <div class="board-header">📊 Comparison Charts</div>
  <div class="board-body">
    <div class="global-charts-grid">
{charts_html}
    </div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Run metadata box
# ---------------------------------------------------------------------------

def _run_metadata_box(run_dir: Path) -> str:
    """Build a metadata box for *run_dir* using ``metrics.json``, if present."""
    meta_path = run_dir / "metrics.json"
    if not meta_path.exists():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    items = [
        ("Algorithm Version", meta.get("algorithm_version", "—")),
        ("Mode", meta.get("mode", "—")),
        ("Games", str(meta.get("games", "—"))),
        ("Parallel Workers", str(meta.get("parallel_workers", "—"))),
        ("Timestamp", meta.get("timestamp", "—")),
        ("Git Commit", meta.get("git_commit", "—")),
    ]

    items_html = "\n".join(
        f'<div class="run-meta-item">'
        f'<div class="meta-key">{html.escape(k)}</div>'
        f'<div class="meta-val">{html.escape(str(v))}</div>'
        f"</div>"
        for k, v in items
    )
    return f'<div class="run-meta-box">{items_html}</div>'


def _stats_grid(df: pd.DataFrame) -> str:
    """Aggregate stats cards computed across *all* stored runs."""
    avg_score = df["score"].mean()
    best_score = df["score"].max()
    best_tile = df["best_tile"].max()
    win_rate = df["won"].mean() * 100
    n_games = len(df)
    return f"""\
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Total Games</div>
    <div class="stat-value">{n_games}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Best Score</div>
    <div class="stat-value">{best_score:,}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Score</div>
    <div class="stat-value">{avg_score:,.0f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Best Tile</div>
    <div class="stat-value">{best_tile}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate</div>
    <div class="stat-value">{win_rate:.1f}%</div>
  </div>
</div>"""


def _results_table(df: pd.DataFrame) -> str:
    """Render the per-game results as an HTML table."""
    rows: list[str] = []
    for _, row in df.iterrows():
        won_badge = (
            '<span class="badge badge-win">WIN</span>'
            if row["won"]
            else '<span class="badge badge-lose">LOSS</span>'
        )
        rows.append(
            f"<tr>"
            f"<td>{int(row['game_index'])}</td>"
            f"<td>{int(row['score']):,}</td>"
            f"<td>{_tile_chip(int(row['best_tile']))}</td>"
            f"<td>{int(row['moves'])}</td>"
            f"<td>{float(row['duration']):.1f}s</td>"
            f"<td>{won_badge}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)
    return f"""\
<div class="results-table">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Score</th>
        <th>Best Tile</th>
        <th>Moves</th>
        <th>Duration</th>
        <th>Result</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</div>"""


def _run_accordion_item(
    run_dir: Path,
    is_latest: bool,
) -> str:
    """Build one collapsible ``<details>`` accordion item for a single run.

    Parameters
    ----------
    run_dir:
        Run subdirectory (e.g. ``results/Random/run_20260307_120000/``).
        Must contain ``results.csv``; optionally ``chart.png``.
    is_latest:
        When ``True`` the item is rendered pre-opened and carries a "latest" badge.
    """
    csv_path = run_dir / "results.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return ""

    n = len(df)
    avg_score = df["score"].mean()
    best_tile = df["best_tile"].max()
    win_rate = df["won"].mean() * 100

    # Derive a display-friendly timestamp from the dir name (strip "run_" prefix)
    stem = run_dir.name.removeprefix("run_")
    ts_display = stem.replace("_", " ")
    anchor = f"run-{html.escape(stem)}"

    # Chips shown in the summary row
    latest_chip = '<span class="chip chip-latest">latest</span>' if is_latest else ""
    chips_html = (
        f'<div class="run-chips">'
        f'{latest_chip}'
        f'<span class="chip">{n} games</span>'
        f'<span class="chip">avg {avg_score:,.0f}</span>'
        f'<span class="chip">best tile {best_tile}</span>'
        f'<span class="chip">{win_rate:.0f}% wins</span>'
        f"</div>"
    )

    # Embedded chart (if chart.png exists inside the run dir)
    img_src = _embed_image(run_dir / "chart.png")
    chart_html = (
        f'<div class="chart-wrap"><img src="{img_src}" '
        f'alt="Results chart for {html.escape(stem)}"></div>'
        if img_src
        else ""
    )

    table_html = _results_table(df)
    meta_html = _run_metadata_box(run_dir)

    open_attr = " open" if is_latest else ""

    return f"""\
<details class="run-item" id="{anchor}"{open_attr}>
  <summary class="run-summary">
    <span class="run-arrow">▶</span>
    <span class="run-ts">📅 {html.escape(ts_display)}</span>
    {chips_html}
  </summary>
  <div class="run-body">
    {meta_html}
    {chart_html}
    {table_html}
  </div>
</details>"""


def _algo_section(algo_name: str, algo_dir: Path) -> str:
    """Build the full HTML section for one algorithm.

    Includes:
    * Aggregate stats cards across all stored runs
    * Inline tile distribution bar chart (best tile vs game count)
    * Inline run stability chart (run_id vs avg_score; only when ≥ 2 runs)
    * Run-history accordion — every stored run as a collapsible item,
      latest run pre-opened
    """
    run_dirs = sorted(
        d for d in algo_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ) if algo_dir.exists() else []

    if not run_dirs:
        return ""

    frames = []
    for d in run_dirs:
        csv_path = d / "results.csv"
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:
            continue

    if not frames:
        return ""

    df_all = pd.concat(frames, ignore_index=True)
    stats_html = _stats_grid(df_all)

    # ── Inline summary charts ──────────────────────────────────────────
    tile_src = _tile_dist_chart_src(df_all)
    stability_src = _run_stability_chart_src(run_dirs)

    chart_cards: list[str] = []
    if tile_src:
        chart_cards.append(
            f'<div class="summary-chart-card">'
            f'<div class="chart-title">Tile Distribution</div>'
            f'<img src="{tile_src}" alt="Tile distribution chart">'
            f"</div>"
        )
    if stability_src:
        chart_cards.append(
            f'<div class="summary-chart-card">'
            f'<div class="chart-title">Run Stability</div>'
            f'<img src="{stability_src}" alt="Run stability chart">'
            f"</div>"
        )
    summary_charts_html = (
        f'<div class="summary-charts">{"".join(chart_cards)}</div>'
        if chart_cards
        else ""
    )

    # Build the accordion — newest run first, auto-opened
    accordion_items: list[str] = []
    for run_dir in reversed(run_dirs):
        is_latest = run_dir == run_dirs[-1]
        item = _run_accordion_item(run_dir, is_latest=is_latest)
        if item:
            accordion_items.append(item)

    accordion_html = "\n".join(accordion_items)
    runs_label = f"{len(run_dirs)} run{'s' if len(run_dirs) != 1 else ''} stored"
    section_id = f"algo-{html.escape(algo_name.lower().replace(' ', '-'))}"

    return f"""\
<section class="algo-section" id="{section_id}">
  <div class="algo-section-header">
    🤖 {html.escape(algo_name)}
    <span style="font-size:0.75rem;font-weight:400;opacity:.8;margin-left:8px">({runs_label})</span>
  </div>
  <div class="algo-section-body">
    {stats_html}
    {summary_charts_html}
    <div class="runs-heading">Run History</div>
    {accordion_html}
  </div>
</section>"""


def _algo_nav(algo_dirs: list[Path], has_leaderboard: bool = False) -> str:
    """Build the sticky navigation bar with global-section and per-algorithm links."""
    if not algo_dirs:
        return ""

    global_links: list[str] = []
    if has_leaderboard:
        for anchor, label in [
            ("leaderboard", "🏆 Leaderboard"),
            ("stability", "📈 Stability"),
            ("efficiency", "⚡ Efficiency"),
            ("comparison-charts", "📊 Charts"),
        ]:
            global_links.append(
                f'<a href="#{anchor}"><span class="nav-dot"></span>{label}</a>'
            )

    algo_links = "\n    ".join(
        f'<a href="#algo-{html.escape(d.name.lower().replace(" ", "-"))}">'
        f'<span class="nav-dot"></span>{html.escape(d.name)}</a>'
        for d in algo_dirs
    )
    all_links = "\n    ".join(global_links) + ("\n    " if global_links else "") + algo_links
    return f"""\
<nav class="algo-nav">
  <span class="nav-label">Jump to</span>
    {all_links}
</nav>"""


def _comparison_section(algo_dirs: list[Path]) -> str:
    """Build a side-by-side comparison table for all algorithms.

    Only rendered when two or more algorithms have result data.  The cell with
    the best value in each metric column is highlighted.
    """
    rows_data = []
    for d in algo_dirs:
        run_dirs = sorted(
            rd for rd in d.iterdir()
            if rd.is_dir() and rd.name.startswith("run_")
        ) if d.exists() else []
        if not run_dirs:
            continue
        frames = []
        for rd in run_dirs:
            csv_path = rd / "results.csv"
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception:
                continue
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        rows_data.append(
            {
                "name": d.name,
                "total_games": len(df),
                "avg_score":    df["score"].mean(),
                "median_score": df["score"].median(),
                "p90_score":    df["score"].quantile(0.9),
                "max_score":    df["score"].max(),
                "best_tile":    df["best_tile"].max(),
                "win_rate":     df["won"].mean() * 100,
            }
        )

    if len(rows_data) < 2:
        return ""

    best_avg    = max(r["avg_score"]    for r in rows_data)
    best_median = max(r["median_score"] for r in rows_data)
    best_p90    = max(r["p90_score"]    for r in rows_data)
    best_max    = max(r["max_score"]    for r in rows_data)
    best_tile   = max(r["best_tile"]    for r in rows_data)
    best_win    = max(r["win_rate"]     for r in rows_data)

    def _cell(value: str, is_best: bool) -> str:
        cls = ' class="cmp-best"' if is_best else ""
        return f"<td{cls}>{value}</td>"

    rows_html = []
    for r in rows_data:
        rows_html.append(
            "<tr>"
            f"<td class=\"cmp-algo\">{html.escape(r['name'])}</td>"
            + _cell(f"{r['total_games']}", False)
            + _cell(f"{r['avg_score']:,.0f}",    r["avg_score"]    == best_avg)
            + _cell(f"{r['median_score']:,.0f}", r["median_score"] == best_median)
            + _cell(f"{r['p90_score']:,.0f}",    r["p90_score"]    == best_p90)
            + _cell(f"{r['max_score']:,}",       r["max_score"]    == best_max)
            + _cell(f"{r['best_tile']}",         r["best_tile"]    == best_tile)
            + _cell(f"{r['win_rate']:.1f}%",     r["win_rate"]     == best_win)
            + "</tr>"
        )

    rows_html_str = "\n".join(rows_html)
    return f"""\
<section class="cmp-section">
  <div class="cmp-header">📊 Algorithm Comparison</div>
  <div class="cmp-body">
    <div class="results-table">
      <table>
        <thead>
          <tr>
            <th>Algorithm</th>
            <th>Games</th>
            <th>Avg</th>
            <th>Median</th>
            <th>P90</th>
            <th>Max</th>
            <th>Best Tile</th>
            <th>Win %</th>
          </tr>
        </thead>
        <tbody>
{rows_html_str}
        </tbody>
      </table>
    </div>
  </div>
</section>"""


def generate_html_report(
    results_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate a self-contained HTML dashboard from all results in *results_dir*.

    Each sub-directory of *results_dir* is treated as an algorithm results
    folder (matching the directory layout created by ``main.py``).

    The page includes:
    * A sticky navigation bar linking to global sections and each algorithm
    * Hero summary cards (best avg score, best tile, most stable, fastest)
    * Main leaderboard table sorted by avg_score
    * Stability board table (algorithms with multiple runs)
    * Efficiency board table
    * Comparison charts (avg/median/p90, score distribution, best tile, run stability)
    * Per-algorithm aggregate stats cards and run-history accordion

    Parameters
    ----------
    results_dir:
        Root results directory (e.g. ``Path("results")``).
    output_path:
        Destination for the generated HTML file.

    Returns
    -------
    Path
        Absolute path to the written HTML file.
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)

    algo_dirs = sorted(
        p for p in results_dir.iterdir() if p.is_dir()
    ) if results_dir.exists() else []

    # Collect all aggregate data once
    leaderboard_data = _collect_leaderboard_data(algo_dirs)
    has_leaderboard = bool(leaderboard_data)

    nav_html = _algo_nav(algo_dirs, has_leaderboard=has_leaderboard)
    hero_html = _hero_section(leaderboard_data)
    leaderboard_html = _leaderboard_section(leaderboard_data)
    stability_html = _stability_section(leaderboard_data)
    efficiency_html = _efficiency_section(leaderboard_data)
    comparison_charts_html = _global_comparison_charts(leaderboard_data, algo_dirs)
    sections_html = "\n".join(_algo_section(d.name, d) for d in algo_dirs)

    if not sections_html.strip():
        sections_html = (
            '<p style="text-align:center;padding:40px;opacity:.5">'
            "No results found yet.  Run <code>python main.py</code> to generate some."
            "</p>"
        )

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>pw2048 — Results Dashboard</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <h1>🎮 pw2048 Results Dashboard</h1>
  <div class="meta">
    Generated: {now_utc}<br>
    {len(algo_dirs)} algorithm(s)
  </div>
</header>
{nav_html}
<main>
{hero_html}
{leaderboard_html}
{stability_html}
{efficiency_html}
{comparison_charts_html}
{sections_html}
</main>
<footer>pw2048 · Auto-generated results dashboard</footer>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path.resolve()
