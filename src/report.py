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

    open_attr = " open" if is_latest else ""

    return f"""\
<details class="run-item" id="{anchor}"{open_attr}>
  <summary class="run-summary">
    <span class="run-arrow">▶</span>
    <span class="run-ts">📅 {html.escape(ts_display)}</span>
    {chips_html}
  </summary>
  <div class="run-body">
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


def _algo_nav(algo_dirs: list[Path]) -> str:
    """Build the sticky algorithm navigation bar."""
    if not algo_dirs:
        return ""
    links = "\n    ".join(
        f'<a href="#algo-{html.escape(d.name.lower().replace(" ", "-"))}">'
        f'<span class="nav-dot"></span>{html.escape(d.name)}</a>'
        for d in algo_dirs
    )
    return f"""\
<nav class="algo-nav">
  <span class="nav-label">Jump to</span>
    {links}
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
    * A sticky navigation bar linking to each algorithm's section
    * Per-algorithm aggregate stats cards
    * A run-history accordion showing every stored run with its chart and
      per-game results table; the most recent run is pre-expanded

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

    nav_html = _algo_nav(algo_dirs)
    comparison_html = _comparison_section(algo_dirs)
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
{comparison_html}
{sections_html}
</main>
<footer>pw2048 · Auto-generated results dashboard</footer>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path.resolve()
