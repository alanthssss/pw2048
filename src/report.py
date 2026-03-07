"""Generate a self-contained HTML results dashboard for pw2048.

The produced page is fully self-contained: chart images are embedded as
base64 data URIs so the file works both locally (file://) and when hosted
on S3.
"""

from __future__ import annotations

import base64
import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# 2048-inspired colour palette
# ---------------------------------------------------------------------------
_PALETTE = {
    "bg": "#faf8ef",
    "header_bg": "#bbada0",
    "header_fg": "#f9f6f2",
    "card_bg": "#eee4da",
    "card_accent": "#f59563",
    "tile_2": "#eee4da",
    "tile_4": "#ede0c8",
    "tile_8": "#f2b179",
    "tile_16": "#f59563",
    "tile_32": "#f67c5f",
    "tile_64": "#f65e3b",
    "tile_128": "#edcf72",
    "tile_256": "#edcc61",
    "tile_512": "#edc850",
    "tile_1024": "#edc53f",
    "tile_2048": "#edc22e",
    "text_dark": "#776e65",
    "text_light": "#f9f6f2",
    "table_even": "#f3ede4",
    "win_badge": "#8ec07c",
    "lose_badge": "#cc241d",
}

_CSS = """\
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: "Clear Sans", "Helvetica Neue", Arial, sans-serif;
    background: {bg};
    color: {text_dark};
    line-height: 1.5;
}}
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
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }}
.algo-section {{
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    margin-bottom: 40px;
    overflow: hidden;
}}
.algo-header {{
    background: {header_bg};
    color: {header_fg};
    padding: 14px 24px;
    font-size: 1.3rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.algo-body {{ padding: 24px; }}
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}}
.stat-card {{
    background: {card_bg};
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}}
.stat-card .stat-label {{ font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; opacity: 0.7; margin-bottom: 4px; }}
.stat-card .stat-value {{ font-size: 1.6rem; font-weight: 700;
    color: {card_accent}; }}
.chart-wrap {{ text-align: center; margin-bottom: 28px; }}
.chart-wrap img {{
    max-width: 100%;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,.12);
}}
.section-title {{
    font-size: 1rem;
    font-weight: 600;
    color: {text_dark};
    margin-bottom: 12px;
    padding-bottom: 4px;
    border-bottom: 2px solid {card_bg};
}}
.results-table {{ overflow-x: auto; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}}
thead th {{
    background: {header_bg};
    color: {header_fg};
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}}
tbody tr:nth-child(even) {{ background: {table_even}; }}
tbody td {{ padding: 9px 14px; }}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
}}
.badge-win  {{ background: {win_badge};  }}
.badge-lose {{ background: {lose_badge}; }}
.tile-chip {{
    display: inline-block;
    min-width: 48px;
    padding: 3px 8px;
    border-radius: 4px;
    text-align: center;
    font-weight: 700;
    font-size: 0.85rem;
}}
footer {{
    text-align: center;
    font-size: 0.8rem;
    color: {text_dark};
    opacity: 0.6;
    padding: 16px;
    border-top: 1px solid {card_bg};
    margin-top: 8px;
}}
@media (max-width: 600px) {{
    header h1 {{ font-size: 1.4rem; }}
    .stat-card .stat-value {{ font-size: 1.2rem; }}
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


def _stats_grid(df: pd.DataFrame) -> str:
    avg_score = df["score"].mean()
    best_score = df["score"].max()
    best_tile = df["max_tile"].max()
    win_rate = df["won"].mean() * 100
    n_games = len(df)
    return f"""\
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Games</div>
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
            f"<td>{_tile_chip(int(row['max_tile']))}</td>"
            f"<td>{int(row['move_count'])}</td>"
            f"<td>{float(row['duration_s']):.1f}s</td>"
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


def _algo_section(algo_name: str, algo_dir: Path) -> str:
    """Build the HTML section for one algorithm.

    Reads all CSV files in *algo_dir*, combines them, and renders:
    * summary stats cards
    * the most recent chart (embedded as base64)
    * a table of the most recent individual game results
    """
    csv_files = sorted(algo_dir.glob("*.csv"))
    if not csv_files:
        return ""

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue

    if not frames:
        return ""

    df_all = pd.concat(frames, ignore_index=True)

    # Stats are computed over *all* available runs in the directory.
    stats_html = _stats_grid(df_all)

    # Chart: use the PNG from the most recent CSV stem.
    latest_stem = csv_files[-1].stem
    png_path = algo_dir / f"{latest_stem}.png"
    img_src = _embed_image(png_path)
    chart_html = (
        f'<div class="chart-wrap"><img src="{img_src}" '
        f'alt="Results chart for {html.escape(algo_name)}"></div>'
        if img_src
        else ""
    )

    # Recent games table: show the last run's individual game rows.
    df_latest = pd.read_csv(csv_files[-1])
    table_html = f"""\
<p class="section-title">Latest Run — {html.escape(latest_stem.replace("_", " "))}</p>
{_results_table(df_latest)}"""

    runs_label = f"{len(csv_files)} run{'s' if len(csv_files) != 1 else ''}"

    return f"""\
<section class="algo-section">
  <div class="algo-header">
    🤖 {html.escape(algo_name)}
    <span style="font-size:0.75rem;font-weight:400;opacity:.8;margin-left:8px">({runs_label} stored)</span>
  </div>
  <div class="algo-body">
    {stats_html}
    {chart_html}
    {table_html}
  </div>
</section>"""


def generate_html_report(
    results_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate a self-contained HTML dashboard from all results in *results_dir*.

    Each sub-directory of *results_dir* is treated as an algorithm results
    folder (matching the directory layout created by ``main.py``).

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

    sections_html = "\n".join(
        _algo_section(d.name, d) for d in algo_dirs
    )

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
<main>
{sections_html}
</main>
<footer>pw2048 · Auto-generated results dashboard</footer>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path.resolve()
