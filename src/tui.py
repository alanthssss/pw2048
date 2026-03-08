"""Interactive TUI launcher for pw2048.

Presents an arrow-key-navigable questionary-based wizard that collects all
run parameters and returns a ``list[str]`` compatible with ``main.parse_args``.
"""

from __future__ import annotations

import os

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_STYLE = Style(
    [
        ("qmark", "fg:#5f9ea0 bold"),
        ("question", "bold"),
        ("answer", "fg:#5f9ea0 bold"),
        ("pointer", "fg:#5f9ea0 bold"),
        ("highlighted", "fg:#5f9ea0 bold"),
        ("selected", "fg:#5f9ea0"),
        ("separator", "fg:#cc5454"),
        ("instruction", "fg:#aaaaaa italic"),
    ]
)

_console = Console()

# These mirror the constants in main.py to avoid a circular import.
_ALGORITHMS = [
    "random", "greedy", "heuristic", "expectimax",
    "mcts-v1", "mcts-v2", "mcts",
    "dqn-v1", "dqn-v2", "dqn",
    "ppo-v1", "ppo-v2", "ppo",
]
_DEFAULT_KEEP = 10
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1
_MODE_PRESETS = {
    "dev":       {"games": 100,  "runs": 1, "parallel": os.cpu_count() or 1},
    "release":   {"games": 1000, "runs": 1, "parallel": os.cpu_count() or 1},
    "benchmark": {"games": 500,  "runs": 5, "parallel": os.cpu_count() or 1},
}


def _pos_int(val: str) -> bool | str:
    """Return True if *val* is a positive integer, otherwise an error string."""
    return (val.isdigit() and int(val) > 0) or "Please enter a positive integer."


def _non_neg_int(val: str) -> bool | str:
    """Return True if *val* is a non-negative integer, otherwise an error string."""
    return (val.isdigit() and int(val) >= 0) or "Please enter 0 or a positive integer."


def run_tui() -> list[str]:
    """Run the interactive wizard and return an ``argv`` list for :func:`main.parse_args`.

    Returns
    -------
    list[str]
        Argument list that can be passed directly to :func:`main.main`.

    Raises
    ------
    SystemExit
        When the user cancels at any prompt (Ctrl-C / Ctrl-D / answering "No" to proceed).
    """
    _console.print(
        Panel.fit(
            "[bold cyan]pw2048[/] – Interactive Launcher\n"
            "[dim]Use arrow keys to select, Enter to confirm.[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    _console.print()

    # ── Algorithm ─────────────────────────────────────────────────────────────
    algorithm = questionary.select(
        "Algorithm:",
        choices=_ALGORITHMS,
        style=_STYLE,
    ).ask()
    if algorithm is None:
        raise SystemExit(0)

    # ── Version tag ───────────────────────────────────────────────────────────
    version_tag = questionary.text(
        "Version tag (leave blank to use the algorithm's default):",
        default="",
        style=_STYLE,
    ).ask()
    if version_tag is None:
        raise SystemExit(0)

    # ── Run mode ──────────────────────────────────────────────────────────────
    mode_choice = questionary.select(
        "Run mode:",
        choices=[
            questionary.Choice(
                "custom     – set games / runs / parallel manually",
                value="custom",
            ),
            questionary.Choice(
                f"dev        – {_MODE_PRESETS['dev']['games']} games · "
                f"{_MODE_PRESETS['dev']['runs']} run · auto parallel",
                value="dev",
            ),
            questionary.Choice(
                f"release    – {_MODE_PRESETS['release']['games']:,} games · "
                f"{_MODE_PRESETS['release']['runs']} run · auto parallel",
                value="release",
            ),
            questionary.Choice(
                f"benchmark  – {_MODE_PRESETS['benchmark']['games']} games × "
                f"{_MODE_PRESETS['benchmark']['runs']} runs · auto parallel",
                value="benchmark",
            ),
        ],
        style=_STYLE,
    ).ask()
    if mode_choice is None:
        raise SystemExit(0)

    # ── Custom settings ────────────────────────────────────────────────────────
    games = runs = parallel = None
    if mode_choice == "custom":
        games = questionary.text(
            "Number of games per run:",
            default=str(_DEFAULT_GAMES),
            validate=_pos_int,
            style=_STYLE,
        ).ask()
        if games is None:
            raise SystemExit(0)

        runs = questionary.text(
            "Number of runs:",
            default=str(_DEFAULT_RUNS),
            validate=_pos_int,
            style=_STYLE,
        ).ask()
        if runs is None:
            raise SystemExit(0)

        parallel = questionary.text(
            "Parallel browser workers:",
            default=str(_DEFAULT_PARALLEL),
            validate=_pos_int,
            style=_STYLE,
        ).ask()
        if parallel is None:
            raise SystemExit(0)

    # ── Output directory ──────────────────────────────────────────────────────
    output = questionary.text(
        "Output directory:",
        default="results",
        style=_STYLE,
    ).ask()
    if output is None:
        raise SystemExit(0)

    # ── Misc options ──────────────────────────────────────────────────────────
    show = questionary.confirm(
        "Show browser window while playing?",
        default=False,
        style=_STYLE,
    ).ask()
    if show is None:
        raise SystemExit(0)

    keep_str = questionary.text(
        "Keep N most-recent runs per algorithm (0 = keep all):",
        default=str(_DEFAULT_KEEP),
        validate=_non_neg_int,
        style=_STYLE,
    ).ask()
    if keep_str is None:
        raise SystemExit(0)

    report = questionary.confirm(
        "Generate HTML report?",
        default=False,
        style=_STYLE,
    ).ask()
    if report is None:
        raise SystemExit(0)

    # ── S3 options ────────────────────────────────────────────────────────────
    use_s3 = questionary.confirm(
        "Upload results to S3?",
        default=False,
        style=_STYLE,
    ).ask()
    if use_s3 is None:
        raise SystemExit(0)

    s3_bucket = s3_prefix = None
    s3_public = False
    if use_s3:
        s3_bucket = questionary.text(
            "S3 bucket name:",
            validate=lambda v: bool(v.strip()) or "Bucket name cannot be empty.",
            style=_STYLE,
        ).ask()
        if s3_bucket is None:
            raise SystemExit(0)

        s3_prefix = questionary.text(
            "S3 key prefix:",
            default="results",
            style=_STYLE,
        ).ask()
        if s3_prefix is None:
            raise SystemExit(0)

        s3_public = questionary.confirm(
            "Apply public-read ACL to uploaded objects?",
            default=False,
            style=_STYLE,
        ).ask()
        if s3_public is None:
            raise SystemExit(0)

    # ── Summary table ─────────────────────────────────────────────────────────
    _console.print()
    table = Table(
        title="Configuration Summary",
        show_header=False,
        border_style="cyan",
        padding=(0, 1),
    )
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Algorithm", algorithm)
    table.add_row("Version tag", version_tag.strip() if version_tag.strip() else "(default)")
    if mode_choice == "custom":
        table.add_row("Games / run", games)
        table.add_row("Runs", runs)
        table.add_row("Workers", parallel)
    else:
        p = _MODE_PRESETS[mode_choice]
        table.add_row(
            "Mode",
            f"{mode_choice}  ({p['games']} games × {p['runs']} run(s), auto parallel)",
        )
    table.add_row("Output dir", output + "/")
    table.add_row("Show browser", "yes" if show else "no")
    table.add_row("Keep N runs", keep_str)
    table.add_row("HTML report", "yes" if report else "no")
    table.add_row(
        "S3 bucket",
        f"{s3_bucket}  (prefix: {s3_prefix}{'  public-read' if s3_public else ''})"
        if s3_bucket
        else "–",
    )

    _console.print(table)
    _console.print()

    proceed = questionary.confirm("Proceed?", default=True, style=_STYLE).ask()
    if not proceed:
        _console.print("[yellow]Aborted.[/]")
        raise SystemExit(0)

    # ── Build argv ────────────────────────────────────────────────────────────
    argv: list[str] = [
        "--algorithm", algorithm,
        "--output", output,
        "--keep", keep_str,
    ]

    if version_tag.strip():
        argv += ["--algo-version", version_tag.strip()]

    if mode_choice != "custom":
        argv += ["--mode", mode_choice]
    else:
        argv += [
            "--games", str(games),
            "--runs", str(runs),
            "--parallel", str(parallel),
        ]

    if show:
        argv.append("--show")
    if report:
        argv.append("--report")
    if use_s3 and s3_bucket:
        argv += ["--s3-bucket", s3_bucket, "--s3-prefix", str(s3_prefix)]
        if s3_public:
            argv.append("--s3-public")

    return argv
