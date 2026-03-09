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
    "dqn-v1", "dqn-v2", "dqn-v3", "dqn",
    "ppo-v1", "ppo-v2", "ppo-v3", "ppo",
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

    # ── RL Training (learning algorithms only) ────────────────────────────────
    _is_rl = algorithm.startswith("dqn") or algorithm.startswith("ppo")
    train_games_str = "0"
    eval_freq_str = "50"
    n_eval_games_str = "20"
    tensorboard_dir = ""
    early_stopping_patience_str = "0"
    early_stopping_min_delta_str = "1"
    if _is_rl:
        _console.print(
            "\n[dim]── RL Training (DQN / PPO only) ──[/]\n",
        )
        train_games_str = questionary.text(
            "Fast training games — in-process, no browser\n"
            "  (0 = skip training, or auto-stop if patience > 0):",
            default="0",
            validate=_non_neg_int,
            style=_STYLE,
        ).ask()
        if train_games_str is None:
            raise SystemExit(0)

        if int(train_games_str) > 0:
            eval_freq_str = questionary.text(
                "Eval frequency — run EvalCallback every N games:",
                default="50",
                validate=_pos_int,
                style=_STYLE,
            ).ask()
            if eval_freq_str is None:
                raise SystemExit(0)

            n_eval_games_str = questionary.text(
                "Eval games per round:",
                default="20",
                validate=_pos_int,
                style=_STYLE,
            ).ask()
            if n_eval_games_str is None:
                raise SystemExit(0)

            tensorboard_dir = questionary.text(
                "TensorBoard / CSV log directory (leave blank to disable):",
                default="",
                style=_STYLE,
            ).ask()
            if tensorboard_dir is None:
                raise SystemExit(0)

        # Early stopping (shown whenever RL is active, even for auto mode).
        _console.print(
            "\n[dim]── Early stopping (optional) ──[/]\n"
            "[dim]Stops training automatically when score stops improving.[/]\n"
        )
        early_stopping_patience_str = questionary.text(
            "Early-stopping patience (eval rounds without improvement, 0 = off):",
            default="0",
            validate=_non_neg_int,
            style=_STYLE,
        ).ask()
        if early_stopping_patience_str is None:
            raise SystemExit(0)

        if int(early_stopping_patience_str) > 0:
            # When user chose auto mode (train_games=0), show eval_freq prompt.
            if int(train_games_str) == 0:
                eval_freq_str = questionary.text(
                    "Eval frequency — run EvalCallback every N games:",
                    default="50",
                    validate=_pos_int,
                    style=_STYLE,
                ).ask()
                if eval_freq_str is None:
                    raise SystemExit(0)
                n_eval_games_str = questionary.text(
                    "Eval games per round:",
                    default="20",
                    validate=_pos_int,
                    style=_STYLE,
                ).ask()
                if n_eval_games_str is None:
                    raise SystemExit(0)
            early_stopping_min_delta_str = questionary.text(
                "Min score improvement to reset patience counter:",
                default="1",
                style=_STYLE,
            ).ask()
            if early_stopping_min_delta_str is None:
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
    _es_patience = int(early_stopping_patience_str)
    if _is_rl and int(train_games_str) > 0:
        table.add_row("Train games", train_games_str)
        table.add_row("Eval freq", eval_freq_str)
        table.add_row("Eval games", n_eval_games_str)
        table.add_row(
            "TensorBoard dir",
            tensorboard_dir.strip() + "/" if tensorboard_dir.strip() else "–",
        )
    elif _is_rl and _es_patience > 0:
        table.add_row("Train mode", "auto (train until stable)")
        table.add_row("Eval freq", eval_freq_str)
        table.add_row("Eval games", n_eval_games_str)
    if _is_rl and _es_patience > 0:
        table.add_row("Early stopping", f"patience={_es_patience}, min_delta={early_stopping_min_delta_str}")
    table.add_row("Show browser", "yes" if show else "no")
    table.add_row("Keep N runs", keep_str)
    table.add_row("HTML report", "yes" if report else "no")

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

    if _is_rl and int(train_games_str) > 0:
        argv += ["--train-games", train_games_str]
        argv += ["--eval-freq", eval_freq_str]
        argv += ["--n-eval-games", n_eval_games_str]
        if tensorboard_dir.strip():
            argv += ["--tensorboard-dir", tensorboard_dir.strip()]

    if _is_rl and _es_patience > 0:
        argv += ["--early-stopping-patience", early_stopping_patience_str]
        argv += ["--early-stopping-min-delta", early_stopping_min_delta_str]
        argv += ["--eval-freq", eval_freq_str]
        argv += ["--n-eval-games", n_eval_games_str]

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

    return argv
