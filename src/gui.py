"""Desktop GUI launcher for pw2048 using tkinter.

Opens a self-contained form window where the user configures all run parameters
and clicks **Launch ▶**.  Returns an argv list accepted by
:func:`main.parse_args`.

tkinter is part of the Python standard library but is packaged separately on
some Linux distributions.  If it is not available, :func:`run_gui` raises a
friendly :exc:`SystemExit` with installation instructions.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Constants (mirror main.py to avoid circular imports)
# ---------------------------------------------------------------------------

_ALGORITHMS = ["random", "greedy", "heuristic", "expectimax", "mcts"]
_DEFAULT_KEEP = 10
_DEFAULT_GAMES = 20
_DEFAULT_RUNS = 1
_DEFAULT_PARALLEL = 1


# ---------------------------------------------------------------------------
# Pure helper – testable without tkinter
# ---------------------------------------------------------------------------

def _build_argv(
    algorithm: str,
    mode_choice: str,
    games: str,
    runs: str,
    parallel: str,
    output: str,
    show: bool,
    keep: str,
    report: bool,
    s3_bucket: str,
    s3_prefix: str,
    s3_public: bool,
) -> list[str]:
    """Convert GUI form values to an argv list for :func:`main.parse_args`.

    Parameters
    ----------
    algorithm:
        One of ``"random"``, ``"greedy"``, ``"heuristic"``, ``"expectimax"``,
        ``"mcts"``.
    mode_choice:
        ``"custom"`` or one of the preset names (``"dev"``, ``"release"``,
        ``"benchmark"``).
    games, runs, parallel:
        String representations of the corresponding integer counts.  Only
        used when *mode_choice* is ``"custom"``.
    output:
        Output directory path string.
    show:
        Whether to pass ``--show``.
    keep:
        String representation of the keep-N-runs integer.
    report:
        Whether to pass ``--report``.
    s3_bucket:
        S3 bucket name (empty string → omit S3 flags).
    s3_prefix:
        S3 key prefix.
    s3_public:
        Whether to pass ``--s3-public``.

    Returns
    -------
    list[str]
        argv list suitable for :func:`main.parse_args`.
    """
    argv: list[str] = [
        "--algorithm", algorithm,
        "--output", output or "results",
        "--keep", keep,
    ]
    if mode_choice != "custom":
        argv += ["--mode", mode_choice]
    else:
        argv += [
            "--games", games,
            "--runs", runs,
            "--parallel", parallel,
        ]
    if show:
        argv.append("--show")
    if report:
        argv.append("--report")
    bucket = s3_bucket.strip()
    if bucket:
        argv += ["--s3-bucket", bucket, "--s3-prefix", s3_prefix or "results"]
        if s3_public:
            argv.append("--s3-public")
    return argv


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui() -> list[str]:
    """Open the tkinter GUI launcher and return an argv list for parse_args().

    Blocks until the user clicks **Launch ▶** or closes the window.

    Returns
    -------
    list[str]
        Argument list that can be passed directly to :func:`main.parse_args`.

    Raises
    ------
    SystemExit
        When the user closes the window without launching, or when tkinter is
        not available in the current Python installation.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, ttk
    except ImportError as exc:
        raise SystemExit(
            "The GUI launcher requires tkinter, which is not installed.\n"
            "  Debian/Ubuntu : sudo apt-get install python3-tk\n"
            "  macOS (Homebrew): brew install python-tk\n"
            "  Windows       : re-run the Python installer and tick 'tcl/tk'.\n"
        ) from exc

    result: list[list[str]] = []

    root = tk.Tk()
    root.title("pw2048 – GUI Launcher")
    root.resizable(False, False)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    PAD: dict = {"padx": 8, "pady": 4}

    outer = ttk.Frame(root, padding=16)
    outer.grid(row=0, column=0, sticky="nsew")

    # ── Heading ───────────────────────────────────────────────────────────
    ttk.Label(
        outer,
        text="pw2048 – GUI Launcher",
        font=("", 13, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

    row = 1

    def _label(text: str) -> None:
        nonlocal row
        ttk.Label(outer, text=text).grid(row=row, column=0, sticky="w", **PAD)

    def _sep() -> None:
        nonlocal row
        ttk.Separator(outer, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

    # ── Algorithm ─────────────────────────────────────────────────────────
    _label("Algorithm:")
    algo_var = tk.StringVar(value=_ALGORITHMS[0])
    ttk.Combobox(
        outer, textvariable=algo_var, values=_ALGORITHMS, state="readonly", width=14
    ).grid(row=row, column=1, sticky="w", **PAD)
    row += 1

    _sep()

    # ── Mode ──────────────────────────────────────────────────────────────
    _label("Mode:")
    mode_var = tk.StringVar(value="custom")
    mode_frame = ttk.Frame(outer)
    mode_frame.grid(row=row, column=1, columnspan=2, sticky="w", **PAD)
    for m in ("custom", "dev", "release", "benchmark"):
        ttk.Radiobutton(
            mode_frame, text=m, variable=mode_var, value=m
        ).pack(side="left", padx=4)
    row += 1

    # ── Custom sub-fields ─────────────────────────────────────────────────
    custom_frame = ttk.Frame(outer)
    custom_frame.grid(row=row, column=0, columnspan=3, sticky="ew")
    row += 1

    def _int_entry(
        parent: ttk.Frame, label_text: str, default: int, row_idx: int
    ) -> tk.StringVar:
        ttk.Label(parent, text=label_text).grid(
            row=row_idx, column=0, sticky="w", **PAD
        )
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=8).grid(
            row=row_idx, column=1, sticky="w", **PAD
        )
        return var

    games_var = _int_entry(custom_frame, "Games / run:", _DEFAULT_GAMES, 0)
    runs_var = _int_entry(custom_frame, "Runs:", _DEFAULT_RUNS, 1)
    parallel_var = _int_entry(custom_frame, "Workers:", _DEFAULT_PARALLEL, 2)

    def _toggle_custom(*_args: object) -> None:
        if mode_var.get() == "custom":
            custom_frame.grid()
        else:
            custom_frame.grid_remove()

    mode_var.trace_add("write", _toggle_custom)
    _toggle_custom()

    _sep()

    # ── Output directory ──────────────────────────────────────────────────
    _label("Output dir:")
    output_var = tk.StringVar(value="results")
    ttk.Entry(outer, textvariable=output_var, width=24).grid(
        row=row, column=1, sticky="w", **PAD
    )
    ttk.Button(
        outer,
        text="Browse…",
        command=lambda: output_var.set(
            filedialog.askdirectory(initialdir=output_var.get() or ".")
            or output_var.get()
        ),
    ).grid(row=row, column=2, sticky="w", **PAD)
    row += 1

    _sep()

    # ── Misc options ──────────────────────────────────────────────────────
    _label("Show browser:")
    show_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(outer, variable=show_var).grid(row=row, column=1, sticky="w", **PAD)
    row += 1

    _label("Keep N runs:")
    keep_var = tk.StringVar(value=str(_DEFAULT_KEEP))
    ttk.Entry(outer, textvariable=keep_var, width=8).grid(
        row=row, column=1, sticky="w", **PAD
    )
    row += 1

    _label("HTML report:")
    report_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(outer, variable=report_var).grid(row=row, column=1, sticky="w", **PAD)
    row += 1

    _sep()

    # ── S3 options ────────────────────────────────────────────────────────
    ttk.Label(outer, text="S3 (optional)", font=("", 9, "italic")).grid(
        row=row, column=0, columnspan=3, sticky="w", **PAD
    )
    row += 1

    _label("S3 bucket:")
    s3_bucket_var = tk.StringVar(value="")
    ttk.Entry(outer, textvariable=s3_bucket_var, width=28).grid(
        row=row, column=1, columnspan=2, sticky="w", **PAD
    )
    row += 1

    _label("S3 prefix:")
    s3_prefix_var = tk.StringVar(value="results")
    ttk.Entry(outer, textvariable=s3_prefix_var, width=28).grid(
        row=row, column=1, columnspan=2, sticky="w", **PAD
    )
    row += 1

    _label("Public-read:")
    s3_public_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(outer, variable=s3_public_var).grid(
        row=row, column=1, sticky="w", **PAD
    )
    row += 1

    _sep()

    # ── Error label ───────────────────────────────────────────────────────
    err_var = tk.StringVar(value="")
    ttk.Label(outer, textvariable=err_var, foreground="red").grid(
        row=row, column=0, columnspan=3, sticky="w", **PAD
    )
    row += 1

    # ── Buttons ───────────────────────────────────────────────────────────
    btn_frame = ttk.Frame(outer)
    btn_frame.grid(row=row, column=0, columnspan=3, sticky="e", pady=8)

    def _on_launch() -> None:
        # Validate numeric inputs
        for field_label, var, positive_only in [
            ("Games", games_var, True),
            ("Runs", runs_var, True),
            ("Workers", parallel_var, True),
            ("Keep N", keep_var, False),
        ]:
            val = var.get().strip()
            if not val.isdigit():
                err_var.set(f"'{field_label}' must be an integer (got '{val}').")
                return
            if positive_only and int(val) <= 0:
                err_var.set(f"'{field_label}' must be greater than 0.")
                return
            if not positive_only and int(val) < 0:
                err_var.set(f"'{field_label}' must be 0 or greater.")
                return

        argv = _build_argv(
            algorithm=algo_var.get(),
            mode_choice=mode_var.get(),
            games=games_var.get().strip(),
            runs=runs_var.get().strip(),
            parallel=parallel_var.get().strip(),
            output=output_var.get().strip(),
            show=bool(show_var.get()),
            keep=keep_var.get().strip(),
            report=bool(report_var.get()),
            s3_bucket=s3_bucket_var.get(),
            s3_prefix=s3_prefix_var.get(),
            s3_public=bool(s3_public_var.get()),
        )
        result.append(argv)
        root.destroy()

    def _on_cancel() -> None:
        root.destroy()

    ttk.Button(btn_frame, text="Cancel", command=_on_cancel).pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Launch ▶", command=_on_launch).pack(side="left", padx=4)

    root.mainloop()

    if not result:
        raise SystemExit(0)
    return result[0]
