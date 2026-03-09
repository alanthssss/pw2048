# UI Wizards — TUI, GUI, and Web UI

Three interactive launchers let you configure a run without memorising flags.
All three support the full feature set including the new `--train-games` RL
training pipeline.

---

## Interactive TUI wizard

```bash
python main.py --tui
```

Arrow-key-navigable questionary wizard.  Walks through every option step-by-step.

```
╭───────────────────────────────────────────────╮
│  pw2048 – Interactive Launcher                │
│  Use arrow keys to select, Enter to confirm.  │
╰───────────────────────────────────────────────╯

? Algorithm:        dqn
? Version tag:      (blank → default)
? Run mode:         custom
? Number of games per run:   50
? Number of runs:            1
? Parallel browser workers:  1
? Output directory:          results
? ── RL Training (DQN / PPO only) ──
? Fast training games (0 = skip):  5000
? Eval frequency (games):          50
? Eval games per round:            20
? TensorBoard / CSV log directory: tb_logs
? Show browser window?    No
? Keep N most-recent runs: 10
? Generate HTML report?   Yes

   Configuration Summary
┌──────────────────┬──────────┐
│ Algorithm        │ dqn      │
│ Games / run      │ 50       │
│ Output dir       │ results/ │
│ Train games      │ 5000     │
│ Eval freq        │ 50       │
│ Eval games       │ 20       │
│ TensorBoard dir  │ tb_logs/ │
│ Show browser     │ no       │
│ Keep N runs      │ 10       │
│ HTML report      │ yes      │
└──────────────────┴──────────┘

? Proceed? Yes
```

Press <kbd>Ctrl-C</kbd> at any prompt, or answer **No** at the final
confirmation, to abort without running any games.

### Wizard steps → CLI flags

| Wizard step | Equivalent flag(s) |
|---|---|
| Algorithm | `--algorithm` |
| Run mode (preset) | `--mode` |
| Games / runs / workers (custom) | `--games`, `--runs`, `--parallel` |
| Output directory | `--output` |
| Fast training games | `--train-games` |
| Eval frequency | `--eval-freq` |
| Eval games per round | `--n-eval-games` |
| TensorBoard log dir | `--tensorboard-dir` |
| Show browser | `--show` |
| Keep N runs | `--keep` |
| HTML report | `--report` |

---

## Desktop GUI wizard

```bash
python main.py --gui
```

Native tkinter window with the same fields as the TUI — includes the new
**RL Training** section with train-games, eval-freq, eval-games, and
tensorboard-dir fields (shown only when a learning algorithm is selected).

**Prerequisites:** tkinter ships with Python on Windows and macOS.  On
Debian/Ubuntu install it with:

```bash
sudo apt-get install python3-tk
```

---

## Web UI wizard

```bash
python main.py --web
```

pw2048 starts an HTTP server on a random free port, prints the URL, and opens
it in your default browser automatically:

```
  Web UI → http://127.0.0.1:54321/
  (fill in the form and click Launch — check your terminal for progress)
```

The form includes a dedicated **RL Training** section for the `--train-games`
pipeline.  Fill in the form and click **Launch ▶** — the server shuts down
automatically and runs start in your terminal.

![pw2048 Web Launcher](https://github.com/user-attachments/assets/bbfba22f-7cdc-44b3-82f5-46dc3f8c983b)

The web UI requires **no third-party packages** — only the Python standard
library (`http.server`, `threading`, `webbrowser`).
