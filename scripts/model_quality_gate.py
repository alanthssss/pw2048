#!/usr/bin/env python3
"""Fail a pipeline when candidate model quality violates policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality_gate import evaluate_quality_gate, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--policy", default="configs/quality-gate.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_quality_gate(load_json(args.stable), load_json(args.candidate), load_json(args.policy))
    rendered = json.dumps(result.to_dict(), indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
