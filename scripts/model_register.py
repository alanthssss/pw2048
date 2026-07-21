#!/usr/bin/env python3
"""Create a reproducibility manifest and register a model locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_registry import FileSystemModelRegistry, create_manifest


def _object(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--experiment-id")
    parser.add_argument("--registry", default="model-registry")
    args = parser.parse_args()
    manifest = create_manifest(
        args.artifact, model_name=args.name, model_version=args.version,
        random_seed=args.seed, configuration=_object(args.config),
        metrics=_object(args.metrics) if args.metrics else {},
        experiment_id=args.experiment_id,
    )
    uri = FileSystemModelRegistry(args.registry).register(args.artifact, manifest)
    print(json.dumps({"uri": uri, "manifest": manifest.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
