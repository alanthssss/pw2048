"""Compare stable and candidate model metrics using declarative thresholds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": list(self.checks)}


def _metrics(document: Mapping[str, Any]) -> Mapping[str, Any]:
    value = document.get("metrics", document)
    if not isinstance(value, Mapping):
        raise ValueError("metrics JSON must be an object or contain a metrics object")
    return value


def evaluate_quality_gate(
    stable: Mapping[str, Any], candidate: Mapping[str, Any], policy: Mapping[str, Any]
) -> GateResult:
    """Evaluate all policy metrics; missing or non-numeric values fail closed.

    Policy rules support ``direction`` (``maximize``/``minimize``),
    ``max_regression_percent``, ``max_regression_absolute``, ``minimum`` and
    ``maximum``.  Every configured constraint must pass.
    """
    stable_metrics, candidate_metrics = _metrics(stable), _metrics(candidate)
    rules = policy.get("metrics")
    if not isinstance(rules, Mapping) or not rules:
        raise ValueError("quality-gate policy requires a non-empty metrics object")
    checks: list[dict[str, Any]] = []
    for name, raw_rule in rules.items():
        rule = raw_rule if isinstance(raw_rule, Mapping) else {}
        baseline, proposed = stable_metrics.get(name), candidate_metrics.get(name)
        reasons: list[str] = []
        if not isinstance(baseline, (int, float)) or isinstance(baseline, bool):
            reasons.append("stable metric is missing or non-numeric")
        if not isinstance(proposed, (int, float)) or isinstance(proposed, bool):
            reasons.append("candidate metric is missing or non-numeric")
        if not reasons:
            direction = rule.get("direction", "maximize")
            if direction not in {"maximize", "minimize"}:
                raise ValueError(f"{name}: direction must be maximize or minimize")
            regression = (baseline - proposed) if direction == "maximize" else (proposed - baseline)
            if "max_regression_absolute" in rule and regression > float(rule["max_regression_absolute"]):
                reasons.append(f"absolute regression {regression:.6g} exceeds limit")
            if "max_regression_percent" in rule:
                if baseline == 0:
                    if regression > 0:
                        reasons.append("percent regression is undefined from zero baseline")
                else:
                    percent = regression / abs(baseline) * 100
                    if percent > float(rule["max_regression_percent"]):
                        reasons.append(f"regression {percent:.3f}% exceeds limit")
            if "minimum" in rule and proposed < float(rule["minimum"]):
                reasons.append(f"value is below minimum {rule['minimum']}")
            if "maximum" in rule and proposed > float(rule["maximum"]):
                reasons.append(f"value is above maximum {rule['maximum']}")
        checks.append({
            "metric": name, "stable": baseline, "candidate": proposed,
            "passed": not reasons, "reasons": reasons,
        })
    required = int(policy.get("minimum_evaluation_games", 0))
    games = candidate_metrics.get("evaluation_games")
    if required:
        passed = isinstance(games, (int, float)) and not isinstance(games, bool) and games >= required
        checks.append({"metric": "evaluation_games", "stable": stable_metrics.get("evaluation_games"),
                       "candidate": games, "passed": passed,
                       "reasons": [] if passed else [f"requires at least {required} games"]})
    return GateResult(all(check["passed"] for check in checks), tuple(checks))


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
