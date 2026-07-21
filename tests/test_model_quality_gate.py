from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src.quality_gate import evaluate_quality_gate


POLICY = {
    "minimum_evaluation_games": 500,
    "metrics": {
        "mean_score": {"direction": "maximize", "max_regression_percent": 3},
        "invalid_action_rate": {"direction": "minimize", "max_regression_absolute": 0},
    },
}


def test_quality_gate_passes_acceptable_candidate():
    stable = {"metrics": {"mean_score": 1000, "invalid_action_rate": 0.02}}
    candidate = {"metrics": {"mean_score": 980, "invalid_action_rate": 0.02,
                              "evaluation_games": 500}}
    result = evaluate_quality_gate(stable, candidate, POLICY)
    assert result.passed
    assert all(check["passed"] for check in result.checks)


def test_quality_gate_fails_regression_and_small_sample():
    stable = {"metrics": {"mean_score": 1000, "invalid_action_rate": 0.02}}
    candidate = {"metrics": {"mean_score": 900, "invalid_action_rate": 0.03,
                              "evaluation_games": 100}}
    result = evaluate_quality_gate(stable, candidate, POLICY)
    assert not result.passed
    assert sum(not check["passed"] for check in result.checks) == 3


def test_quality_gate_fails_closed_on_missing_metric():
    result = evaluate_quality_gate(
        {"metrics": {"mean_score": 1000, "invalid_action_rate": 0.01}},
        {"metrics": {"mean_score": 1100, "evaluation_games": 500}}, POLICY,
    )
    assert not result.passed
    assert "missing" in result.checks[1]["reasons"][0]


def test_quality_gate_rejects_invalid_direction():
    with pytest.raises(ValueError, match="direction"):
        evaluate_quality_gate({"score": 1}, {"score": 1}, {
            "metrics": {"score": {"direction": "sideways"}}
        })


def test_quality_gate_cli_returns_nonzero(tmp_path):
    stable = tmp_path / "stable.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    stable.write_text(json.dumps({"mean_score": 100}), encoding="utf-8")
    candidate.write_text(json.dumps({"mean_score": 50}), encoding="utf-8")
    policy.write_text(json.dumps({"metrics": {"mean_score": {
        "direction": "maximize", "max_regression_percent": 1
    }}}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/model_quality_gate.py", "--stable", str(stable),
         "--candidate", str(candidate), "--policy", str(policy)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert '"passed": false' in result.stdout
