from __future__ import annotations

import json

import pytest

from src.model_registry import FileSystemModelRegistry, create_manifest, sha256_file


def test_manifest_records_reproducibility_metadata(tmp_path, monkeypatch):
    artifact = tmp_path / "weights.npz"
    artifact.write_bytes(b"model-weights")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("definitely-missing-package>=1\n", encoding="utf-8")
    monkeypatch.setenv("GIT_SHA", "abc123")

    manifest = create_manifest(
        artifact, model_name="dqn", model_version="v1", random_seed=42,
        configuration={"games": 100}, metrics={"mean_score": 1234},
        experiment_id="exp-test", requirements=requirements,
    )

    assert manifest.git_sha == "abc123"
    assert manifest.random_seed == 42
    assert manifest.configuration == {"games": 100}
    assert manifest.artifact_sha256 == sha256_file(artifact)
    assert manifest.dependencies["definitely-missing-package"] == "not-installed"
    assert manifest.hardware["accelerator"] in {"cpu", "cuda", "mps"}


def test_filesystem_registry_registers_and_verifies(tmp_path):
    artifact = tmp_path / "weights.npz"
    artifact.write_bytes(b"weights")
    manifest = create_manifest(
        artifact, model_name="ppo", model_version="v2", random_seed=7,
        configuration={}, requirements=None,
    )
    registry = FileSystemModelRegistry(tmp_path / "registry")

    uri = registry.register(artifact, manifest)

    assert uri.startswith("file://")
    assert registry.resolve("ppo", "v2").read_bytes() == b"weights"
    assert registry.load_manifest("ppo", "v2") == manifest
    with pytest.raises(FileExistsError):
        registry.register(artifact, manifest)


def test_filesystem_registry_detects_corruption(tmp_path):
    artifact = tmp_path / "weights.npz"
    artifact.write_bytes(b"weights")
    manifest = create_manifest(
        artifact, model_name="dqn", model_version="v1", random_seed=1,
        configuration={}, requirements=None,
    )
    registry = FileSystemModelRegistry(tmp_path / "registry")
    registry.register(artifact, manifest)
    registered = tmp_path / "registry" / "dqn" / "v1" / "weights.npz"
    registered.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        registry.resolve("dqn", "v1")


def test_manifest_json_round_trip(tmp_path):
    artifact = tmp_path / "weights.npz"
    artifact.write_bytes(b"weights")
    manifest = create_manifest(
        artifact, model_name="dqn", model_version="v1", random_seed=1,
        configuration={"nested": {"batch": 8}}, requirements=None,
    )
    assert json.loads(json.dumps(manifest.to_dict()))["configuration"]["nested"]["batch"] == 8


def test_filesystem_registry_rejects_path_traversal(tmp_path):
    registry = FileSystemModelRegistry(tmp_path / "registry")
    with pytest.raises(ValueError):
        registry.load_manifest("../outside", "v1")
