"""Reproducible model manifests and pluggable model registries.

The filesystem backend is deliberately dependency-free and suitable for local
development and CI.  The :class:`ModelRegistry` protocol is the boundary used
by future S3/MLflow adapters, so callers do not depend on storage details.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol


MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repo: str | Path = ".") -> str:
    """Return the current commit, or an explicit ``GIT_SHA`` fallback."""
    override = os.getenv("GIT_SHA")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dependency_versions(requirements: str | Path | None = None) -> dict[str, str]:
    """Capture installed versions for dependencies named in requirements.txt."""
    result: dict[str, str] = {"python": platform.python_version()}
    if requirements is None or not Path(requirements).exists():
        return result
    for raw in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(";", 1)[0]
        for token in ("===", ">=", "<=", "==", "~=", ">", "<", "["):
            name = name.split(token, 1)[0]
        name = name.strip()
        if not name:
            continue
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return dict(sorted(result.items()))


def hardware_info() -> dict[str, Any]:
    """Collect portable hardware/runtime facts without optional dependencies."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # type: ignore[import]
        info["accelerator"] = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        info["accelerator"] = "cpu"
    return info


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    model_name: str
    model_version: str
    artifact_filename: str
    artifact_sha256: str
    artifact_size_bytes: int
    created_at: str
    git_sha: str
    random_seed: int
    configuration: Mapping[str, Any]
    dependencies: Mapping[str, str]
    hardware: Mapping[str, Any]
    metrics: Mapping[str, Any]
    experiment_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})


def create_manifest(
    artifact: str | Path,
    *,
    model_name: str,
    model_version: str,
    random_seed: int,
    configuration: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    experiment_id: str | None = None,
    repo: str | Path = ".",
    requirements: str | Path | None = "requirements.txt",
) -> ModelManifest:
    """Build a complete, serializable manifest for a model artifact."""
    path = Path(artifact)
    if not path.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {path}")
    created = datetime.now(timezone.utc)
    return ModelManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        model_name=model_name,
        model_version=model_version,
        artifact_filename=path.name,
        artifact_sha256=sha256_file(path),
        artifact_size_bytes=path.stat().st_size,
        created_at=created.isoformat(),
        git_sha=git_sha(repo),
        random_seed=random_seed,
        configuration=dict(configuration),
        dependencies=dependency_versions(requirements),
        hardware=hardware_info(),
        metrics=dict(metrics or {}),
        experiment_id=experiment_id or created.strftime("exp-%Y%m%dT%H%M%S.%fZ"),
    )


class ModelRegistry(Protocol):
    """Storage-neutral model registry contract (filesystem, S3, MLflow, ...)."""

    def register(self, artifact: str | Path, manifest: ModelManifest) -> str: ...
    def load_manifest(self, model_name: str, model_version: str) -> ModelManifest: ...
    def resolve(self, model_name: str, model_version: str) -> Path: ...


class FileSystemModelRegistry:
    """Immutable local registry at ``root/<model>/<version>/``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _version_dir(self, name: str, version: str) -> Path:
        safe = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
        if not safe.fullmatch(name) or not safe.fullmatch(version):
            raise ValueError("model name and version may contain only letters, digits, dot, dash and underscore")
        return self.root / name / version

    def register(self, artifact: str | Path, manifest: ModelManifest) -> str:
        source = Path(artifact)
        if sha256_file(source) != manifest.artifact_sha256:
            raise ValueError("artifact checksum does not match manifest")
        target_dir = self._version_dir(manifest.model_name, manifest.model_version)
        if target_dir.exists():
            raise FileExistsError(f"model version already exists: {target_dir}")
        target_dir.mkdir(parents=True)
        try:
            shutil.copy2(source, target_dir / manifest.artifact_filename)
            (target_dir / "manifest.json").write_text(
                json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        return f"file://{target_dir.resolve()}"

    def load_manifest(self, model_name: str, model_version: str) -> ModelManifest:
        path = self._version_dir(model_name, model_version) / "manifest.json"
        return ModelManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def resolve(self, model_name: str, model_version: str) -> Path:
        manifest = self.load_manifest(model_name, model_version)
        path = self._version_dir(model_name, model_version) / manifest.artifact_filename
        if sha256_file(path) != manifest.artifact_sha256:
            raise ValueError(f"registered artifact failed integrity check: {path}")
        return path


class S3ModelRegistry:
    """Explicit extension point for an S3 implementation.

    Production code can implement :class:`ModelRegistry` using boto3 while
    retaining the same manifest layout.  It is intentionally not silently
    enabled: credentials, encryption and retention are deployment decisions.
    """

    def __init__(self, bucket: str, prefix: str = "models") -> None:
        self.bucket, self.prefix = bucket, prefix

    def register(self, artifact: str | Path, manifest: ModelManifest) -> str:
        raise NotImplementedError("configure an organization-specific S3 registry adapter")

    def load_manifest(self, model_name: str, model_version: str) -> ModelManifest:
        raise NotImplementedError("configure an organization-specific S3 registry adapter")

    def resolve(self, model_name: str, model_version: str) -> Path:
        raise NotImplementedError("S3 artifacts must be downloaded to a verified local cache")


class MLflowModelRegistry:
    """Explicit extension point for MLflow model stages and aliases."""

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri

    def register(self, artifact: str | Path, manifest: ModelManifest) -> str:
        raise NotImplementedError("install mlflow and provide a project-specific adapter")

    def load_manifest(self, model_name: str, model_version: str) -> ModelManifest:
        raise NotImplementedError("install mlflow and provide a project-specific adapter")

    def resolve(self, model_name: str, model_version: str) -> Path:
        raise NotImplementedError("MLflow artifacts must be downloaded to a verified local cache")
