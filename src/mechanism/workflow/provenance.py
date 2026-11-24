"""Provenance manifests for declared trajectory workflows."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from .schema import WorkflowConfig
from .validation import ValidationResult


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def build_manifest(
    config: WorkflowConfig,
    validation: ValidationResult,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic provenance record without executing analysis."""

    root = repo_root or Path(__file__).resolve().parents[3]
    files: dict[str, dict[str, str]] = {
        "workflow_config": {
            "path": str(config.source_path),
            "sha256": sha256(config.source_path),
        }
    }
    for run in config.runs:
        files[f"runs/{run.identifier}/topology"] = {
            "path": str(run.topology),
            "sha256": sha256(run.topology) if run.topology.is_file() else "unavailable",
        }
        for index, trajectory in enumerate(run.trajectories):
            files[f"runs/{run.identifier}/trajectories/{index}"] = {
                "path": str(trajectory),
                "sha256": sha256(trajectory) if trajectory.is_file() else "unavailable",
            }
    if config.hierarchy_config is not None:
        files["hierarchy_config"] = {
            "path": str(config.hierarchy_config),
            "sha256": sha256(config.hierarchy_config)
            if config.hierarchy_config.is_file()
            else "unavailable",
        }
    if config.pocket.povme_executable is not None:
        files["povme_executable"] = {
            "path": str(config.pocket.povme_executable),
            "sha256": sha256(config.pocket.povme_executable)
            if config.pocket.povme_executable.is_file()
            else "unavailable",
        }

    return {
        "manifest_version": 1,
        "purpose": "workflow_plan_not_scientific_output",
        "workflow": config.canonical_dict(),
        "validation": {
            "valid": validation.valid,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "files": dict(sorted(files.items())),
        "runtime": {
            "mechanism_commit": _git_value(root, "rev-parse", "HEAD"),
            "mechanism_git_status": _git_value(root, "status", "--porcelain"),
            "python": _version("mechanism"),
            "MDAnalysis": _version("MDAnalysis"),
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "PyYAML": _version("PyYAML"),
        },
    }


def write_manifest(manifest: dict[str, Any], output_dir: str | Path) -> Path:
    """Serialize a plan manifest deterministically."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "workflow_provenance.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
