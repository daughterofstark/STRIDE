"""Validation for explicit trajectory-workflow configurations."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import WorkflowConfig


@dataclass(frozen=True)
class ValidationResult:
    """Validation findings without executing any trajectory analysis."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_workflow_config(config: WorkflowConfig, *, require_povme: bool = False) -> ValidationResult:
    """Validate files and identifiers required by a declared workflow.

    The default permits a plan to be prepared before POVME is installed.  An
    executor should pass ``require_povme=True`` before beginning a run.
    """

    errors: list[str] = []
    warnings: list[str] = []
    identifiers = [run.identifier for run in config.runs]
    for identifier in identifiers:
        if identifier in {".", ".."} or "/" in identifier or "\\" in identifier:
            errors.append(f"run identifier must be a single safe directory name: {identifier!r}")
    duplicate_ids = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicate_ids:
        errors.append(f"run identifiers must be unique: {', '.join(duplicate_ids)}")
    if len(config.runs) < 2:
        errors.append("at least two runs are required for replicate-aware summaries")

    for run in config.runs:
        for label, path in (("topology", run.topology), *(
            (f"trajectory[{index}]", path) for index, path in enumerate(run.trajectories)
        )):
            if not path.is_file():
                errors.append(f"run '{run.identifier}' {label} does not exist: {path}")

    if config.hierarchy_config is not None and not config.hierarchy_config.is_file():
        errors.append(f"hierarchy_config does not exist: {config.hierarchy_config}")
    if (
        config.downstream_export is not None
        and not config.downstream_export.coordinate_map.is_file()
    ):
        errors.append(
            "downstream_export.coordinate_map does not exist: "
            f"{config.downstream_export.coordinate_map}"
        )

    executable = config.pocket.povme_executable
    if executable is None:
        message = "pocket.povme_executable is not set; an executor must supply POVME explicitly"
        (errors if require_povme else warnings).append(message)
    elif not executable.is_file():
        errors.append(f"pocket.povme_executable does not exist: {executable}")

    return ValidationResult(tuple(errors), tuple(warnings))
