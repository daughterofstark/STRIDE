"""Versioned configuration for the explicit trajectory-workflow contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class WorkflowConfigError(ValueError):
    """Raised when a workflow configuration is incomplete or malformed."""


@dataclass(frozen=True)
class RunInput:
    """One named trajectory/topology pair belonging to a single system."""

    identifier: str
    topology: Path
    trajectories: tuple[Path, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    protein_selection: str
    alignment_selection: str
    alignment_reference: str
    frame_stride: int
    frame_interval_ps: float


@dataclass(frozen=True)
class PocketConfig:
    method: str
    radius_angstrom: float
    grid_spacing_angstrom: float
    distance_cutoff_angstrom: float
    num_processors: int
    reference_residues: tuple[str, ...]
    reference_selection: str
    reference_center_mode: str
    reference_time_mode: str
    reference_drift_threshold_angstrom: float
    povme_executable: Path | None


@dataclass(frozen=True)
class DownstreamExportConfig:
    """Explicit mapping needed to export run effects to STRIDE_downstream."""

    dataset_id: str
    coordinate_map: Path


@dataclass(frozen=True)
class UncertaintyConfig:
    """Explicit parameters for per-run effective-size and bootstrap uncertainty."""

    bootstrap_method: str
    bootstrap_replicates: int
    bootstrap_alpha: float
    bootstrap_seed: int


@dataclass(frozen=True)
class AggregationConfig:
    """Provisional settings passed unchanged to the established STRIDE aggregator."""

    rho_star: float
    alpha: float
    coherence_threshold: float


@dataclass(frozen=True)
class WorkflowConfig:
    """All inputs required to describe, validate, and audit a trajectory run."""

    workflow_version: int
    system_id: str
    runs: tuple[RunInput, ...]
    preprocessing: PreprocessingConfig
    pocket: PocketConfig
    uncertainty: UncertaintyConfig
    aggregation: AggregationConfig
    hierarchy_config: Path | None
    downstream_export: DownstreamExportConfig | None
    source_path: Path

    def canonical_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""

        return {
            "workflow_version": self.workflow_version,
            "system_id": self.system_id,
            "runs": [
                {
                    "id": run.identifier,
                    "topology": str(run.topology),
                    "trajectories": [str(path) for path in run.trajectories],
                }
                for run in self.runs
            ],
            "preprocessing": asdict(self.preprocessing),
            "pocket": {
                "method": self.pocket.method,
                "radius_angstrom": self.pocket.radius_angstrom,
                "grid_spacing_angstrom": self.pocket.grid_spacing_angstrom,
                "distance_cutoff_angstrom": self.pocket.distance_cutoff_angstrom,
                "num_processors": self.pocket.num_processors,
                "reference_residues": list(self.pocket.reference_residues),
                "reference_selection": self.pocket.reference_selection,
                "reference_center_mode": self.pocket.reference_center_mode,
                "reference_time_mode": self.pocket.reference_time_mode,
                "reference_drift_threshold_angstrom": self.pocket.reference_drift_threshold_angstrom,
                "povme_executable": (
                    str(self.pocket.povme_executable)
                    if self.pocket.povme_executable is not None
                    else None
                ),
            },
            "uncertainty": asdict(self.uncertainty),
            "aggregation": asdict(self.aggregation),
            "hierarchy_config": (
                str(self.hierarchy_config) if self.hierarchy_config is not None else None
            ),
            "downstream_export": (
                {
                    "dataset_id": self.downstream_export.dataset_id,
                    "coordinate_map": str(self.downstream_export.coordinate_map),
                }
                if self.downstream_export is not None
                else None
            ),
        }


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowConfigError(f"{location} must be a mapping")
    return value


def _text(mapping: dict[str, Any], key: str, location: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowConfigError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _number(mapping: dict[str, Any], key: str, location: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WorkflowConfigError(f"{location}.{key} must be numeric")
    return float(value)


def _optional_number(mapping: dict[str, Any], key: str, location: str, default: float) -> float:
    if key not in mapping:
        return default
    return _number(mapping, key, location)


def _positive_int(value: float, location: str) -> int:
    if not value.is_integer() or value < 1:
        raise WorkflowConfigError(f"{location} must be a positive integer")
    return int(value)


def _choice(
    mapping: dict[str, Any], key: str, location: str, *, default: str, choices: set[str]
) -> str:
    if key not in mapping:
        return default
    value = _text(mapping, key, location).lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise WorkflowConfigError(f"{location}.{key} must be one of: {allowed}")
    return value


def _path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _trajectories(entry: dict[str, Any], location: str, config_dir: Path) -> tuple[Path, ...]:
    """Accept one trajectory path or an ordered list, never both."""

    singular = entry.get("trajectory")
    plural = entry.get("trajectories")
    if singular is not None and plural is not None:
        raise WorkflowConfigError(f"{location} may declare trajectory or trajectories, not both")
    if singular is not None:
        if not isinstance(singular, str) or not singular.strip():
            raise WorkflowConfigError(f"{location}.trajectory must be a non-empty path string")
        return (_path(singular.strip(), config_dir),)
    if not isinstance(plural, list) or not plural or not all(
        isinstance(value, str) and value.strip() for value in plural
    ):
        raise WorkflowConfigError(
            f"{location}.trajectories must be a non-empty ordered list of path strings"
        )
    return tuple(_path(value.strip(), config_dir) for value in plural)


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Load an explicit trajectory-workflow YAML configuration.

    Relative paths resolve relative to the YAML file.  This deliberately avoids
    hidden directory conventions and machine-specific defaults.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise WorkflowConfigError(f"workflow configuration does not exist: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise WorkflowConfigError("PyYAML is required to read workflow YAML") from exc

    raw = _mapping(yaml.safe_load(config_path.read_text()) or {}, "root")
    version = raw.get("workflow_version")
    if version != 1:
        raise WorkflowConfigError("workflow_version must be 1")

    project = _mapping(raw.get("project"), "project")
    system_id = _text(project, "system_id", "project")

    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise WorkflowConfigError("runs must be a non-empty list")
    runs: list[RunInput] = []
    for index, value in enumerate(raw_runs):
        entry = _mapping(value, f"runs[{index}]")
        runs.append(
            RunInput(
                identifier=_text(entry, "id", f"runs[{index}]"),
                topology=_path(_text(entry, "topology", f"runs[{index}]"), config_path.parent),
                trajectories=_trajectories(entry, f"runs[{index}]", config_path.parent),
            )
        )

    prep = _mapping(raw.get("preprocessing"), "preprocessing")
    frame_stride = _positive_int(_number(prep, "frame_stride", "preprocessing"), "preprocessing.frame_stride")
    frame_interval_ps = _number(prep, "frame_interval_ps", "preprocessing")
    if frame_interval_ps <= 0:
        raise WorkflowConfigError("preprocessing.frame_interval_ps must be positive")
    alignment_reference = _choice(
        prep,
        "alignment_reference",
        "preprocessing",
        default="topology",
        choices={"topology", "trajectory_first_frame"},
    )

    raw_pocket = _mapping(raw.get("pocket"), "pocket")
    method = _text(raw_pocket, "method", "pocket").lower()
    if method != "povme":
        raise WorkflowConfigError("pocket.method must be 'povme' in workflow version 1")
    radius = _number(raw_pocket, "radius_angstrom", "pocket")
    if radius <= 0:
        raise WorkflowConfigError("pocket.radius_angstrom must be positive")
    grid_spacing = _optional_number(raw_pocket, "grid_spacing_angstrom", "pocket", 1.0)
    if grid_spacing <= 0:
        raise WorkflowConfigError("pocket.grid_spacing_angstrom must be positive")
    distance_cutoff = _optional_number(raw_pocket, "distance_cutoff_angstrom", "pocket", 1.09)
    if distance_cutoff <= 0:
        raise WorkflowConfigError("pocket.distance_cutoff_angstrom must be positive")
    num_processors = _positive_int(
        _optional_number(raw_pocket, "num_processors", "pocket", 1.0), "pocket.num_processors"
    )
    reference = _mapping(raw_pocket.get("reference"), "pocket.reference")
    residue_values = reference.get("residues")
    if not isinstance(residue_values, list) or not residue_values or not all(
        isinstance(value, str) and value.strip() for value in residue_values
    ):
        raise WorkflowConfigError("pocket.reference.residues must be a non-empty list of strings")
    normalized_reference_residues = tuple(value.strip() for value in residue_values)
    if len(set(normalized_reference_residues)) != len(normalized_reference_residues):
        raise WorkflowConfigError("pocket.reference.residues must not contain duplicates")
    reference_center_mode = _choice(
        reference,
        "center_mode",
        "pocket.reference",
        default="atom_group_com",
        choices={"atom_group_com", "mean_residue_com"},
    )
    reference_time_mode = _choice(
        reference,
        "time_mode",
        "pocket.reference",
        default="fixed",
        choices={"fixed", "dynamic", "auto_drift"},
    )
    reference_drift_threshold_angstrom = _optional_number(
        reference, "drift_threshold_angstrom", "pocket.reference", 1.5
    )
    if reference_drift_threshold_angstrom < 0:
        raise WorkflowConfigError(
            "pocket.reference.drift_threshold_angstrom must be non-negative"
        )
    executable_value = raw_pocket.get("povme_executable")
    if executable_value is not None and not isinstance(executable_value, str):
        raise WorkflowConfigError("pocket.povme_executable must be a path string when provided")

    hierarchy_value = raw.get("hierarchy_config")
    if hierarchy_value is not None and not isinstance(hierarchy_value, str):
        raise WorkflowConfigError("hierarchy_config must be a path string when provided")

    export_value = raw.get("downstream_export")
    downstream_export = None
    if export_value is not None:
        export = _mapping(export_value, "downstream_export")
        downstream_export = DownstreamExportConfig(
            dataset_id=_text(export, "dataset_id", "downstream_export"),
            coordinate_map=_path(_text(export, "coordinate_map", "downstream_export"), config_path.parent),
        )

    raw_uncertainty = _mapping(raw.get("uncertainty", {}), "uncertainty")
    bootstrap_method = _choice(
        raw_uncertainty,
        "bootstrap_method",
        "uncertainty",
        default="circular",
        choices={"circular", "stationary"},
    )
    bootstrap_replicates = _positive_int(
        _optional_number(raw_uncertainty, "bootstrap_replicates", "uncertainty", 1000.0),
        "uncertainty.bootstrap_replicates",
    )
    bootstrap_alpha = _optional_number(raw_uncertainty, "bootstrap_alpha", "uncertainty", 0.05)
    if not 0 < bootstrap_alpha < 1:
        raise WorkflowConfigError("uncertainty.bootstrap_alpha must be between 0 and 1")
    bootstrap_seed = _optional_number(raw_uncertainty, "bootstrap_seed", "uncertainty", 42.0)
    if not bootstrap_seed.is_integer() or bootstrap_seed < 0:
        raise WorkflowConfigError("uncertainty.bootstrap_seed must be a non-negative integer")

    raw_aggregation = _mapping(raw.get("aggregation", {}), "aggregation")
    rho_star = _optional_number(raw_aggregation, "rho_star", "aggregation", 0.5)
    if not 0 <= rho_star <= 1:
        raise WorkflowConfigError("aggregation.rho_star must be between 0 and 1")
    aggregation_alpha = _optional_number(raw_aggregation, "alpha", "aggregation", 0.05)
    if not 0 < aggregation_alpha < 1:
        raise WorkflowConfigError("aggregation.alpha must be between 0 and 1")
    coherence_threshold = _optional_number(
        raw_aggregation, "coherence_threshold", "aggregation", 0.6
    )
    if not 0 <= coherence_threshold <= 1:
        raise WorkflowConfigError("aggregation.coherence_threshold must be between 0 and 1")

    return WorkflowConfig(
        workflow_version=version,
        system_id=system_id,
        runs=tuple(runs),
        preprocessing=PreprocessingConfig(
            protein_selection=_text(prep, "protein_selection", "preprocessing"),
            alignment_selection=_text(prep, "alignment_selection", "preprocessing"),
            alignment_reference=alignment_reference,
            frame_stride=frame_stride,
            frame_interval_ps=frame_interval_ps,
        ),
        pocket=PocketConfig(
            method=method,
            radius_angstrom=radius,
            grid_spacing_angstrom=grid_spacing,
            distance_cutoff_angstrom=distance_cutoff,
            num_processors=num_processors,
            reference_residues=normalized_reference_residues,
            reference_selection=_text(reference, "selection", "pocket.reference"),
            reference_center_mode=reference_center_mode,
            reference_time_mode=reference_time_mode,
            reference_drift_threshold_angstrom=reference_drift_threshold_angstrom,
            povme_executable=(
                _path(executable_value, config_path.parent)
                if executable_value is not None
                else None
            ),
        ),
        uncertainty=UncertaintyConfig(
            bootstrap_method=bootstrap_method,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_alpha=bootstrap_alpha,
            bootstrap_seed=int(bootstrap_seed),
        ),
        aggregation=AggregationConfig(
            rho_star=rho_star,
            alpha=aggregation_alpha,
            coherence_threshold=coherence_threshold,
        ),
        hierarchy_config=(
            _path(hierarchy_value, config_path.parent)
            if hierarchy_value is not None
            else None
        ),
        downstream_export=downstream_export,
        source_path=config_path,
    )
