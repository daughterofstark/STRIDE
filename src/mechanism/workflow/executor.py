"""Explicit trajectory-to-effect execution for declared POVME workflows.

This module produces one per-run residue correlation table.  It deliberately
does not fit STRIDE's replicate-aware estimator: that remains a downstream
operation over the produced run-item effect fields.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mechanism.statistics.bootstrap import attach_bootstrap_ci
from mechanism.statistics.correlation import pearson_both
from mechanism.statistics.neff import attach_effective_sample_size

from .provenance import build_manifest, sha256, write_manifest
from .schema import RunInput, WorkflowConfig
from .validation import validate_workflow_config


class WorkflowExecutionError(RuntimeError):
    """Raised when declared trajectory inputs cannot produce an effect field."""


@dataclass(frozen=True)
class RunExecution:
    """Files produced for one declared run."""

    run_id: str
    effects_path: Path
    volume_path: Path
    frame_count: int
    reference_time_mode_effective: str
    reference_drift_mean_angstrom: float
    reference_drift_max_angstrom: float


def _reference_center(reference, mode: str) -> np.ndarray:
    """Return the declared fixed reference centre for a trajectory frame."""

    if mode == "atom_group_com":
        return np.asarray(reference.center_of_mass(), dtype=float)
    if mode == "mean_residue_com":
        residue_centres = [residue.atoms.center_of_mass() for residue in reference.residues]
        if not residue_centres:
            raise WorkflowExecutionError("reference selection did not contain any residues")
        return np.mean(np.asarray(residue_centres, dtype=float), axis=0)
    raise WorkflowExecutionError(f"unsupported reference centre mode: {mode}")


def _effective_reference_time_mode(
    requested: str, max_drift_angstrom: float, threshold_angstrom: float
) -> str:
    """Resolve the declared distance-reference mode without hiding the decision."""

    if requested == "fixed":
        return "fixed"
    if requested == "dynamic":
        return "dynamic"
    if requested == "auto_drift":
        return "dynamic" if max_drift_angstrom > threshold_angstrom else "fixed"
    raise WorkflowExecutionError(f"unsupported reference time mode: {requested}")


def _load_mdanalysis():
    try:
        import MDAnalysis as mda
        from MDAnalysis.analysis import align
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise WorkflowExecutionError(
            "MDAnalysis is required for 'stride run'; install STRIDE with its runtime dependencies"
        ) from exc
    return mda, align


def _universe_for_run(mda, run: RunInput):
    """Open one run's declared trajectory segments in their listed order."""

    coordinates = [str(path) for path in run.trajectories]
    if len(coordinates) == 1:
        return mda.Universe(str(run.topology), coordinates[0])
    return mda.Universe(str(run.topology), coordinates)


def _write_aligned_protein(config: WorkflowConfig, run: RunInput, run_dir: Path) -> Path:
    mda, align = _load_mdanalysis()
    aligned_path = run_dir / "aligned_protein.pdb"
    universe = _universe_for_run(mda, run)
    if config.preprocessing.alignment_reference == "topology":
        reference = mda.Universe(str(run.topology))
    else:
        reference = _universe_for_run(mda, run)
        reference.trajectory[0]
    protein = universe.select_atoms(config.preprocessing.protein_selection)
    if not protein.n_atoms:
        raise WorkflowExecutionError(
            f"run '{run.identifier}' protein_selection produced no atoms: "
            f"{config.preprocessing.protein_selection!r}"
        )
    if not universe.select_atoms(config.preprocessing.alignment_selection).n_atoms:
        raise WorkflowExecutionError(
            f"run '{run.identifier}' alignment_selection produced no atoms: "
            f"{config.preprocessing.alignment_selection!r}"
        )

    with mda.Writer(str(aligned_path), n_atoms=protein.n_atoms, multiframe=True) as writer:
        for _ in universe.trajectory[:: config.preprocessing.frame_stride]:
            align.alignto(universe, reference, select=config.preprocessing.alignment_selection)
            writer.write(protein)
    if not aligned_path.is_file():  # pragma: no cover - MDAnalysis contract guard
        raise WorkflowExecutionError(f"run '{run.identifier}' did not produce aligned coordinates")
    return aligned_path


def _sphere_points(center: np.ndarray, radius: float, spacing: float) -> np.ndarray:
    offsets = np.arange(-radius, radius + spacing * 0.5, spacing)
    grid = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid[np.linalg.norm(grid, axis=1) <= radius] + center


def _write_povme_input(
    aligned_path: Path,
    run_dir: Path,
    center: np.ndarray,
    radius: float,
    grid_spacing: float,
    distance_cutoff: float,
    num_processors: int,
) -> tuple[Path, Path, Path]:
    points_path = run_dir / f"povme_points_{radius:g}A.npy"
    prefix = (run_dir / "povme_").resolve()
    ini_path = run_dir / "povme.ini"
    np.save(points_path, _sphere_points(center, radius, grid_spacing))
    ini_path.write_text(
        "\n".join(
            [
                f"PDBFileName {aligned_path.resolve()}",
                f"LoadPointsFilename {points_path.resolve()}",
                f"DistanceCutoff {distance_cutoff:g}",
                f"OutputFilenamePrefix {prefix}",
                "SaveVolumetricDensityMap false",
                "SaveTabbedVolumeFile true",
                "ConvexHullExclusion true",
                f"NumProcessors {num_processors}",
                "",
            ]
        )
    )
    return ini_path, points_path, run_dir / "povme_volumes.tabbed.txt"


def _run_povme(executable: Path, ini_path: Path, volume_path: Path, run_dir: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(executable), str(ini_path)],
        cwd=run_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    (run_dir / "povme.stdout.log").write_text(completed.stdout)
    (run_dir / "povme.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise WorkflowExecutionError(
            f"POVME failed with exit status {completed.returncode}; see {run_dir / 'povme.stderr.log'}"
        )
    generated = sorted(run_dir.glob("povme_*volumes*.tabbed.txt"))
    if volume_path not in generated:
        if len(generated) != 1:
            raise WorkflowExecutionError(
                f"POVME did not produce exactly one tabulated volume file in {run_dir}"
            )
        generated[0].replace(volume_path)
    if not volume_path.is_file():  # pragma: no cover - defensive contract check
        raise WorkflowExecutionError("POVME completed but no volume table was available")


def _load_volumes(path: Path, expected_frames: int) -> np.ndarray:
    table = pd.read_csv(path, sep="\t", header=None).dropna(how="all")
    column = 1 if table.shape[1] > 1 else 0
    volumes = pd.to_numeric(table.iloc[:, column], errors="coerce").dropna().to_numpy(dtype=float)
    if volumes.size != expected_frames:
        raise WorkflowExecutionError(
            f"POVME volume count ({volumes.size}) does not match aligned frame count ({expected_frames})"
        )
    if not np.isfinite(volumes).all() or np.var(volumes) <= 1e-12:
        raise WorkflowExecutionError("POVME volume trace must be finite and non-constant")
    return volumes


def _residue_distance_matrix(
    config: WorkflowConfig, aligned_path: Path
) -> tuple[list[object], np.ndarray, str, float, float]:
    mda, _ = _load_mdanalysis()
    universe = mda.Universe(str(aligned_path))
    reference = universe.select_atoms(config.pocket.reference_selection)
    residues = list(universe.select_atoms(config.preprocessing.protein_selection).residues)
    if not reference.n_atoms:
        raise WorkflowExecutionError(
            "pocket.reference.selection produced no atoms after preprocessing: "
            f"{config.pocket.reference_selection!r}"
        )
    if not residues:
        raise WorkflowExecutionError("protein_selection produced no residues after preprocessing")
    residue_ids = [int(residue.resid) for residue in residues]
    if len(residue_ids) != len(set(residue_ids)):
        raise WorkflowExecutionError(
            "protein_selection contains duplicate residue identifiers; use an input topology "
            "with unique residue IDs before producing per-run effects"
        )

    atom_indices: list[int] = []
    for residue in residues:
        atom = residue.atoms.select_atoms("name CA")
        atom_indices.append(int((atom[0] if atom.n_atoms else residue.atoms[0]).index))

    universe.trajectory[0]
    fixed_center = _reference_center(reference, config.pocket.reference_center_mode)
    reference_centers = np.empty((len(universe.trajectory), 3), dtype=float)
    for frame_index, _ in enumerate(universe.trajectory):
        reference_centers[frame_index] = _reference_center(
            reference, config.pocket.reference_center_mode
        )
    drifts = np.linalg.norm(reference_centers - fixed_center, axis=1)
    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))
    effective_mode = _effective_reference_time_mode(
        config.pocket.reference_time_mode,
        max_drift,
        config.pocket.reference_drift_threshold_angstrom,
    )
    distances = np.empty((len(universe.trajectory), len(residues)), dtype=float)
    for frame_index, _ in enumerate(universe.trajectory):
        positions = universe.atoms.positions[atom_indices]
        center = reference_centers[frame_index] if effective_mode == "dynamic" else fixed_center
        distances[frame_index] = np.linalg.norm(positions - center, axis=1)
    return residues, distances, effective_mode, mean_drift, max_drift


def _effect_table(
    config: WorkflowConfig, residues: list[object], distances: np.ndarray, volumes: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, residue in enumerate(residues):
        correlation, absolute, p_raw, p_bonf, _, sig_bonf = pearson_both(
            volumes, distances[:, index], len(residues)
        )
        rows.append(
            {
                "file_resid": int(residue.resid),
                "resname": str(residue.resname),
                "r": correlation,
                "abs_r": absolute,
                "p_raw": p_raw,
                "p_bonf": p_bonf,
                "sig_bonf": sig_bonf,
            }
        )
    effects = pd.DataFrame(rows).sort_values("file_resid", kind="stable").reset_index(drop=True)
    # The uncertainty helpers preserve existing effect values and append only
    # their documented uncertainty columns.
    effects = attach_effective_sample_size(effects, residues, distances, volumes)
    return attach_bootstrap_ci(
        effects,
        residues,
        distances,
        volumes,
        method=config.uncertainty.bootstrap_method,
        B=config.uncertainty.bootstrap_replicates,
        alpha=config.uncertainty.bootstrap_alpha,
        seed=config.uncertainty.bootstrap_seed,
    )


def _execute_run(config: WorkflowConfig, run: RunInput, output_dir: Path) -> RunExecution:
    run_dir = output_dir / "runs" / run.identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    aligned_path = _write_aligned_protein(config, run, run_dir)
    mda, _ = _load_mdanalysis()
    aligned = mda.Universe(str(aligned_path))
    reference = aligned.select_atoms(config.pocket.reference_selection)
    if not reference.n_atoms:
        raise WorkflowExecutionError(
            f"run '{run.identifier}' reference_selection produced no atoms in the aligned coordinates"
        )
    if len(reference.residues) != len(config.pocket.reference_residues):
        raise WorkflowExecutionError(
            f"run '{run.identifier}' reference_selection resolved to {len(reference.residues)} residues, "
            f"but pocket.reference.residues declares {len(config.pocket.reference_residues)}"
        )
    aligned.trajectory[0]
    fixed_center = _reference_center(reference, config.pocket.reference_center_mode)
    ini_path, _points_path, volume_path = _write_povme_input(
        aligned_path,
        run_dir,
        fixed_center,
        config.pocket.radius_angstrom,
        config.pocket.grid_spacing_angstrom,
        config.pocket.distance_cutoff_angstrom,
        config.pocket.num_processors,
    )
    if config.pocket.povme_executable is None:  # checked earlier, retained for type narrowing
        raise WorkflowExecutionError("POVME executable is required for execution")
    _run_povme(config.pocket.povme_executable, ini_path, volume_path, run_dir)
    residues, distances, effective_mode, mean_drift, max_drift = _residue_distance_matrix(
        config, aligned_path
    )
    volumes = _load_volumes(volume_path, distances.shape[0])
    effects = _effect_table(config, residues, distances, volumes)
    effects_path = run_dir / "item_effects.csv"
    effects.to_csv(effects_path, index=False)
    return RunExecution(
        run.identifier,
        effects_path,
        volume_path,
        int(distances.shape[0]),
        effective_mode,
        mean_drift,
        max_drift,
    )


def execute_workflow(config: WorkflowConfig, output_dir: str | Path) -> tuple[RunExecution, ...]:
    """Run the declared POVME-to-per-residue effect workflow.

    Results are intentionally limited to one effect field per run.  They can
    then be passed to STRIDE's replicate-aware processing with an explicit
    downstream data contract.
    """

    validation = validate_workflow_config(config, require_povme=True)
    if not validation.valid:
        raise WorkflowExecutionError("workflow configuration is invalid: " + "; ".join(validation.errors))
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise WorkflowExecutionError(f"output directory must not already exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        provenance_path = write_manifest(build_manifest(config, validation), staging)
        completed = tuple(_execute_run(config, run, staging) for run in config.runs)
        summary = {
            "manifest_version": 2,
            "purpose": "trajectory_to_per_run_item_effects",
            "system_id": config.system_id,
            "workflow_config_sha256": sha256(config.source_path),
            "workflow_provenance_sha256": sha256(provenance_path),
            "reference_selection": config.pocket.reference_selection,
            "reference_center_mode": config.pocket.reference_center_mode,
            "reference_time_mode_requested": config.pocket.reference_time_mode,
            "reference_drift_threshold_angstrom": config.pocket.reference_drift_threshold_angstrom,
            "runs": [
                {
                    "id": result.run_id,
                    "effects_path": str(result.effects_path.relative_to(staging)),
                    "effects_sha256": sha256(result.effects_path),
                    "volume_path": str(result.volume_path.relative_to(staging)),
                    "volume_sha256": sha256(result.volume_path),
                    "aligned_protein_path": str((result.effects_path.parent / "aligned_protein.pdb").relative_to(staging)),
                    "aligned_protein_sha256": sha256(result.effects_path.parent / "aligned_protein.pdb"),
                    "povme_input_path": str((result.effects_path.parent / "povme.ini").relative_to(staging)),
                    "povme_input_sha256": sha256(result.effects_path.parent / "povme.ini"),
                    "frame_count": result.frame_count,
                    "reference_time_mode_effective": result.reference_time_mode_effective,
                    "reference_drift_mean_angstrom": result.reference_drift_mean_angstrom,
                    "reference_drift_max_angstrom": result.reference_drift_max_angstrom,
                }
                for result in completed
            ],
        }
        for run_record in summary["runs"]:
            run_dir = staging / "runs" / str(run_record["id"])
            points_files = sorted(run_dir.glob("povme_points_*.npy"))
            if len(points_files) != 1:  # pragma: no cover - executor contract guard
                raise WorkflowExecutionError(f"expected one POVME point file for run {run_record['id']}")
            run_record["povme_points_path"] = str(points_files[0].relative_to(staging))
            run_record["povme_points_sha256"] = sha256(points_files[0])
            for log_name in ("povme.stdout.log", "povme.stderr.log"):
                log_path = run_dir / log_name
                run_record[f"{log_name}_sha256"] = sha256(log_path)
        (staging / "workflow_execution.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(destination)
        return tuple(
            RunExecution(
                result.run_id,
                destination / result.effects_path.relative_to(staging),
                destination / result.volume_path.relative_to(staging),
                result.frame_count,
                result.reference_time_mode_effective,
                result.reference_drift_mean_angstrom,
                result.reference_drift_max_angstrom,
            )
            for result in completed
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
