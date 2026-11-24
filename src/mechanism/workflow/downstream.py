"""Explicit handoff of trajectory-derived effects to ``STRIDE_downstream``.

The downstream project owns canonical table construction.  This module only
creates its documented Level-1 input layout after an explicit coordinate map is
provided.  It does not infer structural or cross-dataset correspondence.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256
from .schema import WorkflowConfig


class DownstreamExportError(RuntimeError):
    """Raised when per-run effects cannot be exported without ambiguity."""


_REQUIRED_MAP_COLUMNS = ("file_resid", "canon_resid", "chain", "label")
_REQUIRED_EFFECT_COLUMNS = ("file_resid", "resname", "r", "abs_r")


def _integer_column(mapping: pd.DataFrame, column: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(mapping[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DownstreamExportError(
            f"coordinate map {column} values must be integer-valued"
        ) from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.trunc(values)).all():
        raise DownstreamExportError(f"coordinate map {column} values must be integer-valued")
    return numeric.astype(int)


def _coordinate_map(config: WorkflowConfig) -> pd.DataFrame:
    if config.downstream_export is None:
        raise DownstreamExportError(
            "downstream_export is required; provide a dataset identifier and coordinate map"
        )
    path = config.downstream_export.coordinate_map
    try:
        mapping = pd.read_csv(path)
    except Exception as exc:
        raise DownstreamExportError(f"could not read coordinate map {path}: {exc}") from exc
    missing = [column for column in _REQUIRED_MAP_COLUMNS if column not in mapping.columns]
    if missing:
        raise DownstreamExportError(f"coordinate map is missing required columns: {missing}")
    if mapping["file_resid"].isna().any():
        raise DownstreamExportError("coordinate map file_resid values must be present and unique")
    if mapping.loc[:, list(_REQUIRED_MAP_COLUMNS[1:])].isna().any().any():
        raise DownstreamExportError("coordinate map canonical identity values must be present")
    mapping["file_resid"] = _integer_column(mapping, "file_resid")
    mapping["canon_resid"] = _integer_column(mapping, "canon_resid")
    if mapping["file_resid"].duplicated().any():
        raise DownstreamExportError("coordinate map file_resid values must be present and unique")
    canonical_labels = mapping["chain"].astype(str) + ":" + mapping["canon_resid"].astype(str)
    if not mapping["label"].astype(str).eq(canonical_labels).all():
        raise DownstreamExportError(
            "coordinate map label must equal the explicit chain:canon_resid canonical key"
        )
    return mapping.copy()


def _run_effect_files(run_effects_root: Path) -> list[Path]:
    runs_root = run_effects_root / "runs"
    files = sorted(runs_root.glob("*/item_effects.csv"))
    if not files:
        raise DownstreamExportError(
            f"no per-run item_effects.csv files found beneath {runs_root}"
        )
    run_ids = [path.parent.name for path in files]
    if len(run_ids) != len(set(run_ids)):
        raise DownstreamExportError("run-effect directory contains duplicate run identifiers")
    return files


def _bundle_path(root: Path, raw_path: object, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise DownstreamExportError(f"workflow execution manifest has no valid {field}")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DownstreamExportError(f"workflow execution manifest {field} escapes its root") from exc
    return path


def validate_execution_bundle(config: WorkflowConfig, run_effects_root: str | Path) -> list[Path]:
    """Verify hashes and declared run identities before downstream handoff."""

    root = Path(run_effects_root).expanduser().resolve()
    execution_path = root / "workflow_execution.json"
    provenance_path = root / "workflow_provenance.json"
    if not execution_path.is_file() or not provenance_path.is_file():
        raise DownstreamExportError(
            "run-effects directory must be produced by 'stride run' and contain both workflow manifests"
        )
    try:
        manifest = json.loads(execution_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DownstreamExportError(f"could not read workflow execution manifest: {exc}") from exc
    if manifest.get("manifest_version") != 2 or manifest.get("purpose") != "trajectory_to_per_run_item_effects":
        raise DownstreamExportError("workflow execution manifest has an unsupported schema or purpose")
    if manifest.get("workflow_config_sha256") != sha256(config.source_path):
        raise DownstreamExportError("workflow execution manifest does not match the supplied workflow configuration")
    if manifest.get("workflow_provenance_sha256") != sha256(provenance_path):
        raise DownstreamExportError("workflow provenance manifest hash does not match the execution manifest")

    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise DownstreamExportError("workflow execution manifest runs must be a list")
    configured_ids = {run.identifier for run in config.runs}
    observed_ids = [record.get("id") for record in runs if isinstance(record, dict)]
    if len(runs) != len(observed_ids) or set(observed_ids) != configured_ids or len(set(observed_ids)) != len(observed_ids):
        raise DownstreamExportError("workflow execution manifest run identifiers do not match the configuration")

    effects_paths: list[Path] = []
    for record in runs:
        assert isinstance(record, dict)  # established by the identifier check above
        run_id = str(record["id"])
        effects_path = _bundle_path(root, record.get("effects_path"), field="effects_path")
        volume_path = _bundle_path(root, record.get("volume_path"), field="volume_path")
        expected_effects = root / "runs" / run_id / "item_effects.csv"
        if effects_path != expected_effects.resolve() or not effects_path.is_file():
            raise DownstreamExportError(f"workflow execution manifest has an invalid effects path for run {run_id}")
        if not volume_path.is_file():
            raise DownstreamExportError(f"workflow execution manifest has no volume table for run {run_id}")
        if record.get("effects_sha256") != sha256(effects_path):
            raise DownstreamExportError(f"item-effects hash mismatch for run {run_id}")
        if record.get("volume_sha256") != sha256(volume_path):
            raise DownstreamExportError(f"volume-table hash mismatch for run {run_id}")
        effects_paths.append(effects_path)
    return sorted(effects_paths)


def _export_frame(effects_path: Path, mapping: pd.DataFrame) -> pd.DataFrame:
    try:
        effects = pd.read_csv(effects_path)
    except Exception as exc:
        raise DownstreamExportError(f"could not read effects table {effects_path}: {exc}") from exc
    missing = [column for column in _REQUIRED_EFFECT_COLUMNS if column not in effects.columns]
    if missing:
        raise DownstreamExportError(f"effects table {effects_path} is missing required columns: {missing}")
    if effects["file_resid"].isna().any() or effects["file_resid"].duplicated().any():
        raise DownstreamExportError(f"effects table {effects_path} has non-unique file_resid values")
    frame = effects.merge(mapping, on="file_resid", how="left", validate="one_to_one")
    missing_map = frame["canon_resid"].isna()
    if missing_map.any():
        residues = frame.loc[missing_map, "file_resid"].astype(str).tolist()
        raise DownstreamExportError(
            f"coordinate map does not cover run residues in {effects_path}: {', '.join(residues[:10])}"
        )
    frame = frame.rename(columns={"resname": "name"})
    leading = ["file_resid", "canon_resid", "name", "label", "r", "abs_r"]
    remaining = [column for column in frame.columns if column not in leading]
    return frame.loc[:, leading + remaining]


def export_downstream_layout(
    config: WorkflowConfig,
    run_effects_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Write a native Level-1 layout accepted by ``STRIDE_downstream``.

    The output intentionally contains no Level-2 summaries.  Run S0 with
    ``--no-require-summaries`` to canonicalize the exported Level-1 data.
    """

    mapping = _coordinate_map(config)
    source_root = Path(run_effects_root).expanduser().resolve()
    files = validate_execution_bundle(config, source_root)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise DownstreamExportError(f"output directory must not already exist: {destination}")
    assert config.downstream_export is not None  # narrowed by _coordinate_map
    dataset_id = config.downstream_export.dataset_id

    prepared: list[tuple[str, Path, pd.DataFrame]] = []
    coordinate_sets: dict[str, set[str]] = {}
    for effects_path in files:
        run_id = effects_path.parent.name
        exported = _export_frame(effects_path, mapping)
        coordinate_sets[run_id] = set(exported["label"].astype(str))
        prepared.append((run_id, effects_path, exported))
    baseline_run, baseline_coordinates = next(iter(sorted(coordinate_sets.items())))
    inconsistent = {
        run_id: sorted(coordinates.symmetric_difference(baseline_coordinates))[:10]
        for run_id, coordinates in coordinate_sets.items()
        if coordinates != baseline_coordinates
    }
    if inconsistent:
        raise DownstreamExportError(
            "declared runs do not contain the same mapped coordinate set; "
            f"baseline={baseline_run}, differences={inconsistent}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        outputs: list[dict[str, str]] = []
        for run_id, effects_path, exported in prepared:
            target_dir = staging / "replicates" / run_id / dataset_id / "analysis_output"
            target_dir.mkdir(parents=True, exist_ok=False)
            target = target_dir / f"{dataset_id}_correlations_v5.csv"
            exported.to_csv(target, index=False)
            outputs.append(
                {
                    "run_id": run_id,
                    "source_path": str(effects_path),
                    "source_sha256": sha256(effects_path),
                    "output_path": str(target.relative_to(staging)),
                    "output_sha256": sha256(target),
                }
            )
        manifest = {
            "manifest_version": 2,
            "purpose": "stride_to_stride_downstream_level_1_export",
            "dataset_id": dataset_id,
            "coordinate_map_path": str(config.downstream_export.coordinate_map),
            "coordinate_map_sha256": sha256(config.downstream_export.coordinate_map),
            "run_effects_root": str(source_root),
            "workflow_execution_sha256": sha256(source_root / "workflow_execution.json"),
            "outputs": outputs,
            "next_command": (
                "python -m stride_analysis --data-root <this-export-directory> "
                "--output-dir <canonical-output-directory> --no-require-summaries"
            ),
        }
        (staging / "downstream_export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
