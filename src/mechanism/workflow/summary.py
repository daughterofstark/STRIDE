"""Build a complete downstream-ready STRIDE dataset from declared run effects."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from mechanism.config.hierarchy_schema import HierarchyConfig
from mechanism.replicate.aggregator import GateConfig, run_aggregation
from mechanism.reports.mechanism_report import write_reports

from .downstream import (
    _coordinate_map,
    _export_frame,
    export_downstream_layout,
    validate_execution_bundle,
)
from .provenance import sha256
from .schema import WorkflowConfig


class WorkflowSummaryError(RuntimeError):
    """Raised when a declared hierarchy cannot summarize run effect fields."""


def _aggregation_frame(effects_path: Path, coordinate_map: pd.DataFrame) -> pd.DataFrame:
    """Make a canonical-coordinate copy for the established aggregation API.

    The production API reconstructs hierarchy membership from ``file_resid``
    using a constant offset.  The public workflow permits an explicit mapping,
    so aggregation receives a private normalized copy with ``file_resid`` equal
    to the supplied canonical id.  Exported Level-1 files retain their original
    structural residue identifiers.
    """

    frame = _export_frame(effects_path, coordinate_map)
    frame["original_file_resid"] = frame["file_resid"]
    frame["file_resid"] = frame["canon_resid"].astype(int)
    frame["canon_resid"] = frame["canon_resid"].astype(int)
    return frame


def _validate_map_chains(coordinate_map: pd.DataFrame, hierarchy: HierarchyConfig) -> None:
    """Refuse a mapping whose chain fields conflict with configured ranges."""

    for chain in hierarchy.chains:
        mask = coordinate_map["canon_resid"].between(*chain.canonical_range)
        conflicting = coordinate_map.loc[mask & coordinate_map["chain"].ne(chain.name)]
        if not conflicting.empty:
            examples = conflicting.loc[:, ["file_resid", "canon_resid", "chain"]].head(3)
            raise WorkflowSummaryError(
                "coordinate map chain conflicts with hierarchy_config ranges: "
                f"{examples.to_dict('records')}"
            )


def summarize_to_downstream_dataset(
    config: WorkflowConfig,
    run_effects_root: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create Level-1 inputs plus Level-2 summaries for ``STRIDE_downstream``.

    The result is a self-contained dataset root accepted by S0 with its default
    requirement for both Level-1 replicate inputs and Level-2 summaries. Gate
    settings remain the established provisional defaults in :class:`GateConfig`.
    """

    if config.hierarchy_config is None:
        raise WorkflowSummaryError(
            "hierarchy_config is required to build replicate-aware support summaries"
        )
    try:
        hierarchy = HierarchyConfig.load(config.hierarchy_config)
    except Exception as exc:
        raise WorkflowSummaryError(
            f"could not load hierarchy_config {config.hierarchy_config}: {exc}"
        ) from exc

    mapping = _coordinate_map(config)
    _validate_map_chains(mapping, hierarchy)
    effects_root = Path(run_effects_root).expanduser().resolve()
    files = validate_execution_bundle(config, effects_root)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise WorkflowSummaryError(f"output directory must not already exist: {destination}")
    assert config.downstream_export is not None  # narrowed by _coordinate_map

    aggregation_frames = [_aggregation_frame(path, mapping) for path in files]
    try:
        profile, mechanisms, unresolved, meta = run_aggregation(
            aggregation_frames,
            hierarchy,
            GateConfig(
                rho_star=config.aggregation.rho_star,
                alpha=config.aggregation.alpha,
                coherence_threshold=config.aggregation.coherence_threshold,
            ),
            protein=config.downstream_export.dataset_id,
        )
    except Exception as exc:
        raise WorkflowSummaryError(f"replicate-aware aggregation failed: {exc}") from exc
    if profile.empty:
        raise WorkflowSummaryError("replicate-aware aggregation returned an empty profile")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    shutil.rmtree(staging)
    try:
        export_downstream_layout(config, effects_root, staging)
        summaries = staging / "summaries"
        paths = write_reports(
            profile,
            mechanisms,
            unresolved,
            meta,
            str(summaries),
            prefix=config.downstream_export.dataset_id,
        )
        manifest = {
            "manifest_version": 2,
            "purpose": "trajectory_to_replicate_aware_stride_dataset",
            "dataset_id": config.downstream_export.dataset_id,
            "workflow_config_sha256": sha256(config.source_path),
            "workflow_execution_sha256": sha256(effects_root / "workflow_execution.json"),
            "hierarchy_config_path": str(config.hierarchy_config),
            "hierarchy_config_sha256": sha256(config.hierarchy_config),
            "normalized_aggregation_identity": "file_resid=canon_resid",
            "profile_path": str(Path(paths["profile"]).relative_to(staging)),
            "profile_sha256": sha256(Path(paths["profile"])),
            "mechanism_path": str(Path(paths["mechanism"]).relative_to(staging)),
            "mechanism_sha256": sha256(Path(paths["mechanism"])),
            "gate": {
                "rho_star": meta["rho_star"],
                "alpha": meta["alpha"],
                "coherence_threshold": meta["coherence_threshold"],
                "calibrated": meta["calibrated"],
            },
            "next_command": (
                "python -m stride_analysis --data-root <this-dataset-directory> "
                "--output-dir <canonical-output-directory>"
            ),
        }
        (staging / "replicate_aggregation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
