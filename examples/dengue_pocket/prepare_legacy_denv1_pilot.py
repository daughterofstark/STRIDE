#!/usr/bin/env python3
"""Prepare an explicit DENV1 triplicate pilot from the archived v5 layout.

This helper does not run POVME or STRIDE. It only verifies that the three
historical correlation files agree on their file-to-canonical coordinate map,
then writes that map and a new workflow YAML. The later workflow reruns all
trajectory-derived effects from the declared topology and fitted trajectory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


RUNS = ("1st_run", "2nd_run", "3rd_run")
TRIAD_CANONICAL_IDS = (51, 75, 135)


def _read_mapping(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"file_resid", "canon_resid"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        rows = [(int(row["file_resid"]), int(row["canon_resid"])) for row in reader]
    if not rows or len({file_resid for file_resid, _ in rows}) != len(rows):
        raise ValueError(f"{path} has no mapping rows or duplicate file_resid values")
    if len({canonical for _, canonical in rows}) != len(rows):
        raise ValueError(f"{path} has duplicate canon_resid values")
    return sorted(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument("--povme-executable", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--frame-interval-ps",
        required=True,
        type=float,
        help="Raw saved-frame spacing; obtain it from the trajectory/protocol record.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=20,
        help="Keep every Nth saved frame (20 reproduces the historical 20,001 -> 1,001 sampling).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    legacy_root = args.legacy_root.expanduser().resolve()
    workflow_root = args.workflow_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if args.frame_interval_ps <= 0 or args.frame_stride < 1:
        raise SystemExit("--frame-interval-ps and --frame-stride must be positive")

    mappings: dict[str, list[tuple[int, int]]] = {}
    for run in RUNS:
        path = legacy_root / run / "DENV1" / "analysis_output" / "DENV1_correlations_v5.csv"
        if not path.is_file():
            raise SystemExit(f"missing archived mapping source: {path}")
        mappings[run] = _read_mapping(path)
    baseline = mappings[RUNS[0]]
    disagreements = [run for run in RUNS[1:] if mappings[run] != baseline]
    if disagreements:
        raise SystemExit(
            "historical file-to-canonical coordinate maps disagree across runs: "
            + ", ".join(disagreements)
        )

    canonical_to_file = {canonical: file_resid for file_resid, canonical in baseline}
    missing_triad = [value for value in TRIAD_CANONICAL_IDS if value not in canonical_to_file]
    if missing_triad:
        raise SystemExit(f"historical mapping has no canonical triad coordinates: {missing_triad}")
    triad_file_ids = [canonical_to_file[value] for value in TRIAD_CANONICAL_IDS]

    topology_trajectory: list[tuple[str, Path, Path]] = []
    for run in RUNS:
        run_dir = legacy_root / run / "DENV1"
        topology = run_dir / "33-md.gro"
        trajectory = run_dir / "36-md-fit.xtc"
        if not topology.is_file() or not trajectory.is_file():
            raise SystemExit(f"missing historical input for {run}: {topology} / {trajectory}")
        topology_trajectory.append((run, topology, trajectory))
    povme = args.povme_executable.expanduser().resolve()
    if not povme.is_file():
        raise SystemExit(f"POVME executable does not exist: {povme}")
    hierarchy = workflow_root / "src" / "mechanism" / "configs" / "denv_hierarchy.yaml"
    if not hierarchy.is_file():
        raise SystemExit(f"could not find DENV hierarchy config: {hierarchy}")

    output_dir.mkdir(parents=True)
    coordinate_map = output_dir / "DENV1_coordinate_map.csv"
    with coordinate_map.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_resid", "canon_resid", "chain", "label"))
        writer.writeheader()
        for file_resid, canonical in baseline:
            chain = "NS2B" if canonical <= 0 else "NS3"
            writer.writerow(
                {
                    "file_resid": file_resid,
                    "canon_resid": canonical,
                    "chain": chain,
                    "label": f"{chain}:{canonical}",
                }
            )

    selection_ids = " ".join(str(value) for value in triad_file_ids)
    residue_labels = ", ".join(f'"NS3:{value}"' for value in TRIAD_CANONICAL_IDS)
    lines = [
        "workflow_version: 1",
        "",
        "project:",
        "  system_id: DENV1_NS2B_NS3_triplicate_pilot",
        "",
        "runs:",
    ]
    for run, topology, trajectory in topology_trajectory:
        lines.extend(
            [
                f"  - id: {run}",
                f"    topology: {topology}",
                f"    trajectory: {trajectory}",
            ]
        )
    lines.extend(
        [
            "",
            "preprocessing:",
            "  protein_selection: protein",
            "  alignment_selection: backbone",
            "  alignment_reference: topology",
            f"  frame_stride: {args.frame_stride}",
            f"  frame_interval_ps: {args.frame_interval_ps:g}",
            "",
            "pocket:",
            "  method: povme",
            f"  povme_executable: {povme}",
            "  radius_angstrom: 14.0",
            "  grid_spacing_angstrom: 1.0",
            "  distance_cutoff_angstrom: 1.09",
            "  num_processors: 1",
            "  reference:",
            f"    residues: [{residue_labels}]",
            f"    selection: 'protein and resid {selection_ids}'",
            "    center_mode: mean_residue_com",
            "    time_mode: auto_drift",
            "    drift_threshold_angstrom: 1.5",
            "",
            "uncertainty:",
            "  bootstrap_method: circular",
            "  bootstrap_replicates: 1000",
            "  bootstrap_alpha: 0.05",
            "  bootstrap_seed: 42",
            "",
            "aggregation:",
            "  rho_star: 0.5",
            "  alpha: 0.05",
            "  coherence_threshold: 0.6",
            "",
            f"hierarchy_config: {hierarchy}",
            "downstream_export:",
            "  dataset_id: DENV1",
            f"  coordinate_map: {coordinate_map}",
            "",
        ]
    )
    config = output_dir / "DENV1_stride.yaml"
    config.write_text("\n".join(lines))
    print(f"verified {len(baseline)} identical coordinate mappings across {len(RUNS)} runs")
    print(f"canonical triad {TRIAD_CANONICAL_IDS} maps to file residues {triad_file_ids}")
    print(f"wrote {coordinate_map}")
    print(f"wrote {config}")
    print("next: stride validate --config " + str(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
