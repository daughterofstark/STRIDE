"""Tests for the explicit trajectory-workflow contract and provenance plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mechanism.workflow.__main__ import main as module_main
from mechanism.workflow.cli import main
from mechanism.workflow.downstream import DownstreamExportError, export_downstream_layout
from mechanism.workflow.executor import (
    _effective_reference_time_mode,
    _reference_center,
    execute_workflow,
)
from mechanism.workflow.provenance import sha256
from mechanism.workflow.summary import summarize_to_downstream_dataset
from mechanism.workflow.provenance import build_manifest, write_manifest
from mechanism.workflow.schema import WorkflowConfigError, load_workflow_config
from mechanism.workflow.validation import validate_workflow_config


def _write_config(tmp_path, *, povme="tools/POVME2.py", run_count=3):
    for index in range(run_count):
        run_dir = tmp_path / "inputs" / f"run_{index + 1}"
        run_dir.mkdir(parents=True)
        (run_dir / "protein.gro").write_text(f"topology-{index}\n")
        (run_dir / "trajectory.xtc").write_bytes(f"trajectory-{index}".encode())
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    if povme is not None:
        (tools / "POVME2.py").write_text("print('POVME fixture')\n")
    runs = "\n".join(
        f"  - id: run_{index + 1}\n"
        f"    topology: inputs/run_{index + 1}/protein.gro\n"
        f"    trajectory: inputs/run_{index + 1}/trajectory.xtc"
        for index in range(run_count)
    )
    povme_line = "" if povme is None else f"  povme_executable: {povme}\n"
    config = tmp_path / "stride.yaml"
    config.write_text(
        "workflow_version: 1\n"
        "project:\n  system_id: fixture\n"
        f"runs:\n{runs}\n"
        "preprocessing:\n"
        "  protein_selection: protein\n"
        "  alignment_selection: backbone\n"
        "  frame_stride: 20\n"
        "  frame_interval_ps: 10.0\n"
        "pocket:\n"
        "  method: povme\n"
        f"{povme_line}"
        "  radius_angstrom: 14.0\n"
        "  reference:\n"
        "    residues: ['NS3:51', 'NS3:75', 'NS3:135']\n"
        "    selection: 'resid 51 75 135 and name CA'\n"
    )
    return config


def test_load_and_validate_explicit_workflow(tmp_path):
    config = load_workflow_config(_write_config(tmp_path))
    assert config.system_id == "fixture"
    assert [run.identifier for run in config.runs] == ["run_1", "run_2", "run_3"]
    assert validate_workflow_config(config).valid
    assert config.preprocessing.alignment_reference == "topology"
    assert config.pocket.reference_center_mode == "atom_group_com"
    assert config.pocket.grid_spacing_angstrom == 1.0
    assert config.uncertainty.bootstrap_replicates == 1000
    assert config.aggregation.rho_star == 0.5


def test_missing_trajectory_is_a_validation_error(tmp_path):
    config_path = _write_config(tmp_path)
    (tmp_path / "inputs" / "run_2" / "trajectory.xtc").unlink()
    result = validate_workflow_config(load_workflow_config(config_path))
    assert not result.valid
    assert any("run 'run_2' trajectory[0] does not exist" in message for message in result.errors)


def test_duplicate_run_identifiers_are_rejected(tmp_path):
    config_path = _write_config(tmp_path)
    config_path.write_text(config_path.read_text().replace("id: run_2", "id: run_1"))
    result = validate_workflow_config(load_workflow_config(config_path))
    assert not result.valid
    assert any("run identifiers must be unique: run_1" == message for message in result.errors)


def test_ordered_trajectory_segments_are_accepted(tmp_path):
    config_path = _write_config(tmp_path)
    segment = tmp_path / "inputs" / "run_1" / "trajectory_2.xtc"
    segment.write_bytes(b"second segment")
    config_path.write_text(
        config_path.read_text().replace(
            "trajectory: inputs/run_1/trajectory.xtc",
            "trajectories: [inputs/run_1/trajectory.xtc, inputs/run_1/trajectory_2.xtc]",
        )
    )
    config = load_workflow_config(config_path)
    assert len(config.runs[0].trajectories) == 2
    assert validate_workflow_config(config).valid


def test_one_run_is_not_enough_for_replicate_aware_summaries(tmp_path):
    config = load_workflow_config(_write_config(tmp_path, run_count=1))
    result = validate_workflow_config(config)
    assert not result.valid
    assert "at least two runs are required for replicate-aware summaries" in result.errors


def test_missing_povme_is_a_warning_for_planning_and_error_for_execution(tmp_path):
    config = load_workflow_config(_write_config(tmp_path, povme=None))
    assert validate_workflow_config(config).valid
    assert validate_workflow_config(config).warnings
    assert not validate_workflow_config(config, require_povme=True).valid


def test_manifest_hashes_inputs_and_writes_deterministically(tmp_path):
    config = load_workflow_config(_write_config(tmp_path))
    manifest = build_manifest(config, validate_workflow_config(config))
    first = write_manifest(manifest, tmp_path / "plan")
    second = write_manifest(manifest, tmp_path / "plan")
    assert first == second
    payload = json.loads(first.read_text())
    assert payload["purpose"] == "workflow_plan_not_scientific_output"
    assert payload["files"]["runs/run_1/trajectories/0"]["sha256"] != "unavailable"
    assert payload["workflow"]["pocket"]["method"] == "povme"


def test_cli_validate_and_plan(tmp_path):
    config = _write_config(tmp_path)
    assert main(["validate", "--config", str(config)]) == 0
    output = tmp_path / "output"
    assert main(["plan", "--config", str(config), "--output", str(output)]) == 0
    assert (output / "workflow_provenance.json").is_file()


def test_invalid_version_fails_fast(tmp_path):
    config = _write_config(tmp_path)
    config.write_text(config.read_text().replace("workflow_version: 1", "workflow_version: 2"))
    try:
        load_workflow_config(config)
    except WorkflowConfigError as exc:
        assert "workflow_version" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported version did not fail")


def test_duplicate_reference_labels_fail_fast(tmp_path):
    config = _write_config(tmp_path)
    config.write_text(config.read_text().replace("'NS3:135']", "'NS3:135', 'NS3:135']"))
    with pytest.raises(WorkflowConfigError, match="must not contain duplicates"):
        load_workflow_config(config)


def test_module_entrypoint_reexports_the_workflow_cli():
    assert module_main is main


def test_reference_center_modes_are_explicit_and_distinct():
    class Atoms:
        def __init__(self, centre):
            self.centre = np.asarray(centre, dtype=float)

        def center_of_mass(self):
            return self.centre

    class Residue:
        def __init__(self, centre):
            self.atoms = Atoms(centre)

    class Reference(Atoms):
        def __init__(self):
            super().__init__([3.0, 0.0, 0.0])
            self.residues = [Residue([0.0, 0.0, 0.0]), Residue([4.0, 0.0, 0.0])]

    reference = Reference()
    assert np.allclose(_reference_center(reference, "atom_group_com"), [3.0, 0.0, 0.0])
    assert np.allclose(_reference_center(reference, "mean_residue_com"), [2.0, 0.0, 0.0])


def test_reference_time_modes_keep_the_historical_switch_explicit():
    assert _effective_reference_time_mode("fixed", 9.0, 1.5) == "fixed"
    assert _effective_reference_time_mode("dynamic", 0.0, 1.5) == "dynamic"
    assert _effective_reference_time_mode("auto_drift", 1.5, 1.5) == "fixed"
    assert _effective_reference_time_mode("auto_drift", 1.51, 1.5) == "dynamic"


def _pdb_atom(serial, residue, x, y, z):
    return (
        f"ATOM  {serial:5d}  CA  ALA A{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    )


def _write_multiframe_pdb(path: Path, frames: int) -> None:
    lines = []
    for frame in range(frames):
        lines.append(f"MODEL     {frame + 1:4d}\n")
        lines.append(_pdb_atom(1, 1, 0.0, 0.0, 0.0))
        lines.append(_pdb_atom(2, 2, 10.0 + frame * 0.2, 0.0, 0.0))
        lines.append(_pdb_atom(3, 3, 0.0, 8.0 + (frame % 5), 0.0))
        lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_execution_fixture(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    topology = inputs / "topology.pdb"
    trajectory = inputs / "trajectory.pdb"
    _write_multiframe_pdb(topology, 1)
    _write_multiframe_pdb(trajectory, 30)
    povme = tmp_path / "fake_povme.py"
    povme.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "values = {}\n"
        "for line in Path(sys.argv[1]).read_text().splitlines():\n"
        "    if ' ' in line:\n"
        "        key, value = line.split(' ', 1)\n"
        "        values[key] = value\n"
        "n = Path(values['PDBFileName']).read_text().count('ENDMDL')\n"
        "out = Path(values['OutputFilenamePrefix'] + 'volumes.tabbed.txt')\n"
        "out.write_text(''.join(f'{i}\\t{100.0 + i}\\n' for i in range(n)))\n"
    )
    coordinate_map = tmp_path / "coordinates.csv"
    coordinate_map.write_text(
        "file_resid,canon_resid,chain,label\n"
        "1,1,A,A:1\n"
        "2,2,A,A:2\n"
        "3,3,A,A:3\n"
    )
    hierarchy = tmp_path / "hierarchy.yaml"
    hierarchy.write_text(
        "name: synthetic\n"
        "complex_name: synthetic-complex\n"
        "protein_name: synthetic-protein\n"
        "levels: [complex, protein, chain, domain, motif, secondary_structure, residue]\n"
        "numbering:\n  scheme: identity\n"
        "chains:\n  - {name: A, canonical_range: [1, 3]}\n"
        "domains:\n  - {name: test-group, chain: A, residues: [{range: [1, 3]}]}\n"
        "motifs: []\n"
        "secondary_structure:\n  method: none\n"
    )
    config = tmp_path / "stride.yaml"
    config.write_text(
        "workflow_version: 1\n"
        "project:\n  system_id: synthetic-pocket\n"
        "runs:\n"
        "  - id: run_1\n    topology: inputs/topology.pdb\n    trajectory: inputs/trajectory.pdb\n"
        "  - id: run_2\n    topology: inputs/topology.pdb\n    trajectory: inputs/trajectory.pdb\n"
        "preprocessing:\n"
        "  protein_selection: all\n"
        "  alignment_selection: name CA\n"
        "  frame_stride: 1\n"
        "  frame_interval_ps: 10.0\n"
        "pocket:\n"
        "  method: povme\n"
        "  povme_executable: fake_povme.py\n"
        "  radius_angstrom: 4.0\n"
        "  reference:\n"
        "    residues: ['A:1']\n"
        "    selection: 'resid 1 and name CA'\n"
        "hierarchy_config: hierarchy.yaml\n"
        "downstream_export:\n"
        "  dataset_id: synthetic\n"
        "  coordinate_map: coordinates.csv\n"
    )
    return config


def test_executor_writes_per_run_effect_fields_from_declared_inputs(tmp_path):
    config = load_workflow_config(_write_execution_fixture(tmp_path))
    completed = execute_workflow(config, tmp_path / "run-output")
    assert [result.run_id for result in completed] == ["run_1", "run_2"]
    assert all(result.frame_count == 30 for result in completed)
    for result in completed:
        table = __import__("pandas").read_csv(result.effects_path)
        assert len(table) == 3
        assert {"r", "theta_se", "theta_bootstrap_se"}.issubset(table.columns)
        assert (result.effects_path.parent / "povme.ini").is_file()
    summary = json.loads((tmp_path / "run-output" / "workflow_execution.json").read_text())
    assert summary["purpose"] == "trajectory_to_per_run_item_effects"
    assert summary["workflow_config_sha256"] == sha256(config.source_path)
    assert summary["reference_center_mode"] == "atom_group_com"
    assert summary["reference_time_mode_requested"] == "fixed"
    assert len(summary["runs"]) == 2
    assert {row["reference_time_mode_effective"] for row in summary["runs"]} == {"fixed"}
    assert all(row["reference_drift_max_angstrom"] >= 0 for row in summary["runs"])
    assert (tmp_path / "run-output" / "workflow_provenance.json").is_file()


def test_cli_run_executes_declared_workflow(tmp_path):
    config = _write_execution_fixture(tmp_path)
    output = tmp_path / "cli-run-output"
    assert main(["run", "--config", str(config), "--output", str(output)]) == 0
    assert (output / "workflow_execution.json").is_file()


def test_exporter_writes_downstream_level_1_layout_from_explicit_map(tmp_path):
    config = load_workflow_config(_write_execution_fixture(tmp_path))
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    export_root = export_downstream_layout(config, run_effects, tmp_path / "downstream-input")
    for run_id in ("run_1", "run_2"):
        table_path = (
            export_root / "replicates" / run_id / "synthetic" / "analysis_output"
            / "synthetic_correlations_v5.csv"
        )
        table = __import__("pandas").read_csv(table_path)
        assert {"file_resid", "canon_resid", "name", "label", "r", "abs_r"}.issubset(table.columns)
        assert table["label"].tolist() == ["A:1", "A:2", "A:3"]
    manifest = json.loads((export_root / "downstream_export_manifest.json").read_text())
    assert manifest["dataset_id"] == "synthetic"
    assert len(manifest["outputs"]) == 2


def test_cli_exports_downstream_layout(tmp_path):
    config = _write_execution_fixture(tmp_path)
    run_effects = tmp_path / "run-effects"
    assert main(["run", "--config", str(config), "--output", str(run_effects)]) == 0
    export_root = tmp_path / "downstream-input"
    assert main(
        [
            "export-downstream",
            "--config",
            str(config),
            "--run-effects",
            str(run_effects),
            "--output",
            str(export_root),
        ]
    ) == 0
    assert (export_root / "downstream_export_manifest.json").is_file()


def test_exporter_rejects_coordinate_label_that_is_not_a_canonical_key(tmp_path):
    config_path = _write_execution_fixture(tmp_path)
    coordinate_map = tmp_path / "coordinates.csv"
    coordinate_map.write_text(coordinate_map.read_text().replace("A:2", "incorrect-label"))
    config = load_workflow_config(config_path)
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    try:
        export_downstream_layout(config, run_effects, tmp_path / "downstream-input")
    except Exception as exc:
        assert "label must equal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid canonical label was exported")


def test_exporter_rejects_non_integral_coordinate_identifiers(tmp_path):
    config_path = _write_execution_fixture(tmp_path)
    coordinate_map = tmp_path / "coordinates.csv"
    coordinate_map.write_text(coordinate_map.read_text().replace("2,2,A,A:2", "1.5,2,A,A:2"))
    config = load_workflow_config(config_path)
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    with pytest.raises(DownstreamExportError, match="integer-valued"):
        export_downstream_layout(config, run_effects, tmp_path / "downstream-input")
    assert not (tmp_path / "downstream-input").exists()


def test_exporter_rejects_tampered_execution_effects(tmp_path):
    config = load_workflow_config(_write_execution_fixture(tmp_path))
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    effects_path = run_effects / "runs" / "run_1" / "item_effects.csv"
    effects_path.write_text(effects_path.read_text() + "\n")
    with pytest.raises(DownstreamExportError, match="item-effects hash mismatch"):
        export_downstream_layout(config, run_effects, tmp_path / "downstream-input")
    assert not (tmp_path / "downstream-input").exists()


def test_exporter_rejects_execution_from_a_different_configuration(tmp_path):
    config_path = _write_execution_fixture(tmp_path)
    config = load_workflow_config(config_path)
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    config_path.write_text(config_path.read_text().replace("frame_interval_ps: 10.0", "frame_interval_ps: 20.0"))
    changed = load_workflow_config(config_path)
    with pytest.raises(DownstreamExportError, match="does not match the supplied workflow configuration"):
        export_downstream_layout(changed, run_effects, tmp_path / "downstream-input")
    assert not (tmp_path / "downstream-input").exists()


def test_summarizer_writes_level_1_and_level_2_dataset(tmp_path):
    config = load_workflow_config(_write_execution_fixture(tmp_path))
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    dataset = summarize_to_downstream_dataset(config, run_effects, tmp_path / "stride-dataset")
    assert (dataset / "replicates" / "run_1" / "synthetic" / "analysis_output"
            / "synthetic_correlations_v5.csv").is_file()
    assert (dataset / "summaries" / "synthetic_profile.csv").is_file()
    assert (dataset / "summaries" / "synthetic_mechanism.json").is_file()
    manifest = json.loads((dataset / "replicate_aggregation_manifest.json").read_text())
    assert manifest["gate"]["calibrated"] is False
    assert manifest["normalized_aggregation_identity"] == "file_resid=canon_resid"


def test_cli_summarize_creates_complete_dataset(tmp_path):
    config = _write_execution_fixture(tmp_path)
    run_effects = tmp_path / "run-effects"
    assert main(["run", "--config", str(config), "--output", str(run_effects)]) == 0
    dataset = tmp_path / "stride-dataset"
    assert main(
        [
            "summarize",
            "--config",
            str(config),
            "--run-effects",
            str(run_effects),
            "--output",
            str(dataset),
        ]
    ) == 0
    assert (dataset / "replicate_aggregation_manifest.json").is_file()


def test_summary_does_not_leave_partial_dataset_when_report_writing_fails(tmp_path, monkeypatch):
    config = load_workflow_config(_write_execution_fixture(tmp_path))
    run_effects = tmp_path / "run-effects"
    execute_workflow(config, run_effects)
    destination = tmp_path / "stride-dataset"

    def fail_reports(*args, **kwargs):
        raise RuntimeError("forced report failure")

    monkeypatch.setattr("mechanism.workflow.summary.write_reports", fail_reports)
    with pytest.raises(Exception, match="forced report failure"):
        summarize_to_downstream_dataset(config, run_effects, destination)
    assert not destination.exists()
