"""Focused invariants for the pre-registered V9A characterization."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation import v9a_frozen_design as v9a


ARTIFACT_DIR = Path(__file__).parents[1] / "artifacts" / "v9a"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(
        (ARTIFACT_DIR / v9a.SNAPSHOT_FILENAME).read_text(encoding="ascii")
    )


def test_frozen_snapshot_schema_counts_and_k(snapshot):
    verified = v9a.verify_snapshot(snapshot)
    assert snapshot["schema_version"] == "v9a.snapshot.1"
    assert verified["dataset_item_counts"] == [236, 247, 242, 199]
    assert verified["dataset_items_total"] == 924
    assert verified["run_items_total"] == 2772
    assert all(dataset["K"] == 3 for dataset in snapshot["datasets"])


def test_uncertainty_preference_and_fallback():
    assert v9a._resolve_uncertainty(0.2, 0.3) == (
        0.2, "theta_bootstrap_se"
    )
    assert v9a._resolve_uncertainty(np.nan, 0.3) == (0.3, "theta_se")
    with pytest.raises(v9a.SnapshotError, match="no finite uncertainty"):
        v9a._resolve_uncertainty(np.nan, np.nan)
    with pytest.raises(v9a.SnapshotError, match="nonpositive uncertainty"):
        v9a._resolve_uncertainty(0.0, 0.3)


def test_support_membership_preserved(snapshot):
    for dataset in snapshot["datasets"]:
        catalog = {
            support["support_id"]: support
            for support in dataset["support_catalog"]
        }
        for path in dataset["item_paths"]:
            for support_id in path["support_by_level"].values():
                support = catalog[support_id]
                assert path["item_id"] in support["members"]
                assert support["support_size"] == len(support["members"])


def test_duplicate_equivalence_detection(snapshot):
    verified = v9a.verify_snapshot(snapshot)
    assert verified["multi_name_equivalence_classes"] == 36
    assert verified["equivalence_class_counts"] == {
        "secondary_structure=motif=domain": 28,
        "secondary_structure=motif=domain=chain": 4,
        "protein=complex": 4,
    }


def test_collapsed_support_construction_uses_unique_memberships(snapshot):
    for dataset in snapshot["datasets"]:
        collapsed = v9a._collapsed_paths(dataset)
        for rows in collapsed.values():
            support_ids = [row["support_id"] for row in rows]
            assert len(support_ids) == len(set(support_ids))
            assert rows[0]["collapsed_level"] == "item"
            assert rows[-1]["collapsed_level"] == "whole_system_root"


def test_alias_comparison_denominators_distinguish_structure_from_checks(snapshot):
    named = sum(
        dataset["verification"]["named_support_object_count"]
        for dataset in snapshot["datasets"]
    )
    unique = sum(
        dataset["verification"]["unique_support_count"]
        for dataset in snapshot["datasets"]
    )
    assert (named, unique, named - unique) == (1036, 964, 72)
    per_dataset_duplicates = [
        dataset["verification"]["named_support_object_count"]
        - dataset["verification"]["unique_support_count"]
        for dataset in snapshot["datasets"]
    ]
    assert per_dataset_duplicates == [18, 18, 18, 18]
    # Equality of selected support/rho is structural after exact-support
    # deduplication. The empirical checks are the duplicate named rho
    # comparisons performed while resolving production output.
    assert sum(
        duplicates * len(v9a.REGIMES) * v9a.N_DRAWS
        for duplicates in per_dataset_duplicates
    ) == 72000


def test_seedsequence_children_are_deterministic_and_distinct():
    first = v9a.deterministic_child_seeds(
        v9a.MASTER_SEED, ["dataset_1", "dataset_2"], v9a.REGIMES, 5
    )
    second = v9a.deterministic_child_seeds(
        v9a.MASTER_SEED, ["dataset_1", "dataset_2"], v9a.REGIMES, 5
    )
    assert first == second
    flattened = [seed for values in first.values() for seed in values]
    assert len(flattened) == len(set(flattened))


def test_tiny_deterministic_draw_fixture():
    dataset = {
        "items": [
            {
                "item_id": "item_1",
                "canon_resid": 1001,
                "chain_id": "chain_1",
                "mu_frozen": 0.25,
                "sigma2_by_run": [0.01, 0.04, 0.09],
            },
            {
                "item_id": "item_2",
                "canon_resid": 1002,
                "chain_id": "chain_1",
                "mu_frozen": -0.5,
                "sigma2_by_run": [0.04, 0.01, 0.04],
            },
        ]
    }
    first = v9a._draw_frames(dataset, "frozen_scale_fixed_mean", 17)
    second = v9a._draw_frames(dataset, "frozen_scale_fixed_mean", 17)
    assert len(first) == len(second) == 3
    for observed, repeated in zip(first, second):
        pd.testing.assert_frame_equal(observed, repeated)


def test_crossing_semantics_are_first_finite_value_at_point_five():
    rows = [
        {"named_level": "residue", "support_id": "a", "rho": np.nan},
        {"named_level": "secondary_structure", "support_id": "b", "rho": 0.49},
        {"named_level": "motif", "support_id": "b", "rho": 0.50},
        {"named_level": "domain", "support_id": "b", "rho": 0.90},
    ]
    assert v9a._first_crossing(rows, "named_level") == {
        "level": "motif", "support_id": "b", "rho": 0.5
    }


def test_failure_ledger_retains_failed_realization():
    rows = [
        {
            "dataset_id": "dataset_1", "regime": "zero_mean",
            "draw_index": 0, "seed": 11, "success": True,
            "error_type": None, "error_message": None,
        },
        {
            "dataset_id": "dataset_1", "regime": "zero_mean",
            "draw_index": 1, "seed": 12, "success": False,
            "error_type": "RuntimeError", "error_message": "non-finite",
        },
    ]
    assert v9a.build_failure_ledger(rows) == [{
        "dataset_id": "dataset_1",
        "regime": "zero_mean",
        "draw_index": 1,
        "seed": 12,
        "error_type": "RuntimeError",
        "error_message": "non-finite",
    }]


def test_artifact_item_decisions_record_labels_not_empirical_invariance(snapshot):
    crossing = pd.read_csv(
        ARTIFACT_DIR / v9a.FIRST_CROSSING_FILENAME
    )
    item_counts = {
        dataset["dataset_id"]: dataset["item_count"]
        for dataset in snapshot["datasets"]
    }
    expected = crossing["dataset_id"].map(item_counts) * v9a.N_DRAWS
    assert (crossing["expected_item_decisions"] == expected).all()

    alias = pd.read_csv(ARTIFACT_DIR / v9a.ALIAS_FILENAME)
    paired = alias[alias["record_type"] == "paired_metric"]
    membership = paired[
        paired["metric_or_alias_block"] == "selected_membership_changed"
    ]
    rho = paired[
        paired["metric_or_alias_block"] == "selected_rho_changed_gt_1e-12"
    ]
    labels = paired[
        paired["metric_or_alias_block"] == "reported_label_changed"
    ]
    assert (membership["count"] == 0).all()
    assert (rho["count"] == 0).all()
    assert (labels["count"] > 0).all()


def _copy_production_sources(destination: Path) -> None:
    source_root = Path(v9a.__file__).resolve().parents[1]
    manifest = json.loads(v9a.PRODUCTION_SOURCE_MANIFEST.read_text())
    for relative in manifest["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)


def test_provenance_descendant_full_clone_and_detached_head(tmp_path):
    repo = Path(v9a.__file__).resolve().parents[1]
    full = v9a.verify_execution_provenance(repo)
    assert full["git_history_status"] == "frozen_anchor_ancestor_of_head"
    assert full["production_source_status"] == "verified_against_frozen_hashes"

    detached = tmp_path / "detached"
    subprocess.run(["git", "clone", "-q", str(repo), str(detached)], check=True)
    subprocess.run(
        ["git", "checkout", "-q", "--detach", "HEAD"], cwd=detached, check=True
    )
    observed = v9a.verify_execution_provenance(detached)
    assert observed["git_history_status"] == "frozen_anchor_ancestor_of_detached_head"
    assert observed["head_detached"] is True


def test_provenance_exact_anchor_checkout_with_external_v9a(tmp_path):
    repo = Path(v9a.__file__).resolve().parents[1]
    checkout = tmp_path / "anchor"
    subprocess.run(["git", "clone", "-q", str(repo), str(checkout)], check=True)
    subprocess.run(
        ["git", "checkout", "-q", "--detach", v9a.STRIDE_COMMIT],
        cwd=checkout,
        check=True,
    )
    observed = v9a.verify_execution_provenance(checkout)
    assert observed["git_history_status"] == "exact_frozen_anchor"


def test_provenance_shallow_clone_uses_source_hashes(tmp_path):
    repo = Path(v9a.__file__).resolve().parents[1]
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)],
        check=True,
    )
    observed = v9a.verify_execution_provenance(shallow)
    assert observed["git_history_status"] == "anchor_unavailable_in_shallow_checkout"
    assert observed["production_source_status"] == "verified_against_frozen_hashes"


def test_provenance_source_export_without_git_uses_source_hashes(tmp_path):
    export = tmp_path / "export"
    _copy_production_sources(export)
    observed = v9a.verify_execution_provenance(export)
    assert observed["git_history_status"] == "git_metadata_unavailable"
    assert observed["production_source_status"] == "verified_against_frozen_hashes"


def test_provenance_rejects_unrelated_history_and_modified_source(tmp_path):
    unrelated = tmp_path / "unrelated"
    _copy_production_sources(unrelated)
    subprocess.run(["git", "init", "-q", str(unrelated)], check=True)
    subprocess.run(["git", "config", "user.email", "audit@example.invalid"], cwd=unrelated, check=True)
    subprocess.run(["git", "config", "user.name", "Audit"], cwd=unrelated, check=True)
    subprocess.run(["git", "add", "."], cwd=unrelated, check=True)
    subprocess.run(["git", "commit", "-qm", "fabricated"], cwd=unrelated, check=True)
    with pytest.raises(RuntimeError, match="unavailable in non-shallow Git history"):
        v9a.verify_execution_provenance(unrelated)

    export = tmp_path / "modified"
    _copy_production_sources(export)
    source = export / "src/mechanism/statistics/reproducibility.py"
    source.write_text(source.read_text() + "\n# modified\n")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        v9a.verify_execution_provenance(export)


def test_execute_end_to_end_on_committed_snapshot(monkeypatch, snapshot, tmp_path):
    repo = Path(v9a.__file__).resolve().parents[1]
    monkeypatch.setattr(v9a, "extract_snapshot", lambda _root: snapshot)
    monkeypatch.setattr(v9a, "N_DRAWS", 1)
    hashes = v9a.execute(
        tmp_path / "unused-frozen-root",
        tmp_path / "output",
        draws=1,
        workers=1,
        repo_root=repo,
    )
    assert set(hashes) == {
        "snapshot", "results", "support_size", "first_crossing",
        "alias_effects", "manifest", "runtime_provenance",
    }
    runtime = json.loads(
        (tmp_path / "output" / v9a.RUNTIME_PROVENANCE_FILENAME).read_text()
    )
    assert runtime["git_and_source_provenance"]["production_source_status"] == (
        "verified_against_frozen_hashes"
    )
