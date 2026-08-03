"""Focused invariants for the pre-registered V9A characterization."""
from __future__ import annotations

import json
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
