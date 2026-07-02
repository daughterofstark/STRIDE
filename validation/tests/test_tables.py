"""V8 tests: manuscript tables (validation.tables) read frozen artifacts only.

Verify tables build from persisted artifacts and that their cells match the underlying
stored records exactly. No recomputation, no frozen empirical value invented.
"""
import pytest

from validation import tables as T
from validation._artifacts import (
    load_metrics, load_method_comparison, load_sweep_records,
)


def test_all_tables_build():
    tabs = T.all_tables()
    assert len(tabs) == 5
    for tbl in tabs:
        assert tbl.header and tbl.rows
        # every row has the same arity as the header
        assert all(len(r) == len(tbl.header) for r in tbl.rows)


def test_markdown_and_csv_render():
    tbl = T.table_calibration()
    md = tbl.to_markdown()
    assert md.startswith("**") and "|" in md
    csv = tbl.to_csv()
    assert csv.count("\n") >= len(tbl.rows)


def test_empirical_table_matches_metrics_artifact():
    m = load_metrics()
    tbl = T.table_empirical_vs_predicted()
    # one row per operating point
    assert len(tbl.rows) == len(m["operating_points"])
    # spot-check: the emp power cell equals the stored value for some (K, beta2)
    op = m["operating_points"][0]
    target = f"{op['empirical_power']:.3f}"
    assert any(target in row for row in tbl.rows)


def test_over_resolution_table_matches_comparison_artifact():
    c = load_method_comparison()
    tbl = T.table_over_resolution()
    # rows = sum over K of (#baselines)
    n_expected = sum(len(cell["comparisons"])
                     for cell in c["comparisons_by_K"].values())
    assert len(tbl.rows) == n_expected


def test_coverage_table_matches_artifact():
    c = load_method_comparison()
    tbl = T.table_coverage()
    assert len(tbl.rows) == len(c["naive_coverage_by_K"])
    for K, cov in c["naive_coverage_by_K"].items():
        assert any(f"{cov:.3f}" in row for row in tbl.rows)


def test_hierarchy_table_covers_all_systems():
    recs = load_sweep_records()
    tbl = T.table_hierarchy_sensitivity()
    systems_in_table = {row[0] for row in tbl.rows}
    systems_in_store = {r["system"] for r in recs}
    assert systems_in_table == systems_in_store
    assert len(tbl.rows) == len(recs)


def test_tables_deterministic():
    a = [t.to_markdown() for t in T.all_tables()]
    b = [t.to_markdown() for t in T.all_tables()]
    assert a == b


def test_write_tables(tmp_path):
    paths = T.write_tables(str(tmp_path))
    assert len(paths) == 5
    for p in paths:
        assert p.endswith(".md")
        with open(p) as fh:
            assert fh.read().startswith("**")
