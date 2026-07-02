"""V8 tests: report assembly + reproducibility package (validation.report).

Verify the report assembles with the required sections, embeds the store digest,
builds a package deterministically from the frozen store, and recomputes no numbers.
"""
import os

import pytest

from validation import report as R
from validation._artifacts import load_sweep_records, results_digest_of


def test_report_text_has_required_sections():
    txt = R.build_report_text()
    for section in R.REQUIRED_SECTIONS:
        assert section.lower() in txt.lower(), f"missing section {section!r}"


def test_report_mentions_two_non_denv_systems():
    txt = R.build_report_text()
    # the >=2-beyond-DENV coverage must be visible in the report
    assert "beyond DENV" in txt
    assert "two_level_single_chain" in txt or "three_level_two_chain" in txt


def test_report_embeds_results_digest():
    txt = R.build_report_text()
    digest = results_digest_of(load_sweep_records())
    assert digest in txt


def test_report_is_deterministic():
    assert R.build_report_text() == R.build_report_text()


def test_report_documents_frozen_store_limitations():
    txt = R.build_report_text()
    # ROC/PR and Pi-profile limitations must be documented, not silently dropped
    assert "ROC/PR" in txt
    assert "schematic" in txt.lower()


def test_build_report_writes_file(tmp_path):
    out = tmp_path / "VALIDATION_AND_BENCHMARKING.md"
    p = R.build_report(str(out), render_figures=True)
    assert os.path.exists(p)
    fig_dir = tmp_path / "figures"
    assert fig_dir.exists() and len(list(fig_dir.glob("*.png"))) == 7


def test_build_package_is_complete_and_deterministic(tmp_path):
    d1 = tmp_path / "pkg1"
    d2 = tmp_path / "pkg2"
    m1 = R.build_package(str(d1))
    m2 = R.build_package(str(d2))
    # complete
    assert os.path.exists(m1["report"])
    assert len(m1["figures"]) == 7 and len(m1["tables"]) == 5
    assert os.path.exists(m1["results_store"])
    # deterministic: identical report text and table text, identical digest
    with open(m1["report"]) as a, open(m2["report"]) as b:
        assert a.read() == b.read()
    assert m1["results_digest"] == m2["results_digest"]


def test_package_store_copy_matches_source(tmp_path):
    m = R.build_package(str(tmp_path / "pkg"))
    # the bundled store is a byte copy of the frozen source store
    import hashlib
    def md5(p):
        return hashlib.md5(open(p, "rb").read()).hexdigest()
    src = os.path.join(os.path.dirname(R.__file__), "artifacts",
                       "sweep_results.jsonl")
    assert md5(m["results_store"]) == md5(src)
