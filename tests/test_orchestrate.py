"""M6 tests: the orchestration/wiring layer.

These exercise the integration only -- the statistics are covered by the M4/M5
suites. They run without MDAnalysis/POVME/data by laying synthetic per-run
``{proj}_correlations_v5.csv`` files into a run-dir tree on disk, exactly as the
v5 engine would, then driving the orchestration. Key properties checked: per-run
CSVs are read and grouped per protein from disk; the M5 reports are written
alongside existing outputs; existing files are left byte-identical; the K>=2 guard,
the protein whitelist, and the Config plumbing all behave; and the uncalibrated
flag is present.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from mechanism.config import Config
from mechanism.replicate.aggregator import GateConfig
from mechanism.replicate.orchestrate import (
    aggregate_from_rundirs, run_aggregation_tail, gate_config_from,
    _collect_per_protein,
)

_CANON = np.array([51, 75, 135, 152, 153, 154, 10, 20])


def _write_run_csv(base, run, proj, sig_canon, seed, offset=47, signal=0.9):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1e-3, len(_CANON))
    if sig_canon is not None:
        r[np.where(_CANON == sig_canon)[0]] += signal
    df = pd.DataFrame(dict(
        file_resid=_CANON + offset, canon_resid=_CANON, name=["ALA"] * len(_CANON),
        chain=["NS3"] * len(_CANON), r=r, abs_r=np.abs(r),
        theta_se=np.full(len(_CANON), 1e-2),
        theta_bootstrap_se=np.full(len(_CANON), 1e-2)))
    outdir = os.path.join(base, run, proj, "analysis_output")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{proj}_correlations_v5.csv")
    df.to_csv(p, index=False)
    return p


def _hcfg_json(tmp_path, name="DENV2"):
    cfg = {"name": name,
           "chains": [{"name": "NS3", "canonical_range": [1, 999]}],
           "domains": [{"name": "Triad", "residues": [51, 75, 135], "chain": "NS3"}]}
    p = os.path.join(str(tmp_path), "h.json")
    with open(p, "w") as fh:
        json.dump(cfg, fh)
    return p


def _three_runs(base, proj="DENV2"):
    # Triad carrier permutes across the three runs (reproducible at region scale)
    _write_run_csv(base, "1st_run", proj, 51, 0)
    _write_run_csv(base, "2nd_run", proj, 75, 1)
    _write_run_csv(base, "3rd_run", proj, 135, 2)


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ── reading & grouping from disk ─────────────────────────────────────────────
def test_collect_groups_per_protein_in_run_order(tmp_path):
    base = str(tmp_path)
    _three_runs(base)
    per = _collect_per_protein(base, ["1st_run", "2nd_run", "3rd_run"], None)
    assert set(per) == {"DENV2"}
    assert len(per["DENV2"]) == 3
    assert all(isinstance(d, pd.DataFrame) for d in per["DENV2"])


# ── happy path: reports written ──────────────────────────────────────────────
def test_reports_written_with_uncalibrated_flag(tmp_path):
    base = str(tmp_path)
    _three_runs(base)
    hcfg = _hcfg_json(tmp_path)
    written = aggregate_from_rundirs(
        base, ["1st_run", "2nd_run", "3rd_run"], ["DENV2"],
        gate_config=GateConfig(rho_star=0.5), hierarchy_config_path=hcfg)
    assert "DENV2" in written
    assert os.path.exists(written["DENV2"]["profile"])
    assert os.path.exists(written["DENV2"]["mechanism"])
    payload = json.load(open(written["DENV2"]["mechanism"]))
    assert payload["calibrated"] is False
    assert payload["gate"]["rho_star"] == 0.5
    prof = pd.read_csv(written["DENV2"]["profile"])
    assert {"locus", "scale_index", "rho", "gated"}.issubset(prof.columns)


# ── additive only: existing files untouched ──────────────────────────────────
def test_existing_outputs_left_byte_identical(tmp_path):
    base = str(tmp_path)
    _three_runs(base)
    hcfg = _hcfg_json(tmp_path)
    # simulate pre-existing engine artefacts at the triplicate level
    legacy_csv = os.path.join(base, "TriplicateSummary_v5.csv")
    legacy_png = os.path.join(base, "TriplicateDomainHeatmap_v5.png")
    with open(legacy_csv, "w") as fh:
        fh.write("Run,Protein\n1st_run,DENV2\n")
    with open(legacy_png, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n-not-a-real-image-but-fixed-bytes")
    before = {p: _md5(p) for p in (legacy_csv, legacy_png)}
    per_run_before = _md5(os.path.join(
        base, "1st_run", "DENV2", "analysis_output", "DENV2_correlations_v5.csv"))

    aggregate_from_rundirs(base, ["1st_run", "2nd_run", "3rd_run"], ["DENV2"],
                           gate_config=GateConfig(rho_star=0.5),
                           hierarchy_config_path=hcfg)

    after = {p: _md5(p) for p in (legacy_csv, legacy_png)}
    assert after == before, "existing triplicate outputs were modified"
    per_run_after = _md5(os.path.join(
        base, "1st_run", "DENV2", "analysis_output", "DENV2_correlations_v5.csv"))
    assert per_run_after == per_run_before, "per-run CSV was modified"
    # new outputs are present alongside the untouched ones
    assert os.path.exists(os.path.join(base, "DENV2_mechanism.json"))
    assert os.path.exists(os.path.join(base, "DENV2_profile.csv"))


# ── guards ───────────────────────────────────────────────────────────────────
def test_single_replicate_is_skipped(tmp_path):
    base = str(tmp_path)
    _write_run_csv(base, "1st_run", "DENV2", 51, 0)  # only one run
    hcfg = _hcfg_json(tmp_path)
    written = aggregate_from_rundirs(base, ["1st_run", "2nd_run"], ["DENV2"],
                                     gate_config=GateConfig(),
                                     hierarchy_config_path=hcfg)
    assert "DENV2" not in written
    assert not os.path.exists(os.path.join(base, "DENV2_mechanism.json"))


def test_protein_whitelist(tmp_path):
    base = str(tmp_path)
    _three_runs(base, proj="DENV2")
    _three_runs(base, proj="ZIKV")
    hcfg = _hcfg_json(tmp_path)
    written = aggregate_from_rundirs(
        base, ["1st_run", "2nd_run", "3rd_run"], ["DENV2"],
        gate_config=GateConfig(), hierarchy_config_path=hcfg)
    assert set(written) == {"DENV2"}
    assert not os.path.exists(os.path.join(base, "ZIKV_mechanism.json"))


def test_empty_base_is_safe(tmp_path):
    written = aggregate_from_rundirs(str(tmp_path), ["nope"], None)
    assert written == {}


# ── Config plumbing ──────────────────────────────────────────────────────────
def test_gate_config_from_config():
    c = Config(rho_star=0.7, alpha=0.1, coherence_threshold=0.4)
    gc = gate_config_from(c)
    assert (gc.rho_star, gc.alpha, gc.coherence_threshold) == (0.7, 0.1, 0.4)


def test_run_aggregation_tail_uses_config(tmp_path):
    base = str(tmp_path)
    _three_runs(base)
    hcfg = _hcfg_json(tmp_path)
    cfg = Config(base_dir=base, run_dirs=["1st_run", "2nd_run", "3rd_run"],
                 proteins=["DENV2"], hierarchy_config=hcfg, rho_star=0.5)
    written = run_aggregation_tail(cfg)
    assert "DENV2" in written
    payload = json.load(open(written["DENV2"]["mechanism"]))
    assert payload["gate"]["rho_star"] == 0.5
    assert payload["calibrated"] is False


def test_config_defaults_backward_compatible():
    # existing construction still works; new fields have defaults
    c = Config(base_dir="/tmp/x")
    assert c.rho_star == 0.5 and c.alpha == 0.05 and c.coherence_threshold == 0.6
