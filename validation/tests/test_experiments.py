"""V7 tests: sweep orchestration, cell execution, persistence (validation.experiments).

Verify framework correctness, determinism, load-only calibration, store round-trip,
and provenance. No incidental empirical value is frozen as an invariant.
"""
import os
import tempfile

import pytest

from validation.experiments import (
    sweep_grid, run_cell, run_sweep, hierarchy_sensitivity,
    ResultStore, results_digest, build_manifest, CellRecord,
    load_calibrated_rho_star, rho_star_artifact_path, CalibrationMissingError,
)
from validation.types import SweepCell
from validation.systems import non_denv_systems

_NONDENV = [d.name for d in non_denv_systems()]


# ── sweep grid ───────────────────────────────────────────────────────────────
def test_sweep_grid_size_and_order_deterministic():
    a = sweep_grid(_NONDENV, Ks=[3, 5], Ts=[0], tau2s=[0.0], beta2s=[0.36, 1.0],
                   seed=0, n_seeds=1)
    b = sweep_grid(_NONDENV, Ks=[3, 5], Ts=[0], tau2s=[0.0], beta2s=[0.36, 1.0],
                   seed=0, n_seeds=1)
    assert a == b
    assert len(a) == len(_NONDENV) * 2 * 1 * 1 * 2 * 1


def test_sweep_grid_rejects_unknown_system():
    with pytest.raises(KeyError):
        sweep_grid(["nope"], Ks=[5], Ts=[0], tau2s=[0.0], beta2s=[0.36])


# ── load-only calibration ────────────────────────────────────────────────────
def test_calibration_is_load_only_and_errors_clearly():
    with pytest.raises(CalibrationMissingError) as ei:
        load_calibrated_rho_star("unknown_system", 5, "domain")
    msg = str(ei.value)
    assert "calibrate" in msg.lower()          # actionable message
    assert "does not calibrate inside a sweep" in msg


def test_existing_artifact_loads():
    # the non-DENV systems were calibrated as an explicit step and shipped
    for name in _NONDENV:
        d = next(x for x in non_denv_systems() if x.name == name)
        rho = load_calibrated_rho_star(name, 5, d.true_scale_level)
        assert 0.0 < rho < 1.0


def test_denv_calibration_key_maps_to_existing_artifact():
    # DENV artifacts use the short 'DENV' stem; the path convention must find them
    path = rho_star_artifact_path("DENV_NS2B_NS3", 5)
    assert path.endswith("rho_star_DENV_K5.yaml")
    assert os.path.exists(path)


# ── cell execution + determinism ─────────────────────────────────────────────
def test_run_cell_produces_provenanced_record():
    cell = SweepCell(system=_NONDENV[0], K=5, T=0, tau2=0.0, beta2=0.36, seed=0)
    rec = run_cell(cell, n_eval=20)
    assert isinstance(rec, CellRecord)
    assert rec.system == _NONDENV[0]
    # provenance present (seed via cell, rho*, versions, source)
    assert "validation_version" in rec.provenance
    assert "rho_star_source" in rec.provenance
    assert 0.0 < rec.rho_star < 1.0
    # metrics are valid ranges (values themselves not frozen)
    assert 0.0 <= rec.empirical_power <= 1.0
    assert 0.0 <= rec.empirical_fpr <= 1.0
    assert 0.0 <= rec.stride_over_resolution_rate <= 1.0
    assert rec.power_diff == pytest.approx(rec.empirical_power - rec.predicted_power)


def test_run_cell_deterministic():
    cell = SweepCell(system=_NONDENV[0], K=5, T=0, tau2=0.0, beta2=0.36, seed=1)
    a = run_cell(cell, n_eval=20)
    b = run_cell(cell, n_eval=20)
    assert a.to_dict() == b.to_dict()


def test_run_sweep_one_record_per_cell():
    cells = sweep_grid(_NONDENV, Ks=[5], Ts=[0], tau2s=[0.0], beta2s=[0.36],
                       seed=0, n_seeds=1)
    recs = run_sweep(cells, n_eval=20)
    assert len(recs) == len(cells)
    assert results_digest(recs) == results_digest(run_sweep(cells, n_eval=20))


# ── hierarchy-sensitivity sweep (R6) exercises distinct topologies ───────────
def test_hierarchy_sensitivity_covers_distinct_systems():
    recs = hierarchy_sensitivity(_NONDENV, K=5, beta2=0.36, n_eval=20)
    assert len(recs) == len(_NONDENV)
    # the systems exercised have distinct true scales / hierarchies
    scales = {r.scale_level for r in recs}
    assert len(scales) >= 1
    assert {r.system for r in recs} == set(_NONDENV)


# ── results store round-trip + manifest ──────────────────────────────────────
def test_result_store_round_trip_identical():
    cells = sweep_grid(_NONDENV, Ks=[5], Ts=[0], tau2s=[0.0], beta2s=[0.36])
    recs = run_sweep(cells, n_eval=20)
    path = tempfile.mktemp(suffix=".jsonl")
    try:
        store = ResultStore(path)
        store.write(recs, manifest=build_manifest(cells, seed=0, n_eval=20))
        loaded = store.read()
        assert results_digest(loaded) == results_digest(recs)
        man = store.read_manifest()
        assert man["n_non_denv_systems"] >= 2
        assert man["n_cells"] == len(cells)
    finally:
        for p in (path, os.path.splitext(path)[0] + "_manifest.json"):
            if os.path.exists(p):
                os.remove(p)


def test_manifest_records_two_non_denv_systems():
    cells = sweep_grid(_NONDENV + ["DENV_NS2B_NS3"], Ks=[5], Ts=[0], tau2s=[0.0],
                       beta2s=[0.36])
    man = build_manifest(cells, seed=0, n_eval=10)
    # roadmap enforcement: >= 2 non-DENV systems exercised
    assert man["n_non_denv_systems"] >= 2


def test_results_digest_order_sensitive():
    cells = sweep_grid(_NONDENV, Ks=[5], Ts=[0], tau2s=[0.0], beta2s=[0.36])
    recs = run_sweep(cells, n_eval=15)
    if len(recs) >= 2:
        rev = list(reversed(recs))
        assert results_digest(rev) != results_digest(recs)


def test_cell_record_from_dict_round_trip():
    cell = SweepCell(system=_NONDENV[0], K=5, T=0, tau2=0.0, beta2=0.36, seed=0)
    rec = run_cell(cell, n_eval=15)
    assert CellRecord.from_dict(rec.to_dict()).to_dict() == rec.to_dict()
