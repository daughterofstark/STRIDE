"""V5 tests: empirical operating characteristics + empirical-vs-predicted framework.

These tests verify the **correctness and determinism of the evaluation framework** —
not particular empirical values of the current estimator. Per the V5 mandate:
prediction (V3), calibration (V4), and empirical evaluation (V5) are independent; a
disagreement is *reported*, never encoded as a required behavior. No test asserts that
empirical power exceeds (or falls below) predicted power by any amount or direction.
"""
import os
import tempfile

import numpy as np
import pytest

from validation.generate import (
    SyntheticSystemSpec, SynChain, SynDomain, Driver, NullRegion,
    generate_system, region_path,
)
from validation.adapters import to_hierarchy_config, aggregate_via_production
from validation._seed import spawn_seeds
from validation.calibrate import load_rho_star_yaml
from validation.predicted import ScalePrediction
from validation.metrics import (
    empirical_crossing_rate, empirical_rho_recovery, empirical_coverage,
    empirical_hierarchy_recovery, empirical_over_resolution_rate,
    check_I2_upward_closed, check_I3_standardization_invariance, roc_auc,
    operating_point, ell_min_grid, OperatingPoint, MetricsReport,
    write_metrics_report, load_metrics_report,
)

_LEVELS = ("complex", "protein", "chain", "domain", "residue")
_ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                         "rho_star_DENV_K5.yaml")


def _spec(seed, beta, tau2=0.0, K=5, sigma2=0.04):
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    drivers = () if beta == 0 else (
        Driver(support=triad, scale_level="domain",
               region_id=region_path(chain="NS3", domain="Triad"),
               beta=beta, tau2=tau2, carrier_mode="distributed"),)
    nulls = (NullRegion(triad if beta == 0 else oxy, "domain",
             region_path(chain="NS3", domain="Triad" if beta == 0 else "Oxy")),)
    return SyntheticSystemSpec(
        name="DENV_NS2B_NS3", levels=_LEVELS, chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"), SynDomain("Oxy", oxy, "NS3")),
        residues=triad + oxy, drivers=drivers, nulls=nulls, K=K, sigma2=sigma2,
        offset=0, seed=seed, true_ell_star=1)


def _cfg():
    return to_hierarchy_config(_spec(1, 1.0))


def _nullf(s):
    return list(generate_system(_spec(s, 0)).per_run_dfs)


def _drivf(beta=0.8, K=5):
    def build(s):
        return list(generate_system(_spec(s, beta, K=K)).per_run_dfs)
    return build


# ── empirical crossing rate (FPR / power) is a correct rate in [0,1] ─────────
def test_empirical_crossing_rate_in_unit_interval():
    r = empirical_crossing_rate(_nullf, _cfg(), seeds=spawn_seeds(1, 30),
                                rho_star=0.8, scale_level="domain", protein="DENV_NS2B_NS3")
    assert 0.0 <= r <= 1.0


def test_empirical_crossing_rate_monotone_in_threshold():
    # a higher threshold can only lower (or equal) the crossing rate — framework check
    seeds = spawn_seeds(2, 40)
    f = _drivf(beta=0.6)
    low = empirical_crossing_rate(f, _cfg(), seeds=seeds, rho_star=0.5,
                                  scale_level="domain", label="Triad", protein="DENV_NS2B_NS3")
    high = empirical_crossing_rate(f, _cfg(), seeds=seeds, rho_star=0.95,
                                   scale_level="domain", label="Triad", protein="DENV_NS2B_NS3")
    assert high <= low + 1e-12


def test_empirical_crossing_rate_deterministic():
    seeds = spawn_seeds(3, 25)
    a = empirical_crossing_rate(_nullf, _cfg(), seeds=seeds, rho_star=0.8,
                                scale_level="domain", protein="DENV_NS2B_NS3")
    b = empirical_crossing_rate(_nullf, _cfg(), seeds=seeds, rho_star=0.8,
                                scale_level="domain", protein="DENV_NS2B_NS3")
    assert a == b


# ── FPR at the calibrated rho* is controlled (reproduces V4 through V5 layer) ─
def test_empirical_fpr_at_calibrated_rho_star_controlled():
    cal = load_rho_star_yaml(_ARTIFACT)
    fpr = empirical_crossing_rate(
        _nullf, _cfg(), seeds=spawn_seeds(9999, 200),
        rho_star=cal.rho_star["domain"], scale_level="domain",
        protein="DENV_NS2B_NS3")
    # the (Cal) guarantee from V4 should hold through the metrics layer (MC slack)
    assert fpr <= cal.alpha + 0.03


# ── consistency (C): rho_hat -> rho_true as sigma^2 shrinks ──────────────────
def test_rho_recovery_converges_to_truth():
    def mfs(sig2, s):
        return list(generate_system(_spec(s, 1.0, K=10, sigma2=sig2)).per_run_dfs)
    rows = empirical_rho_recovery(mfs, _cfg(), sigma2_grid=[0.2, 0.04, 0.01],
                                  beta2=1.0, tau2=0.0, seeds_per_point=20, seed=8000,
                                  scale_level="domain", label="Triad",
                                  protein="DENV_NS2B_NS3")
    # the gap |rho_hat - rho_true| shrinks as sigma^2 -> 0 (consistency framework)
    gaps = [abs(r["mean_rho_hat"] - r["rho_true"]) for r in rows]
    assert gaps[-1] <= gaps[0] + 1e-9
    # and rho_hat approaches 1 in the low-noise limit
    assert rows[-1]["mean_rho_hat"] > 0.95


# ── coverage is a valid fraction; framework returns a well-formed summary ────
def test_coverage_is_valid_fraction():
    cov = empirical_coverage(
        lambda s: list(generate_system(_spec(s, 1.0, tau2=0.05, K=5)).per_run_dfs),
        _cfg(), seeds=spawn_seeds(9000, 60), scale_level="domain", label="Triad",
        protein="DENV_NS2B_NS3")
    assert 0.0 <= cov["coverage"] <= 1.0
    assert cov["n"] > 0 and np.isfinite(cov["mean_beta"])


# ── hierarchy recovery returns coherent counts; recall in [0,1] ──────────────
def test_hierarchy_recovery_counts_coherent():
    cal = load_rho_star_yaml(_ARTIFACT)
    hr = empirical_hierarchy_recovery(
        _drivf(beta=1.0), _cfg(), seeds=spawn_seeds(7000, 40),
        rho_star=cal.rho_star["domain"], true_scale_level="domain",
        driver_region_substr="Triad", protein="DENV_NS2B_NS3")
    assert 0.0 <= hr["recall"] <= 1.0
    # emitted decomposes exactly into correct + finer + coarser
    assert hr["emitted"] == hr["correct_scale"] + hr["finer_than_truth"] + hr["coarser_than_truth"]


def test_over_resolution_rate_in_unit_interval():
    cal = load_rho_star_yaml(_ARTIFACT)
    r = empirical_over_resolution_rate(
        _drivf(beta=1.0), _cfg(), seeds=spawn_seeds(7000, 40),
        rho_star=cal.rho_star["domain"], true_scale_level="domain",
        driver_region_substr="Triad", protein="DENV_NS2B_NS3")
    assert 0.0 <= r <= 1.0


# ── I2 (upward-closed passable set) and I3 (standardization invariance) ──────
def test_I2_passable_set_upward_closed():
    cal = load_rho_star_yaml(_ARTIFACT)
    frac = check_I2_upward_closed(_drivf(beta=1.0), _cfg(),
                                  seeds=spawn_seeds(7000, 30),
                                  rho_star=cal.rho_star["domain"], protein="DENV_NS2B_NS3")
    # I2 is a structural property of the nested hierarchy; must hold on all draws
    assert frac == pytest.approx(1.0)


def test_I3_standardization_invariance_holds():
    worst = check_I3_standardization_invariance(_drivf(beta=0.8), _cfg(),
                                                seeds=spawn_seeds(7000, 15),
                                                scale_factor=7.0, protein="DENV_NS2B_NS3")
    # rescaling the effect field + its uncertainty must not change rho_hat (I3)
    assert worst < 1e-9


# ── ROC/AUC correctness on constructed cases (reporting) ─────────────────────
def test_roc_auc_perfect_separation():
    assert roc_auc([0.9, 0.8, 0.95], [0.1, 0.2, 0.05]) == pytest.approx(1.0)


def test_roc_auc_chance_on_identical():
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)


def test_roc_auc_reversed_is_zero():
    assert roc_auc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)


def test_roc_auc_driver_beats_null_on_data():
    # constructed empirical case: driver-region rho vs null-region rho
    cfg = _cfg()
    pos, neg = [], []
    for s in spawn_seeds(1234, 30):
        o = aggregate_via_production(_drivf(beta=1.0)(s), cfg, protein="DENV_NS2B_NS3")
        pos.append(float(o.query("scale_level=='domain' and label=='Triad'")["rho"].iloc[0]))
        neg.append(float(o.query("scale_level=='domain' and label=='Oxy'")["rho"].iloc[0]))
    assert roc_auc(pos, neg) > 0.9


# ── the empirical-vs-predicted operating point: framework correctness only ───
def test_operating_point_pairs_empirical_and_predicted():
    cal = load_rho_star_yaml(_ARTIFACT)
    op = operating_point(
        _nullf, _drivf(beta=0.8), _cfg(), K=5, T=0, tau2=0.0, beta2=0.64,
        sigma2_bar=0.04, rho_star=cal.rho_star["domain"], scale_level="domain",
        driver_label="Triad", seeds_null=spawn_seeds(9999, 60),
        seeds_driver=spawn_seeds(6000, 60), protein="DENV_NS2B_NS3")
    # framework correctness: both values present, in range, diff is their difference
    assert 0.0 <= op.empirical_power <= 1.0
    assert 0.0 <= op.predicted_power <= 1.0
    assert op.power_diff == pytest.approx(op.empirical_power - op.predicted_power)
    assert op.predicted_fpr == pytest.approx(0.05)
    # NOTE: deliberately no assertion on the sign/size of power_diff — the
    # empirical-vs-predicted gap is a characterization of the current estimator,
    # not a repository invariant (V5 mandate).


def test_operating_point_deterministic():
    cal = load_rho_star_yaml(_ARTIFACT)
    kw = dict(K=5, T=0, tau2=0.0, beta2=0.64, sigma2_bar=0.04,
              rho_star=cal.rho_star["domain"], scale_level="domain",
              driver_label="Triad", seeds_null=spawn_seeds(9999, 40),
              seeds_driver=spawn_seeds(6000, 40), protein="DENV_NS2B_NS3")
    a = operating_point(_nullf, _drivf(beta=0.8), _cfg(), **kw)
    b = operating_point(_nullf, _drivf(beta=0.8), _cfg(), **kw)
    assert a.to_dict() == b.to_dict()


# ── ell_min grid uses the fixed V3 reference ─────────────────────────────────
def test_ell_min_grid_uses_predicted_reference():
    def scales_KT(K, T):
        # finer scales gain predicted rho with K (monotone construction)
        return [ScalePrediction(0, 0.3 + 0.03 * K),
                ScalePrediction(1, 0.85),
                ScalePrediction(2, 0.9)]
    grid = ell_min_grid(scales_KT, 0.8, [3, 5, 10], [100, 200])
    assert len(grid) == 3 * 2
    for row in grid:
        assert set(row) == {"K", "T", "ell_min"}


# ── report artifact round-trip with provenance ───────────────────────────────
def test_metrics_report_round_trip():
    cal = load_rho_star_yaml(_ARTIFACT)
    op = operating_point(
        _nullf, _drivf(beta=0.8), _cfg(), K=5, T=0, tau2=0.0, beta2=0.64,
        sigma2_bar=0.04, rho_star=cal.rho_star["domain"], scale_level="domain",
        driver_label="Triad", seeds_null=spawn_seeds(9999, 30),
        seeds_driver=spawn_seeds(6000, 30), protein="DENV_NS2B_NS3")

    def scales_KT(K, T):
        return [ScalePrediction(0, 0.4), ScalePrediction(1, 0.85)]
    rep = MetricsReport(system="DENV_NS2B_NS3", alpha=0.05, seed=1234,
                        points=(op,),
                        ell_min_grid=tuple(ell_min_grid(scales_KT, 0.8, [3, 5], [100])))
    path = tempfile.mktemp(suffix=".yaml")
    try:
        write_metrics_report(rep, path)
        loaded = load_metrics_report(path)
        assert loaded["provenance"]["system"] == "DENV_NS2B_NS3"
        assert len(loaded["operating_points"]) == 1
        assert loaded["operating_points"][0]["empirical_power"] == pytest.approx(op.empirical_power)
        assert "power_diff" in loaded["operating_points"][0]
        assert len(loaded["ell_min_grid"]) == 2
    finally:
        if os.path.exists(path):
            os.remove(path)
