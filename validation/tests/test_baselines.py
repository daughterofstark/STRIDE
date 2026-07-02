"""V6 tests for Part VI baselines and the comparison framework (validation.baselines).

These verify the **correctness of the baseline estimators and the comparison
pipeline** — the known structural degeneracies of each baseline, and that the
comparison machinery computes and is deterministic. Per the V6 mandate, NO test
encodes today's empirical superiority of STRIDE over any baseline as a permanent
invariant; that result lives in the generated artifact/report.
"""
import numpy as np
import pytest

from validation.generate import (
    SyntheticSystemSpec, SynChain, SynDomain, Driver, NullRegion,
    generate_system, region_path,
)
from validation.adapters import to_hierarchy_config
from validation._seed import spawn_seeds
from validation.calibrate import load_rho_star_yaml
from validation.baselines import (
    single_trajectory_claim, single_trajectory_over_resolves,
    naive_ensemble_claim, naive_ensemble_over_resolves, naive_coverage,
    residue_ranking_claim, gtheory_coefficient,
    baseline_over_resolution_rates, build_method_comparison,
)

import os

_LEVELS = ("complex", "protein", "chain", "domain", "residue")
_ARTIFACT = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                         "rho_star_DENV_K5.yaml")


def _spec(seed, beta, K=5, sigma2=0.04, mode="distributed", tau2=0.0):
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    drivers = () if beta == 0 else (
        Driver(support=triad, scale_level="domain",
               region_id=region_path(chain="NS3", domain="Triad"),
               beta=beta, tau2=tau2, carrier_mode=mode),)
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


# ── single-trajectory baseline: ALWAYS residue-scale (structural degeneracy) ─
def test_single_trajectory_always_residue_scale():
    for s in range(5):
        claim = single_trajectory_claim(_nullf(s))
        assert claim.scale_level == "residue"


def test_single_trajectory_over_resolves_on_coarse_truth():
    # by construction the claim is residue-scale, so any coarser truth is over-resolved
    frames = _nullf(0)
    assert single_trajectory_over_resolves(frames, true_scale_level="domain") is True
    assert single_trajectory_over_resolves(frames, true_scale_level="chain") is True
    # if the truth really is residue-scale, it is not over-resolution
    assert single_trajectory_over_resolves(frames, true_scale_level="residue") is False


def test_single_trajectory_picks_max_magnitude():
    frames = _nullf(1)
    claim = single_trajectory_claim(frames)
    abs0 = np.abs(frames[0]["r"].to_numpy())
    assert claim.magnitude == pytest.approx(abs0.max())


# ── naive ensemble baseline: fixed residue scale, SD/sqrt(K) interval ────────
def test_naive_ensemble_reports_per_residue_stats():
    frames = _nullf(2)
    claim = naive_ensemble_claim(frames)
    n = len(frames[0])
    assert claim.mean.shape == (n,) and claim.se.shape == (n,)
    assert claim.K == len(frames)


def test_naive_ensemble_never_over_resolves_on_residue_truth():
    assert naive_ensemble_over_resolves(_nullf(0), true_scale_level="residue") is False


def test_naive_coverage_undercovers_at_small_K_is_computable():
    # framework check: naive_coverage returns a valid fraction; we do NOT freeze a
    # particular coverage value (that is an empirical result for the report).
    means = np.array([[0.0, 0.1], [0.05, -0.1], [0.2, 0.0]])
    ses = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    target = np.array([0.0, 0.0])
    cov = naive_coverage(target, means, ses)
    assert 0.0 <= cov <= 1.0


# ── residue ranking (IDR relative): fixed resolution, sorted by |theta| ──────
def test_residue_ranking_sorted_descending():
    ranking = residue_ranking_claim(_nullf(3))
    mags = [m for _, m in ranking]
    assert mags == sorted(mags, reverse=True)
    assert len(ranking) == len(_nullf(3)[0])


# ── G-theory coefficient: reliability in [0,1], high for reproducible signal ─
def test_gtheory_in_unit_interval():
    g = gtheory_coefficient(_nullf(0))
    assert 0.0 <= g <= 1.0


def test_gtheory_higher_for_reproducible_than_null():
    # G-theory reliability (objects = residues, facet = replicates) is high when
    # residues have DISTINCT magnitudes that are stable across replicates, and low for
    # pure noise. This is a structural property of the reliability coefficient (not a
    # STRIDE-vs-baseline claim). Construct a controlled reproducible field vs a null
    # field directly on the frame schema.
    import pandas as pd

    def _frame(rvals):
        n = len(rvals)
        return pd.DataFrame({
            "file_resid": np.arange(n), "canon_resid": np.arange(n),
            "name": ["ALA"] * n, "chain": ["A"] * n,
            "r": rvals, "abs_r": np.abs(rvals),
            "theta_se": np.full(n, 0.05), "theta_bootstrap_se": np.full(n, 0.05)})

    rng = np.random.default_rng(0)
    base = np.array([0.1, 0.4, 0.8, 1.2, 1.7])         # distinct residue magnitudes
    # reproducible: same pattern each replicate + tiny noise -> high between/low within
    repro = [_frame(base + rng.normal(0, 0.03, base.size)) for _ in range(5)]
    # null: independent noise each replicate -> no stable between-residue structure
    nul = [_frame(rng.normal(0, 0.6, base.size)) for _ in range(5)]
    g_repro = gtheory_coefficient(repro)
    g_null = gtheory_coefficient(nul)
    assert g_repro > g_null


def test_gtheory_nan_for_single_replicate():
    frames = list(generate_system(_spec(0, 1.0, K=1)).per_run_dfs)
    assert np.isnan(gtheory_coefficient(frames))


# ── baseline over-resolution rates: pipeline correctness + determinism ───────
def test_baseline_over_resolution_rates_structure():
    rates = baseline_over_resolution_rates(
        _nullf, seeds=spawn_seeds(1, 20), true_scale_level="domain")
    assert set(rates) == {"single_trajectory", "naive_ensemble"}
    # single-trajectory ALWAYS over-resolves on coarse truth (structural) -> rate 1.0
    assert rates["single_trajectory"]["rate"] == pytest.approx(1.0)
    # rates are valid fractions
    for info in rates.values():
        assert 0.0 <= info["rate"] <= 1.0


def test_baseline_rates_deterministic():
    seeds = spawn_seeds(2, 15)
    a = baseline_over_resolution_rates(_nullf, seeds=seeds, true_scale_level="domain")
    b = baseline_over_resolution_rates(_nullf, seeds=seeds, true_scale_level="domain")
    assert np.array_equal(a["naive_ensemble"]["indicators"],
                          b["naive_ensemble"]["indicators"])


# ── method comparison pipeline: computes, is well-formed, is deterministic ───
def test_build_method_comparison_is_wellformed_and_deterministic():
    cal = load_rho_star_yaml(_ARTIFACT)
    seeds = spawn_seeds(9999, 40)
    kw = dict(seeds=seeds, rho_star=cal.rho_star["domain"], true_scale_level="domain",
              driver_region_substr="Triad", protein="DENV_NS2B_NS3")
    rep = build_method_comparison(_nullf, _nullf, _cfg(), **kw)
    # well-formed: rates present and in range, comparisons carry the test outputs
    assert 0.0 <= rep["stride_over_resolution_rate"] <= 1.0
    for name in ("single_trajectory", "naive_ensemble"):
        c = rep["comparisons"][name]
        assert 0.0 <= c["baseline_rate"] <= 1.0
        assert "mcnemar_p" in c and 0.0 <= c["mcnemar_p"] <= 1.0
        assert len(c["paired_bootstrap_ci"]) == 2
    # deterministic: identical inputs -> identical report
    rep2 = build_method_comparison(_nullf, _nullf, _cfg(), **kw)
    assert rep == rep2
    # NOTE: no assertion that STRIDE's rate is below any baseline's — that empirical
    # result belongs to the artifact/report, not the test suite (V6 mandate).
