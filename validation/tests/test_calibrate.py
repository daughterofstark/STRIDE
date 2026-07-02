"""V4 tests for empirical rho* calibration (validation.calibrate).

Covers the (Cal) FPR-control guarantee (out-of-sample FPR <= alpha on DISJOINT
beta=0 draws), the train/test seed split, determinism, alpha-responsiveness (no
hard-coding), the single-base under-coverage limitation, and the rho_star.yaml
artifact round-trip. rho_hat always comes from the production estimator via the
adapter bridge; generators are reused, never re-implemented.
"""
import os
import tempfile

import numpy as np
import pytest

from validation.generate import (
    SyntheticSystemSpec, SynChain, SynDomain, NullRegion,
    generate_system, region_path,
)
from validation.adapters import to_hierarchy_config
from validation._seed import spawn_seeds
from validation.calibrate import (
    CalibrationResult, upper_alpha_quantile,
    surrogate_null_rho, ensemble_surrogate_null_rho, calibrate_rho_star,
    generator_null_rho, empirical_fpr,
    write_rho_star_yaml, load_rho_star_yaml,
)

_LEVELS = ("complex", "protein", "chain", "domain", "residue")


def _null_spec(seed, K=5, sigma2=0.04):
    """A pure beta=0 system (two null domains)."""
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    return SyntheticSystemSpec(
        name="DENVX", levels=_LEVELS, chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"),
                 SynDomain("Oxy", oxy, "NS3")),
        residues=triad + oxy, drivers=(),
        nulls=(NullRegion(triad, "domain", region_path(chain="NS3", domain="Triad")),
               NullRegion(oxy, "domain", region_path(chain="NS3", domain="Oxy"))),
        K=K, sigma2=sigma2, offset=0, seed=seed, true_ell_star=1)


def _cfg():
    return to_hierarchy_config(_null_spec(1))


def _make_base(K=5):
    def build(seed):
        return list(generate_system(_null_spec(seed, K=K)).per_run_dfs)
    return build


# ── upper-alpha quantile is the (1-alpha) quantile ───────────────────────────
def test_upper_alpha_quantile():
    vals = np.linspace(0.0, 1.0, 1001)
    assert upper_alpha_quantile(vals, 0.05) == pytest.approx(0.95, abs=1e-3)
    assert upper_alpha_quantile(vals, 0.01) == pytest.approx(0.99, abs=1e-3)


def test_upper_alpha_quantile_ignores_nans():
    vals = np.array([0.1, 0.5, np.nan, 0.9])
    assert np.isfinite(upper_alpha_quantile(vals, 0.5))


# ── canonical calibration produces rho* from the surrogate null (no hardcode) ─
def test_calibrate_produces_result_with_provenance():
    make_base = _make_base(K=5)
    base_seeds = spawn_seeds(1234, 20)
    res = calibrate_rho_star(make_base, _cfg(), system="DENVX", K=5, T=0,
                             base_seeds=base_seeds, surr_per_base=8,
                             alpha=0.05, seed=1234, protein="DENVX")
    assert isinstance(res, CalibrationResult)
    assert res.system == "DENVX" and res.K == 5 and res.alpha == 0.05
    assert res.B == 20 * 8
    # rho* exists for the standard scales and lies in (0, 1)
    for scale in ("residue", "domain", "chain"):
        assert 0.0 < res.rho_star[scale] < 1.0
    # calibration uncertainty CI present and ordered
    for scale, (lo, hi) in res.rho_star_ci.items():
        assert lo <= hi


def test_calibrated_rho_star_is_not_hardcoded_responds_to_alpha():
    make_base = _make_base(K=5)
    base_seeds = spawn_seeds(1234, 20)
    strict = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                                base_seeds=base_seeds, surr_per_base=8,
                                alpha=0.01, seed=1234, protein="DENVX")
    loose = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                               base_seeds=base_seeds, surr_per_base=8,
                               alpha=0.10, seed=1234, protein="DENVX")
    # smaller alpha -> stricter (higher) rho*
    assert strict.rho_star["domain"] >= loose.rho_star["domain"]


def test_calibrated_rho_star_reflects_null_not_provisional_half():
    # the folded-energy null concentrates well above 0; the calibrated domain rho*
    # is materially higher than the provisional 0.5 (evidence it came from the null)
    make_base = _make_base(K=5)
    res = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                             base_seeds=spawn_seeds(1234, 30), surr_per_base=8,
                             alpha=0.05, seed=1234, protein="DENVX")
    assert res.rho_star["domain"] > 0.6


# ── determinism (R5) ─────────────────────────────────────────────────────────
def test_calibration_deterministic_in_seed():
    make_base = _make_base(K=5)
    base_seeds = spawn_seeds(1234, 15)
    a = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                           base_seeds=base_seeds, surr_per_base=6, seed=1234,
                           protein="DENVX")
    b = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                           base_seeds=base_seeds, surr_per_base=6, seed=1234,
                           protein="DENVX")
    assert a.rho_star == b.rho_star
    assert a.rho_star_ci == b.rho_star_ci


def test_different_seed_changes_null_draws():
    make_base = _make_base(K=5)
    a = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                           base_seeds=spawn_seeds(1, 15), surr_per_base=6, seed=1,
                           protein="DENVX")
    b = calibrate_rho_star(make_base, _cfg(), system="D", K=5, T=0,
                           base_seeds=spawn_seeds(2, 15), surr_per_base=6, seed=2,
                           protein="DENVX")
    assert a.rho_star["domain"] != b.rho_star["domain"]


# ── the (Cal) property: out-of-sample FPR <= alpha on DISJOINT beta=0 draws ───
@pytest.mark.parametrize("K", [3, 5, 10])
def test_out_of_sample_fpr_control(K):
    make_base = _make_base(K=K)
    train_base = spawn_seeds(1234, 30)     # calibration ensemble (train stream)
    test_seeds = spawn_seeds(9999, 300)    # DISJOINT evaluation draws (test stream)
    assert set(train_base).isdisjoint(test_seeds)   # train/test split enforced

    res = calibrate_rho_star(make_base, _cfg(), system="DENVX", K=K, T=0,
                             base_seeds=train_base, surr_per_base=8,
                             alpha=0.05, seed=1234, protein="DENVX")
    gnull = generator_null_rho(make_base, _cfg(), seeds=test_seeds, protein="DENVX")
    # domain and residue pool many regions -> stable; assert tight control.
    for scale in ("domain", "residue"):
        fpr = empirical_fpr(gnull[scale], res.rho_star[scale])
        assert fpr <= 0.05 + 0.02, (K, scale, fpr)   # (Cal): FPR <= alpha (+MC slack)
    # chain is a single region -> higher MC variance; assert a wider but still
    # controlled bound (documented: fewer pooled null values at coarse scales).
    fpr_chain = empirical_fpr(gnull["chain"], res.rho_star["chain"])
    assert fpr_chain <= 0.05 + 0.04, (K, "chain", fpr_chain)


def test_train_test_seed_streams_are_disjoint():
    train = spawn_seeds(1234, 200)
    test = spawn_seeds(9999, 200)
    assert set(train).isdisjoint(set(test))


# ── single-base under-coverage is a documented limitation ────────────────────
def test_single_base_null_undercovers_relative_to_ensemble():
    # [KNOWN LIMITATION] one base's surrogate null is narrower than the ensemble
    # null, so its upper-alpha quantile is lower -> higher FPR. Assert the ensemble
    # rho* is >= the single-base rho* at the domain scale.
    make_base = _make_base(K=5)
    single = surrogate_null_rho(make_base(1234), _cfg(), B=150, seed=1234,
                                protein="DENVX", scales=["domain"])
    ens = ensemble_surrogate_null_rho(make_base, _cfg(),
                                      base_seeds=spawn_seeds(1234, 30),
                                      surr_per_base=10, seed=1234,
                                      protein="DENVX", scales=["domain"])
    q_single = upper_alpha_quantile(single["domain"], 0.05)
    q_ens = upper_alpha_quantile(ens["domain"], 0.05)
    assert q_ens >= q_single - 1e-9


# ── artifact round-trip (rho_star.yaml), provenance keyed correctly ──────────
def test_rho_star_yaml_round_trip():
    make_base = _make_base(K=5)
    res = calibrate_rho_star(make_base, _cfg(), system="DENVX", K=5, T=200,
                             base_seeds=spawn_seeds(1234, 15), surr_per_base=6,
                             alpha=0.05, seed=1234, protein="DENVX")
    path = tempfile.mktemp(suffix=".yaml")
    try:
        write_rho_star_yaml(res, path)
        loaded = load_rho_star_yaml(path)
        assert loaded.system == res.system
        assert loaded.K == res.K and loaded.T == res.T
        assert loaded.alpha == res.alpha and loaded.B == res.B
        assert loaded.seed == res.seed and loaded.surrogate == res.surrogate
        for scale in res.rho_star:
            assert loaded.rho_star[scale] == pytest.approx(res.rho_star[scale])
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_rho_star_yaml_has_provenance_block():
    import yaml
    make_base = _make_base(K=5)
    res = calibrate_rho_star(make_base, _cfg(), system="DENVX", K=3, T=100,
                             base_seeds=spawn_seeds(7, 10), surr_per_base=5,
                             alpha=0.05, seed=7, protein="DENVX")
    path = tempfile.mktemp(suffix=".yaml")
    try:
        write_rho_star_yaml(res, path)
        payload = yaml.safe_load(open(path))
        prov = payload["provenance"]
        # keyed by (system, K, T, alpha, B, seed) exactly as the roadmap requires
        assert set(["system", "K", "T", "alpha", "B", "seed", "surrogate"]).issubset(prov)
        assert prov["system"] == "DENVX" and prov["K"] == 3 and prov["T"] == 100
    finally:
        if os.path.exists(path):
            os.remove(path)


# ── empirical_fpr helper ─────────────────────────────────────────────────────
def test_empirical_fpr_basic():
    null = np.array([0.1, 0.4, 0.6, 0.9])
    assert empirical_fpr(null, 0.5) == pytest.approx(0.5)   # 0.6 and 0.9 >= 0.5
    assert empirical_fpr(null, 1.0) == 0.0
    assert empirical_fpr(null, 0.0) == 1.0
