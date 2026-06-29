"""M4 tests: aggregation operators, the reproducibility coefficient, the I1
invariance property, and the multi-scale hierarchy driver.

Key property under test (spec Part III, I1): when a single-residue effect of
magnitude ``a`` permutes its supporting residue within a region across replicates,
the energy aggregate ``A_en`` gives ``Theta_R^(k) = a`` for every ``k`` so
``tau^2 = 0`` and ``rho_R = 1``, while at residue level the effect is
near-orthogonal across replicates so ``rho_{i} -> 0``. The proposition is exact
in the noiseless limit, which is the regime asserted; the formal test uses the
noiseless planted signal (roadmap tolerance: residue rho < 0.2, region rho > 0.8).
"""
import numpy as np
import pandas as pd
import pytest

from mechanism.config.hierarchy_schema import (
    HierarchyConfig, ChainSpec, GroupSpec,
)
from mechanism.statistics import reproducibility as rep
from mechanism.statistics.reproducibility import (
    pooled_scale, energy, signed_mean, directional_coherence,
    propagate_energy_sigma2, propagate_signed_sigma2,
    beta2_bias_corrected, reproducibility_coefficient,
    region_reproducibility, aggregate_reproducibility,
    write_reproducibility_tables,
)


# ── aggregation operators ────────────────────────────────────────────────────
def test_energy_matches_l2_norm():
    th = np.array([0.3, -0.4, 0.0])
    assert energy(th) == pytest.approx(0.5)  # 3-4-5


def test_energy_support_permutation_invariant():
    base = np.array([0.8, 0.0, 0.0, 0.0])
    for shift in range(4):
        assert energy(np.roll(base, shift)) == pytest.approx(energy(base))


def test_signed_mean_and_coherence():
    assert signed_mean(np.array([1.0, -1.0, 2.0])) == pytest.approx(2.0 / 3.0)
    assert directional_coherence(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert directional_coherence(np.array([1.0, -1.0])) == pytest.approx(0.0)


def test_pooled_scale_is_scale_only():
    runs = [np.array([1.0, -1.0, 2.0]), np.array([0.0, 3.0, -2.0])]
    s = pooled_scale(runs)
    assert s == pytest.approx(np.std(np.concatenate(runs)))


def test_propagate_energy_delta_method():
    th = np.array([0.3, 0.4])
    sig2 = np.array([0.01, 0.02])
    Theta = energy(th)
    expected = np.sum((th ** 2) * sig2) / (Theta ** 2)
    assert propagate_energy_sigma2(th, sig2) == pytest.approx(expected)


def test_propagate_energy_zero_field_conservative():
    th = np.zeros(3)
    sig2 = np.array([0.01, 0.02, 0.03])
    # at Theta ~ 0 we return sum(sigma2) (conservative -> drives rho down)
    assert propagate_energy_sigma2(th, sig2) == pytest.approx(sig2.sum())


def test_propagate_signed():
    sig2 = np.array([0.04, 0.04, 0.04])
    assert propagate_signed_sigma2(sig2) == pytest.approx(0.04 * 3 / 9)


# ── reproducibility coefficient: limits and guards ───────────────────────────
def test_rho_limits():
    assert reproducibility_coefficient(1.0, 0.0, 0.0) == pytest.approx(1.0)
    assert reproducibility_coefficient(0.0, 1.0, 0.0) == pytest.approx(0.0)
    assert reproducibility_coefficient(1.0, 9.0, 0.0) == pytest.approx(0.1)
    assert reproducibility_coefficient(0.0, 0.0, 0.0) == 0.0       # 0/0 guard
    assert np.isnan(reproducibility_coefficient(1.0, float("nan"), 0.0))


def test_rho_in_unit_interval_random():
    rng = np.random.default_rng(0)
    for _ in range(500):
        b2, t2, s2 = rng.uniform(0, 5, 3)
        r = reproducibility_coefficient(b2, t2, s2)
        assert 0.0 <= r <= 1.0


def test_beta2_bias_corrected_floor():
    assert beta2_bias_corrected(1.0, 2.0) == 0.0          # clipped at 0
    assert beta2_bias_corrected(2.0, 1.0) == pytest.approx(3.0)


# ── I1: the load-bearing invariance ──────────────────────────────────────────
def _planted_region(seed, n_res=8, K=3, a=1.0, noise=0.0, sig=1e-6):
    rng = np.random.default_rng(seed)
    carriers = rng.choice(n_res, size=K, replace=False)
    runs = []
    for k in range(K):
        th = rng.normal(0.0, noise, n_res) if noise > 0 else np.zeros(n_res)
        th[carriers[k]] += a
        runs.append(th)
    scale = pooled_scale(runs)
    theta = [t / scale for t in runs]
    sig2 = [np.full(n_res, (sig / scale) ** 2) for _ in range(K)]
    return theta, sig2


def test_I1_region_dominates_residue_noiseless():
    for seed in range(5):
        theta, sig2 = _planted_region(seed, noise=0.0)
        region = region_reproducibility(theta, sig2)
        res_rhos = [
            region_reproducibility([np.array([theta[k][i]]) for k in range(3)],
                                   [np.array([sig2[k][i]]) for k in range(3)]).rho
            for i in range(8)
        ]
        assert region.rho > 0.8, f"region rho {region.rho} (seed {seed})"
        assert np.nanmax(res_rhos) < 0.2, f"residue rho {np.nanmax(res_rhos)}"


def test_I1_region_tau2_zero_and_beta_equals_a():
    # exact proposition: Theta_R^(k) = a for all k -> tau^2 = 0
    theta, sig2 = _planted_region(0, noise=0.0)
    region = region_reproducibility(theta, sig2)
    assert region.tau2 == pytest.approx(0.0, abs=1e-6)
    Theta = np.array([energy(t) for t in theta])
    assert np.allclose(Theta, Theta[0])  # constant across replicates


def test_I1_energy_invariant_under_carrier_permutation():
    # region energy identical regardless of which residue carries the signal
    a = 1.0
    energies = []
    for carrier in range(4):
        th = np.zeros(4)
        th[carrier] = a
        energies.append(energy(th))
    assert np.allclose(energies, energies[0])


# ── large-tau (DENV2-type) and consistent-signal regimes ─────────────────────
def test_large_tau_low_rho_even_with_big_per_run_effect():
    # each run shows a big regional energy, but the magnitude wanders a lot
    # across runs (large tau^2) -> rho is low.
    rng = np.random.default_rng(2)
    K = 6
    theta = [np.array([m]) for m in [0.2, 3.0, 0.3, 2.8, 0.25, 3.1]]
    sig2 = [np.full(1, 1e-4) for _ in range(K)]
    r = region_reproducibility(theta, sig2)
    assert r.tau2 > 0.5
    assert r.rho < 0.5


def test_consistent_strong_signal_high_rho():
    K = 6
    theta = [np.array([2.0, 0.1]) for _ in range(K)]
    sig2 = [np.full(2, 1e-3) for _ in range(K)]
    r = region_reproducibility(theta, sig2)
    assert r.rho > 0.9


def test_functional_invariance_global_rescale():
    # rho is invariant to a global rescaling of the effect field (I3, scale-only
    # standardization). Multiply every value by c -> same standardized field.
    theta, sig2 = _planted_region(1, noise=1e-3)
    r1 = region_reproducibility(theta, sig2)
    c = 7.3
    r2 = region_reproducibility([t * c for t in theta],
                                [s * c * c for s in sig2])
    assert r2.rho == pytest.approx(r1.rho, abs=1e-9)


# ── multi-scale hierarchy driver (integration) ───────────────────────────────
def _denv_like_cfg():
    return HierarchyConfig(
        name="t",
        chains=(ChainSpec("NS2B", (-999, 0)), ChainSpec("NS3", (1, 999))),
        domains=(
            GroupSpec("Triad", (51, 75, 135), chain="NS3", order=0),
            GroupSpec("Oxy", tuple(range(152, 160)), chain="NS3", order=1),
        ),
    )


def _make_run_df(canon_signal, offset=47, seed=0, signal=1.0, n=12):
    """Synthetic per-run correlation table with M1/M2/M3-style columns.

    A single residue (``canon_signal``, in the Triad domain) carries the effect;
    all others are ~0. ``canon_signal`` permutes across runs (I1 at domain scale).
    """
    rng = np.random.default_rng(seed)
    canon = np.array([51, 75, 135, 152, 153, 154, 155, 156, 157, 158, 10, 20])[:n]
    r = rng.normal(0.0, 1e-4, len(canon))
    r[np.where(canon == canon_signal)[0]] += signal
    return pd.DataFrame(dict(
        file_resid=canon + offset,
        canon_resid=canon,
        name=["ALA"] * len(canon),
        chain=["NS3"] * len(canon),
        r=r,
        abs_r=np.abs(r),
        theta_se=np.full(len(canon), 1e-3),
        theta_bootstrap_se=np.full(len(canon), 1e-3),
    ))


def test_aggregate_reproducibility_multiscale():
    cfg = _denv_like_cfg()
    # Triad carrier permutes across the 3 runs: 51 -> 75 -> 135
    dfs = [_make_run_df(51, seed=0), _make_run_df(75, seed=1),
           _make_run_df(135, seed=2)]
    out = aggregate_reproducibility(dfs, cfg, protein="DENV")
    assert not out.empty
    # required columns present
    for c in ("scale_level", "scale_index", "region_id", "rho", "tau2", "beta"):
        assert c in out.columns
    # residue level is the finest (scale_index 0)
    assert out.loc[out.scale_level == "residue", "scale_index"].iloc[0] == 0
    # domain index > residue index (coarser)
    dom_idx = out.loc[out.scale_level == "domain", "scale_index"].iloc[0]
    assert dom_idx > 0
    # I1 at the hierarchy level: the Triad domain is reproducible...
    triad = out[(out.scale_level == "domain") & (out.label == "Triad")]
    assert triad["rho"].iloc[0] > 0.8
    # ...while no individual residue is (the carrier permutes across runs, so
    # every singleton effect vector is near-orthogonal across replicates).
    res = out[out.scale_level == "residue"]
    assert res["rho"].max() < 0.3


def test_aggregate_handles_ragged_membership():
    # a residue present in only some runs must not crash the energy aggregation
    cfg = _denv_like_cfg()
    d0 = _make_run_df(51, seed=0)
    d1 = _make_run_df(75, seed=1).iloc[:-1].copy()  # drop one residue
    d2 = _make_run_df(135, seed=2)
    out = aggregate_reproducibility([d0, d1, d2], cfg)
    assert not out.empty and out["rho"].notna().any()


def test_writer_outputs(tmp_path):
    cfg = _denv_like_cfg()
    dfs = [_make_run_df(51, seed=0), _make_run_df(75, seed=1),
           _make_run_df(135, seed=2)]
    out = aggregate_reproducibility(dfs, cfg, protein="DENV")
    paths = write_reproducibility_tables(out, str(tmp_path), prefix="DENV")
    vc = pd.read_csv(paths["varcomp"])
    rh = pd.read_csv(paths["rho_by_scale"])
    assert {"region_id", "beta", "tau2", "sigma2_bar"}.issubset(vc.columns)
    assert {"region_id", "rho", "scale_index"}.issubset(rh.columns)
    assert len(vc) == len(out) and len(rh) == len(out)
