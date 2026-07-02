"""V4 tests for null-surrogate generators (validation.surrogates).

Pure tests of the surrogate transforms plus their key null-generating properties,
driven where needed through the production estimator via the adapter bridge.
"""
import numpy as np
import pandas as pd
import pytest

from validation.surrogates import (
    permute_replicate_labels, phase_randomize, phase_randomize_pairs, power_spectrum,
)
from validation._seed import make_rng

# generators (reused) + production bridge for the "destroys reproducibility" checks
from validation.generate import (
    SyntheticSystemSpec, SynChain, SynDomain, Driver, NullRegion,
    generate_system, region_path,
    TierBSystemSpec, SeriesResidueSpec, generate_series_replicates,
)
from validation.adapters import to_hierarchy_config, aggregate_via_production
from mechanism.statistics import integrated_autocorr_time, pearson_both

_LEVELS = ("complex", "protein", "chain", "domain", "residue")


def _driver_system(seed=1, K=5, mode="distributed"):
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    d = Driver(support=triad, scale_level="domain",
               region_id=region_path(chain="NS3", domain="Triad"),
               beta=1.0, tau2=0.0, carrier_mode=mode)
    spec = SyntheticSystemSpec(
        name="SIG", levels=_LEVELS, chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"), SynDomain("Oxy", oxy, "NS3")),
        residues=triad + oxy, drivers=(d,),
        nulls=(NullRegion(oxy, "domain", region_path(chain="NS3", domain="Oxy")),),
        K=K, sigma2=0.04, offset=0, seed=seed, true_ell_star=1)
    return generate_system(spec)


# ── replicate-label permutation: shape, marginals, inputs untouched ──────────
def test_permute_preserves_shape_and_columns():
    g = _driver_system()
    surr = permute_replicate_labels(list(g.per_run_dfs), make_rng(0))
    assert len(surr) == len(g.per_run_dfs)
    for a, b in zip(surr, g.per_run_dfs):
        assert list(a.columns) == list(b.columns)
        assert len(a) == len(b)


def test_permute_preserves_per_residue_multiset():
    # for each residue, the multiset of K effect values is unchanged (only the
    # replicate assignment is permuted)
    g = _driver_system()
    surr = permute_replicate_labels(list(g.per_run_dfs), make_rng(3))
    orig = np.stack([f["r"].to_numpy() for f in g.per_run_dfs])   # (K, n)
    new = np.stack([f["r"].to_numpy() for f in surr])
    for j in range(orig.shape[1]):
        assert np.allclose(np.sort(orig[:, j]), np.sort(new[:, j]))


def test_permute_does_not_mutate_inputs():
    g = _driver_system()
    before = [f["r"].to_numpy().copy() for f in g.per_run_dfs]
    _ = permute_replicate_labels(list(g.per_run_dfs), make_rng(1))
    after = [f["r"].to_numpy() for f in g.per_run_dfs]
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_permute_deterministic_in_seed():
    g = _driver_system()
    a = permute_replicate_labels(list(g.per_run_dfs), make_rng(7))
    b = permute_replicate_labels(list(g.per_run_dfs), make_rng(7))
    for x, y in zip(a, b):
        assert np.array_equal(x["r"].to_numpy(), y["r"].to_numpy())


def test_permute_requires_equal_length_frames():
    g = _driver_system()
    bad = list(g.per_run_dfs)
    bad[1] = bad[1].iloc[:-1].copy()
    with pytest.raises(ValueError):
        permute_replicate_labels(bad, make_rng(0))


# ── replicate-label permutation destroys a COHERENT driver's reproducibility ─
def test_permute_nulls_coherent_driver():
    # A distributed (non-permuting) coherent driver IS reproducible; permuting
    # replicate labels should lower its domain rho toward the null band.
    g = _driver_system(mode="distributed")
    cfg = to_hierarchy_config(g.spec)
    obs = aggregate_via_production(list(g.per_run_dfs), cfg, protein="SIG")
    obs_rho = float(obs[(obs.scale_level == "domain") & (obs.label == "Triad")]["rho"].iloc[0])
    # average surrogate rho over several permutations
    rng = make_rng(11)
    surr_rhos = []
    for _ in range(30):
        s = permute_replicate_labels(list(g.per_run_dfs), rng)
        o = aggregate_via_production(s, cfg, protein="SIG")
        surr_rhos.append(float(o[(o.scale_level == "domain") & (o.label == "Triad")]["rho"].iloc[0]))
    # the observed (coherent) reproducibility exceeds the typical surrogate value
    assert obs_rho > np.mean(surr_rhos)


# ── phase randomization: spectrum preserved, mean preserved, real output ─────
def test_phase_randomize_preserves_power_spectrum():
    rng = make_rng(0)
    x = rng.normal(size=4096)
    xs = phase_randomize(x, rng)
    assert xs.shape == x.shape
    assert np.allclose(power_spectrum(x), power_spectrum(xs), rtol=1e-6, atol=1e-6)


def test_phase_randomize_preserves_mean_and_is_real():
    rng = make_rng(1)
    x = rng.normal(size=2048) + 3.0
    xs = phase_randomize(x, rng)
    assert np.isrealobj(xs)
    assert xs.mean() == pytest.approx(x.mean(), abs=1e-9)


def test_phase_randomize_preserves_tau_int():
    # AR(1)-like autocorrelation is a spectral property, so tau_int is preserved.
    from validation.processes import ar1_series
    x = ar1_series(20000, 0.8, make_rng(0))
    xs = phase_randomize(x, make_rng(1))
    tau_x, _ = integrated_autocorr_time(x)
    tau_s, _ = integrated_autocorr_time(xs)
    assert tau_s == pytest.approx(tau_x, rel=0.15)


def test_phase_randomize_changes_the_series():
    rng = make_rng(2)
    x = rng.normal(size=1024)
    xs = phase_randomize(x, rng)
    assert not np.allclose(x, xs)   # phases actually randomized


def test_phase_randomize_short_series_is_safe():
    x = np.array([1.0, 2.0])
    assert np.array_equal(phase_randomize(x, make_rng(0)), x)


# ── phase randomization breaks V–d coupling (theta -> ~0) ────────────────────
def test_phase_randomize_pairs_destroys_coupling():
    triad = (51, 75, 135)
    spec = TierBSystemSpec(
        name="B", levels=_LEVELS, chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"),),
        residues=tuple(SeriesResidueSpec(c, target_r=0.6, process="ar1", phi=0.7)
                       for c in triad),
        K=1, T=6000, v_phi=0.7, offset=0, seed=0)
    reps = generate_series_replicates(spec)
    rep = reps[0]
    # original coupling is strong
    r_orig = [pearson_both(rep.V, rep.d_by_canon[c], 1)[0] for c in triad]
    Vs, ds = phase_randomize_pairs(rep.V, rep.d_by_canon, make_rng(0))
    r_surr = [pearson_both(Vs, ds[c], 1)[0] for c in triad]
    assert np.mean(np.abs(r_orig)) > 0.4
    assert np.mean(np.abs(r_surr)) < 0.15    # coupling destroyed under the null
