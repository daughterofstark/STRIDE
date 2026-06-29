"""M2 tests: block-bootstrap confidence intervals."""
import numpy as np
import pytest

from mechanism.statistics.bootstrap import (
    bootstrap_correlation,
    attach_bootstrap_ci,
    select_block_length,
    circular_block_indices,
    stationary_block_indices,
    BootstrapResult,
)
from mechanism.statistics.neff import effective_sample_size


# ── helpers ─────────────────────────────────────────────────────────────────
def _ar1_pair(n, phi, c, seed, burn=2000):
    """Two AR(1) series with the same phi driven by innovations of correlation c.
    Their lag-0 (population) correlation equals c."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, c], [c, 1.0]])
    e = rng.multivariate_normal([0, 0], cov, size=n + burn)
    v = np.empty(n + burn); d = np.empty(n + burn)
    v[0], d[0] = e[0]
    for t in range(1, n + burn):
        v[t] = phi * v[t - 1] + e[t, 0]
        d[t] = phi * d[t - 1] + e[t, 1]
    return v[burn:], d[burn:]


def _neff(v, d):
    e = effective_sample_size(v, d)
    return e.tau_int, e.n_eff


# ── block-length selection ──────────────────────────────────────────────────
def test_block_length_scales_with_tau():
    L1, ok1 = select_block_length(1.0, 2000)
    L2, ok2 = select_block_length(10.0, 2000)
    assert ok1 and ok2 and L2 > L1


def test_block_length_caps_when_too_few_blocks():
    L, ok = select_block_length(tau_int=1e6, n=100)  # absurd autocorrelation
    assert L >= 1 and ok in (True, False)
    assert L <= 100  # cannot exceed series length


# ── index generators ────────────────────────────────────────────────────────
def test_circular_indices_shape_and_range():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(100, L=7, B=50, rng=rng)
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_circular_blocks_are_contiguous_mod_n():
    rng = np.random.default_rng(1)
    idx = circular_block_indices(20, L=5, B=10, rng=rng)
    # within each length-5 block, indices increase by 1 mod n
    for row in idx:
        for b in range(0, 20, 5):
            block = row[b:b + 5]
            diffs = (np.diff(block)) % 20
            assert np.all(diffs == 1)


def test_stationary_indices_shape_and_range():
    rng = np.random.default_rng(2)
    idx = stationary_block_indices(200, mean_L=8.0, B=30, rng=rng)
    assert idx.shape == (30, 200) and idx.min() >= 0 and idx.max() < 200


# ── core behaviour ──────────────────────────────────────────────────────────
def test_independent_data_ci_contains_zero():
    rng = np.random.default_rng(3)
    v = rng.standard_normal(3000); d = rng.standard_normal(3000)
    tau, neff = _neff(v, d)
    res = bootstrap_correlation(v, d, tau, neff, B=600, seed=1)
    assert res.method == "circular"
    assert res.ci_lower < 0 < res.ci_upper


def test_known_correlation_recovered():
    v, d = _ar1_pair(4000, phi=0.5, c=0.4, seed=4)
    tau, neff = _neff(v, d)
    res = bootstrap_correlation(v, d, tau, neff, B=800, seed=2)
    assert res.ci_lower < 0.4 < res.ci_upper


def test_autocorrelation_widens_ci_vs_naive_fisher():
    """Bootstrap CI under autocorrelation must be wider than a naive Fisher CI
    that pretends all N frames are independent."""
    v, d = _ar1_pair(3000, phi=0.85, c=0.4, seed=5)
    tau, neff = _neff(v, d)
    res = bootstrap_correlation(v, d, tau, neff, B=800, seed=3)
    r = float(np.corrcoef(v, d)[0, 1])
    n = len(v)
    z = np.arctanh(r); se_z_naive = 1 / np.sqrt(n - 3)
    naive_lo, naive_hi = np.tanh(z - 1.96 * se_z_naive), np.tanh(z + 1.96 * se_z_naive)
    boot_w = res.ci_upper - res.ci_lower
    naive_w = naive_hi - naive_lo
    assert boot_w > naive_w  # autocorrelation must inflate the interval


def test_independent_ci_close_to_fisher():
    rng = np.random.default_rng(6)
    # independent draws with a real correlation
    x = rng.standard_normal(4000)
    v = x + rng.standard_normal(4000)
    d = x + rng.standard_normal(4000)
    tau, neff = _neff(v, d)
    res = bootstrap_correlation(v, d, tau, neff, B=800, seed=4)
    r = float(np.corrcoef(v, d)[0, 1]); n = len(v)
    z = np.arctanh(r); se_z = 1 / np.sqrt(n - 3)
    fisher_w = np.tanh(z + 1.96 * se_z) - np.tanh(z - 1.96 * se_z)
    boot_w = res.ci_upper - res.ci_lower
    assert 0.6 * fisher_w < boot_w < 1.6 * fisher_w  # comparable when ~independent


def test_constant_signal_degenerate_no_fake_ci():
    res = bootstrap_correlation(np.ones(500), np.arange(500.0), 0.5, 500.0, seed=1)
    assert res.method == "degenerate"
    assert np.isnan(res.ci_lower) and np.isnan(res.ci_upper)


def test_short_trajectory_falls_back_to_fisher():
    rng = np.random.default_rng(7)
    v = rng.standard_normal(15); d = v + 0.1 * rng.standard_normal(15)
    e = effective_sample_size(v, d)
    res = bootstrap_correlation(v, d, e.tau_int, e.n_eff, seed=1)
    assert res.method in ("fisher_neff", "degenerate")


def test_determinism_same_seed_identical():
    v, d = _ar1_pair(2000, 0.6, 0.3, seed=8)
    tau, neff = _neff(v, d)
    a = bootstrap_correlation(v, d, tau, neff, B=400, seed=99)
    b = bootstrap_correlation(v, d, tau, neff, B=400, seed=99)
    assert a == b


def test_stationary_method_runs():
    v, d = _ar1_pair(2000, 0.6, 0.3, seed=9)
    tau, neff = _neff(v, d)
    res = bootstrap_correlation(v, d, tau, neff, B=400, method="stationary", seed=1)
    assert res.method == "stationary" and res.ci_lower < res.ci_upper


# ── empirical coverage (the key validation) ─────────────────────────────────
@pytest.mark.parametrize("phi", [0.0, 0.7])
def test_empirical_coverage_near_nominal(phi):
    """95% CIs should cover the true correlation ~95% of the time, for both
    independent (phi=0) and autocorrelated (phi=0.7) data."""
    true_c, n, M, B = 0.4, 1500, 160, 400
    hits = 0
    for m in range(M):
        v, d = _ar1_pair(n, phi=phi, c=true_c, seed=1000 + m)
        tau, neff = _neff(v, d)
        res = bootstrap_correlation(v, d, tau, neff, B=B, seed=m)
        if res.method in ("circular",) and res.ci_lower <= true_c <= res.ci_upper:
            hits += 1
    coverage = hits / M
    # generous band: Monte-Carlo noise (M=160) + bootstrap approximation
    assert 0.86 <= coverage <= 0.995, f"phi={phi} coverage={coverage:.3f}"


# ── integration helper ──────────────────────────────────────────────────────
class _Res:
    def __init__(self, rid): self.resid = rid


def test_attach_appends_only_and_preserves_existing():
    import pandas as pd
    from mechanism.statistics.neff import attach_effective_sample_size
    rng = np.random.default_rng(0)
    n, nres = 1500, 6
    vol = rng.standard_normal(n).cumsum()
    dm = rng.standard_normal((n, nres)).cumsum(axis=0)
    res = [_Res(98 + i) for i in range(nres)]
    rows = [dict(file_resid=r.resid, canon_resid=r.resid - 47, name="ALA",
                 r=float(np.corrcoef(vol, dm[:, i])[0, 1]), abs_r=0.0,
                 label=f"A{i}") for i, r in enumerate(res)]
    df = pd.DataFrame(rows)
    df = attach_effective_sample_size(df, res, dm, vol)   # M1 first
    before = df.copy(deep=True)
    out = attach_bootstrap_ci(df, res, dm, vol, B=200, seed=42)
    for col in before.columns:
        pd.testing.assert_series_equal(out[col], before[col])
    added = ["theta_bootstrap_se", "theta_bootstrap_ci_lower",
             "theta_bootstrap_ci_upper", "bootstrap_method",
             "bootstrap_block_length", "bootstrap_replicates"]
    assert list(out.columns) == list(before.columns) + added


def test_attach_is_deterministic():
    import pandas as pd
    from mechanism.statistics.neff import attach_effective_sample_size
    rng = np.random.default_rng(1)
    n, nres = 1200, 5
    vol = rng.standard_normal(n).cumsum()
    dm = rng.standard_normal((n, nres)).cumsum(axis=0)
    res = [_Res(10 + i) for i in range(nres)]
    def build():
        rows = [dict(file_resid=r.resid, canon_resid=r.resid, name="A",
                     r=float(np.corrcoef(vol, dm[:, i])[0, 1]), abs_r=0.0,
                     label=f"A{i}") for i, r in enumerate(res)]
        df = pd.DataFrame(rows)
        df = attach_effective_sample_size(df, res, dm, vol)
        return attach_bootstrap_ci(df, res, dm, vol, B=200, seed=7)
    a, b = build(), build()
    pd.testing.assert_frame_equal(a, b)
