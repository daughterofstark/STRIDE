"""M1 tests: integrated autocorrelation time / effective sample size.

Theoretical anchors
-------------------
AR(1) with parameter phi:
    rho(Delta) = phi**Delta
    tau_int    = 0.5 * (1 + phi) / (1 - phi)
    N_eff      = N * (1 - phi) / (1 + phi)
White noise: tau_int -> 0.5, N_eff -> N.
"""
import numpy as np
import pytest

from mechanism.statistics.neff import (
    autocorr_fft,
    integrated_autocorr_time,
    effective_sample_size,
    corrected_standard_error,
    NeffResult,
)


def _ar1(n, phi, seed=0, burn=2000):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def _tau_theory(phi):
    return 0.5 * (1 + phi) / (1 - phi)


# ── autocorr_fft ────────────────────────────────────────────────────────────
def test_acf_white_noise_decays():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(8000)
    acf = autocorr_fft(x)
    assert acf[0] == pytest.approx(1.0)
    assert abs(acf[1:50].mean()) < 0.05  # near-zero lags for white noise


def test_acf_constant_signals_zero():
    acf = autocorr_fft(np.full(100, 3.0))
    assert acf[0] == 0.0  # sentinel for constant


# ── integrated_autocorr_time vs AR(1) theory ───────────────────────────────
@pytest.mark.parametrize("phi", [0.5, 0.8, 0.9])
def test_iat_ar1_matches_theory(phi):
    x = _ar1(40000, phi, seed=2)
    tau, status = integrated_autocorr_time(x)
    assert status in ("ok", "white_noise")
    assert tau == pytest.approx(_tau_theory(phi), rel=0.20)


def test_iat_white_noise_floor():
    rng = np.random.default_rng(3)
    tau, status = integrated_autocorr_time(rng.standard_normal(8000))
    assert tau == pytest.approx(0.5, abs=0.15)
    assert status in ("ok", "white_noise")


def test_iat_constant_signal():
    tau, status = integrated_autocorr_time(np.ones(500))
    assert status == "constant_signal" and tau == 0.5


def test_iat_short_trajectory_flagged():
    tau, status = integrated_autocorr_time(np.array([1.0, 2.0, 1.5, 2.5, 1.0]))
    assert status == "short_trajectory"
    assert tau >= 0.5


def test_iat_strong_autocorr_capped_safely():
    # phi very close to 1 on a short series -> tau capped so n_eff stays >= 2
    x = _ar1(200, 0.98, seed=4)
    tau, status = integrated_autocorr_time(x)
    assert tau <= len(x) / 4.0 + 1e-9


# ── effective_sample_size ───────────────────────────────────────────────────
def test_neff_white_noise_near_full():
    rng = np.random.default_rng(5)
    n = 6000
    v = rng.standard_normal(n)
    d = rng.standard_normal(n)
    res = effective_sample_size(v, d)
    assert isinstance(res, NeffResult)
    assert res.n_eff > 0.5 * n  # little autocorrelation -> most samples count


def test_neff_strong_autocorr_reduced():
    n = 20000
    base = _ar1(n, 0.9, seed=6)
    v = base + 0.01 * np.random.default_rng(7).standard_normal(n)
    d = base + 0.01 * np.random.default_rng(8).standard_normal(n)
    res = effective_sample_size(v, d)
    assert res.n_eff < n          # autocorrelation must reduce effective N
    assert res.n_eff >= 2.0
    assert res.tau_int > 0.5


def test_neff_constant_signal():
    res = effective_sample_size(np.ones(500), np.arange(500.0))
    assert res.status == "constant_signal"
    assert res.n_eff >= 2.0


def test_neff_short_trajectory():
    res = effective_sample_size(np.array([1.0, 2, 3, 2]), np.array([2.0, 1, 3, 4]))
    assert res.n_eff >= 2.0
    assert res.n_frames == 4


def test_neff_monotone_in_autocorrelation():
    n = 20000
    weak = _ar1(n, 0.3, seed=9)
    strong = _ar1(n, 0.9, seed=9)
    r_weak = effective_sample_size(weak, weak + 0.01 * _ar1(n, 0.3, seed=10))
    r_strong = effective_sample_size(strong, strong + 0.01 * _ar1(n, 0.9, seed=11))
    assert r_strong.n_eff < r_weak.n_eff  # more autocorrelation -> fewer effective


# ── corrected_standard_error ────────────────────────────────────────────────
def test_corrected_se_scales_with_neff():
    se_small = corrected_standard_error(0.5, n_eff=100)
    se_large = corrected_standard_error(0.5, n_eff=400)
    assert se_large == pytest.approx(se_small / 2, rel=1e-9)  # SE ~ 1/sqrt(n)


def test_corrected_se_perfect_corr_zero():
    assert corrected_standard_error(1.0, n_eff=100) == pytest.approx(0.0)


def test_corrected_se_bad_neff_nan():
    assert np.isnan(corrected_standard_error(0.5, n_eff=0))
