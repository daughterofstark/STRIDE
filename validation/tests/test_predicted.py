"""V3 tests: closed-form predicted operating characteristics (Part IV).

Pure analytic-property checks — **no simulation, no generators, no production**.
Every assertion verifies a property the specification *requires* (a limit, a
monotonicity, or an identity); none pins an arbitrary numeric value of the
[CHOICE] power/coverage approximations.
"""
import math

import pytest

from validation.predicted import (
    lambda_snr, rho_from_lambda, rho_from_params, lambda_star,
    n_eff_from_T, sigma2_bar_from_neff,
    predicted_fpr, predicted_power, predicted_coverage,
    ScalePrediction, ell_min, over_resolution_bound,
    predicted_reference_table,
)
from validation.types import RegionTruth


# ── rho = lambda/(1+lambda): identity and limits [SPEC] ──────────────────────
def test_rho_lambda_identity():
    for beta2, tau2, sig2 in [(1.0, 0.2, 0.1), (4.0, 0.5, 0.3), (0.0, 1.0, 1.0)]:
        lam = lambda_snr(beta2, tau2, sig2)
        assert rho_from_lambda(lam) == pytest.approx(rho_from_params(beta2, tau2, sig2))


def test_rho_to_one_as_lambda_to_infinity():
    # ρ → 1 as λ → ∞
    prev = -1.0
    for lam in [1, 10, 100, 1e4, 1e8]:
        r = rho_from_lambda(lam)
        assert r > prev            # strictly increasing in λ
        prev = r
    assert rho_from_lambda(float("inf")) == 1.0
    assert rho_from_lambda(1e12) > 0.999


def test_rho_to_zero_as_lambda_to_zero():
    assert rho_from_lambda(0.0) == 0.0
    assert rho_from_lambda(1e-9) < 1e-6


def test_lambda_snr_noiseless_is_infinite():
    assert lambda_snr(1.0, 0.0, 0.0) == float("inf")
    assert lambda_snr(0.0, 0.0, 0.0) == 0.0


def test_rho_matches_region_truth_formula():
    # V3 does not introduce a second copy of the ρ formula: it must equal the
    # ground-truth RegionTruth.rho for the same parameters.
    for beta, tau2, sig2 in [(2.0, 0.3, 0.2), (0.5, 1.0, 0.5), (0.0, 0.4, 0.1)]:
        rt = RegionTruth("R", 1, beta=beta, tau2=tau2, sigma2_bar=sig2,
                         is_driver=(beta > 0))
        assert rho_from_params(beta * beta, tau2, sig2) == pytest.approx(rt.rho)


def test_lambda_star_inverts_rho_star():
    for rho_star in [0.1, 0.5, 0.9]:
        lam = lambda_star(rho_star)
        assert rho_from_lambda(lam) == pytest.approx(rho_star)


# ── sampling-noise link: N_eff = T/(2 tau_int); sigma_bar^2 ∝ 1/N_eff [SPEC] ─
def test_n_eff_formula():
    assert n_eff_from_T(200.0, 5.0) == pytest.approx(20.0)
    assert n_eff_from_T(1000.0, 2.5) == pytest.approx(200.0)


def test_n_eff_decreases_with_tau_int():
    vals = [n_eff_from_T(1000.0, t) for t in [1.0, 2.0, 5.0, 10.0]]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_sigma2_bar_inversely_proportional_to_neff():
    # doubling N_eff halves sigma_bar^2 (the ∝ 1/N_eff law)
    s1 = sigma2_bar_from_neff(100.0, theta=0.0)
    s2 = sigma2_bar_from_neff(200.0, theta=0.0)
    assert s2 == pytest.approx(s1 / 2.0)
    # explicit constant mode
    assert sigma2_bar_from_neff(50.0, k_proportional=3.0) == pytest.approx(0.06)


def test_sigma2_bar_fisher_form_shrinks_with_theta():
    # (1 - theta^2)^2 / N_eff decreases as |theta| grows toward 1
    base = sigma2_bar_from_neff(100.0, theta=0.0)
    hi = sigma2_bar_from_neff(100.0, theta=0.9)
    assert hi < base


# ── FPR = alpha by construction [SPEC] ───────────────────────────────────────
def test_predicted_fpr_is_alpha():
    for a in [0.01, 0.05, 0.1]:
        assert predicted_fpr(a) == a


def test_predicted_fpr_rejects_out_of_range():
    with pytest.raises(ValueError):
        predicted_fpr(1.5)


# ── power: spec-required monotonicities and limits [SPEC] ────────────────────
def test_power_increasing_in_beta2():
    vals = [predicted_power(b, 0.2, 0.1, K=5, rho_star=0.5)
            for b in [0.1, 0.5, 1.0, 2.0, 5.0]]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[0] < vals[-1]            # strictly increases overall


def test_power_increasing_in_K():
    vals = [predicted_power(1.0, 0.2, 0.1, K=k, rho_star=0.5)
            for k in [2, 3, 5, 10, 20]]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_power_increasing_in_neff():
    # larger N_eff -> smaller sigma_bar^2 -> more power
    vals = [predicted_power(1.0, 0.2, s, K=5, rho_star=0.5)
            for s in [1.0, 0.5, 0.1, 0.01]]  # decreasing sigma_bar^2 == increasing N_eff
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_power_decreasing_in_tau2():
    vals = [predicted_power(1.0, t, 0.1, K=5, rho_star=0.5)
            for t in [0.05, 0.1, 0.5, 1.0, 2.0]]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]


def test_power_to_one_as_beta2_to_infinity():
    assert predicted_power(1e6, 0.2, 0.1, K=5, rho_star=0.5) == pytest.approx(1.0, abs=1e-6)


def test_power_small_at_null():
    # β²→0: power collapses to the model's own (small) null crossing rate; the V4
    # calibration is what pins that rate to alpha. We assert only that it is small,
    # not that it equals alpha (calibration is V4).
    p = predicted_power(0.0, 0.2, 0.1, K=5, rho_star=0.5)
    assert p < 0.1


def test_power_noiseless_is_deterministic():
    # no noise: power is 1 iff rho(=1) >= rho_star, else 0
    assert predicted_power(1.0, 0.0, 0.0, K=3, rho_star=0.5) == 1.0
    assert predicted_power(1.0, 0.0, 0.0, K=3, rho_star=1.0 - 1e-9) == 1.0


def test_power_in_unit_interval():
    for b2 in [0.0, 0.5, 5.0]:
        for K in [2, 5, 20]:
            p = predicted_power(b2, 0.3, 0.2, K, 0.5)
            assert 0.0 <= p <= 1.0


# ── coverage: spec-required properties (nominal as K grows; honest at small K) ─
def test_coverage_converges_to_nominal_from_above():
    nominal = 0.95
    vals = [predicted_coverage(K, nominal) for K in [1, 2, 5, 20, 100, 1000]]
    # every value is >= nominal (honest / never anticonservative)
    assert all(v >= nominal - 1e-12 for v in vals)
    # monotonically decreasing toward nominal as K grows
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(nominal, abs=1e-2)


def test_coverage_small_K_is_wider():
    # small K interval is wider (higher coverage) than large K
    assert predicted_coverage(2) > predicted_coverage(50)


# ── ell_min: definition + non-increasing in K and T [SPEC] ───────────────────
def test_ell_min_picks_finest_passing_scale():
    scales = [ScalePrediction(0, 0.30), ScalePrediction(1, 0.70),
              ScalePrediction(2, 0.95)]
    assert ell_min(scales, 0.5) == 1        # finest (smallest index) with rho>=0.5
    assert ell_min(scales, 0.9) == 2        # stricter -> coarser
    assert ell_min(scales, 0.2) == 0        # looser -> finer


def test_ell_min_none_when_nothing_passes():
    scales = [ScalePrediction(0, 0.10), ScalePrediction(1, 0.20)]
    assert ell_min(scales, 0.5) is None


def test_ell_min_non_increasing_in_K():
    # As K grows, sigma_bar^2 unchanged but the ESTIMATOR resolves finer; here we
    # model that by finer scales' predicted rho rising with K (more N_eff). ell_min
    # (the finest passing index) must be non-increasing (finer or equal) as K grows.
    def scales_at_K(K):
        # finer scales gain rho as K increases (monotone construction)
        return [ScalePrediction(0, 0.3 + 0.05 * K),   # residue
                ScalePrediction(1, 0.6 + 0.02 * K),   # domain
                ScalePrediction(2, 0.9)]              # chain (already high)
    prev = None
    for K in [2, 3, 5, 8]:
        lm = ell_min(scales_at_K(K), 0.5)
        if prev is not None:
            assert lm <= prev                 # finer-or-equal (index non-increasing)
        prev = lm


def test_ell_min_non_increasing_in_T():
    # Same monotonicity via T (through N_eff): more T -> finer scales pass.
    def scales_at_T(T):
        gain = 0.0005 * T
        return [ScalePrediction(0, 0.2 + gain),
                ScalePrediction(1, 0.55 + 0.0002 * T),
                ScalePrediction(2, 0.9)]
    prev = None
    for T in [200, 500, 1000, 2000]:
        lm = ell_min(scales_at_T(T), 0.5)
        if prev is not None:
            assert lm <= prev
        prev = lm


# ── over-resolution bound: decreasing in K and gap; in [0,1]; ->0 [SPEC] ─────
def test_over_resolution_decreasing_in_K():
    vals = [over_resolution_bound(K, gap=0.2) for K in [1, 3, 5, 10, 50]]
    assert all(a >= b for a, b in zip(vals, vals[1:]))
    assert vals[-1] < vals[0]


def test_over_resolution_decreasing_in_gap():
    vals = [over_resolution_bound(5, gap=g) for g in [0.05, 0.1, 0.2, 0.4]]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_over_resolution_to_zero_as_K_grows():
    assert over_resolution_bound(10_000, gap=0.2) == pytest.approx(0.0, abs=1e-9)


def test_over_resolution_in_unit_interval():
    for K in [1, 5, 100]:
        for g in [0.0, 0.1, 0.5]:
            v = over_resolution_bound(K, g)
            assert 0.0 <= v <= 1.0
    # gap 0 -> bound is 1 (no separation, no protection)
    assert over_resolution_bound(5, 0.0) == pytest.approx(1.0)


def test_over_resolution_c_scales_decay():
    # larger model constant c -> faster decay (smaller bound)
    assert over_resolution_bound(5, 0.2, c=2.0) < over_resolution_bound(5, 0.2, c=1.0)


# ── reference table: pure structure, internally consistent [ROADMAP] ─────────
def test_reference_table_is_consistent():
    rows = predicted_reference_table([0.1, 1.0, 4.0], tau2=0.2, sigma2_bar=0.1,
                                     Ks=[3, 5], rho_star=0.5, alpha=0.05)
    assert len(rows) == 3 * 2
    for row in rows:
        # rho equals the closed-form from its own params
        assert row["rho"] == pytest.approx(
            rho_from_params(row["beta2"], row["tau2"], row["sigma2_bar"]))
        assert row["fpr"] == 0.05
        assert 0.0 <= row["power"] <= 1.0
        assert row["coverage"] >= 0.95


def test_reference_table_power_increases_with_beta2_within_K():
    rows = predicted_reference_table([0.1, 1.0, 4.0], 0.2, 0.1, [5], 0.5)
    powers = [r["power"] for r in sorted(rows, key=lambda r: r["beta2"])]
    assert all(a <= b + 1e-12 for a, b in zip(powers, powers[1:]))


# ── purity: V3 introduces no randomness and no mechanism dependency ──────────
def test_predicted_module_is_deterministic():
    # identical inputs -> identical outputs (no hidden RNG)
    a = predicted_power(1.0, 0.2, 0.1, 5, 0.5)
    b = predicted_power(1.0, 0.2, 0.1, 5, 0.5)
    assert a == b


def test_predicted_module_does_not_import_mechanism():
    import validation.predicted as pred
    import sys
    src = open(pred.__file__, encoding="utf-8").read()
    assert "import mechanism" not in src and "from mechanism" not in src
