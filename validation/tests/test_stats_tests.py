"""V6 tests for the paired comparative statistical tests (validation.stats_tests).

These verify the **correctness of the test implementations on constructed inputs** —
known-answer cases and mathematical properties. They do not touch STRIDE or any
baseline result; they are pure statistics checks.
"""
import numpy as np
import pytest

from validation.stats_tests import (
    mcnemar_test, wilcoxon_signed_rank, delong_auc_test,
    paired_bootstrap_diff, benjamini_hochberg,
)


# ── McNemar ──────────────────────────────────────────────────────────────────
def test_mcnemar_symmetric_is_not_significant():
    r = mcnemar_test(20, 20)
    # perfectly symmetric discordance -> far from significant
    assert r.p_value > 0.5


def test_mcnemar_large_chi2_matches_formula():
    r = mcnemar_test(40, 10)          # b+c=50 > exact threshold -> chi2 w/ correction
    expected = (abs(40 - 10) - 1.0) ** 2 / 50.0
    assert r.statistic == pytest.approx(expected)
    assert 0.0 <= r.p_value <= 1.0
    assert r.p_value < 0.001


def test_mcnemar_small_uses_exact_binomial():
    r = mcnemar_test(10, 2)           # b+c=12 <= threshold -> exact
    assert r.corrected is False
    # exact two-sided binomial at p=0.5, k=2, n=12
    from scipy.stats import binom
    assert r.p_value == pytest.approx(min(1.0, 2 * binom.cdf(2, 12, 0.5)))


def test_mcnemar_zero_discordant_is_p1():
    r = mcnemar_test(0, 0)
    assert r.p_value == 1.0


def test_mcnemar_direction_symmetry():
    # swapping b and c gives the same two-sided p-value
    assert mcnemar_test(30, 12).p_value == pytest.approx(mcnemar_test(12, 30).p_value)


# ── Wilcoxon signed-rank ─────────────────────────────────────────────────────
def test_wilcoxon_detects_consistent_shift():
    x = [2, 3, 4, 5, 6, 7, 8]
    y = [1, 1, 1, 1, 1, 1, 1]
    r = wilcoxon_signed_rank(x, y)
    assert r.n_nonzero == 7
    assert r.p_value < 0.05


def test_wilcoxon_all_zero_differences():
    r = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert r.n_nonzero == 0 and r.p_value == 1.0


def test_wilcoxon_matches_scipy():
    from scipy.stats import wilcoxon as sp_wilcoxon
    x = np.array([1.2, 2.3, 0.7, 4.1, 3.3])
    y = np.array([0.5, 1.1, 0.9, 2.0, 2.9])
    r = wilcoxon_signed_rank(x, y)
    stat, p = sp_wilcoxon(x, y, zero_method="wilcox")
    assert r.statistic == pytest.approx(stat)
    assert r.p_value == pytest.approx(p)


# ── DeLong AUC ───────────────────────────────────────────────────────────────
def test_delong_identical_scores_z_zero():
    labels = [1, 1, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.2, 0.3, 0.1]
    r = delong_auc_test(s, s, labels)
    assert r.auc_a == pytest.approx(r.auc_b)
    assert r.statistic == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)


def test_delong_auc_values_correct():
    # perfectly separating scores -> AUC 1.0; chance scores -> AUC ~0.5
    labels = [1, 1, 1, 0, 0, 0]
    perfect = [0.9, 0.8, 0.7, 0.2, 0.3, 0.1]
    r = delong_auc_test(perfect, perfect, labels)
    assert r.auc_a == pytest.approx(1.0)


def test_delong_better_method_has_higher_auc():
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    good = [0.9, 0.85, 0.8, 0.7, 0.2, 0.25, 0.3, 0.1]
    poor = [0.5, 0.55, 0.45, 0.5, 0.5, 0.52, 0.48, 0.5]
    r = delong_auc_test(good, poor, labels)
    assert r.auc_a > r.auc_b


def test_delong_degenerate_labels():
    r = delong_auc_test([0.1, 0.2], [0.3, 0.4], [1, 1])   # no negatives
    assert r.p_value == 1.0


# ── paired bootstrap ─────────────────────────────────────────────────────────
def test_paired_bootstrap_recovers_known_difference():
    a = np.array([1.0] * 10)
    b = np.array([0.0] * 10)
    r = paired_bootstrap_diff(a, b, seed=0)
    assert r.diff == pytest.approx(1.0)
    assert r.ci_lower == pytest.approx(1.0) and r.ci_upper == pytest.approx(1.0)


def test_paired_bootstrap_deterministic():
    a = [1, 0, 1, 1, 0, 1, 0, 1]
    b = [0, 0, 1, 0, 0, 1, 0, 0]
    r1 = paired_bootstrap_diff(a, b, seed=7)
    r2 = paired_bootstrap_diff(a, b, seed=7)
    assert r1 == r2


def test_paired_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.5, 100)
    b = rng.normal(0.0, 0.5, 100)
    r = paired_bootstrap_diff(a, b, seed=1)
    assert r.ci_lower <= r.diff <= r.ci_upper


def test_paired_bootstrap_rejects_mismatched():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1], seed=0)


# ── Benjamini-Hochberg ───────────────────────────────────────────────────────
def test_bh_rejects_small_pvalues():
    r = benjamini_hochberg([0.001, 0.008, 0.02, 0.5, 0.9], alpha=0.05)
    assert r.rejected[0] and r.rejected[1] and r.rejected[2]
    assert not r.rejected[3] and not r.rejected[4]
    assert r.n_rejected == 3


def test_bh_all_null_rejects_none():
    r = benjamini_hochberg([0.5, 0.6, 0.7, 0.8], alpha=0.05)
    assert r.n_rejected == 0
    assert r.threshold == 0.0


def test_bh_all_significant_rejects_all():
    r = benjamini_hochberg([0.0001, 0.0002, 0.0003], alpha=0.05)
    assert r.n_rejected == 3


def test_bh_adjusted_pvalues_monotone_and_bounded():
    raw = [0.04, 0.01, 0.03, 0.005]
    r = benjamini_hochberg(raw, alpha=0.05)
    adj = np.array(r.pvals_adjusted)
    assert np.all((adj >= 0) & (adj <= 1))
    # adjusted p-values are non-decreasing when taken in ascending raw-p order
    order = np.argsort(raw)
    adj_in_order = adj[order]
    assert np.all(np.diff(adj_in_order) >= -1e-12)


def test_bh_preserves_input_order():
    r = benjamini_hochberg([0.9, 0.001, 0.5, 0.002], alpha=0.05)
    # positions 1 and 3 are the small p-values -> rejected; 0 and 2 not
    assert r.rejected[1] and r.rejected[3]
    assert not r.rejected[0] and not r.rejected[2]


def test_bh_empty():
    r = benjamini_hochberg([], alpha=0.05)
    assert r.n_rejected == 0 and r.rejected == tuple()
