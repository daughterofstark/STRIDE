"""Paired comparative statistical tests for the V6 baseline comparison.

**Pure** module: numpy/scipy only, no ``mechanism``. These are the pre-registered
tests the roadmap lists for comparing STRIDE against the Part VI baselines on the same
seeded ensembles:

* ``mcnemar_test`` — paired *binary* outcomes (e.g. correct/incorrect scale decision):
  tests whether two methods' discordant outcomes are symmetric.
* ``wilcoxon_signed_rank`` — paired *continuous* outcomes (e.g. per-seed metric):
  non-parametric paired location test.
* ``delong_auc_test`` — paired comparison of two ROC AUCs on the *same* labels
  (DeLong, DeLong & Clarke-Pearson 1988), via placement/structural components.
* ``paired_bootstrap_diff`` — bootstrap CI for the difference of a paired metric,
  deterministic in its seed.
* ``benjamini_hochberg`` — BH-FDR control across the grid of per-claim p-values.

All randomness flows through an explicit seed (``paired_bootstrap_diff``); the rest are
deterministic closed forms. Results are returned as small frozen dataclasses so the
comparison pipeline and its artifact are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy import stats


# ── McNemar (paired binary) ──────────────────────────────────────────────────
@dataclass(frozen=True)
class McNemarResult:
    b: int          # method A correct, B incorrect
    c: int          # method A incorrect, B correct
    statistic: float
    p_value: float
    corrected: bool


def mcnemar_test(b: int, c: int, *, correction: bool = True,
                 exact_threshold: int = 25) -> McNemarResult:
    """McNemar's test on the two discordant counts ``b`` and ``c``.

    ``b`` = #(A right, B wrong); ``c`` = #(A wrong, B right). For small discordant
    totals (``b + c <= exact_threshold``) an exact binomial test is used; otherwise the
    chi-square approximation (with continuity correction by default). Concordant pairs
    do not enter the statistic. Returns statistic and two-sided p-value.
    """
    b = int(b)
    c = int(c)
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, correction)
    if n <= exact_threshold:
        # exact two-sided binomial test at p = 0.5
        k = min(b, c)
        p = float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))
        stat = float(k)
        return McNemarResult(b, c, stat, p, corrected=False)
    if correction:
        stat = (abs(b - c) - 1.0) ** 2 / n
    else:
        stat = (b - c) ** 2 / n
    p = float(stats.chi2.sf(stat, df=1))
    return McNemarResult(b, c, float(stat), p, correction)


# ── Wilcoxon signed-rank (paired continuous) ─────────────────────────────────
@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_nonzero: int


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float],
                         *, alternative: str = "two-sided") -> WilcoxonResult:
    """Wilcoxon signed-rank test on paired samples ``x`` and ``y``.

    Wraps ``scipy.stats.wilcoxon`` (documented dependency). Pairs with zero difference
    are dropped (``zero_method='wilcox'``). Returns the statistic, p-value, and the
    number of non-zero-difference pairs actually used. If all differences are zero, the
    p-value is 1.0 by convention.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    nz = int(np.count_nonzero(d))
    if nz == 0:
        return WilcoxonResult(0.0, 1.0, 0)
    stat, p = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return WilcoxonResult(float(stat), float(p), nz)


# ── DeLong paired AUC comparison ─────────────────────────────────────────────
@dataclass(frozen=True)
class DeLongResult:
    auc_a: float
    auc_b: float
    statistic: float     # z
    p_value: float


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks of ``x`` (ties averaged) — the DeLong placement primitive."""
    order = np.argsort(x, kind="mergesort")
    ranked = x[order]
    n = x.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1] == ranked[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _auc_and_structural(pos: np.ndarray, neg: np.ndarray):
    """AUC plus the DeLong structural components ``V10`` (pos) and ``V01`` (neg)."""
    m = pos.size
    n = neg.size
    tx = _midrank(pos)
    ty = _midrank(neg)
    txy = _midrank(np.concatenate([pos, neg]))
    txy_pos = txy[:m]
    txy_neg = txy[m:]
    auc = (txy_pos.sum() - m * (m + 1) / 2.0) / (m * n)
    v10 = (txy_pos - tx) / n            # length m
    v01 = 1.0 - (txy_neg - ty) / m      # length n
    return auc, v10, v01


def delong_auc_test(scores_a: Sequence[float], scores_b: Sequence[float],
                    labels: Sequence[int]) -> DeLongResult:
    """DeLong test comparing two AUCs on the **same** labels (paired by sample).

    ``scores_a``/``scores_b`` are two methods' scores for the same items;
    ``labels`` are 0/1 (1 = positive/reproducible). Returns both AUCs, the DeLong
    ``z`` statistic for their difference, and a two-sided p-value. Identical score
    vectors give ``z = 0`` and ``p = 1``.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels).astype(int)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return DeLongResult(float("nan"), float("nan"), 0.0, 1.0)

    auc_a, v10_a, v01_a = _auc_and_structural(a[pos], a[neg])
    auc_b, v10_b, v01_b = _auc_and_structural(b[pos], b[neg])
    m = int(pos.sum())
    n = int(neg.sum())

    # 2x2 covariance of (auc_a, auc_b) via structural components
    def cov(u_a, u_b, k):
        return np.cov(np.stack([u_a, u_b]), ddof=1)[0, 1] / k if k > 1 else 0.0

    s10 = np.cov(np.stack([v10_a, v10_b]), ddof=1) / m if m > 1 else np.zeros((2, 2))
    s01 = np.cov(np.stack([v01_a, v01_b]), ddof=1) / n if n > 1 else np.zeros((2, 2))
    S = s10 + s01
    var_diff = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    if var_diff <= 0:
        # identical or degenerate -> no detectable difference
        z = 0.0 if abs(auc_a - auc_b) < 1e-12 else np.inf * np.sign(auc_a - auc_b)
        p = 1.0 if np.isfinite(z) else 0.0
        return DeLongResult(float(auc_a), float(auc_b), float(z), float(p))
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2.0 * stats.norm.sf(abs(z)))
    return DeLongResult(float(auc_a), float(auc_b), float(z), p)


# ── paired bootstrap difference ──────────────────────────────────────────────
@dataclass(frozen=True)
class PairedBootstrapResult:
    diff: float                 # mean(a - b)
    ci_lower: float
    ci_upper: float
    n_boot: int


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], *,
                          n_boot: int = 2000, ci: float = 0.95,
                          seed: int = 0) -> PairedBootstrapResult:
    """Bootstrap CI for the paired difference ``mean(a - b)`` (deterministic in seed).

    Resamples paired indices with replacement ``n_boot`` times. Returns the observed
    mean difference and a percentile CI. Used to compare a paired per-seed metric
    (e.g. over-resolution indicator) between STRIDE and a baseline.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and the same shape")
    d = a - b
    n = d.size
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 - (1 - ci) / 2)))
    return PairedBootstrapResult(float(d.mean()), lo, hi, int(n_boot))


# ── Benjamini-Hochberg FDR ───────────────────────────────────────────────────
@dataclass(frozen=True)
class BHResult:
    rejected: tuple             # bool per input p-value (original order)
    pvals_adjusted: tuple       # BH-adjusted p-values (original order)
    threshold: float            # largest p passing the BH line (0 if none)
    n_rejected: int


def benjamini_hochberg(pvals: Sequence[float], *, alpha: float = 0.05) -> BHResult:
    """Benjamini-Hochberg FDR control at level ``alpha`` across a p-value grid.

    Returns per-hypothesis rejection flags (in the original order), BH-adjusted
    p-values, and the rejection threshold. Standard step-up procedure.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return BHResult(tuple(), tuple(), 0.0, 0)
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    crit = ranks / m * alpha
    below = p_sorted <= crit
    if below.any():
        kmax = np.max(np.where(below)[0])
        threshold = float(p_sorted[kmax])
        rejected_sorted = np.arange(m) <= kmax
    else:
        threshold = 0.0
        rejected_sorted = np.zeros(m, dtype=bool)
    # BH-adjusted p-values (monotone from the top)
    adj_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    rejected = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adj_sorted
    return BHResult(tuple(bool(x) for x in rejected),
                    tuple(float(x) for x in adjusted),
                    threshold, int(rejected.sum()))
