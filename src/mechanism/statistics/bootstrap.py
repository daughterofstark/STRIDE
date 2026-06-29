"""Block-bootstrap confidence intervals for the correlation effect (milestone M2).

Adds *uncertainty only* (confidence intervals and a bootstrap SE) around the
existing per-residue Pearson correlation. The effect itself is never recomputed
or altered. Implements §2.1 of MATHEMATICAL_SPECIFICATION.md (the block-bootstrap
companion to the analytic effective-N interval of M1). Nothing from later
milestones is implemented here.

Method choice (deviation from the roadmap's "moving-block", justified)
---------------------------------------------------------------------
The default is the **circular block bootstrap** (CBB; Politis & Romano 1992)
rather than the plain moving-block bootstrap (MBB; Künsch 1989):

* Equilibrium MD is assumed (weakly) stationary, so wrapping the series removes
  MBB's endpoint under-weighting and gives every frame equal resampling weight.
* At the optimal block length, block methods (CBB/MBB) have lower variance-
  estimation MSE than the stationary bootstrap (Lahiri 1999), so a block method
  is kept as the default.

The **stationary bootstrap** (SB; Politis & Romano 1994) is provided as an option
because its random (geometric) block lengths make it more robust to block-length
misspecification — relevant because the integrated autocorrelation time inherited
from M1 is itself noisy for short trajectories.

For a *cross*-statistic (a correlation between two paired series) the same time
indices are applied to both signals so that the cross-correlation structure is
preserved. Percentiles are taken on the Fisher-z scale and back-transformed,
which improves coverage for a bounded, skewed statistic at negligible cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Safeguards / defaults ------------------------------------------------------
_VAR_EPS = 1e-12
_MIN_FRAMES = 20          # below this, block bootstrap is unreliable -> Fisher fallback
_MIN_BLOCKS = 10          # require at least this many blocks, else fallback
_BLOCK_C = 2.0            # block length = c * tau_int (one+ decorrelation time)
_R_CLIP = 1.0 - 1e-9      # clip |r| before atanh
_MIN_VALID_FRAC = 0.5     # fraction of finite bootstrap reps required
DEFAULT_B = 1000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap uncertainty for one correlation effect.

    Attributes
    ----------
    se : float
        Bootstrap standard error of ``r`` (std of the bootstrap replicates).
    ci_lower, ci_upper : float
        Confidence-interval bounds at the requested level.
    method : str
        ``circular``, ``stationary``, ``fisher_neff`` (fallback), or
        ``degenerate`` (no estimable interval; bounds are NaN).
    block_length : float
        Block length used (NaN for non-block methods).
    n_rep : int
        Number of finite bootstrap replicates actually used (0 for fallbacks).
    """

    se: float
    ci_lower: float
    ci_upper: float
    method: str
    block_length: float
    n_rep: int


# ── block-length selection (uses M1's tau_int) ──────────────────────────────
def select_block_length(
    tau_int: float, n: int, c: float = _BLOCK_C, min_blocks: int = _MIN_BLOCKS
) -> tuple[int, bool]:
    """Choose a block length from the integrated autocorrelation time.

    Parameters
    ----------
    tau_int : float
        Integrated autocorrelation time from M1.
    n : int
        Series length.
    c : float
        Multiplier; the block spans about ``c`` decorrelation times.
    min_blocks : int
        Minimum number of blocks required for a usable block bootstrap.

    Returns
    -------
    block_length : int
        Chosen block length (>= 1).
    ok : bool
        False if the series cannot support ``min_blocks`` blocks of this length
        (caller should fall back).
    """
    if not np.isfinite(tau_int) or tau_int < 0.5:
        tau_int = 0.5
    L = int(max(1, round(c * tau_int)))
    max_L = n // min_blocks
    if max_L < 1:
        return 1, False
    if L > max_L:
        return max_L, (max_L >= 1 and n // max_L >= min_blocks)
    return L, True


# ── index generators ────────────────────────────────────────────────────────
def circular_block_indices(n: int, L: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index matrix of shape ``(B, n)``."""
    nblocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nblocks))
    offsets = np.arange(L)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(B, nblocks * L)[:, :n]


def stationary_block_indices(n: int, mean_L: float, B: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap index matrix of shape ``(B, n)`` (geometric blocks)."""
    p = 1.0 / max(mean_L, 1.0)
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=B)
    newblock = rng.random((B, n)) < p
    rand_starts = rng.integers(0, n, size=(B, n))
    for t in range(1, n):
        cont = ~newblock[:, t]
        idx[:, t] = np.where(cont, (idx[:, t - 1] + 1) % n, rand_starts[:, t])
    return idx


def _row_pearson(A: np.ndarray, Bm: np.ndarray) -> np.ndarray:
    """Vectorised Pearson correlation for each paired row of ``A`` and ``Bm``."""
    A = A - A.mean(axis=1, keepdims=True)
    Bm = Bm - Bm.mean(axis=1, keepdims=True)
    num = (A * Bm).sum(axis=1)
    den = np.sqrt((A * A).sum(axis=1) * (Bm * Bm).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r


def _fisher_neff_ci(r: float, n_eff: float, alpha: float) -> BootstrapResult:
    """Analytic Fisher/N_eff fallback interval (ties M2 back to M1)."""
    if not np.isfinite(n_eff) or n_eff <= 3 or not np.isfinite(r):
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               "degenerate", float("nan"), 0)
    z = np.arctanh(np.clip(r, -_R_CLIP, _R_CLIP))
    se_z = 1.0 / np.sqrt(n_eff - 3.0)
    from scipy.stats import norm
    zc = norm.ppf(1.0 - alpha / 2.0)
    lo, hi = np.tanh(z - zc * se_z), np.tanh(z + zc * se_z)
    se_r = float((1.0 - r * r) / np.sqrt(n_eff))  # delta-method SE on r scale
    return BootstrapResult(se_r, float(lo), float(hi), "fisher_neff", float("nan"), 0)


def _finalize(r_b: np.ndarray, alpha: float, method: str, L: float):
    """Build a BootstrapResult from replicate correlations, or None to fall back."""
    finite = np.isfinite(r_b)
    if finite.mean() < _MIN_VALID_FRAC:
        return None
    r_b = r_b[finite]
    se = float(np.std(r_b, ddof=1))
    z_b = np.arctanh(np.clip(r_b, -_R_CLIP, _R_CLIP))
    lo_z, hi_z = np.percentile(z_b, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return BootstrapResult(se, float(np.tanh(lo_z)), float(np.tanh(hi_z)),
                           method, float(L), int(r_b.size))


# ── main entry point ────────────────────────────────────────────────────────
def bootstrap_correlation(
    v: np.ndarray,
    d: np.ndarray,
    tau_int: float,
    n_eff: float,
    *,
    method: str = "circular",
    B: int = DEFAULT_B,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> BootstrapResult:
    """Block-bootstrap confidence interval for the correlation between ``v`` and ``d``.

    Parameters
    ----------
    v, d : numpy.ndarray
        Paired 1-D signals (pocket volume and a residue distance).
    tau_int, n_eff : float
        From M1; ``tau_int`` sets the block length, ``n_eff`` powers the Fisher
        fallback.
    method : {"circular", "stationary"}
        Block-bootstrap variant. Default circular (see module docstring).
    B : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (0.05 -> 95% CI).
    seed : int
        Deterministic seed (reproducible CIs).

    Returns
    -------
    BootstrapResult

    Notes
    -----
    Uncertainty only; the correlation point estimate is not produced here and is
    never altered. Graceful fallbacks: constant/near-constant signals ->
    ``degenerate``; too-short series or too-few blocks -> ``fisher_neff``.
    """
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    n = min(v.size, d.size)
    v, d = v[:n], d[:n]

    # Degenerate: no variance -> no estimable interval (never fabricate one)
    if n == 0 or np.var(v) <= _VAR_EPS or np.var(d) <= _VAR_EPS:
        return BootstrapResult(float("nan"), float("nan"), float("nan"),
                               "degenerate", float("nan"), 0)

    r_point = float(np.corrcoef(v, d)[0, 1])

    # Too short for a block bootstrap -> analytic fallback
    if n < _MIN_FRAMES:
        return _fisher_neff_ci(r_point, n_eff, alpha)

    L, ok = select_block_length(tau_int, n)
    if not ok:
        return _fisher_neff_ci(r_point, n_eff, alpha)

    rng = np.random.default_rng(seed)
    if method == "stationary":
        idx = stationary_block_indices(n, L, B, rng)
        method_used = "stationary"
    else:
        idx = circular_block_indices(n, L, B, rng)
        method_used = "circular"

    r_b = _row_pearson(v[idx], d[idx])
    res = _finalize(r_b, alpha, method_used, L)
    if res is None:
        # Pathological resamples (many constant blocks) -> fall back, don't lie
        return _fisher_neff_ci(r_point, n_eff, alpha)
    return res


def attach_bootstrap_ci(
    df_res: pd.DataFrame,
    residues,
    dist_matrix: np.ndarray,
    volumes: np.ndarray,
    *,
    method: str = "circular",
    B: int = DEFAULT_B,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
    resid_attr: str = "resid",
    resid_col: str = "file_resid",
    tau_col: str = "tau_int",
    neff_col: str = "n_eff",
) -> pd.DataFrame:
    """Append block-bootstrap CI columns to a per-residue table.

    Reads the existing ``volumes``, the Cα distance matrix, and the M1 columns
    ``tau_int``/``n_eff`` (read-only). Appends, in order:
    ``theta_bootstrap_se, theta_bootstrap_ci_lower, theta_bootstrap_ci_upper,
    bootstrap_method, bootstrap_block_length, bootstrap_replicates``.
    Existing columns (including the M1 columns) are left untouched.

    Performance: residues that share a block length share a single resampling
    index matrix (and a single ``volumes`` gather), computed once per distinct
    block length. This does not bias any individual interval; it only makes the
    per-residue CIs share a common (deterministic) resampling scheme.

    Determinism: index matrices are drawn from child seeds spawned from ``seed``
    via ``numpy`` ``SeedSequence``, keyed by block length in sorted order, so
    results are fully reproducible and independent of residue ordering.
    """
    vols = np.asarray(volumes, dtype=float)
    n = min(len(vols), dist_matrix.shape[0])
    vols = vols[:n]

    tau_by_fid = dict(zip(df_res[resid_col], df_res[tau_col]))
    neff_by_fid = dict(zip(df_res[resid_col], df_res[neff_col]))

    # Classify residues into block-bootstrap groups (keyed by block length) vs
    # fallback (degenerate / too short / too few blocks).
    recs: dict[int, BootstrapResult] = {}
    groups: dict[int, list[tuple[int, int]]] = {}  # L -> [(fid, col_index)]
    for i, res in enumerate(residues):
        fid = int(getattr(res, resid_attr))
        d = dist_matrix[:n, i]
        if np.var(vols) <= _VAR_EPS or np.var(d) <= _VAR_EPS:
            recs[fid] = BootstrapResult(float("nan"), float("nan"), float("nan"),
                                        "degenerate", float("nan"), 0)
            continue
        tau = float(tau_by_fid.get(fid, np.nan))
        neff = float(neff_by_fid.get(fid, np.nan))
        if n < _MIN_FRAMES:
            recs[fid] = _fisher_neff_ci(float(np.corrcoef(vols, d)[0, 1]), neff, alpha)
            continue
        L, ok = select_block_length(tau, n)
        if not ok:
            recs[fid] = _fisher_neff_ci(float(np.corrcoef(vols, d)[0, 1]), neff, alpha)
            continue
        groups.setdefault(L, []).append((fid, i))

    # One index matrix (and one volume gather) per distinct block length.
    for L in sorted(groups):
        child = np.random.SeedSequence([seed, L]).spawn(1)[0]
        rng = np.random.default_rng(child)
        if method == "stationary":
            idx = stationary_block_indices(n, float(L), B, rng)
            mname = "stationary"
        else:
            idx = circular_block_indices(n, L, B, rng)
            mname = "circular"
        vol_boot = vols[idx]  # gathered once for the whole group
        for fid, i in groups[L]:
            d = dist_matrix[:n, i]
            r_b = _row_pearson(vol_boot, d[idx])
            res = _finalize(r_b, alpha, mname, L)
            if res is None:
                res = _fisher_neff_ci(float(np.corrcoef(vols, d)[0, 1]),
                                      float(neff_by_fid.get(fid, np.nan)), alpha)
            recs[fid] = res

    df_res["theta_bootstrap_se"] = df_res[resid_col].map(lambda f: recs[f].se)
    df_res["theta_bootstrap_ci_lower"] = df_res[resid_col].map(lambda f: recs[f].ci_lower)
    df_res["theta_bootstrap_ci_upper"] = df_res[resid_col].map(lambda f: recs[f].ci_upper)
    df_res["bootstrap_method"] = df_res[resid_col].map(lambda f: recs[f].method)
    df_res["bootstrap_block_length"] = df_res[resid_col].map(lambda f: recs[f].block_length)
    df_res["bootstrap_replicates"] = df_res[resid_col].map(lambda f: recs[f].n_rep)
    return df_res
