"""Null-surrogate generators for empirical rho* calibration (milestone V4).

**Pure** module: imports **nothing** from ``mechanism``. All randomness flows
through :mod:`validation._seed`. These functions build the *null reference*
distribution the specification uses to calibrate the reproducibility threshold
``rho*`` (Part V step 11; the (Cal) property of Part III).

Why surrogates (and not just beta=0 draws)
-------------------------------------------
The specification and the validation roadmap define calibration in terms of **null
surrogates**: ``rho* = upper-alpha quantile of rho_hat under B null surrogates
(permute replicate labels / phase-randomize series)``. A surrogate takes a *given*
dataset (which may contain real signal) and destroys the cross-replicate
reproducibility while preserving the marginal structure that sets the sampling
noise, yielding a draw from the null ``beta_R = 0`` while keeping ``sigma_bar^2``
(hence ``N_eff``) correct. This is the canonical calibration mechanism.

Two schemes are provided, exactly as the roadmap specifies:

* **replicate-label permutation** (field-level; operates on the per-run effect
  frames): for each residue independently, permute its ``K`` per-replicate effect
  values across the replicate labels. This destroys any cross-replicate agreement
  (so the region-level ``beta`` collapses toward 0) while preserving each residue's
  *multiset* of values across replicates (so the per-residue marginal, and hence the
  within/between magnitude budget, is preserved).
* **phase randomization** (series-level; operates on Tier-B ``V(t)`` / ``d_i(t)``):
  replace each series by one with the **same power spectrum** but randomized
  Fourier phases. This preserves the autocorrelation (hence ``tau_int`` and
  ``N_eff``) while destroying the cross-series coupling that produces a non-zero
  Pearson ``theta`` — the null with ``N_eff`` correct, as the roadmap requires.

[CHOICE] The permutation scheme permutes **per residue independently** (not a single
shared permutation across residues). A single shared permutation would preserve the
cross-residue co-variation that ``A_en`` aggregates and would not fully break
region-level reproducibility; independent per-residue permutation is the standard
construction for a reproducibility null and is what makes ``beta_R`` collapse.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ── scheme 1: replicate-label permutation (field level) ──────────────────────
def permute_replicate_labels(
    frames: Sequence[pd.DataFrame], rng: np.random.Generator, *,
    effect_col: str = "r", abs_col: str = "abs_r",
) -> list:
    """Replicate-label-permutation surrogate of K per-run effect frames.

    For each residue (row position, shared across the ``K`` frames) independently,
    permute its ``K`` per-replicate ``effect_col`` values across the replicate
    labels. Returns a new list of ``K`` frames (inputs untouched). Everything except
    ``effect_col`` and ``abs_col`` is copied verbatim, so the frames remain valid
    production-schema inputs for ``aggregate_reproducibility``.

    Preserves each residue's multiset of effect values across replicates (marginal /
    sampling-noise budget) while destroying cross-replicate reproducibility, so the
    region-level ``beta`` collapses toward 0 — a draw from the ``beta_R = 0`` null.

    Requires the frames to share residue order and length (the generators guarantee
    this). Raises ``ValueError`` otherwise.
    """
    frames = list(frames)
    K = len(frames)
    if K == 0:
        return []
    n = len(frames[0])
    for f in frames:
        if len(f) != n:
            raise ValueError("all frames must have the same number of rows")
        if effect_col not in f.columns:
            raise ValueError(f"frame missing effect column {effect_col!r}")
    # (K, n) matrix of effects
    mat = np.stack([f[effect_col].to_numpy(dtype=float) for f in frames])
    out = mat.copy()
    for j in range(n):
        out[:, j] = mat[rng.permutation(K), j]
    new_frames = []
    for k in range(K):
        g = frames[k].copy()
        g[effect_col] = out[k]
        if abs_col in g.columns:
            g[abs_col] = np.abs(out[k])
        new_frames.append(g)
    return new_frames


# ── scheme 2: phase randomization (series level) ─────────────────────────────
def phase_randomize(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate of a 1-D real series (Theiler et al. 1992).

    Returns a real series with (to numerical precision) the **same power spectrum**
    as ``series`` — hence the same autocorrelation and integrated autocorrelation
    time, so ``N_eff`` is preserved under the null — but with randomized Fourier
    phases, which destroys deterministic/coupling structure. The mean is preserved;
    the DC and (for even length) Nyquist components are left unrandomized, and phases
    are made antisymmetric so the inverse transform is real.

    This is the canonical way to build a null that keeps second-order (spectral)
    structure while removing higher-order / cross-series structure.
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 3:
        return x.copy()
    X = np.fft.rfft(x)
    mag = np.abs(X)
    # random phases for the interior frequencies; keep DC (and Nyquist if even) fixed
    phases = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    phases[0] = 0.0  # DC: preserve the mean
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist must be real for an even-length real series
    Xr = mag * np.exp(1j * phases)
    return np.fft.irfft(Xr, n=n)


def phase_randomize_pairs(
    V: np.ndarray, d_by_canon: dict, rng: np.random.Generator,
) -> tuple:
    """Phase-randomize a reference ``V`` and every ``d_i`` independently.

    Independent phase randomization breaks the ``V``–``d_i`` coupling (so the
    recovered Pearson ``theta`` collapses toward 0, i.e. ``beta = 0``) while
    preserving each series' own spectrum (so each ``tau_int`` / ``N_eff`` is intact).
    Returns ``(V_surrogate, {canon: d_surrogate})``.
    """
    V_s = phase_randomize(V, rng)
    d_s = {}
    for cid, d in d_by_canon.items():
        d_s[cid] = phase_randomize(np.asarray(d, dtype=float), rng)
    return V_s, d_s


# ── spectrum-preservation diagnostic (pure; used by tests) ───────────────────
def power_spectrum(series: np.ndarray) -> np.ndarray:
    """One-sided power spectrum ``|rfft|^2`` of a real series (pure helper)."""
    x = np.asarray(series, dtype=float)
    return np.abs(np.fft.rfft(x)) ** 2
