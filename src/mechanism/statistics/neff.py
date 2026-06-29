"""Effective sample size for autocorrelated MD observables (milestone M1).

Implements the within-replicate uncertainty machinery of the mathematical
specification, §2.1 only:

    N_eff = T / (2 * tau_int),     tau_int = 1/2 + sum_{Delta>=1} rho(Delta)

with ``tau_int`` estimated from the per-frame product (covariance integrand)
series of the two signals whose coupling defines the effect, using an
FFT-based autocorrelation and Sokal automatic windowing.

This module computes *uncertainty only*. It never touches the effect field
(the Pearson correlation produced by the existing pipeline). Nothing from
later milestones (block bootstrap, hierarchical models, variance components,
reproducibility coefficient, resolution gate) is implemented here.

References
----------
Flyvbjerg & Petersen (1989); Sokal (1997) automatic windowing; Geyer (1992).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Numerical safeguards -------------------------------------------------------
_MIN_TAU = 0.5          # white-noise floor: a process cannot mix faster than i.i.d.
_MIN_FRAMES = 8         # below this, the IAT estimate is flagged unreliable
_VAR_EPS = 1e-12        # variance below which a signal is treated as constant
_SOKAL_C = 5.0          # Sokal window constant (window M >= C * tau_int(M))


@dataclass(frozen=True)
class NeffResult:
    """Effective-sample-size estimate for one observable or coupling.

    Attributes
    ----------
    tau_int : float
        Integrated autocorrelation time in frames (>= 0.5), convention
        ``tau_int = 1/2 + sum_{Delta>=1} rho(Delta)``.
    n_eff : float
        Effective number of independent samples, ``T / (2 * tau_int)``,
        floored at 2.0.
    n_frames : int
        Number of frames actually used.
    status : str
        One of ``ok``, ``white_noise``, ``constant_signal``,
        ``short_trajectory``, ``undersampled_capped``.
    """

    tau_int: float
    n_eff: float
    n_frames: int
    status: str


def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation function via FFT (linear, not circular).

    Parameters
    ----------
    x : numpy.ndarray
        1-D real signal.

    Returns
    -------
    numpy.ndarray
        ``rho`` with ``rho[0] == 1``; length ``len(x)``. Returns an array of
        zeros (with ``rho[0] == 0``) when the signal has no variance, so callers
        can detect the constant case.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    xc = x - x.mean()
    var0 = float(np.dot(xc, xc))
    if var0 <= _VAR_EPS * max(n, 1):
        out = np.zeros(n)
        return out  # rho[0] == 0 signals "constant"
    # zero-pad to next power of two >= 2n for a non-circular estimate
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.fft(xc, n=nfft)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    return acf


def integrated_autocorr_time(
    x: np.ndarray, c: float = _SOKAL_C
) -> tuple[float, str]:
    """Integrated autocorrelation time with Sokal automatic windowing.

    Parameters
    ----------
    x : numpy.ndarray
        1-D real signal (e.g. the per-frame covariance integrand).
    c : float, optional
        Sokal window constant; the window ``M`` is the smallest index with
        ``M >= c * tau_int(M)``. Default 5.0.

    Returns
    -------
    tau_int : float
        Estimated integrated autocorrelation time (>= 0.5).
    status : str
        ``ok``, ``white_noise``, ``constant_signal``, ``short_trajectory`` or
        ``undersampled_capped``.

    Notes
    -----
    Convention ``tau_int = 1/2 + sum_{Delta>=1} rho(Delta)``. For an AR(1)
    process with parameter ``phi`` the analytic value is
    ``0.5 * (1 + phi) / (1 - phi)``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return _MIN_TAU, "constant_signal"

    acf = autocorr_fft(x)
    if acf[0] == 0.0:  # constant signal
        return _MIN_TAU, "constant_signal"

    # tau_int(M) = 0.5 + sum_{t=1..M} rho(t) = cumsum(rho)[M] - 0.5
    cumsum_rho = np.cumsum(acf)
    tau_window = cumsum_rho - 0.5  # index m -> tau_int using window m

    # Sokal: first window m with m >= c * tau_window[m]
    m_idx = np.arange(n)
    converged = m_idx >= c * tau_window
    if converged.any():
        window = int(np.argmax(converged))  # first True
        status = "ok"
    else:
        window = n - 1  # never converged: very autocorrelated / under-sampled
        status = "undersampled_capped"

    tau = float(tau_window[window])

    # Safeguards
    if tau < _MIN_TAU:
        tau = _MIN_TAU
        if status == "ok":
            status = "white_noise"
    tau_cap = n / 4.0  # ensures n_eff >= 2
    if tau > tau_cap:
        tau = tau_cap
        status = "undersampled_capped"

    if n < _MIN_FRAMES:
        status = "short_trajectory"

    return tau, status


def effective_sample_size(
    v: np.ndarray, d: np.ndarray, c: float = _SOKAL_C
) -> NeffResult:
    """Effective sample size for the coupling between two signals.

    The integrated autocorrelation time is estimated on the per-frame product
    (covariance integrand) series ``z(t) = (v - mean v)(d - mean d)``, which is
    the series governing the sampling variance of a covariance/correlation
    estimate.

    Parameters
    ----------
    v, d : numpy.ndarray
        1-D signals of equal length (e.g. pocket volume and a residue distance).
    c : float, optional
        Sokal window constant.

    Returns
    -------
    NeffResult
        Estimate with ``tau_int``, ``n_eff``, ``n_frames`` and ``status``.

    Notes
    -----
    Uncertainty only. Does not compute or alter any correlation/effect value.
    """
    v = np.asarray(v, dtype=float)
    d = np.asarray(d, dtype=float)
    n = min(v.size, d.size)
    if n == 0:
        return NeffResult(_MIN_TAU, 2.0, 0, "constant_signal")
    v, d = v[:n], d[:n]

    if np.var(v) <= _VAR_EPS or np.var(d) <= _VAR_EPS:
        # No coupling is estimable; report independent-baseline N_eff.
        status = "constant_signal"
        n_eff = float(n)
        return NeffResult(_MIN_TAU, max(n_eff, 2.0), n, status)

    z = (v - v.mean()) * (d - d.mean())
    tau, status = integrated_autocorr_time(z, c=c)
    n_eff = max(n / (2.0 * tau), 2.0)
    return NeffResult(float(tau), float(n_eff), int(n), status)


def corrected_standard_error(r: float, n_eff: float) -> float:
    """Autocorrelation-corrected standard error of a Pearson correlation.

    Uses the Fisher large-sample variance with the effective sample size:
    ``SE = sqrt((1 - r**2)**2 / n_eff)``.

    Parameters
    ----------
    r : float
        The (existing) correlation effect for this residue. Read-only input.
    n_eff : float
        Effective sample size from :func:`effective_sample_size`.

    Returns
    -------
    float
        Standard error; ``nan`` if ``n_eff`` is non-positive.
    """
    if not np.isfinite(n_eff) or n_eff <= 0:
        return float("nan")
    r = float(r)
    return float(np.sqrt(((1.0 - r * r) ** 2) / n_eff))


def attach_effective_sample_size(
    df_res: pd.DataFrame,
    residues,
    dist_matrix: np.ndarray,
    volumes: np.ndarray,
    *,
    resid_attr: str = "resid",
    resid_col: str = "file_resid",
    r_col: str = "r",
    c: float = _SOKAL_C,
) -> pd.DataFrame:
    """Append effective-sample-size uncertainty columns to a per-residue table.

    Uncertainty only: this reads the existing ``volumes`` and the Cα distance
    matrix and the already-computed correlation column ``r_col``; it does not
    modify any existing column or recompute any effect. Adds, in order:
    ``tau_int``, ``n_eff``, ``neff_status``, ``theta_se``.

    Parameters
    ----------
    df_res : pandas.DataFrame
        Existing per-residue results (must contain ``resid_col`` and ``r_col``).
    residues : sequence
        Residue objects in the same column order as ``dist_matrix``; each must
        expose ``resid_attr`` (default ``.resid``).
    dist_matrix : numpy.ndarray
        Frames x residues distance matrix used for the correlations (read-only).
    volumes : numpy.ndarray
        Per-frame pocket volume series (read-only).
    resid_attr, resid_col, r_col : str
        Attribute / column names used to align residues to ``df_res``.
    c : float
        Sokal window constant.

    Returns
    -------
    pandas.DataFrame
        The same DataFrame with four appended columns. Existing columns are
        left byte-for-byte unchanged.
    """
    vols = np.asarray(volumes, dtype=float)
    n_frames = min(len(vols), dist_matrix.shape[0])
    vols = vols[:n_frames]

    recs: dict[int, NeffResult] = {}
    for i, res in enumerate(residues):
        est = effective_sample_size(vols, dist_matrix[:n_frames, i], c=c)
        recs[int(getattr(res, resid_attr))] = est

    df_res["tau_int"] = df_res[resid_col].map(lambda f: recs[f].tau_int)
    df_res["n_eff"] = df_res[resid_col].map(lambda f: recs[f].n_eff)
    df_res["neff_status"] = df_res[resid_col].map(lambda f: recs[f].status)
    df_res["theta_se"] = [
        corrected_standard_error(r, ne)
        for r, ne in zip(df_res[r_col].to_numpy(), df_res["n_eff"].to_numpy())
    ]
    return df_res
