"""Regional aggregation and the reproducibility coefficient (milestone M4).

Implements §2.4 of MATHEMATICAL_SPECIFICATION.md: the two aggregation operators
``A_en`` / ``A_sgn``, the field standardization required by property I3, the
delta-method propagation of within-replicate sampling variance to the regional
statistic, the bias-corrected ``beta^2``, and the **reproducibility coefficient**

    rho_R = beta_R^2 / (beta_R^2 + tau_R^2 + sigma_bar_R^2)  in [0, 1].

It also provides the multi-scale driver ``aggregate_reproducibility`` that, given
the K per-replicate effect fields (the M1/M2 per-run CSVs) and the M3 hierarchy
config, reconstructs the nested partitions and computes ``rho`` for every region
at every scale, yielding ``varcomp.csv`` and ``rho_by_scale.csv``.

Scope (faithful to the roadmap M4 boundary): this is the **per-scale engine**.
It does **not** implement the resolution gate ``hat ell*`` or profile ``Pi``
(§2.5-2.6, M5), the calibrated threshold ``rho*`` (validation phase), or the
directional emit / "mixed" labelling. ``A_sgn`` and a directional-coherence
statistic are provided as informational helpers, not as a gate. M0-M3 are not
modified; this module only reads their outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from . import varcomp

_EPS = 1e-12


# ── field standardization (property I3 / algorithm step 5) ───────────────────
def pooled_scale(theta_runs: Sequence[np.ndarray]) -> float:
    """Pooled dispersion of the effect field across all residues and replicates.

    ``A_en`` is not scale-invariant (spec property I3), so the effect field must
    be standardized to a common scale before aggregation. We use a *scale-only*
    standardization (divide by the pooled SD): mean-centering is intentionally
    omitted because ``A_en`` aggregates squared magnitudes ("energy") and
    subtracting a global mean would distort that quantity. The same scale is
    applied to every replicate, so between-replicate variation (``tau^2``) is
    preserved.
    """
    flat = np.concatenate([np.asarray(t, float).ravel() for t in theta_runs]) \
        if len(theta_runs) else np.asarray([])
    if flat.size == 0:
        return 1.0
    sd = float(np.std(flat))
    return sd if sd > _EPS else 1.0


# ── aggregation operators (spec §2.4) ────────────────────────────────────────
def energy(theta: np.ndarray) -> float:
    """``A_en`` — support-invariant energy ``sqrt(sum_i theta_i^2)``."""
    theta = np.asarray(theta, float).ravel()
    if theta.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(theta ** 2)))


def signed_mean(theta: np.ndarray) -> float:
    """``A_sgn`` — signed mean ``mean_i theta_i`` (direction/magnitude)."""
    theta = np.asarray(theta, float).ravel()
    if theta.size == 0:
        return 0.0
    return float(np.mean(theta))


def directional_coherence(theta: np.ndarray) -> float:
    """Directional-coherence statistic in ``[0, 1]`` (informational, not a gate).

    ``|sum theta| / sum |theta|`` = 1 when all residues agree in sign, → 0 when
    the region is directionally balanced. The spec (§2.4, assumption A4) gates the
    *signed* claim on a homogeneity test; that gating/emit is M5. Here we only
    expose the statistic.
    """
    theta = np.asarray(theta, float).ravel()
    denom = float(np.sum(np.abs(theta)))
    if denom <= _EPS:
        return 0.0
    return float(abs(np.sum(theta)) / denom)


def propagate_energy_sigma2(theta: np.ndarray, sigma2: np.ndarray) -> float:
    """Delta-method variance of ``A_en`` from per-residue variances.

    For ``Theta = sqrt(sum theta_i^2)``, ``d Theta / d theta_i = theta_i / Theta``,
    so ``Var(Theta) ~ sum_i (theta_i / Theta)^2 sigma_i^2``. At ``Theta ~ 0`` the
    gradient is undefined; we return ``sum_i sigma_i^2`` (a conservative upper
    estimate that drives ``rho`` toward 0 for a null region — the safe direction).
    """
    theta = np.asarray(theta, float).ravel()
    sigma2 = np.asarray(sigma2, float).ravel()
    if theta.size == 0:
        return 0.0
    th = energy(theta)
    if th <= _EPS:
        return float(np.sum(sigma2))
    return float(np.sum((theta ** 2) * sigma2) / (th ** 2))


def propagate_signed_sigma2(sigma2: np.ndarray) -> float:
    """Variance of ``A_sgn`` under within-replicate residue independence.

    ``Var(mean_i theta_i) ~ (1/|R|^2) sum_i sigma_i^2``. The independence
    approximation across residues within a replicate is documented; residues in a
    region share the reference signal ``V``, so this is a first-order estimate.
    """
    sigma2 = np.asarray(sigma2, float).ravel()
    n = sigma2.size
    if n == 0:
        return 0.0
    return float(np.sum(sigma2) / (n ** 2))


# ── reproducibility coefficient (spec §2.4) ──────────────────────────────────
def beta2_bias_corrected(theta_bar: float, var_theta_bar: float) -> float:
    """``beta^2`` with the upward bias of ``mean(Theta)^2`` removed (§2.4).

    ``hat beta^2 = (mean(Theta)^2 - Var_hat(mean Theta))_+``.
    """
    return float(max(theta_bar ** 2 - var_theta_bar, 0.0))


def reproducibility_coefficient(beta2_bc: float, tau2: float,
                                sigma2_bar: float) -> float:
    """``rho_R = beta^2 / (beta^2 + tau^2 + sigma_bar^2)`` clipped to ``[0, 1]``.

    Guards: ``nan`` ``tau^2`` (unidentified, ``K < 2``) → ``nan`` (gate-uncertain);
    all-zero denominator → ``0`` (a consistent-but-null region is not reproducible).
    """
    if not np.isfinite(tau2):
        return float("nan")
    denom = beta2_bc + tau2 + sigma2_bar
    if denom <= _EPS:
        return 0.0
    return float(np.clip(beta2_bc / denom, 0.0, 1.0))


@dataclass(frozen=True)
class RegionReproducibility:
    """Per-region reproducibility result at one scale."""

    n_residues: int
    n_replicates: int
    beta: float
    beta_se: float
    tau2: float
    sigma2_bar: float
    beta2_bc: float
    rho: float
    a_signed: float
    coherence: float
    method: str
    status: str


def region_reproducibility(
    theta_runs: Sequence[np.ndarray],
    sigma2_runs: Sequence[np.ndarray],
    *,
    method: str = "auto",
    prior_scale: Optional[float] = None,
) -> RegionReproducibility:
    """Reproducibility coefficient for one region from K replicate effect vectors.

    Parameters
    ----------
    theta_runs : sequence of arrays
        ``theta_runs[k]`` = the (already standardized) effect values of the
        region's residues present in replicate ``k``.
    sigma2_runs : sequence of arrays
        ``sigma2_runs[k]`` = matching per-residue within-replicate variances
        (already on the standardized scale).

    Notes
    -----
    Uses ``A_en`` for the reproducibility decision and the delta-method
    propagation for ``sigma_{R,k}^2``; the variance components come from
    ``varcomp.fit`` (known within-variances). ``beta^2`` for ``rho`` is the
    bias-corrected moment estimate with model-based
    ``Var(mean Theta) = (tau^2 + sigma_bar^2) / K``.
    """
    k = len(theta_runs)
    n_res = max((np.asarray(t).size for t in theta_runs), default=0)

    Theta = np.array([energy(t) for t in theta_runs], float)
    sig2_R = np.array(
        [propagate_energy_sigma2(t, s) for t, s in zip(theta_runs, sigma2_runs)],
        float)

    fit = varcomp.fit(Theta, sig2_R, method=method, prior_scale=prior_scale)

    theta_bar = float(np.mean(Theta)) if k else 0.0
    # Bias correction (spec §2.4 step 9): remove the upward bias of mean(Theta)^2
    # using an *empirical* estimate of Var(mean Theta) = s^2(Theta)/K. This is the
    # unbiased estimator of Var(mean Theta) = (tau^2 + sigma_bar^2)/K and is kept
    # independent of the (small-K Bayesian-shrunk) tau^2, which is over-shrunk at
    # K=3; the shrunk tau^2 enters only the rho denominator (honest widening).
    if k >= 2:
        var_theta_bar = float(np.var(Theta, ddof=1) / k)
    else:
        var_theta_bar = float("nan")
    beta2_bc = (beta2_bias_corrected(theta_bar, var_theta_bar)
                if np.isfinite(var_theta_bar) else float("nan"))
    rho = (reproducibility_coefficient(beta2_bc, fit.tau2, fit.sigma2_bar)
           if np.isfinite(beta2_bc) else float("nan"))

    # informational signed/direction summary (not a gate)
    a_sgn_runs = np.array([signed_mean(t) for t in theta_runs], float)
    coh_runs = np.array([directional_coherence(t) for t in theta_runs], float)
    a_signed = float(np.mean(a_sgn_runs)) if k else 0.0
    coherence = float(np.mean(coh_runs)) if k else 0.0

    return RegionReproducibility(
        n_residues=int(n_res), n_replicates=int(k),
        beta=fit.beta, beta_se=fit.beta_se, tau2=fit.tau2,
        sigma2_bar=fit.sigma2_bar, beta2_bc=float(beta2_bc), rho=float(rho),
        a_signed=a_signed, coherence=coherence,
        method=fit.method, status=fit.status,
    )


# ── multi-scale driver over the M3 hierarchy ─────────────────────────────────
class _Res:
    """Minimal residue stand-in (``.resid``/``.resname``/``.icode``) so the M3
    hierarchy can be rebuilt from a per-run CSV without MDAnalysis."""

    __slots__ = ("resid", "resname", "icode")

    def __init__(self, resid: int, resname: str, icode: str = ""):
        self.resid = int(resid)
        self.resname = str(resname)
        self.icode = icode


def _recover_offset(df: pd.DataFrame) -> int:
    """offset = file_resid - canon_resid (constant per run)."""
    diff = (df["file_resid"].astype(int) - df["canon_resid"].astype(int))
    vals = pd.unique(diff)
    if len(vals) != 1:
        # non-constant offset: fall back to the modal value, do not crash
        return int(diff.mode().iloc[0])
    return int(vals[0])


def _scale_index(level_order: Sequence[str], level: str) -> int:
    """Spec convention: P_0 = residues (finest). level_order is coarse->fine, so
    the residue level gets index 0 and the coarsest level gets the largest index.
    """
    pos = list(level_order).index(level)
    return (len(level_order) - 1) - pos


def aggregate_reproducibility(
    per_run_dfs: Sequence[pd.DataFrame],
    hierarchy_config,
    *,
    theta_col: str = "r",
    sigma_col: str = "auto",
    levels: Optional[Iterable[str]] = None,
    method: str = "auto",
    prior_scale: Optional[float] = None,
    standardize: bool = True,
    protein: str = "",
) -> pd.DataFrame:
    """Compute ``rho`` for every region at every scale from K per-run tables.

    Parameters
    ----------
    per_run_dfs : sequence of DataFrames
        The K per-replicate ``{proj}_correlations_v5.csv`` tables (must contain
        ``file_resid``, ``canon_resid``, ``name``, the effect column and an
        uncertainty column).
    hierarchy_config : HierarchyConfig
        The M3 family config (used to reconstruct the nested partitions).
    theta_col : str
        Signed per-residue effect column (default ``r``).
    sigma_col : str
        Per-residue within-replicate SD column. ``"auto"`` prefers
        ``theta_bootstrap_se`` (M2, refined) and falls back to ``theta_se`` (M1).
    levels : iterable of str, optional
        Restrict to these hierarchy levels; default = all configured levels.

    Returns
    -------
    pandas.DataFrame
        One row per (scale, region) with variance components and ``rho``.
    """
    from ..hierarchy.hierarchy import build_hierarchy

    dfs = [d.reset_index(drop=True) for d in per_run_dfs]
    K = len(dfs)
    if K == 0:
        return pd.DataFrame()

    def _sigma_series(d: pd.DataFrame) -> np.ndarray:
        if sigma_col != "auto":
            return d[sigma_col].to_numpy(float)
        s = d["theta_bootstrap_se"].to_numpy(float) \
            if "theta_bootstrap_se" in d.columns else np.full(len(d), np.nan)
        if "theta_se" in d.columns:
            fb = d["theta_se"].to_numpy(float)
            s = np.where(np.isfinite(s), s, fb)
        return s

    # pooled standardization scale across all residues and replicates (I3)
    theta_lists = [d[theta_col].to_numpy(float) for d in dfs]
    scale = pooled_scale(theta_lists) if standardize else 1.0

    # per-run hierarchies + per-residue lookups keyed by canonical id
    run_hier, run_theta, run_sigma2, run_canon = [], [], [], []
    for d in dfs:
        offset = _recover_offset(d)
        residues = [_Res(int(r.file_resid), str(r.name_))
                    for r in d[["file_resid"]].assign(
                        name_=d["name"]).itertuples(index=False)]
        h = build_hierarchy(residues, hierarchy_config, offset=offset)
        run_hier.append(h)
        canon = d["canon_resid"].to_numpy(int)
        th = d[theta_col].to_numpy(float) / scale
        sg2 = (_sigma_series(d) / scale) ** 2
        run_theta.append(dict(zip(canon, th)))
        run_sigma2.append(dict(zip(canon, sg2)))
        run_canon.append(canon)

    level_order = list(run_hier[0].levels)
    use_levels = list(levels) if levels is not None else level_order

    # map each run's residue key -> canonical id (for region membership lookup)
    key_to_canon = []
    for d in dfs:
        key_to_canon.append(dict(zip(
            [f"{c}:{f}" for c, f in zip(d.get("chain", pd.Series(["?"] * len(d))),
                                        d["file_resid"].astype(int))],
            d["canon_resid"].astype(int))))

    rows = []
    for level in use_levels:
        ell = _scale_index(level_order, level)
        # union of region ids across runs (region paths are canonical-space stable)
        region_members: dict[tuple, list[list[int]]] = {}
        for k in range(K):
            for region_id, member_keys in run_hier[k].aggregate(level).items():
                canon_members = [key_to_canon[k].get(mk) for mk in member_keys]
                canon_members = [c for c in canon_members if c is not None]
                region_members.setdefault(region_id, [[] for _ in range(K)])
                region_members[region_id][k] = canon_members

        for region_id, per_run_members in region_members.items():
            theta_runs, sigma2_runs = [], []
            for k in range(K):
                cm = per_run_members[k]
                th = np.array([run_theta[k][c] for c in cm
                               if c in run_theta[k]], float)
                sg = np.array([run_sigma2[k][c] for c in cm
                               if c in run_sigma2[k]], float)
                theta_runs.append(th)
                sigma2_runs.append(sg)
            if all(t.size == 0 for t in theta_runs):
                continue
            res = region_reproducibility(
                theta_runs, sigma2_runs, method=method, prior_scale=prior_scale)
            rows.append(dict(
                protein=protein,
                scale_level=level,
                scale_index=ell,
                region_id="/".join(str(x) for x in region_id),
                label=str(region_id[-1]),
                n_residues=res.n_residues,
                K=res.n_replicates,
                beta=res.beta,
                beta_se=res.beta_se,
                tau2=res.tau2,
                sigma2_bar=res.sigma2_bar,
                beta2_bc=res.beta2_bc,
                rho=res.rho,
                a_signed=res.a_signed,
                coherence=res.coherence,
                method=res.method,
                status=res.status,
            ))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["scale_index", "region_id"]).reset_index(drop=True)
    return out


# ── output writers (varcomp.csv + rho_by_scale.csv) ──────────────────────────
_VARCOMP_COLS = ["protein", "scale_level", "scale_index", "region_id", "label",
                 "n_residues", "K", "beta", "beta_se", "tau2", "sigma2_bar",
                 "beta2_bc", "method", "status"]
_RHO_COLS = ["protein", "scale_level", "scale_index", "region_id", "label",
             "n_residues", "K", "rho", "a_signed", "coherence", "status"]


def write_reproducibility_tables(df: pd.DataFrame, out_dir: str,
                                 prefix: str = "") -> dict[str, str]:
    """Write ``varcomp.csv`` and ``rho_by_scale.csv`` from the aggregate table."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""
    vpath = os.path.join(out_dir, f"{tag}varcomp.csv")
    rpath = os.path.join(out_dir, f"{tag}rho_by_scale.csv")
    vc = df[[c for c in _VARCOMP_COLS if c in df.columns]] if not df.empty \
        else pd.DataFrame(columns=_VARCOMP_COLS)
    rh = df[[c for c in _RHO_COLS if c in df.columns]] if not df.empty \
        else pd.DataFrame(columns=_RHO_COLS)
    vc.to_csv(vpath, index=False)
    rh.to_csv(rpath, index=False)
    return {"varcomp": vpath, "rho_by_scale": rpath}
