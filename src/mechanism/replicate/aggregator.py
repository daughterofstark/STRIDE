"""Resolution profile, resolution gate, and mechanism assembly (milestone M5).

Implements §2.5-2.6 and Part V (steps 12-15) of MATHEMATICAL_SPECIFICATION.md,
restricted to the M5 scope of IMPLEMENTATION_ROADMAP.md (the gate, the profile and
the mechanism report). It is a **consumer** of the M4 per-scale reproducibility
table (``statistics.reproducibility.aggregate_reproducibility``) and the M3
hierarchy; M0-M4 are not modified.

Estimands
---------
* **Resolution profile** (§2.6): for a locus ``i``, ``Pi_i : ell -> rho_hat(R_ell(i))``
  over scales ``ell = 0..L`` (``ell = 0`` is the residue, the finest scale). This is
  obtained by reading ``rho_hat`` of the region containing ``i`` at each scale.
* **Resolution gate** (§2.5, §1.3): ``ell*(i) = min{ ell : rho_hat(R_ell(i)) >= rho* }``
  (the first up-crossing from the finest scale), ``None`` if no scale qualifies.
* **Emitted mechanism** (§2.5, Part V step 14-15): at ``ell*`` the region, the signed
  direction and effect magnitude ``beta_hat +/- z * se(beta_hat)`` from the **signed**
  aggregate ``A_sgn`` (gated by a directional-homogeneity test, §2.4 / A4), with the
  full profile attached.

Deliberate scope boundaries
---------------------------
* **Calibration is separate.** ``rho*`` is a *provisional, configurable* constant
  (``GateConfig.rho_star``); the empirical calibration of Part IV (null-surrogate
  quantile) is deferred to the validation phase. Every emitted mechanism carries
  ``calibrated = False`` (roadmap risk R8). ``calibrate_rho_star`` is a marked seam
  that returns the provisional value unchanged; it computes no validated threshold.
* **Uncertainty** is the M4 variance-components posterior, used directly. No extra
  small-K widening heuristic is applied; the Bayesian posterior SD is already the
  honest (wider) interval at ``K < 5`` (Part V(a)), and ``gate_uncertain`` is
  surfaced as a flag only.

Implementation decisions beyond the spec are tagged ``[impl-decision]`` in the
docstrings of the relevant functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as _sps

from ..statistics import varcomp
from ..statistics.reproducibility import (
    signed_mean, directional_coherence, propagate_signed_sigma2,
    aggregate_reproducibility,
)
# reuse M4's per-run reconstruction helpers (M4 stays frozen)
from ..statistics.reproducibility import _Res, _recover_offset

_EPS = 1e-12


# ── configuration ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GateConfig:
    """Gate parameters. All thresholds are **provisional and uncalibrated**.

    Attributes
    ----------
    rho_star : float
        Provisional reproducibility threshold for the gate (§2.5). A configured
        constant; NOT the calibrated Part IV threshold. [impl-decision: value]
    alpha : float
        Two-sided level for the emitted CI ``beta_hat +/- z_{1-alpha/2} se`` (§2.5).
    coherence_threshold : float
        Minimum directional coherence ``|sum theta|/sum|theta|`` for a signed claim
        (§2.4 / A4). Below it the region is reported as ``"mixed"`` (energy only).
        [impl-decision: statistic and threshold are not specified by the spec]
    theta_col, sigma_col : str
        Effect and uncertainty columns in the per-run tables (passed to M4).
    """

    rho_star: float = 0.5
    alpha: float = 0.05
    coherence_threshold: float = 0.6
    theta_col: str = "r"
    sigma_col: str = "auto"
    calibrated: bool = False  # always False in M5; flips only after validation


def calibrate_rho_star(per_run_dfs, hierarchy_config, config: GateConfig) -> float:
    """Calibration seam (deferred to the validation phase).

    Part IV defines ``rho*`` as the upper-``alpha`` quantile of ``rho_hat`` under
    null surrogates (replicate-label permutation / phase-randomised series). That
    empirical calibration is intentionally **not** implemented here (roadmap: it is
    the methods contribution of the validation study). This function exists so the
    validation phase can fill it without changing the gate's call sites; for now it
    returns the provisional configured value unchanged.
    """
    return config.rho_star


# ── resolution profile (§2.6) ────────────────────────────────────────────────
def _is_ancestor(anc: str, desc: str) -> bool:
    """True if region id ``anc`` is ``desc`` or a path-prefix of it.

    Region ids are ``"/"``-joined nested path tuples (hierarchy guarantee), so a
    locus's region at every scale is exactly a prefix of its residue-level id.
    """
    return desc == anc or desc.startswith(anc + "/")


def build_profiles(rho_table: pd.DataFrame) -> pd.DataFrame:
    """Per-locus resolution profiles ``Pi_i`` from the M4 per-scale table (§2.6).

    For each residue-level region (a locus, ``scale_index == 0``), collect the
    ``rho_hat`` of every region on its path to the root, ordered by scale. Returns a
    tidy long-form frame with one row per (locus, scale).
    """
    if rho_table.empty:
        return pd.DataFrame(columns=[
            "protein", "locus", "canon_label", "scale_index", "scale_level",
            "region_id", "rho", "beta", "beta_se", "tau2", "sigma2_bar",
            "a_signed", "coherence", "method", "status"])

    loci = rho_table[rho_table["scale_index"] == 0]
    rows = []
    # index regions by id for prefix lookup
    by_region = {r.region_id: r for r in rho_table.itertuples(index=False)}
    for locus in loci.itertuples(index=False):
        for region_id, row in by_region.items():
            if _is_ancestor(region_id, locus.region_id):
                rows.append(dict(
                    protein=getattr(locus, "protein", ""),
                    locus=locus.region_id,
                    canon_label=locus.label,
                    scale_index=row.scale_index,
                    scale_level=row.scale_level,
                    region_id=row.region_id,
                    rho=row.rho,
                    beta=row.beta,
                    beta_se=row.beta_se,
                    tau2=row.tau2,
                    sigma2_bar=row.sigma2_bar,
                    a_signed=row.a_signed,
                    coherence=row.coherence,
                    method=row.method,
                    status=row.status,
                ))
    out = pd.DataFrame(rows)
    return out.sort_values(["locus", "scale_index"]).reset_index(drop=True)


# ── gate (§2.5, §1.3, Part V step 12-13) ─────────────────────────────────────
def gate_profile(profile_locus: pd.DataFrame, rho_star: float) -> Optional[pd.Series]:
    """First up-crossing of ``rho*`` by a single locus profile (§2.5).

    ``ell*(i) = min{ ell : rho_hat(R_ell(i)) >= rho* }`` scanning finest->coarsest
    (``scale_index`` ascending). Returns the gated profile row, or ``None`` when no
    scale qualifies (the "no reproducible mechanism" path, §1.2(iv)).
    """
    pl = profile_locus.sort_values("scale_index")
    for row in pl.itertuples(index=False):
        if np.isfinite(row.rho) and row.rho >= rho_star:
            return pd.Series(row._asdict())
    return None


# ── signed effect for a gated region (§2.5, A_sgn; A4 homogeneity) ────────────
def _per_run_lookup(per_run_dfs: Sequence[pd.DataFrame], hierarchy_config,
                    theta_col: str, sigma_col: str):
    """Build per-run (hierarchy, key->theta, key->sigma2, level->aggregate).

    Native (un-standardized) ``theta`` units are used for the *reported* signed
    effect magnitude, so ``beta_hat`` is interpretable on the original effect scale
    (standardisation is required only for the scale-invariant energy decision, I3).
    [impl-decision: report signed beta in native units, not standardised units]
    """
    from ..hierarchy.hierarchy import build_hierarchy

    runs = []
    for d in per_run_dfs:
        d = d.reset_index(drop=True)
        offset = _recover_offset(d)
        residues = [_Res(int(fid), str(nm))
                    for fid, nm in zip(d["file_resid"], d["name"])]
        h = build_hierarchy(residues, hierarchy_config, offset=offset)
        keys = [f"{c}:{int(f)}" for c, f in zip(
            d.get("chain", pd.Series(["?"] * len(d))), d["file_resid"])]
        theta = d[theta_col].to_numpy(float)
        if sigma_col == "auto":
            s = (d["theta_bootstrap_se"].to_numpy(float)
                 if "theta_bootstrap_se" in d.columns
                 else np.full(len(d), np.nan))
            if "theta_se" in d.columns:
                s = np.where(np.isfinite(s), s, d["theta_se"].to_numpy(float))
        else:
            s = d[sigma_col].to_numpy(float)
        key2theta = dict(zip(keys, theta))
        key2sig2 = dict(zip(keys, s ** 2))
        agg = {lvl: {"/".join(str(x) for x in rid): members
                     for rid, members in h.aggregate(lvl).items()}
               for lvl in h.levels}
        runs.append((key2theta, key2sig2, agg))
    return runs


def _signed_fit(runs, level: str, region_id: str, alpha: float):
    """Fit the signed regional effect ``A_sgn`` across replicates (§2.5 step 14-15).

    Returns ``(beta, se, ci_lo, ci_hi, coherence, method, status, n_rep)`` using the
    M4 variance-components fit on the per-run signed means with delta-method within-
    variances; ``se`` is the M4 posterior SD (no extra widening, per requirement).
    """
    y, v, pooled = [], [], []
    for key2theta, key2sig2, agg in runs:
        members = agg.get(level, {}).get(region_id, [])
        th = np.array([key2theta[m] for m in members if m in key2theta], float)
        s2 = np.array([key2sig2[m] for m in members if m in key2sig2], float)
        if th.size == 0:
            continue
        y.append(signed_mean(th))
        v.append(propagate_signed_sigma2(s2))
        pooled.append(th)
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    if y.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0.0, "undefined", "k_too_small", 0)
    fit = varcomp.fit(y, v)
    z = float(_sps.norm.ppf(1.0 - alpha / 2.0))
    lo = fit.beta - z * fit.beta_se
    hi = fit.beta + z * fit.beta_se
    # Directional coherence pooled over residues AND replicates: |sum theta| /
    # sum|theta| over the whole region across runs. This is 1 when every residue
    # in every replicate agrees in sign, and falls toward 0 under within-run
    # cancellation OR across-run sign flips -- exactly the two ways a signed
    # regional claim becomes unreproducible (A4). [impl-decision: statistic/threshold]
    coherence = directional_coherence(np.concatenate(pooled)) if pooled else 0.0
    return (float(fit.beta), float(fit.beta_se), float(lo), float(hi),
            float(coherence), fit.method, fit.status, int(fit.n_replicates))


# ── mechanism assembly (Part V step 14-15) ───────────────────────────────────
@dataclass
class Mechanism:
    """One emitted mechanism at the gated scale for a group of loci."""

    region_id: str
    label: str
    scale_level: str
    scale_index: int
    n_loci: int
    loci: list
    rho: float
    rho_star: float
    calibrated: bool
    direction: str                # "increase" | "decrease" | "mixed"
    beta_signed: Optional[float]
    beta_ci_lower: Optional[float]
    beta_ci_upper: Optional[float]
    beta_se: Optional[float]
    coherence: float
    reproducible_magnitude_energy: float   # the A_en-scale beta from M4
    method: str
    gate_uncertain: bool
    status: str


def _direction(beta_signed: float, coherence: float, threshold: float) -> str:
    if not np.isfinite(coherence) or coherence < threshold:
        return "mixed"
    if not np.isfinite(beta_signed) or abs(beta_signed) <= _EPS:
        return "mixed"
    return "increase" if beta_signed > 0 else "decrease"


def run_aggregation(
    per_run_dfs: Sequence[pd.DataFrame],
    hierarchy_config,
    config: GateConfig = None,
    *,
    protein: str = "",
):
    """End-to-end M5: profile -> gate -> mechanisms.

    Returns ``(profile_df, mechanisms, unresolved, meta)``:
    ``profile_df`` is the tidy per-locus profile with a ``gated`` flag;
    ``mechanisms`` is a list of :class:`Mechanism` (one per distinct gated region);
    ``unresolved`` is the list of loci with no qualifying scale (§1.2(iv));
    ``meta`` records the (provisional) gate settings and the uncalibrated flag.
    """
    config = config or GateConfig()
    rho_star = calibrate_rho_star(per_run_dfs, hierarchy_config, config)  # provisional

    rho_table = aggregate_reproducibility(
        per_run_dfs, hierarchy_config, theta_col=config.theta_col,
        sigma_col=config.sigma_col, protein=protein)
    profiles = build_profiles(rho_table)

    meta = dict(rho_star=rho_star, alpha=config.alpha,
                coherence_threshold=config.coherence_threshold,
                calibrated=False, n_loci=0, n_unresolved=0,
                n_mechanisms=0, n_gate_uncertain=0)
    if profiles.empty:
        return profiles, [], [], meta

    runs = _per_run_lookup(per_run_dfs, hierarchy_config,
                           config.theta_col, config.sigma_col)

    profiles["gated"] = False
    gated_rows = {}        # locus -> gated profile row (Series)
    unresolved = []
    for locus, pl in profiles.groupby("locus"):
        g = gate_profile(pl, rho_star)
        if g is None:
            unresolved.append(locus)
        else:
            gated_rows[locus] = g
            mask = (profiles["locus"] == locus) & \
                   (profiles["scale_index"] == g["scale_index"])
            profiles.loc[mask, "gated"] = True

    # group loci by their gated region and emit one mechanism each
    region_to_loci: dict[tuple, list] = {}
    for locus, g in gated_rows.items():
        key = (g["scale_level"], g["region_id"])
        region_to_loci.setdefault(key, []).append(locus)

    mechanisms = []
    for (level, region_id), loci in sorted(region_to_loci.items()):
        g = gated_rows[loci[0]]
        beta, se, lo, hi, coh, method, sstatus, _ = _signed_fit(
            runs, level, region_id, config.alpha)
        direction = _direction(beta, coh, config.coherence_threshold)
        energy_status = str(g["status"])
        gate_uncertain = (energy_status == "gate_uncertain"
                          or sstatus == "gate_uncertain")
        if direction == "mixed":
            beta_out = ci_lo = ci_hi = se_out = None
        else:
            beta_out, ci_lo, ci_hi, se_out = beta, lo, hi, se
        mechanisms.append(Mechanism(
            region_id=region_id,
            label=str(g["region_id"]).split("/")[-1],
            scale_level=level,
            scale_index=int(g["scale_index"]),
            n_loci=len(loci),
            loci=sorted(loci),
            rho=float(g["rho"]),
            rho_star=rho_star,
            calibrated=False,
            direction=direction,
            beta_signed=beta_out,
            beta_ci_lower=ci_lo,
            beta_ci_upper=ci_hi,
            beta_se=se_out,
            coherence=float(coh),
            reproducible_magnitude_energy=float(g["beta"]),
            method=method,
            gate_uncertain=bool(gate_uncertain),
            status=energy_status,
        ))

    meta.update(n_loci=int(profiles["locus"].nunique()),
                n_unresolved=len(unresolved),
                n_mechanisms=len(mechanisms),
                n_gate_uncertain=sum(m.gate_uncertain for m in mechanisms))
    return profiles, mechanisms, unresolved, meta
