"""Publication figures for the V8 layer (reads frozen artifacts only).

Every figure is drawn from a *persisted* V4/V5/V6/V7 artifact — **no recomputation, no
new numbers**. Uses a non-interactive Matplotlib backend so rendering is headless and
deterministic in content. Nothing here imports ``mechanism``.

Scope note (frozen-store limitation, honored intentionally)
-----------------------------------------------------------
Two figures named in the roadmap's wish-list cannot be drawn from the frozen store
without introducing new data, so they are handled as follows rather than by extending
any store (V8 must not add presentation-only data):

* **ROC/PR curves.** The frozen artifacts record over-resolution *rates* and paired
  test statistics, not the per-item score vectors a smooth ROC/PR curve needs. We
  therefore present the *stored* separation summary (STRIDE vs baseline over-resolution
  rates) as a bar figure and **document** that full ROC/PR curves are not reconstructible
  from the frozen store. See :func:`fig_over_resolution_comparison`.
* **Production Pi profiles.** Raw per-locus Pi profiles are not persisted (they are a
  production object, not in the results store). We draw a **profile schematic from
  stored per-scale values** (rho* by scale from the calibration artifacts) and label it
  as such; a live production Pi is out of scope for a frozen-store publication step.
  See :func:`fig_profile_schematic`.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless, deterministic-in-content
import matplotlib.pyplot as plt  # noqa: E402

from ._artifacts import (  # noqa: E402
    load_sweep_records, load_metrics, load_method_comparison, load_calibration,
    available_calibrations,
)
from .systems import SYSTEMS  # noqa: E402

_FIGSIZE = (7.0, 4.5)
_DPI = 120


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Figure: calibration curve (rho* by scale across K, per system) ───────────
def fig_calibration_curve(out_path: str, artifact_dir: Optional[str] = None) -> str:
    key_to_system = {(d.calibration_key or n): (n, d.true_scale_level)
                     for n, d in SYSTEMS.items()}
    by_system = {}
    for (key, K) in available_calibrations(artifact_dir):
        system, scale = key_to_system.get(key, (key, "domain"))
        cal = load_calibration(key, K, artifact_dir)
        by_system.setdefault(system, {})[K] = cal["rho_star"].get(scale)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for system in sorted(by_system):
        Ks = sorted(by_system[system])
        ax.plot(Ks, [by_system[system][k] for k in Ks], marker="o", label=system)
    ax.set_xlabel("replicate count K")
    ax.set_ylabel("calibrated rho* (true scale)")
    ax.set_title("Calibrated reproducibility threshold by system and K")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


# ── Figure: empirical vs predicted FPR & power (DENV) ────────────────────────
def fig_empirical_vs_predicted(out_path: str,
                               artifact_dir: Optional[str] = None) -> str:
    m = load_metrics(artifact_dir)
    ops = sorted(m["operating_points"], key=lambda o: (o["K"], o["beta2"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for K in sorted({o["K"] for o in ops}):
        sub = [o for o in ops if o["K"] == K]
        b2 = [o["beta2"] for o in sub]
        ax1.plot(b2, [o["empirical_power"] for o in sub], marker="o",
                 label=f"emp K={K}")
        ax1.plot(b2, [o["predicted_power"] for o in sub], marker="x",
                 linestyle="--", label=f"pred K={K}")
    ax1.set_xlabel("beta^2")
    ax1.set_ylabel("power")
    ax1.set_title("Empirical vs predicted power")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    # FPR panel: empirical vs alpha
    Ks = sorted({o["K"] for o in ops})
    emp_fpr = [max(o["empirical_fpr"] for o in ops if o["K"] == K) for K in Ks]
    ax2.bar([str(k) for k in Ks], emp_fpr, color="steelblue", label="emp FPR")
    ax2.axhline(0.05, color="crimson", linestyle="--", label="alpha=0.05")
    ax2.set_xlabel("K")
    ax2.set_ylabel("empirical FPR")
    ax2.set_title("FPR control at calibrated rho*")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_path)


# ── Figure: naive coverage by K (Part IV coverage bullet) ────────────────────
def fig_coverage(out_path: str, artifact_dir: Optional[str] = None) -> str:
    c = load_method_comparison(artifact_dir)
    cov = c["naive_coverage_by_K"]
    Ks = sorted(cov, key=int)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot([int(k) for k in Ks], [cov[k] for k in Ks], marker="o",
            color="darkorange", label="naive SD/sqrt(K)")
    ax.axhline(0.95, color="green", linestyle="--", label="nominal 0.95")
    ax.set_xlabel("replicate count K")
    ax.set_ylabel("interval coverage")
    ax.set_title("Naive ensemble interval coverage (anticonservative at small K)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


# ── Figure: ell_min heatmap (K x T) ──────────────────────────────────────────
def fig_ell_min_heatmap(out_path: str, artifact_dir: Optional[str] = None) -> str:
    m = load_metrics(artifact_dir)
    grid = m["ell_min_grid"]
    Ks = sorted({g["K"] for g in grid})
    Ts = sorted({g["T"] for g in grid})
    Z = [[next((g["ell_min"] for g in grid if g["K"] == K and g["T"] == T), None)
          for T in Ts] for K in Ks]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(Z, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(Ts)))
    ax.set_xticklabels(Ts)
    ax.set_yticks(range(len(Ks)))
    ax.set_yticklabels(Ks)
    ax.set_xlabel("trajectory length T")
    ax.set_ylabel("replicate count K")
    ax.set_title("Predicted ell_min (finest reproducible scale index)")
    fig.colorbar(im, ax=ax, label="ell_min")
    return _save(fig, out_path)


# ── Figure: over-resolution comparison (STRIDE vs baselines) ─────────────────
def fig_over_resolution_comparison(out_path: str,
                                   artifact_dir: Optional[str] = None) -> str:
    """Bar figure of over-resolution rates: STRIDE vs baselines, by K.

    [KNOWN LIMITATION] Full ROC/PR *curves* are not reconstructible from the frozen
    store (it records rates and paired statistics, not per-item scores). This figure
    presents the stored separation summary instead; the report documents the omission.
    """
    c = load_method_comparison(artifact_dir)
    Ks = sorted(c["comparisons_by_K"], key=int)
    methods = ["stride", "single_trajectory", "naive_ensemble"]
    series = {mth: [] for mth in methods}
    for K in Ks:
        cell = c["comparisons_by_K"][K]
        series["stride"].append(cell["stride_over_resolution_rate"])
        for mth in ("single_trajectory", "naive_ensemble"):
            series[mth].append(cell["comparisons"][mth]["baseline_rate"])
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = range(len(Ks))
    w = 0.25
    for i, mth in enumerate(methods):
        ax.bar([xi + (i - 1) * w for xi in x], series[mth], width=w, label=mth)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylabel("over-resolution rate on planted nulls")
    ax.set_title("STRIDE refuses over-resolution that baselines emit (Part VII)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_path)


# ── Figure: hierarchy-sensitivity panel (>=2 non-DENV systems) ───────────────
def fig_hierarchy_sensitivity(out_path: str,
                              artifact_dir: Optional[str] = None) -> str:
    recs = load_sweep_records(artifact_dir)
    systems = sorted({r["system"] for r in recs})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for system in systems:
        sub = sorted((r for r in recs if r["system"] == system),
                     key=lambda r: (r["K"], r["beta2"]))
        # power vs beta2 at the largest K present, as a representative slice
        Kmax = max(r["K"] for r in sub)
        s2 = [r for r in sub if r["K"] == Kmax]
        ax1.plot([r["beta2"] for r in s2], [r["empirical_power"] for r in s2],
                 marker="o", label=f"{system} (K={Kmax})")
    ax1.set_xlabel("beta^2")
    ax1.set_ylabel("empirical power")
    ax1.set_title("Power across hierarchy topologies")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    # over-resolution rate per system (max across cells)
    over = [max(r["stride_over_resolution_rate"] for r in recs
                if r["system"] == s) for s in systems]
    ax2.bar(range(len(systems)), over, color="slategray")
    ax2.set_xticks(range(len(systems)))
    ax2.set_xticklabels(systems, rotation=20, ha="right", fontsize=7)
    ax2.set_ylabel("max STRIDE over-resolution rate")
    ax2.set_title("Over-resolution across systems")
    ax2.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_path)


# ── Figure: profile schematic (from stored rho* by scale) ────────────────────
def fig_profile_schematic(out_path: str, system_key: str = "DENV", K: int = 5,
                          artifact_dir: Optional[str] = None) -> str:
    """A Pi-style profile schematic drawn from *stored* rho* by scale.

    [KNOWN LIMITATION] Raw production Pi profiles are not persisted in the frozen
    store, so this is a schematic built from the calibration artifact's rho* per scale
    (a stored value), not a live production profile. Documented as such in the report.
    """
    cal = load_calibration(system_key, K, artifact_dir)
    order = ["residue", "domain", "chain", "protein", "complex"]
    scales = [s for s in order if s in cal["rho_star"]]
    vals = [cal["rho_star"][s] for s in scales]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.step(range(len(scales)), vals, where="mid", marker="o",
            label="rho* by scale (stored)")
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales, rotation=20, ha="right")
    ax.set_ylabel("calibrated rho*")
    ax.set_title(f"Threshold profile schematic ({system_key}, K={K})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def build_all_figures(out_dir: str, artifact_dir: Optional[str] = None) -> list:
    """Render every publication figure into ``out_dir``; return the list of paths."""
    return [
        fig_calibration_curve(os.path.join(out_dir, "fig_calibration_curve.png"),
                              artifact_dir),
        fig_empirical_vs_predicted(
            os.path.join(out_dir, "fig_empirical_vs_predicted.png"), artifact_dir),
        fig_coverage(os.path.join(out_dir, "fig_coverage.png"), artifact_dir),
        fig_ell_min_heatmap(os.path.join(out_dir, "fig_ell_min_heatmap.png"),
                            artifact_dir),
        fig_over_resolution_comparison(
            os.path.join(out_dir, "fig_over_resolution_comparison.png"),
            artifact_dir),
        fig_hierarchy_sensitivity(
            os.path.join(out_dir, "fig_hierarchy_sensitivity.png"), artifact_dir),
        fig_profile_schematic(os.path.join(out_dir, "fig_profile_schematic.png"),
                              artifact_dir=artifact_dir),
    ]
