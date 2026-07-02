"""Sweep orchestration, cell execution, and deterministic persistence (milestone V7).

The final **engineering** milestone: drive the sweep over the
``(system, K, T, tau2, beta2, seed)`` grid, execute each cell by *orchestrating* the
already-implemented V1-V6 components, and persist every result with full provenance.
**No new mathematics**; **no figures/report** (those are V8).

Design rules honored here
-------------------------
* **Reuse, never duplicate.** Cells call the V1/V2 generators (via
  ``validation.systems``), the V4 calibration *artifacts* (load-only), the V5 metrics
  (``operating_point``, hierarchy recovery), and the V6 baseline comparison. This
  module contains orchestration and persistence only.
* **Calibration is a separate, explicit step (V4).** A sweep **loads** calibrated
  ``rho*`` artifacts; if a required artifact is missing it **fails with a clear
  error** telling the user to calibrate first. A sweep never calibrates a new system
  on the fly, so a sweep is fully reproducible and V4/V7 stay cleanly separated.
  [CHOICE, per maintainer instruction]
* **Determinism.** Every cell derives its RNG from ``_seed``; the results store is a
  stable, ordered JSON-lines file; ``results_digest`` gives a reproducibility hash.
* **Separation.** Production is reached only through the ``adapters`` bridge, imported
  lazily, so ``import validation.experiments`` stays ``mechanism``-free.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Sequence

from ._seed import spawn_seeds
from .types import SweepCell
from .systems import SYSTEMS, get_system, non_denv_systems


def _validation_version() -> str:
    """Read the package version lazily (avoids an import cycle with __init__)."""
    from . import __version__
    return __version__


# default artifact directory (shipped calibration lives here)
_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class CalibrationMissingError(RuntimeError):
    """Raised when a sweep needs a calibrated rho* artifact that does not exist."""


def rho_star_artifact_path(system: str, K: int, *,
                           artifact_dir: str = _ARTIFACT_DIR) -> str:
    """Path convention for a system's calibrated rho* artifact at replicate count K.

    Uses the system's ``calibration_key`` when set (so the DENV system maps to the
    existing V4 ``rho_star_DENV_K*.yaml`` artifacts without renaming them); otherwise
    the system name is the stem.
    """
    key = system
    if system in SYSTEMS and SYSTEMS[system].calibration_key:
        key = SYSTEMS[system].calibration_key
    return os.path.join(artifact_dir, f"rho_star_{key}_K{K}.yaml")


def load_calibrated_rho_star(system: str, K: int, scale_level: str, *,
                             artifact_dir: str = _ARTIFACT_DIR) -> float:
    """Load a calibrated ``rho*`` for ``(system, K)`` at ``scale_level`` — never calibrate.

    [CHOICE] Orchestration is load-only. If the artifact is missing, raise
    :class:`CalibrationMissingError` with an actionable message. This keeps V4
    (calibration) and V7 (orchestration) cleanly separated and the sweep reproducible.
    """
    from .calibrate import load_rho_star_yaml  # local import: no mechanism here
    path = rho_star_artifact_path(system, K, artifact_dir=artifact_dir)
    if not os.path.exists(path):
        raise CalibrationMissingError(
            f"no calibration artifact for system={system!r}, K={K} at {path!r}. "
            f"Calibration is an explicit V4 step: run "
            f"`python -m validation calibrate --system {system} --K {K}` (or provide "
            f"the artifact) before sweeping. V7 does not calibrate inside a sweep.")
    cal = load_rho_star_yaml(path)
    if scale_level not in cal.rho_star:
        raise CalibrationMissingError(
            f"artifact {path!r} has no rho* for scale {scale_level!r}; "
            f"available: {sorted(cal.rho_star)}")
    return float(cal.rho_star[scale_level])


# ── sweep grid ───────────────────────────────────────────────────────────────
def sweep_grid(systems: Sequence[str], Ks: Sequence[int], Ts: Sequence[int],
               tau2s: Sequence[float], beta2s: Sequence[float], *,
               seed: int = 0, n_seeds: int = 1) -> list:
    """Deterministic ordered list of :class:`SweepCell` over the parameter grid.

    ``n_seeds`` independent per-cell seeds are spawned deterministically from
    ``seed`` (so a sweep can average over replicate ensembles). Ordering is stable
    (systems, then K, T, tau2, beta2, seed) for reproducible stores/digests.
    """
    cells = []
    base = spawn_seeds(seed, max(n_seeds, 1))
    for system in systems:
        get_system(system)  # validate name early
        for K in Ks:
            for T in Ts:
                for tau2 in tau2s:
                    for beta2 in beta2s:
                        for s in base:
                            cells.append(SweepCell(system=system, K=int(K),
                                                   T=int(T), tau2=float(tau2),
                                                   beta2=float(beta2), seed=int(s)))
    return cells


# ── per-cell record ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CellRecord:
    """One sweep cell's orchestrated result, with full provenance."""

    system: str
    K: int
    T: int
    tau2: float
    beta2: float
    seed: int
    scale_level: str                  # the system's true reproducible scale
    rho_star: float                   # loaded calibrated threshold (provenance)
    rho_true: float
    empirical_power: float
    predicted_power: float
    power_diff: float
    empirical_fpr: float
    predicted_fpr: float
    stride_over_resolution_rate: float
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CellRecord":
        return cls(
            system=d["system"], K=int(d["K"]), T=int(d["T"]),
            tau2=float(d["tau2"]), beta2=float(d["beta2"]), seed=int(d["seed"]),
            scale_level=d["scale_level"], rho_star=float(d["rho_star"]),
            rho_true=float(d["rho_true"]),
            empirical_power=float(d["empirical_power"]),
            predicted_power=float(d["predicted_power"]),
            power_diff=float(d["power_diff"]),
            empirical_fpr=float(d["empirical_fpr"]),
            predicted_fpr=float(d["predicted_fpr"]),
            stride_over_resolution_rate=float(d["stride_over_resolution_rate"]),
            provenance=dict(d.get("provenance", {})))


# ── cell execution (orchestration only) ──────────────────────────────────────
def run_cell(cell: SweepCell, *, n_eval: int = 60, sigma2_bar: float = 0.04,
             alpha: float = 0.05, artifact_dir: str = _ARTIFACT_DIR,
             calibration_B: Optional[int] = None) -> CellRecord:
    """Execute one sweep cell by orchestrating V1-V6; return a provenanced record.

    Loads the calibrated ``rho*`` (never calibrates), builds null and driver
    ensembles from the system factory, and computes the empirical-vs-predicted
    operating point (V5) and STRIDE's over-resolution rate. Deterministic in
    ``cell.seed``.
    """
    # lazy imports keep this module importable without mechanism on the path
    from .generate import generate_system
    from .adapters import to_hierarchy_config, gate_via_production
    from .metrics import operating_point

    sysdef = get_system(cell.system)
    scale = sysdef.true_scale_level
    rho_star = load_calibrated_rho_star(cell.system, cell.K, scale,
                                        artifact_dir=artifact_dir)

    cfg = to_hierarchy_config(
        sysdef.build(seed=cell.seed, K=cell.K, T=cell.T, tau2=cell.tau2,
                     beta2=cell.beta2, driver=False))

    def make_null(s):
        return list(generate_system(
            sysdef.build(seed=s, K=cell.K, T=cell.T, tau2=cell.tau2,
                         beta2=0.0, driver=False)).per_run_dfs)

    def make_driver(s):
        return list(generate_system(
            sysdef.build(seed=s, K=cell.K, T=cell.T, tau2=cell.tau2,
                         beta2=cell.beta2, driver=True)).per_run_dfs)

    # disjoint null/driver evaluation seed streams, derived from the cell seed
    seeds_null = spawn_seeds(cell.seed * 2 + 1, n_eval)
    seeds_driver = spawn_seeds(cell.seed * 2 + 2, n_eval)

    op = operating_point(
        make_null, make_driver, cfg, K=cell.K, T=cell.T, tau2=cell.tau2,
        beta2=cell.beta2, sigma2_bar=sigma2_bar, rho_star=rho_star,
        scale_level=scale, driver_label=sysdef.driver_label,
        seeds_null=seeds_null, seeds_driver=seeds_driver, alpha=alpha,
        protein=cell.system)

    # STRIDE over-resolution rate on planted nulls (reuses the gate bridge)
    order = {"residue": 0, "domain": 1, "chain": 2, "protein": 3, "complex": 4}
    true_idx = order.get(scale, 0)
    over = 0
    for s in seeds_null:
        mechs = gate_via_production(make_null(s), cfg, rho_star=rho_star,
                                    protein=cell.system)
        driver = [m for m in mechs
                  if sysdef.driver_region_substr in m["region_id"]]
        if driver and min(m["scale_index"] for m in driver) < true_idx:
            over += 1
    over_rate = float(over / len(seeds_null)) if seeds_null else float("nan")

    provenance = {
        "validation_version": _validation_version(),
        "n_eval": int(n_eval), "alpha": float(alpha),
        "sigma2_bar": float(sigma2_bar),
        "rho_star_source": os.path.basename(
            rho_star_artifact_path(cell.system, cell.K)),
        "true_scale_level": scale,
        "driver_label": sysdef.driver_label,
    }
    return CellRecord(
        system=cell.system, K=cell.K, T=cell.T, tau2=cell.tau2, beta2=cell.beta2,
        seed=cell.seed, scale_level=scale, rho_star=rho_star,
        rho_true=op.rho_true, empirical_power=op.empirical_power,
        predicted_power=op.predicted_power, power_diff=op.power_diff,
        empirical_fpr=op.empirical_fpr, predicted_fpr=op.predicted_fpr,
        stride_over_resolution_rate=over_rate, provenance=provenance)


def run_sweep(cells: Sequence[SweepCell], *, n_eval: int = 60,
              artifact_dir: str = _ARTIFACT_DIR, **cell_kw) -> list:
    """Execute every cell in order; return the list of :class:`CellRecord`."""
    return [run_cell(c, n_eval=n_eval, artifact_dir=artifact_dir, **cell_kw)
            for c in cells]


# ── hierarchy-sensitivity sweep (risk R6) ────────────────────────────────────
def hierarchy_sensitivity(systems: Sequence[str], *, K: int = 5, T: int = 0,
                          tau2: float = 0.0, beta2: float = 0.64, seed: int = 0,
                          n_eval: int = 60, artifact_dir: str = _ARTIFACT_DIR
                          ) -> list:
    """Sweep the *same* driver setting across systems with **distinct hierarchies**.

    Risk R6: how do the gate/operating characteristics depend on hierarchy structure?
    Returns one :class:`CellRecord` per system (at a fixed (K,T,tau2,beta2)), so the
    caller can compare behavior across topologies. Orchestration only; load-only
    calibration.
    """
    out = []
    for system in systems:
        cell = SweepCell(system=system, K=K, T=T, tau2=tau2, beta2=beta2,
                         seed=seed)
        out.append(run_cell(cell, n_eval=n_eval, artifact_dir=artifact_dir))
    return out


# ── deterministic results store (JSON lines + manifest) ──────────────────────
class ResultStore:
    """Append-ordered JSON-lines store of :class:`CellRecord` with a manifest.

    Deterministic: records are written in the order supplied and serialized with
    sorted keys, so two identical sweeps produce byte-identical stores and equal
    digests.
    """

    def __init__(self, path: str):
        self.path = path

    def write(self, records: Sequence[CellRecord], *, manifest: Optional[dict] = None
              ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")
        if manifest is not None:
            with open(self._manifest_path(), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, sort_keys=True, indent=2)

    def read(self) -> list:
        records = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(CellRecord.from_dict(json.loads(line)))
        return records

    def read_manifest(self) -> Optional[dict]:
        p = self._manifest_path()
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _manifest_path(self) -> str:
        root, _ = os.path.splitext(self.path)
        return root + "_manifest.json"


def results_digest(records: Sequence[CellRecord]) -> str:
    """Stable content hash of a record list (order-sensitive), for reproducibility."""
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r.to_dict(), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_manifest(cells: Sequence[SweepCell], *, seed: int, n_eval: int) -> dict:
    """Provenance manifest for a sweep (systems covered, grid size, versions)."""
    systems = sorted({c.system for c in cells})
    non_denv = [s for s in systems if not SYSTEMS[s].is_denv]
    return {
        "validation_version": _validation_version(),
        "n_cells": len(cells),
        "systems": systems,
        "n_non_denv_systems": len(non_denv),
        "non_denv_systems": non_denv,
        "seed": int(seed),
        "n_eval": int(n_eval),
    }
