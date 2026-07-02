"""Tier-A field-level synthetic ground-truth generator (milestone V1).

This module realizes the **field level** of the specification's random-effects
model (§2.2 of ``MATHEMATICAL_SPECIFICATION.md``) so that later validation
milestones have synthetic systems whose planted truth is known by construction.
It is **pure**: it imports **nothing** from ``mechanism`` (the only bridge to the
production package is :mod:`validation.adapters`). All randomness flows through
:mod:`validation._seed`.

What "Tier A" means (and does not)
----------------------------------
Tier A plants the per-replicate, per-residue **effect field** ``theta_i^(k)``
directly and assigns each residue a within-replicate sampling variance
``sigma^2`` **as a free parameter**. It does **not** synthesize autocorrelated
time series and derive ``sigma^2`` from ``(T, tau_int)`` via
``sigma^2 = (1 - theta^2)^2 / N_eff``; that series-level chain is Tier B (V2).
This split is sanctioned by ``MASTER_CONTEXT.md`` §8 (deviation 3) and is the
defining feature of the field-level tier. ``tau_int`` is still recorded on the
:class:`~validation.types.GroundTruthSystem` for provenance, but Tier A does not
sample from it. [CHOICE, flagged for approval — see the V1 design review]

Faithfulness to the *current* estimator (important)
---------------------------------------------------
The production estimator fits the §2.2 model on the **energy** aggregate
``A_en(R) = sqrt(sum_i theta_i^2)`` (see
``mechanism.statistics.reproducibility.region_reproducibility``), which is a
**folded, non-negative** quantity. Two consequences are treated as observed
properties of the *current implementation*, not as theorems and not as spec:

* **[KNOWN LIMITATION]** For a driver region the recovered ``beta_hat`` estimates
  ``E|beta + gamma|`` rather than ``beta`` itself; the two coincide only in the
  low-``tau`` / high-SNR corner. Exact recovery of the planted ``(beta, tau^2)``
  is therefore claimed by this generator only in that corner.
* **[KNOWN LIMITATION]** A pure-null region (``beta = 0``) of ``m >= 2`` residues
  with ordinary within-noise has ``E[A_en] = E sqrt(sum eps_i^2) > 0``, so the
  estimator reports a positive ``beta_hat`` and a non-trivial ``rho_hat`` that
  **grows with region size**. This generator does **not** hide or "correct" that:
  its tests assert the *achievable* property (a driver region separates from
  equal-size null regions) rather than the idealized "nulls read as zero".
  Calibrating ``rho*`` to control this is exactly the deferred V4 work.

The construction (spec §2.2 / Part III I1)
------------------------------------------
For a driver region ``R`` with support ``S`` and replicate ``k``:

    theta_i^(k) = (beta + gamma^(k)) * carrier_weight_i^(k) * sign_i
                  + eps_i^(k),
    gamma^(k) ~ N(0, tau^2),   eps_i^(k) ~ N(0, sigma_i^2),

where ``carrier_weight`` selects which residue(s) carry the effect in replicate
``k`` according to ``carrier_mode``:

* ``"permute"`` (default): one carrier drawn i.i.d.-uniformly from ``S`` each
  replicate — the location of the effect permutes within ``R`` across replicates.
  Under ``A_en`` the regional energy is ``|beta + gamma^(k)|`` every replicate
  (region reproducible) while no single residue is (Property I1). Because draws
  are i.i.d., a residue may be the carrier in more than one replicate; under the
  current (folded-energy) estimator this leaves residue-level ``rho_hat`` around
  ~0.4 rather than ~0, which is the honest realistic behavior. [KNOWN LIMITATION]
* ``"permute_disjoint"``: the carrier is assigned by a **seeded permutation** so
  that, when ``K <= |S|``, no residue is the carrier in more than one replicate
  (carriers are disjoint across replicates). This is the construction the
  production I1 unit test uses (``tests/test_reproducibility.py``); it drives
  residue-level ``rho_hat`` below the roadmap tolerance (< 0.2) while keeping the
  region reproducible. Use this mode for the strict I1 tolerance check.
* ``"fixed"``: the same first support residue carries the effect every replicate
  — the effect is residue-reproducible, so the true finest scale is the residue.
* ``"distributed"``: every support residue carries an equal share ``1/sqrt(|S|)``
  every replicate — a non-permuting region-level effect.

Null regions carry ``beta = 0`` (within-noise only). Multiple drivers and
explicit nulls are supported from the outset.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._seed import make_rng, spawn_seeds
from .types import RegionTruth, GroundTruthSystem

# ── recipe dataclasses (validation-side; distinct from the production schema) ──
# These duplicate a little of mechanism.config.hierarchy_schema (ChainSpec /
# GroupSpec) on purpose: generate.py must not import mechanism, so it cannot
# reference those types. The single translation happens in validation.adapters.
# [CHOICE] the duplication is forced by the separation rule, not incidental.


@dataclass(frozen=True)
class SynChain:
    """A chain: a name and an inclusive canonical residue range.

    Mirrors the information ``mechanism.config.hierarchy_schema.ChainSpec`` needs,
    without importing it (separation rule).
    """

    name: str
    canonical_range: tuple[int, int]


@dataclass(frozen=True)
class SynDomain:
    """A named residue group (domain) as canonical ids, optionally chain-scoped.

    Mirrors ``GroupSpec``. ``residues`` is an explicit tuple of canonical ids.
    """

    name: str
    residues: tuple[int, ...]
    chain: Optional[str] = None


@dataclass(frozen=True)
class Driver:
    """A planted reproducible mechanism at a chosen true scale.

    Attributes
    ----------
    support : tuple[int, ...]
        Canonical residue ids that may carry the effect (the region's carrier
        support). For a domain-scale driver this is the domain's residues; for a
        residue-scale driver a single id; for a chain-scale driver, residues
        spanning >1 domain of the chain (so the *finest reproducible* scale is
        the chain, not any single domain).
    scale_level : str
        The hierarchy level at which this driver is reproducible ("residue",
        "domain", "chain", "protein", "complex", ...). Used to compute the true
        ``ell*`` and to label the ground-truth region.
    region_id : str
        The production-style region id ("/"-joined path) at ``scale_level`` that
        this driver plants into (e.g. ``"NS2B-NS3/protease/NS3/Catalytic Triad"``).
        Provided by the adapter/spec builder so ground truth matches what the
        estimator will report.
    beta : float
        Reproducible population effect magnitude (>= 0 before sign).
    tau2 : float
        Between-replicate variance of the effect (``gamma ~ N(0, tau2)``).
    direction : str
        ``"increase"`` (sign +1) or ``"decrease"`` (sign -1) for a coherent
        driver; ``"mixed"`` to assign per-residue signs from ``sign_pattern``.
    carrier_mode : str
        ``"permute"`` (default) | ``"permute_disjoint"`` | ``"fixed"`` |
        ``"distributed"`` (see module doc).
    sign_pattern : tuple[int, ...], optional
        Per-support-residue signs (+1/-1); only used when ``direction == "mixed"``.
        Must match ``len(support)``.
    """

    support: tuple[int, ...]
    scale_level: str
    region_id: str
    beta: float
    tau2: float
    direction: str = "increase"
    carrier_mode: str = "permute"
    sign_pattern: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class NullRegion:
    """An explicit null region: residues carrying only within-noise (beta = 0)."""

    support: tuple[int, ...]
    scale_level: str
    region_id: str


@dataclass(frozen=True)
class SyntheticSystemSpec:
    """A complete recipe for one synthetic system (multi-driver, heterogeneous-ready).

    Attributes
    ----------
    name : str
        System name (also the ``proj`` used to build per-run frame file stems).
    levels : tuple[str, ...]
        Hierarchy level names, **coarse -> fine** (production convention). Must end
        in ``"residue"``.
    chains : tuple[SynChain, ...]
        Chains partitioning canonical id space.
    domains : tuple[SynDomain, ...]
        Named domains (the production ``domain`` level).
    residues : tuple[int, ...]
        The full canonical residue list of the system (every id that appears in a
        per-run frame). Order is preserved in the emitted frames.
    resnames : tuple[str, ...], optional
        Per-residue 3-letter names aligned to ``residues`` (default all "ALA").
    drivers : tuple[Driver, ...]
        Planted reproducible mechanisms (>= 0).
    nulls : tuple[NullRegion, ...]
        Explicit null regions (informational; any residue not in a driver support
        is implicitly null too).
    K, T : int
        Replicate count and per-replicate length (T is provenance in Tier A).
    tau_int : float
        Integrated autocorrelation time (provenance only in Tier A; sets N_eff in
        Tier B / V2).
    sigma2 : float
        Baseline within-replicate sampling variance applied to every residue.
    sigma2_overrides : dict[int, float]
        Per-canonical-id overrides of ``sigma2`` (heterogeneous-sigma support).
    offset : int
        Numbering offset: ``file_resid = canonical + offset``. Recovered downstream
        by the estimator as ``file_resid - canon_resid``.
    seed : int
        Master seed; all draws derive from it deterministically.
    true_ell_star : int
        The true finest reproducible scale index (residue = 0). Usually the min
        scale index over drivers; provided explicitly so ground truth is unambiguous.
    direction : str
        System-level direction summary for the ground-truth record.
    """

    name: str
    levels: tuple[str, ...]
    chains: tuple[SynChain, ...]
    domains: tuple[SynDomain, ...]
    residues: tuple[int, ...]
    drivers: tuple[Driver, ...] = ()
    nulls: tuple[NullRegion, ...] = ()
    resnames: tuple[str, ...] = ()
    K: int = 3
    T: int = 200
    tau_int: float = 5.0
    sigma2: float = 1e-4
    sigma2_overrides: dict = field(default_factory=dict)
    offset: int = 0
    seed: int = 0
    true_ell_star: int = 0
    direction: str = "increase"


@dataclass(frozen=True)
class GeneratedSystem:
    """The output of :func:`generate_system`.

    A frozen result object (rather than a bare tuple) so later milestones can add
    fields without breaking call sites.
    """

    truth: GroundTruthSystem
    per_run_dfs: tuple  # tuple[pandas.DataFrame, ...]  (one per replicate)
    spec: SyntheticSystemSpec


# ── production-schema per-run frame builder (factored for Tier-B / V2 reuse) ──
# Column set exactly matches what the production M4/M5 consumers read:
#   file_resid, canon_resid, name, chain, r, abs_r, theta_se, theta_bootstrap_se
# (verified against mechanism.statistics.reproducibility.aggregate_reproducibility
#  and mechanism.replicate.aggregator._per_run_lookup). [REPO]
_PROD_COLUMNS = ["file_resid", "canon_resid", "name", "chain",
                 "r", "abs_r", "theta_se", "theta_bootstrap_se"]


def build_per_run_frame(
    canon: Sequence[int],
    theta: Sequence[float],
    sigma: Sequence[float],
    chain: Sequence[str],
    *,
    offset: int,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Assemble one production-schema per-run correlation frame.

    Pure numpy/pandas — no ``mechanism`` import. Factored out so the Tier-B (V2)
    series generator, which produces ``theta``/``sigma`` from autocorrelated
    series through the production M1/M2 stack, can reuse the exact same frame
    layout. [CHOICE: forward-compatible factoring for V2]

    Parameters
    ----------
    canon : sequence of int
        Canonical residue ids.
    theta : sequence of float
        Per-residue effect ``r`` (= theta) for this replicate.
    sigma : sequence of float
        Per-residue within-replicate SD (sqrt of sigma^2); written to both
        ``theta_se`` and ``theta_bootstrap_se`` (the estimator prefers the
        bootstrap column and falls back to ``theta_se``).
    chain : sequence of str
        Per-residue chain label.
    offset : int
        ``file_resid = canonical + offset``.
    names : sequence of str, optional
        3-letter residue names (default "ALA").
    """
    canon = np.asarray(canon, dtype=int)
    theta = np.asarray(theta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = canon.size
    if not (theta.size == n and sigma.size == n and len(chain) == n):
        raise ValueError("canon, theta, sigma, chain must have equal length")
    if names is None:
        names = ["ALA"] * n
    if len(names) != n:
        raise ValueError("names length must match canon length")
    return pd.DataFrame({
        "file_resid": canon + int(offset),
        "canon_resid": canon,
        "name": list(names),
        "chain": list(chain),
        "r": theta,
        "abs_r": np.abs(theta),
        "theta_se": sigma,
        "theta_bootstrap_se": sigma,
    })[_PROD_COLUMNS]


# ── region-path helper (REPO convention, reproduced without importing mechanism) ─
# The production hierarchy builds region ids as "/"-joined path tuples
# (coarsest..level); e.g. a domain region is
#   "{complex_name}/{protein_name}/{chain}/{domain}"
# (verified: mechanism.hierarchy.hierarchy.BiologicalHierarchy._build_path and the
#  aggregate_reproducibility output). We reproduce that string here so ground-truth
# region ids match what the estimator emits, WITHOUT importing mechanism. [REPO]
def region_path(*, complex_name: str = "complex", protein_name: str = "protein",
                chain: Optional[str] = None, domain: Optional[str] = None,
                motif: Optional[str] = None) -> str:
    """Build a production-style region id path at the coarsest..given level.

    Pass only the components down to the desired level. Example (domain scale):
    ``region_path(chain="NS3", domain="Triad")`` ->
    ``"complex/protein/NS3/Triad"``.
    """
    parts = [complex_name, protein_name]
    if chain is not None:
        parts.append(chain)
    if domain is not None:
        parts.append(domain)
    if motif is not None:
        parts.append(motif)
    return "/".join(str(p) for p in parts)


# ── the generator ────────────────────────────────────────────────────────────
def _validate_spec(spec: SyntheticSystemSpec) -> None:
    """Reject ill-formed specs early (duplicate ids, bad support, sign mismatch)."""
    res = list(spec.residues)
    if len(res) != len(set(res)):
        dupes = sorted({r for r in res if res.count(r) > 1})
        raise ValueError(f"duplicate canonical residue ids in spec: {dupes}")
    if not spec.levels or spec.levels[-1] != "residue":
        raise ValueError("levels must be coarse->fine and end in 'residue'")
    if spec.resnames and len(spec.resnames) != len(res):
        raise ValueError("resnames length must match residues length")
    res_set = set(res)
    for d in spec.drivers:
        if not d.support:
            raise ValueError(f"driver {d.region_id!r} has empty support")
        missing = [r for r in d.support if r not in res_set]
        if missing:
            raise ValueError(
                f"driver {d.region_id!r} support has ids not in residues: {missing}")
        if len(set(d.support)) != len(d.support):
            raise ValueError(f"driver {d.region_id!r} support has duplicate ids")
        if d.direction == "mixed":
            if d.sign_pattern is None or len(d.sign_pattern) != len(d.support):
                raise ValueError(
                    f"driver {d.region_id!r} direction='mixed' requires a "
                    f"sign_pattern matching support length")
        if d.carrier_mode not in ("permute", "permute_disjoint", "fixed",
                                  "distributed"):
            raise ValueError(f"unknown carrier_mode {d.carrier_mode!r}")
    for nr in spec.nulls:
        missing = [r for r in nr.support if r not in res_set]
        if missing:
            raise ValueError(
                f"null region {nr.region_id!r} has ids not in residues: {missing}")


def _chain_of(canon_id: int, spec: SyntheticSystemSpec) -> str:
    for c in spec.chains:
        lo, hi = c.canonical_range
        if lo <= canon_id <= hi:
            return c.name
    return "?"


def _sigma2_for(canon_id: int, spec: SyntheticSystemSpec) -> float:
    return float(spec.sigma2_overrides.get(canon_id, spec.sigma2))


def _driver_signs(d: Driver) -> np.ndarray:
    """Per-support-residue sign vector for a driver."""
    if d.direction == "mixed":
        return np.asarray(d.sign_pattern, dtype=float)
    s = 1.0 if d.direction == "increase" else -1.0
    return np.full(len(d.support), s, dtype=float)


def _driver_carrier_plan(d: Driver, K: int, rng: np.random.Generator) -> np.ndarray:
    """Precompute the per-replicate carrier-weight matrix (K x |support|) for a driver.

    Returns an array ``W`` where ``W[k]`` is the carrier-weight vector for replicate
    ``k``. Modes:

    * permute          -> one i.i.d.-uniform carrier per replicate (rows independent).
    * permute_disjoint -> carriers assigned by a seeded permutation of the support so
                          that no residue is carrier twice while ``K <= |S|`` (rows
                          use distinct carriers); if ``K > |S|`` it wraps by reshuffling.
    * fixed            -> the first support residue every replicate.
    * distributed      -> equal 1/sqrt(m) on every support residue every replicate
                          (energy of the pure signal = |beta+gamma|).
    """
    m = len(d.support)
    W = np.zeros((K, m), dtype=float)
    if d.carrier_mode == "permute":
        for k in range(K):
            W[k, int(rng.integers(m))] = 1.0
    elif d.carrier_mode == "permute_disjoint":
        order = []
        while len(order) < K:
            perm = rng.permutation(m)
            order.extend(perm.tolist())
        for k in range(K):
            W[k, order[k]] = 1.0
    elif d.carrier_mode == "fixed":
        W[:, 0] = 1.0
    else:  # distributed
        W[:, :] = 1.0 / np.sqrt(m)
    return W


def generate_system(spec: SyntheticSystemSpec) -> GeneratedSystem:
    """Generate a synthetic system: ground truth + K production-schema frames.

    Deterministic in ``spec.seed``: two calls with the same spec produce
    byte-identical per-run DataFrames.

    RNG structure (kept explicit for reproducibility):
    * one seed stream per replicate (``rep_seeds``) drives the within-noise ``eps``
      and the per-driver between-replicate ``gamma`` for that replicate;
    * one seed stream per driver (``drv_seeds``) drives that driver's carrier plan
      (the K x |support| carrier-weight matrix), so carrier assignment is
      independent of the noise draws and stable under spec edits elsewhere.
    """
    _validate_spec(spec)

    res = np.asarray(spec.residues, dtype=int)
    n = res.size
    resnames = list(spec.resnames) if spec.resnames else ["ALA"] * n
    chains = [_chain_of(int(c), spec) for c in res]
    sigma2 = np.array([_sigma2_for(int(c), spec) for c in res], dtype=float)
    sigma = np.sqrt(sigma2)
    index_of = {int(c): i for i, c in enumerate(res)}

    # independent, reproducible seed streams: K replicates + one per driver
    rep_seeds = spawn_seeds(spec.seed, spec.K)
    # derive driver seeds from a distinct spawn so they don't collide with rep_seeds
    drv_seeds = spawn_seeds(spec.seed + 1, max(len(spec.drivers), 1))

    # carrier plans (K x |support|) per driver, drawn once
    carrier_plans = []
    for di, d in enumerate(spec.drivers):
        drng = make_rng(drv_seeds[di])
        carrier_plans.append(_driver_carrier_plan(d, spec.K, drng))

    per_run = []
    for k in range(spec.K):
        rng = make_rng(rep_seeds[k])
        theta = np.zeros(n, dtype=float)
        # within-replicate sampling noise on every residue
        theta += rng.normal(0.0, 1.0, n) * sigma
        # planted driver contributions
        for di, d in enumerate(spec.drivers):
            gamma = float(rng.normal(0.0, np.sqrt(d.tau2))) if d.tau2 > 0 else 0.0
            amp = d.beta + gamma
            signs = _driver_signs(d)
            weights = carrier_plans[di][k]
            for j, cid in enumerate(d.support):
                theta[index_of[cid]] += amp * weights[j] * signs[j]
        df = build_per_run_frame(
            res, theta, sigma, chains, offset=spec.offset, names=resnames)
        per_run.append(df)

    truth = _build_ground_truth(spec, sigma2, index_of)
    return GeneratedSystem(truth=truth, per_run_dfs=tuple(per_run), spec=spec)


def _build_ground_truth(spec: SyntheticSystemSpec, sigma2: np.ndarray,
                        index_of: dict) -> GroundTruthSystem:
    """Assemble the :class:`GroundTruthSystem` record from the planted parameters.

    ``sigma2_bar`` for a region is the mean of the planted per-residue ``sigma^2``
    over its support (the *planted* within-variance, on the native theta scale).
    Note this is the planted field-level ``sigma^2``; the estimator internally
    propagates it to the energy scale by the delta method, so the ``RegionTruth``
    ``rho`` is the *planted* coefficient, not necessarily the estimator's output.
    That distinction is the [KNOWN LIMITATION] documented at module top.
    """
    regions = []
    for d in spec.drivers:
        supp_sig2 = float(np.mean([sigma2[index_of[c]] for c in d.support]))
        regions.append(RegionTruth(
            region_id=d.region_id,
            scale_index=_scale_index(spec.levels, d.scale_level),
            beta=float(d.beta),
            tau2=float(d.tau2),
            sigma2_bar=supp_sig2,
            is_driver=True,
        ))
    for nr in spec.nulls:
        supp_sig2 = float(np.mean([sigma2[index_of[c]] for c in nr.support])) \
            if nr.support else 0.0
        regions.append(RegionTruth(
            region_id=nr.region_id,
            scale_index=_scale_index(spec.levels, nr.scale_level),
            beta=0.0, tau2=0.0, sigma2_bar=supp_sig2, is_driver=False,
        ))
    return GroundTruthSystem(
        name=spec.name,
        levels=tuple(spec.levels),
        regions=tuple(regions),
        true_ell_star=int(spec.true_ell_star),
        direction=spec.direction,
        K=int(spec.K), T=int(spec.T), tau_int=float(spec.tau_int),
        seed=int(spec.seed),
    )


def _scale_index(levels: Sequence[str], level: str) -> int:
    """Spec convention: residue = 0 (finest). ``levels`` is coarse->fine, so the
    residue level gets index 0 and the coarsest level the largest index. Matches
    ``mechanism.statistics.reproducibility._scale_index``. [REPO]"""
    levels = list(levels)
    if level not in levels:
        raise KeyError(f"level {level!r} not in {levels}")
    pos = levels.index(level)
    return (len(levels) - 1) - pos


# ── determinism aid: a stable hash of a system's per-run frames ──────────────
def frames_digest(gen: GeneratedSystem) -> str:
    """A stable content hash of the per-run frames (for golden determinism tests).

    Serializes each frame's columns in a fixed order to CSV bytes and hashes the
    concatenation. Independent of pandas' repr/formatting drift.
    """
    h = hashlib.sha256()
    for df in gen.per_run_dfs:
        ordered = df[_PROD_COLUMNS]
        h.update(ordered.to_csv(index=False, float_format="%.12g").encode("utf-8"))
    return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# Tier B (V2): series-level generation
# ═════════════════════════════════════════════════════════════════════════════
# Tier B synthesises the *time series* V(t) and d_i(t) (via validation.processes,
# pure) and recovers the effect field theta_i = r(V, d_i), the effective sample
# size, and the sampling variance THROUGH the production M1/M2 stack — validating
# the §2.1 chain end-to-end rather than planting theta directly (that is Tier A).
#
# The pure part lives here (the spec, the per-residue coupling recipe, and the
# raw-series orchestration). The single step that must call ``mechanism`` (running
# the series through the production correlation / effective-N / bootstrap) lives in
# ``validation.adapters`` — the one designated production bridge — so the
# separation boundary stays at {adapters.py}. [CHOICE: keep the mechanism-touching
# recovery in the bridge; keep series synthesis pure]

@dataclass(frozen=True)
class SeriesResidueSpec:
    """One residue's coupling recipe for a Tier-B system.

    Attributes
    ----------
    canonical : int
        Canonical residue id (maps to the hierarchy the same way Tier A does).
    target_r : float
        Planted Pearson correlation between this residue's ``d_i(t)`` and the
        shared reference ``V(t)`` (the mechanistic effect ``theta_i``).
    process : str
        ``"ar1"`` | ``"ou"`` | ``"ar2"`` | ``"heavy_tailed"`` | ``"slow_mixing"``.
    phi : float
        AR(1) parameter (raw-series autocorrelation) for ar1/heavy_tailed/slow_mixing.
    theta_ou, dt : float
        OU parameters (used when ``process == "ou"``).
    a1, a2 : float
        AR(2) coefficients (used when ``process == "ar2"``).
    df : float
        Student-t degrees of freedom (used when ``process == "heavy_tailed"``).
    """

    canonical: int
    target_r: float
    process: str = "ar1"
    phi: float = 0.7
    theta_ou: float = 0.2
    dt: float = 1.0
    a1: float = 0.6
    a2: float = 0.3
    df: float = 3.0


@dataclass(frozen=True)
class TierBSystemSpec:
    """Recipe for a Tier-B (series-level) synthetic system.

    Every residue gets its own ``d_i(t)`` coupled to a **shared** reference
    ``V(t)`` at the planted ``target_r``. ``V(t)`` is generated once per replicate
    with its own AR(1) autocorrelation; each ``d_i`` is coupled to it.
    """

    name: str
    levels: tuple[str, ...]
    chains: tuple[SynChain, ...]
    domains: tuple[SynDomain, ...]
    residues: tuple[SeriesResidueSpec, ...]
    K: int = 3
    T: int = 2000
    v_phi: float = 0.7           # autocorrelation of the shared reference V(t)
    offset: int = 0
    seed: int = 0
    resnames: tuple[str, ...] = ()

    @property
    def canonical_ids(self) -> tuple:
        return tuple(r.canonical for r in self.residues)


@dataclass(frozen=True)
class TierBReplicateSeries:
    """The raw series for one replicate of a Tier-B system (pre-recovery)."""

    V: object                    # numpy.ndarray, shape (T,)
    d_by_canon: dict             # canonical id -> numpy.ndarray, shape (T,)
    tau_int_v_analytic: float    # analytic raw-series tau_int of V (nan if N/A)


def generate_series_replicates(spec: TierBSystemSpec) -> list:
    """Generate the raw ``V(t)`` and ``d_i(t)`` series for every replicate (pure).

    Returns a list (length ``K``) of :class:`TierBReplicateSeries`. Deterministic
    in ``spec.seed``. Imports no ``mechanism``: this is pure series synthesis; the
    production recovery of ``theta`` / ``N_eff`` / ``sigma^2`` happens in the
    adapter.
    """
    from .processes import (
        ar1_series, ar1_tau_int, coupled_ar1_pair, coupled_ou_pair,
        coupled_ar2_pair, coupled_heavy_tailed_pair, coupled_slow_mixing_pair,
    )

    _validate_tierb_spec(spec)
    rep_seeds = spawn_seeds(spec.seed, spec.K)
    out = []
    for k in range(spec.K):
        rk = make_rng(rep_seeds[k])
        # per-replicate child seeds: one for V, one per residue
        child = spawn_seeds(int(rk.integers(0, 2 ** 31 - 1)),
                            1 + len(spec.residues))
        v_rng = make_rng(child[0])
        V = ar1_series(spec.T, spec.v_phi, v_rng)
        d_by_canon = {}
        for j, rspec in enumerate(spec.residues):
            prng = make_rng(child[1 + j])
            pair = _series_pair_for(rspec, spec.T, prng)
            # couple d_i to THIS replicate's V by regenerating the pair's coupling
            # against the shared V: use the pair's d directly but re-anchor so the
            # planted correlation is to the shared V(t). We instead build d_i as a
            # mixture of the shared V and idiosyncratic noise (see helper).
            d_by_canon[rspec.canonical] = _couple_to_reference(
                V, rspec, prng)
        tau_v = ar1_tau_int(spec.v_phi)
        out.append(TierBReplicateSeries(V=V, d_by_canon=d_by_canon,
                                        tau_int_v_analytic=tau_v))
    return out


def _series_pair_for(rspec: SeriesResidueSpec, T: int, rng):
    """Dispatch to the right process family (used for standalone pair fixtures)."""
    from .processes import (
        coupled_ar1_pair, coupled_ou_pair, coupled_ar2_pair,
        coupled_heavy_tailed_pair, coupled_slow_mixing_pair,
    )
    p = rspec.process
    if p == "ar1":
        return coupled_ar1_pair(T, rspec.target_r, rspec.phi, rng)
    if p == "ou":
        return coupled_ou_pair(T, rspec.target_r, rspec.theta_ou, rspec.dt, rng)
    if p == "ar2":
        return coupled_ar2_pair(T, rspec.target_r, rspec.a1, rspec.a2, rng)
    if p == "heavy_tailed":
        return coupled_heavy_tailed_pair(T, rspec.target_r, rspec.phi,
                                         rspec.df, rng)
    if p == "slow_mixing":
        return coupled_slow_mixing_pair(T, rspec.target_r, rng, phi=rspec.phi)
    raise ValueError(f"unknown process {p!r}")


def _couple_to_reference(V, rspec: SeriesResidueSpec, rng):
    """Build ``d_i(t)`` correlated with the shared reference ``V`` at ``target_r``.

    ``d_i = sign(r) * sqrt(|r|) * V_std + sqrt(1-|r|) * noise_i``, where ``V_std``
    is the standardized reference and ``noise_i`` is an autocorrelated series drawn
    from the residue's own process family (so d inherits realistic autocorrelation
    and, for the misspecified families, the intended tail/structure). The sample
    correlation of ``d_i`` with ``V`` concentrates at ``target_r``.
    """
    import numpy as np
    from .processes import (
        ar1_series, ar2_series, gaussian_innovations, student_t_innovations,
    )
    T = V.size
    Vc = V - V.mean()
    Vsd = Vc / (Vc.std() if Vc.std() > 0 else 1.0)
    p = rspec.process
    if p in ("ar1", "slow_mixing"):
        noise = ar1_series(T, rspec.phi, rng)
    elif p == "ou":
        from .processes import ou_phi
        noise = ar1_series(T, ou_phi(rspec.theta_ou, rspec.dt), rng)
    elif p == "ar2":
        noise = ar2_series(T, rspec.a1, rspec.a2, rng)
    elif p == "heavy_tailed":
        noise = ar1_series(T, rspec.phi, rng,
                           innovations=student_t_innovations(df=rspec.df))
    else:
        raise ValueError(f"unknown process {p!r}")
    nsd = noise.std()
    noise = noise / (nsd if nsd > 0 else 1.0)
    r = rspec.target_r
    # Bivariate construction: for standardized, independent V and noise,
    # d = r * Vstd + sqrt(1 - r^2) * noise has population corr(d, V) = r exactly.
    b = np.sqrt(max(1.0 - r * r, 0.0))
    return r * Vsd + b * noise


def _validate_tierb_spec(spec: TierBSystemSpec) -> None:
    ids = [r.canonical for r in spec.residues]
    if len(ids) != len(set(ids)):
        dupes = sorted({r for r in ids if ids.count(r) > 1})
        raise ValueError(f"duplicate canonical ids in Tier-B spec: {dupes}")
    if not spec.levels or spec.levels[-1] != "residue":
        raise ValueError("levels must be coarse->fine and end in 'residue'")
    if spec.resnames and len(spec.resnames) != len(spec.residues):
        raise ValueError("resnames length must match residues length")
    if spec.K < 1 or spec.T < 2:
        raise ValueError("need K >= 1 and T >= 2")


def series_digest(replicates) -> str:
    """Stable content hash of the raw Tier-B series (determinism tests)."""
    h = hashlib.sha256()
    for rep in replicates:
        h.update(np.asarray(rep.V, float).tobytes())
        for cid in sorted(rep.d_by_canon):
            h.update(str(cid).encode("utf-8"))
            h.update(np.asarray(rep.d_by_canon[cid], float).tobytes())
    return h.hexdigest()
