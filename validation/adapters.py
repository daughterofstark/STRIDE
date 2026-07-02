"""The single validation->production bridge (milestone V1).

This is the **only** non-test module in ``validation/`` permitted to import
``mechanism``, and it imports **only** the public hierarchy schema
(``mechanism.config.hierarchy_schema``). It translates a pure
:class:`validation.generate.SyntheticSystemSpec` into a production
:class:`mechanism.config.hierarchy_schema.HierarchyConfig` so that generated
per-run frames round-trip through the frozen production estimator
(``aggregate_reproducibility`` / ``run_aggregation``).

Separation contract (V1, tightened from V0)
-------------------------------------------
* ``generate.py`` stays ``mechanism``-free; only this module bridges.
* This module touches **only** ``config.hierarchy_schema`` — never
  ``build_hierarchy``, never ``_legacy``, never any underscore-prefixed name.
  The estimator rebuilds the hierarchy internally from the frames; validation
  never calls ``build_hierarchy`` itself.
* ``mechanism`` never imports ``validation`` (unchanged from V0).

The static separation test (``validation/tests/test_separation.py``) enforces
that the set of validation modules importing ``mechanism`` is a subset of
``{adapters.py}`` and that only public names are used.
"""
from __future__ import annotations

# Public API only. No underscore-prefixed production internals.
from mechanism.config.hierarchy_schema import (
    HierarchyConfig, ChainSpec, GroupSpec, NumberingSpec,
)
# [V2] the production §2.1 stack (all public exports of mechanism.statistics):
#   pearson_both            -> the production correlation (theta = r)
#   effective_sample_size   -> N_eff / tau_int on the product series (M1)
#   corrected_standard_error-> Fisher sigma from N_eff (M1)
#   bootstrap_correlation   -> block-bootstrap CI / SE (M2)
from mechanism.statistics import (
    pearson_both,
    effective_sample_size,
    corrected_standard_error,
    bootstrap_correlation,
)

from .generate import SyntheticSystemSpec


def to_hierarchy_config(spec: SyntheticSystemSpec) -> HierarchyConfig:
    """Build a production :class:`HierarchyConfig` from a synthetic spec.

    Maps ``SynChain -> ChainSpec`` and ``SynDomain -> GroupSpec`` and sets the
    numbering to ``offset`` with the spec's offset (so canonical ids recover as
    ``file_resid - offset``, exactly as the estimator expects). Levels are carried
    through verbatim (coarse->fine), so the reconstructed nested partitions match
    the ground-truth scale indices.

    The generated frames already carry canonical ids and chain labels; this config
    only needs to reproduce the chain ranges, domain memberships, and level order
    the estimator uses to rebuild regions.
    """
    chains = tuple(
        ChainSpec(name=c.name,
                  canonical_range=(int(c.canonical_range[0]),
                                   int(c.canonical_range[1])))
        for c in spec.chains
    )
    domains = tuple(
        GroupSpec(name=d.name, residues=tuple(int(r) for r in d.residues),
                  chain=d.chain, order=i)
        for i, d in enumerate(spec.domains)
    )
    return HierarchyConfig(
        name=spec.name,
        levels=tuple(spec.levels),
        numbering=NumberingSpec(scheme="offset", offset=int(spec.offset)),
        chains=chains,
        domains=domains,
        motifs=(),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Tier-B (V2) production-recovery bridge
# ═════════════════════════════════════════════════════════════════════════════
# These functions are the single point where Tier-B series meet the production
# §2.1 stack. Keeping them here (not in the pure generate.py) preserves the
# separation boundary: only adapters.py imports mechanism. [CHOICE]

def tierb_hierarchy_config(spec) -> HierarchyConfig:
    """Build a :class:`HierarchyConfig` from a Tier-B system spec.

    Mirrors :func:`to_hierarchy_config` but reads a ``TierBSystemSpec`` (whose
    residues are :class:`~validation.generate.SeriesResidueSpec`).
    """
    chains = tuple(
        ChainSpec(name=c.name,
                  canonical_range=(int(c.canonical_range[0]),
                                   int(c.canonical_range[1])))
        for c in spec.chains
    )
    domains = tuple(
        GroupSpec(name=d.name, residues=tuple(int(r) for r in d.residues),
                  chain=d.chain, order=i)
        for i, d in enumerate(spec.domains)
    )
    return HierarchyConfig(
        name=spec.name,
        levels=tuple(spec.levels),
        numbering=NumberingSpec(scheme="offset", offset=int(spec.offset)),
        chains=chains,
        domains=domains,
        motifs=(),
    )


def recover_residue_effect(V, d, *, bootstrap: bool = False, B: int = 400,
                           seed: int = 0):
    """Recover one residue's effect + uncertainty through the production §2.1 stack.

    Runs the **production** functions, never a reimplementation:

    * ``theta = r`` via ``pearson_both`` (the production correlation);
    * ``tau_int`` / ``n_eff`` via ``effective_sample_size`` (M1, product series);
    * ``theta_se`` via ``corrected_standard_error`` (M1 Fisher/N_eff);
    * optionally a block-bootstrap SE/CI via ``bootstrap_correlation`` (M2).

    Returns a dict with keys ``r, tau_int, n_eff, neff_status, theta_se`` and, if
    ``bootstrap``, ``theta_bootstrap_se, boot_ci_lower, boot_ci_upper,
    bootstrap_method, bootstrap_block_length, bootstrap_replicates``.
    """
    r, _abs_r, _p, _pb, _pf, _sig = pearson_both(V, d, 1)
    neff = effective_sample_size(V, d)
    se = corrected_standard_error(r, neff.n_eff)
    out = dict(r=float(r), tau_int=float(neff.tau_int), n_eff=float(neff.n_eff),
               neff_status=neff.status, theta_se=float(se))
    if bootstrap:
        bs = bootstrap_correlation(V, d, neff.tau_int, neff.n_eff, B=B, seed=seed)
        out.update(theta_bootstrap_se=float(bs.se),
                   boot_ci_lower=float(bs.ci_lower),
                   boot_ci_upper=float(bs.ci_upper),
                   bootstrap_method=bs.method,
                   bootstrap_block_length=float(bs.block_length),
                   bootstrap_replicates=int(bs.n_rep))
    return out


def recover_frames_from_series(spec, replicates, *, bootstrap: bool = True,
                               B: int = 400, seed: int = 0):
    """Recover K production-schema per-run frames from Tier-B raw series.

    For each replicate and residue, recover ``theta``/``N_eff``/``sigma`` through
    the production stack, then assemble a production-schema frame using the V1
    :func:`validation.generate.build_per_run_frame` (reuse, not duplication).

    ``theta_se`` (M1 Fisher) is written to the ``theta_se`` column; the
    block-bootstrap SE (M2) is written to ``theta_bootstrap_se`` when
    ``bootstrap`` is set (else the Fisher SE is copied there, matching the
    production preference order used downstream). Returns
    ``(per_run_dfs, recovery_records)`` where ``recovery_records[k][canon]`` holds
    the full per-residue recovery dict (for validation/reporting).
    """
    import numpy as np
    from .generate import build_per_run_frame, SynChain

    def _chain_of(cid):
        for c in spec.chains:
            lo, hi = c.canonical_range
            if lo <= cid <= hi:
                return c.name
        return "?"

    canon_order = list(spec.canonical_ids)
    names = list(spec.resnames) if spec.resnames else ["ALA"] * len(canon_order)
    per_run = []
    records = []
    for k, rep in enumerate(replicates):
        theta = np.empty(len(canon_order))
        se_fisher = np.empty(len(canon_order))
        se_boot = np.empty(len(canon_order))
        rec = {}
        for j, cid in enumerate(canon_order):
            d = rep.d_by_canon[cid]
            r = recover_residue_effect(rep.V, d, bootstrap=bootstrap, B=B,
                                       seed=(seed * 100003 + k * 997 + j))
            theta[j] = r["r"]
            se_fisher[j] = r["theta_se"]
            se_boot[j] = r.get("theta_bootstrap_se", r["theta_se"])
            rec[cid] = r
        chains = [_chain_of(cid) for cid in canon_order]
        # build the base frame (writes sigma to both se columns), then overwrite
        # theta_bootstrap_se with the true M2 bootstrap SE where available.
        df = build_per_run_frame(canon_order, theta, se_fisher, chains,
                                 offset=spec.offset, names=names)
        boot_col = np.where(np.isfinite(se_boot), se_boot, se_fisher)
        df["theta_bootstrap_se"] = boot_col
        per_run.append(df)
        records.append(rec)
    return per_run, records
