"""V1 adapter tests: SyntheticSystemSpec -> HierarchyConfig correctness.

Verifies the single validation->production bridge builds a valid public
``HierarchyConfig`` whose chains, domains, levels, and numbering reproduce the
partitions the estimator will rebuild from the generated frames — and that the
residue->region mapping is correct through the estimator.
"""
import numpy as np
import pytest

from validation.generate import (
    SynChain, SynDomain, Driver, NullRegion,
    SyntheticSystemSpec, generate_system, region_path,
)
from validation.adapters import to_hierarchy_config

from mechanism.config.hierarchy_schema import HierarchyConfig, ChainSpec, GroupSpec
from mechanism.statistics.reproducibility import aggregate_reproducibility

_LEVELS = ("complex", "protein", "chain", "domain", "residue")


def _spec():
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    residues = triad + oxy
    return SyntheticSystemSpec(
        name="DENVX", levels=_LEVELS,
        chains=(SynChain("NS2B", (-999, 0)), SynChain("NS3", (1, 9999))),
        domains=(SynDomain("Triad", triad, "NS3"),
                 SynDomain("Oxy", oxy, "NS3")),
        residues=residues, offset=1000, seed=0)


def test_adapter_returns_public_hierarchy_config():
    cfg = to_hierarchy_config(_spec())
    assert isinstance(cfg, HierarchyConfig)


def test_adapter_maps_chains():
    cfg = to_hierarchy_config(_spec())
    names = {c.name: c.canonical_range for c in cfg.chains}
    assert names == {"NS2B": (-999, 0), "NS3": (1, 9999)}
    assert all(isinstance(c, ChainSpec) for c in cfg.chains)


def test_adapter_maps_domains_in_order():
    cfg = to_hierarchy_config(_spec())
    assert [g.name for g in cfg.domains] == ["Triad", "Oxy"]
    assert all(isinstance(g, GroupSpec) for g in cfg.domains)
    triad = [g for g in cfg.domains if g.name == "Triad"][0]
    assert triad.residues == (51, 75, 135)
    assert triad.chain == "NS3"
    # config order preserved for deterministic overlap tie-breaks
    assert cfg.domains[0].order == 0 and cfg.domains[1].order == 1


def test_adapter_sets_offset_numbering():
    cfg = to_hierarchy_config(_spec())
    assert cfg.numbering.scheme == "offset"
    assert cfg.numbering.offset == 1000


def test_adapter_carries_levels_verbatim():
    cfg = to_hierarchy_config(_spec())
    assert cfg.levels == _LEVELS


def test_adapter_maps_non_denv_system():
    # family-agnostic: a single-chain non-DENV system maps cleanly.
    spec = SyntheticSystemSpec(
        name="KINASE", levels=_LEVELS,
        chains=(SynChain("A", (1, 300)),),
        domains=(SynDomain("Activation Loop", tuple(range(150, 170)), "A"),),
        residues=tuple(range(140, 180)), offset=0, seed=0)
    cfg = to_hierarchy_config(spec)
    assert cfg.name == "KINASE"
    assert [c.name for c in cfg.chains] == ["A"]
    assert cfg.domains[0].name == "Activation Loop"


def test_residue_region_mapping_correct_through_estimator():
    # The estimator, using the adapter's config, must place each Triad residue in
    # the Triad domain region and each Oxy residue in the Oxy region.
    spec = _spec()
    g = generate_system(spec)
    cfg = to_hierarchy_config(spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    res = out[out.scale_level == "residue"]
    # every triad residue's region_id contains "/Triad/"; oxy contains "/Oxy/"
    triad_files = {51 + spec.offset, 75 + spec.offset, 135 + spec.offset}
    for row in res.itertuples(index=False):
        # region_id form: complex/protein/NS3/<Domain>/NS3:<file_resid>
        file_resid = int(row.region_id.split(":")[-1])
        if file_resid in triad_files:
            assert "/Triad/" in row.region_id, row.region_id
        else:
            assert "/Oxy/" in row.region_id, row.region_id


def test_offset_recovered_by_estimator():
    # generated frames carry file_resid = canonical + offset; the estimator
    # recovers the offset internally. A non-zero offset must still map correctly.
    spec = SyntheticSystemSpec(
        name="OFF", levels=_LEVELS,
        chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", (51, 75, 135), "NS3"),),
        residues=(51, 75, 135),
        drivers=(Driver(support=(51, 75, 135), scale_level="domain",
                        region_id=region_path(chain="NS3", domain="Triad"),
                        beta=1.0, tau2=0.0, carrier_mode="permute"),),
        K=3, sigma2=1e-8, offset=4242, seed=0, true_ell_star=1)
    g = generate_system(spec)
    # confirm the frames really carry the offset
    assert (g.per_run_dfs[0]["file_resid"] - g.per_run_dfs[0]["canon_resid"]
            == 4242).all()
    cfg = to_hierarchy_config(spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="OFF")
    triad = out[(out.scale_level == "domain") & (out.label == "Triad")]
    assert len(triad) == 1
    assert float(triad["rho"].iloc[0]) > 0.8
