"""M3 tests: biological hierarchy (structural groupings only)."""
import numpy as np
import pandas as pd
import pytest

from mechanism.config.hierarchy_schema import (
    HierarchyConfig, NumberingSpec, ChainSpec, GroupSpec, SecondaryStructureSpec,
)
from mechanism.hierarchy import (
    Residue, ResidueMapper, BiologicalHierarchy, build_hierarchy,
    attach_structural_metadata, NullAssigner, MappingAssigner, SSEType,
)
from mechanism.hierarchy.domain import GroupResolver


# ── ResidueMapper: numbering schemes ────────────────────────────────────────
def test_identity_numbering():
    m = ResidueMapper(NumberingSpec(scheme="identity"))
    assert m.to_canonical(51) == 51


def test_offset_numbering():
    m = ResidueMapper(NumberingSpec(scheme="offset", offset=47))
    assert m.to_canonical(98) == 51  # His51 at file 98


def test_explicit_numbering_and_missing():
    m = ResidueMapper(NumberingSpec(scheme="explicit", explicit_map={10: 1, 11: 2}))
    assert m.to_canonical(10) == 1
    assert m.to_canonical(999) is None  # missing -> None, never fabricated


def test_alternative_numbering_via_alignment_map():
    m = ResidueMapper(NumberingSpec(scheme="identity"))
    m.set_alignment_map("A", {200: 51, 201: 52})  # homolog with different numbering
    assert m.to_canonical(200, chain="A") == 51
    assert m.to_canonical(999, chain="A") is None


# ── insertion codes ─────────────────────────────────────────────────────────
def test_insertion_codes_distinct_keys():
    m = ResidueMapper(NumberingSpec(scheme="identity"))
    r1 = m.build_residue(100, "ALA", chain="A", insertion="")
    r2 = m.build_residue(100, "GLY", chain="A", insertion="A")  # 100A insertion
    assert r1.key != r2.key
    assert r1.key == "A:100" and r2.key == "A:100A"


# ── multiple chains ─────────────────────────────────────────────────────────
def test_multiple_chains_by_canonical_range():
    cfg = HierarchyConfig(
        name="x",
        chains=(ChainSpec("COF", (-999, 0)), ChainSpec("CAT", (1, 999))),
    )
    m = ResidueMapper(NumberingSpec(scheme="offset", offset=47), chains=cfg.chains)
    cof = m.build_residue(40, "GLY")   # canon -7 -> COF
    cat = m.build_residue(98, "HIS")   # canon 51 -> CAT
    assert cof.chain == "COF" and cat.chain == "CAT"


# ── GroupResolver: overlap + coverage ───────────────────────────────────────
def test_overlapping_domains_smallest_wins():
    specs = [
        GroupSpec("Big", tuple(range(1, 100)), order=0),
        GroupSpec("Small", (51, 52, 53), order=1),
    ]
    res = GroupResolver(specs)
    assert res.resolve(52) == "Small"   # most specific wins
    assert res.resolve(10) == "Big"
    assert res.resolve(500) is None     # not covered -> None


def test_overlap_tie_break_by_order():
    specs = [GroupSpec("First", (5, 6), order=0), GroupSpec("Second", (5, 6), order=1)]
    assert GroupResolver(specs).resolve(5) == "First"


# ── secondary structure (pluggable) ─────────────────────────────────────────
def test_null_assigner_all_unknown():
    rs = [Residue("A", 1, 1, "ALA"), Residue("A", 2, 2, "GLY")]
    a = NullAssigner().assign(rs)
    assert all(v == SSEType.UNKNOWN for v in a.values())


def test_mapping_assigner_one_letter():
    rs = [Residue("A", 1, 1, "ALA"), Residue("A", 2, 2, "GLY"), Residue("A", 3, 3, "VAL")]
    a = MappingAssigner({1: "H", 2: "E", 3: "C"}).assign(rs)
    assert a[rs[0].key] == SSEType.HELIX
    assert a[rs[1].key] == SSEType.SHEET
    assert a[rs[2].key] == SSEType.COIL


# ── hierarchy: nesting + partition invariants ───────────────────────────────
def _denv_like_cfg():
    return HierarchyConfig(
        name="t",
        chains=(ChainSpec("NS2B", (-999, 0)), ChainSpec("NS3", (1, 999))),
        domains=(
            GroupSpec("Triad", (51, 75, 135), chain="NS3", order=0),
            GroupSpec("Oxy", tuple(range(152, 160)), chain="NS3", order=1),
        ),
    )


class _MD:
    def __init__(self, resid, resname="ALA", icode=""):
        self.resid = resid; self.resname = resname; self.icode = icode


def test_nesting_and_partition():
    cfg = _denv_like_cfg()
    mds = [_MD(98, "HIS"), _MD(205, "SER"), _MD(40), _MD(0), _MD(-30)]  # offset 47
    h = build_hierarchy(mds, cfg, offset=47)
    # partition: each residue in exactly one region per level
    for lvl in h.levels:
        cov = sum(len(v) for v in h.aggregate(lvl).values())
        assert cov == len(h.residues), f"{lvl} not a partition"
    # nesting: coarser region is a prefix of finer region
    order = h.levels
    for r in h.residues:
        for i in range(len(order) - 1):
            coarse = h.region_of(r.key, order[i])
            fine = h.region_of(r.key, order[i + 1])
            assert fine[: len(coarse)] == coarse


def test_discontinuous_numbering_with_gaps():
    cfg = _denv_like_cfg()
    # gaps in residue numbering (missing residues) must not break the hierarchy
    mds = [_MD(98, "HIS"), _MD(122, "ASP"), _MD(150), _MD(300)]  # 300 -> canon 253 unassigned
    h = build_hierarchy(mds, cfg, offset=47)
    assert len(h.residues) == 4
    # the unmodelled-gap residue still gets a complete path
    last = [r for r in h.residues if r.file_resid == 300][0]
    assert h.annotate(last.key)["domain"] == "unassigned"


def test_aggregate_returns_groupings_not_stats():
    cfg = _denv_like_cfg()
    mds = [_MD(98, "HIS"), _MD(122, "ASP"), _MD(182, "SER"), _MD(40)]
    h = build_hierarchy(mds, cfg, offset=47)
    agg = h.aggregate("domain")
    # values are lists of residue keys (structural), never numbers
    for region_id, members in agg.items():
        assert isinstance(region_id, tuple)
        assert all(isinstance(k, str) for k in members)


def test_arbitrary_custom_level_supported():
    cfg = HierarchyConfig(
        name="t",
        levels=("complex", "protein", "chain", "subsystem", "domain",
                "secondary_structure", "residue"),  # 'subsystem' is custom
        chains=(ChainSpec("A", (-999, 999)),),
        domains=(GroupSpec("D", (1, 2, 3), order=0),),
    )
    h = build_hierarchy([_MD(1), _MD(2)], cfg, offset=0)
    assert "subsystem" in h.levels
    # custom level yields a valid partition
    cov = sum(len(v) for v in h.aggregate("subsystem").values())
    assert cov == 2


# ── pipeline helper: additive + preserves existing ──────────────────────────
def test_attach_appends_only_and_preserves_existing():
    cfg = _denv_like_cfg()
    mds = [_MD(98, "HIS"), _MD(205, "SER"), _MD(40), _MD(0)]
    df = pd.DataFrame([
        dict(file_resid=m.resid, canon_resid=m.resid - 47, name=m.resname,
             r=0.1 * i, abs_r=abs(0.1 * i), label=f"X{i}")
        for i, m in enumerate(mds)
    ])
    before = df.copy(deep=True)
    out = attach_structural_metadata(df, mds, cfg, offset=47)
    for col in before.columns:
        pd.testing.assert_series_equal(out[col], before[col])
    added = ["chain", "domain", "motif", "secondary_structure", "region_id"]
    assert list(out.columns) == list(before.columns) + added
    # His51 row annotated as Triad / NS3
    his = out[out["file_resid"] == 98].iloc[0]
    assert his["chain"] == "NS3" and his["domain"] == "Triad"


def test_generic_fallback_for_unknown_family():
    from mechanism.hierarchy.config_resolver import resolve_hierarchy_config
    cfg = resolve_hierarchy_config("SARS")  # no bundled config -> generic
    h = build_hierarchy([_MD(41, "HIS"), _MD(145, "CYS")], cfg, offset=0)
    # single generic chain, everything resolves, partition holds
    assert sum(len(v) for v in h.aggregate("chain").values()) == 2


# ── config loading (YAML/JSON) ──────────────────────────────────────────────
def test_resolve_bundled_denv_config():
    from mechanism.hierarchy.config_resolver import resolve_hierarchy_config
    cfg = resolve_hierarchy_config("DENV2")
    assert cfg.name == "DENV2"
    assert {c.name for c in cfg.chains} == {"NS2B", "NS3"}
    assert any(d.name == "Catalytic Triad" for d in cfg.domains)


def test_load_json_config(tmp_path):
    import json
    d = {"name": "Z",
         "chains": [{"name": "A", "canonical_range": [-9, 9]}],
         "domains": [{"name": "D", "residues": [{"range": [1, 3]}], "chain": "A"}],
         "numbering": {"scheme": "offset", "offset": 0}}
    p = tmp_path / "z.json"; p.write_text(json.dumps(d))
    cfg = HierarchyConfig.load(p)
    assert cfg.name == "Z" and cfg.domains[0].residues == (1, 2, 3)
