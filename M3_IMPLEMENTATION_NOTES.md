# M3 implementation notes — biological hierarchy

Implements the nested partition data model of MATHEMATICAL_SPECIFICATION.md §1.1.
**No inference** — no variance components, no reproducibility coefficient, no
gate, no aggregation of statistics.

## Modules
- `config/hierarchy_schema.py` — `HierarchyConfig` (+ from_dict / YAML+JSON load).
- `hierarchy/residue.py` — `Residue` (chain, file id, canonical, insertion).
- `hierarchy/mapping.py` — `ResidueMapper` (numbering, insertion codes, missing,
  multi-chain, alignment hook).
- `hierarchy/secondary_structure.py` — pluggable `SSEAssigner` (Null / Mapping),
  DSSP never hard-coded.
- `hierarchy/domain.py` — `GroupResolver` (overlap = smallest-wins, coverage).
- `hierarchy/region.py` — `Region` (path-id, members, parent).
- `hierarchy/hierarchy.py` — `BiologicalHierarchy` (path-based nesting,
  `aggregate`/`region_of`/`regions`/`annotate`) + pipeline helper.
- `hierarchy/config_resolver.py` — selects a config by name (data, not biology).
- `configs/denv_hierarchy.yaml` — the DENV biology (only place it lives).

## Integration (additive, family-agnostic)
`_legacy.run_pipeline`: one call `attach_structural_metadata(df_res, all_res,
resolve_hierarchy_config(proj, HIERARCHY_CONFIG), offset=offset)` after the M2
call, wrapped in try/except so metadata can never break the pipeline. Appends
`chain, domain, motif, secondary_structure, region_id`. `Config` gains
`hierarchy_config` (optional path).

## Design choices
- **Path-based regions** guarantee nesting + partition with no extra bookkeeping
  and directly yield the spec's P_0 ≺ … ≺ P_L.
- **Canonical space**: chains/domains are resolved on canonical ids (file − offset,
  the pipeline's universal convention), so the same config works for any numbering.
- **Graceful by default**: unknown families → generic single-chain config; missing
  residues → `None`; uncovered → `unassigned`. The pipeline always runs.

## Explicitly deferred (NOT in M3)
variance components, hierarchical inference, reproducibility coefficient,
resolution gate, regional aggregation of statistics, calibration, coronavirus
comparison (the mapping infrastructure exists; comparison does not).
