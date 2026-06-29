# Biological hierarchy (M3)

A protein-family-agnostic structural data model. **Structural groupings only —
no inference, no statistics, no reproducibility computation, no resolution gate.**
It provides the nested partitions that later inference milestones (M4+) will use.

## Levels (coarse → fine, arbitrary depth)
```
complex > protein > chain > domain > motif > secondary_structure > residue
```
Any subset can be configured, and custom level names are accepted (they receive a
deterministic placeholder so the partition stays valid). `atom` is reserved for
future use.

## Nesting & partition guarantee
Each residue is assigned a **path** (a tuple from the coarsest configured level to
the finest). A region at level ℓ is the path prefix up to ℓ. Therefore:
- every level is a **partition** (each residue lies in exactly one region), and
- levels are **nested** by construction (a coarser region is a prefix of a finer
  one).
This is exactly the nested partition family the mathematical specification (§1.1)
requires.

## Family-agnostic configuration
All biology lives in a YAML/JSON config, never in source. A config declares:
`numbering` (identity / offset / explicit), `chains` (by canonical range),
`domains` and `motifs` (named canonical residue sets, optionally per chain),
`catalytic_residues`, `secondary_structure` (method), and `aliases`. The bundled
`configs/denv_hierarchy.yaml` is the DENV1-4 implementation; a new family needs
only its own config file. Unknown families fall back to a generic single-chain
config, so the pipeline never breaks.

## Residue mapping
`ResidueMapper` resolves file→canonical numbering and supports: insertion codes
(distinct keys), missing residues (returns `None`, never fabricated), alternative
numbering (explicit maps), multiple chains (by canonical range), and an
alignment-map hook (`set_alignment_map`) reserved for future cross-family
comparison. No comparison is performed in M3.

## Secondary structure (pluggable, DSSP not hard-coded)
An `SSEAssigner` interface with interchangeable backends: `NullAssigner`
(default; everything `unknown`, no dependency) and `MappingAssigner` (from a
precomputed `canonical → type` mapping, e.g. DSSP/STRIDE output supplied in the
config). An external DSSP backend can be registered later without changing
callers.

## Aggregation API (groupings only)
```python
h = build_hierarchy(residues, config, offset=offset)
h.aggregate(level="domain")   # {region_path: [residue_key, ...]}  (a partition)
h.aggregate(level="chain")
h.region_of(residue_key, "domain")   # the residue's region path at that level
h.regions("domain")                  # Region objects (id, label, members, parent)
h.annotate(residue_key)              # {chain, domain, motif, secondary_structure, region_id}
```
These return **structural groupings only**; they never compute statistics.

## Pipeline integration
`attach_structural_metadata(df_res, residues, config, offset=offset)` appends
`chain, domain, motif, secondary_structure, region_id` to the per-residue table.
Read-only: no existing column or effect/uncertainty value is touched.

## Overlap & coverage rules
Overlapping domains: the **most specific** (smallest) match wins; ties broken by
config order. Uncovered residues: `unassigned` placeholder (keeps the partition
complete).
