# Regression report — M3 (biological hierarchy)

## Verdict
M3 adds five structural-metadata columns and **changes no existing output**.

## Existing outputs unchanged (source-level proof)
- Function audit `_legacy` vs `reference/v5_final_code_piece.py`: **0 v5
  functions differ** (the 6 M0-extracted ones separately proven identical); only
  new function is `run_pipeline`.
- M3 footprint in `_legacy`: **two imports + one global + one call**
  (lines 36–37, 52, 1853), inserted after the M2 block, before `to_csv`, wrapped
  in try/except. Effect loop, POVME, plotting, triplicate tail, and the M1/M2
  blocks are untouched.
- M1/M2 columns are read-only and not modified (asserted by the additive
  preservation tests in each milestone's suite).
- No plotting code touched → figures identical.

## Tests
`69 passed, 2 skipped` (skips = end-to-end golden; no data here).
- M0: 13. M1: 21. M2: 17. **M3: 18.**
- M3 coverage of required cases: identity/offset/explicit numbering, missing
  residues, insertion codes, multiple chains (by canonical range), overlapping
  domains (smallest-wins + order tie-break), discontinuous numbering, alternative
  numbering (alignment map), arbitrary custom level, nesting + partition
  invariants, additive-only pipeline helper, generic fallback, YAML/JSON loading.

## Coverage (code)
New hierarchy modules 84–100% (`hierarchy.py` 93%, `mapping.py` 89%,
`domain.py` 92%, `secondary_structure.py` 84%); config schema/resolver exercised
by loader tests. `_legacy.py` 7% (MD pipeline needs MDAnalysis+POVME+data;
covered by the golden suite on the data machine).

## Numerical differences in existing values
**None.** Only additive structural-metadata columns
(`chain, domain, motif, secondary_structure, region_id`). The golden-CSV subset
comparison (introduced in M1) already permits additive columns; no harness change
needed for M3.

## Outstanding (data machine)
End-to-end byte-identity of existing columns + figures on real trajectories via
the golden harness.
