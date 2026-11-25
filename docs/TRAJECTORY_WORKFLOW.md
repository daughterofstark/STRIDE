# Trajectory workflow contract

`stride run` converts declared trajectories into one per-run correlation field.
It is intended for a system with at least two repeated runs. The command does
not run molecular dynamics and does not alter STRIDE's replicate-aware
estimator.

## Required inputs

Each workflow YAML declares a topology and one trajectory or an ordered list of
trajectory segments for every run, a protein selection, an alignment selection,
and a POVME reference selection. It also declares the coordinate map and
hierarchy required for the optional downstream handoff. Use `trajectory` for
one file or `trajectories` for multiple files; never declare both for a run.

The current v1 file-residue contract requires globally unique residue IDs in
the selected protein. This is checked before effect fields are written. A
multi-chain topology whose chains reuse residue numbers must be renumbered or
given a later chain-aware input contract before it is used with this workflow.

## Explicit choices

The following values are recorded in `workflow_provenance.json` and control a
run:

- `preprocessing.alignment_reference`: `topology` or `trajectory_first_frame`.
  The topology option matches the historical DENV route. The first-frame option
  is available for a separately declared workflow.
- `preprocessing.alignment_mode`: `per_frame` (the default) applies the
  declared alignment to each selected frame. `use_supplied_fitted_trajectory`
  writes the selected coordinates as supplied and is reserved for a trajectory
  whose upstream fitting is itself part of the declared input contract. The
  DENV compatibility example uses the latter because the archived v5
  `aligned_clean.pdb` was verified to equal every twentieth frame of
  `36-md-fit.xtc` to PDB precision; it did not carry a second alignment.
- `pocket.reference.center_mode`: `atom_group_com` or `mean_residue_com`.
  It defines how the configured reference centre is measured in each frame.
- `pocket.reference.time_mode`: `fixed` (the default), `dynamic`, or
  `auto_drift`. POVME points are always fixed from the first aligned frame.
  `fixed` also uses that centre for every residue-distance trace. `dynamic` uses
  the per-frame centre for distance traces. `auto_drift` selects `dynamic` only
  when the maximum observed reference-centre drift exceeds the declared
  `drift_threshold_angstrom`. The requested mode, effective mode, and mean/max
  drift are recorded per run. The DENV example uses `auto_drift` solely to make
  the historical calculation explicit; it is not the default for new studies.
- `pocket.grid_spacing_angstrom`, `distance_cutoff_angstrom`, and
  `num_processors`: passed into the POVME input file.
- `uncertainty`: block-bootstrap method, replicate count, confidence level, and
  deterministic seed.
- `aggregation`: the existing provisional `rho_star`, Wald `alpha`, and
  coherence eligibility setting used only by `stride summarize`.

The configuration should identify a compatible Python `POVME2.py` script.
This v1 runner invokes it with the same Python interpreter that invoked
STRIDE. Its exact file hash is recorded in the provenance plan.

## Outputs and integrity checks

`stride run` writes a transactional output directory. It contains aligned
coordinates, the POVME input and points, POVME logs and volume table, one
`item_effects.csv` per run, `workflow_provenance.json`, and
`workflow_execution.json`. The execution manifest records hashes for the
configuration, provenance manifest, aligned coordinates, POVME inputs and
points, logs, volume tables, and item-effect tables.

`stride export-downstream` and `stride summarize` first check that the run
directory matches the declared configuration and that every recorded effect and
volume-table hash still matches. They refuse missing, tampered, or
configuration-mismatched run outputs. Their own output directories are written
transactionally.

If a long `stride run` is interrupted, retain its `.run-effects.tmp-*` staging
directory and use `stride resume --staging <directory>` with the same YAML and
a still-unused final output directory. Resume verifies the saved provenance and
configuration hash and all recorded source-file hashes, retains complete run
directories, and rebuilds any incomplete run from its declared inputs. Partial
run directories are preserved under `interrupted_runs/`. Failed executions
retain their staging directory for inspection and resume. Run inside a session manager such
as `tmux` for long calculations; do not manually copy partial run files.

## Scope

The synthetic fixture proves command, file-layout, provenance, and downstream
handoff behavior. It uses a POVME-format stub. It does not validate POVME, a
force field, an MD protocol, or the scientific operating characteristics of a
new system. A release-ready real-POVME fixture must meet the separate checklist
in [REAL_POVME_FIXTURE.md](REAL_POVME_FIXTURE.md).

## Historical DENV pilot

`examples/dengue_pocket/prepare_legacy_denv1_pilot.py` prepares a new DENV1
triplicate pilot from the archived v5 directory. It reads the three historical
correlation CSVs only to verify and recover their existing file-to-canonical
coordinate map; it does not reuse their effect values. The subsequent
`stride run` command recomputes the aligned coordinates, POVME volume traces,
and per-residue effects from `33-md.gro` and `36-md-fit.xtc` for each run.

The helper requires an explicit raw saved-frame interval because that provenance
must be declared rather than guessed. Its default stride of 20 matches the
historical 20,001-frame DENV trajectories and 1,001-row volume traces. It
writes a new YAML with the historical topology alignment, a fixed POVME sphere,
and the explicitly recorded `auto_drift` residue-distance compatibility mode.
