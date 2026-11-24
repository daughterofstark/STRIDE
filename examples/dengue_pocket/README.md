# DENV pocket-workflow configuration example

This directory documents the explicit input contract for the trajectory route.
It contains no trajectories, topology files, or frozen scientific outputs.

Provide one topology and one trajectory for each repeated run, then update the
paths in `stride.yaml` and point `povme_executable` at a verified POVME2.py
installation. Validate the configuration before execution, then create one
per-run item-effect table:

```bash
stride validate --config stride.yaml
stride plan --config stride.yaml --output workflow-plan
stride run --config stride.yaml --output run-effects
```

The plan writes hashes and runtime information only. The run command writes
`run-effects/runs/<run-id>/item_effects.csv`, with the per-residue correlation
between the POVME pocket-volume trace and the residue's C-alpha distance to the
declared reference selection. It also writes within-run uncertainty columns.

These per-run tables are not yet a cross-run STRIDE result. A later, explicit
data-contract step supplies them to replicate-aware aggregation. To create the
native Level-1 input layout for the separate `STRIDE_downstream` repository,
add a `downstream_export` block to `stride.yaml`, provide a complete coordinate
map, and run:

```bash
stride export-downstream --config stride.yaml \
  --run-effects run-effects --output downstream-input
python -m stride_analysis --data-root downstream-input \
  --output-dir canonical-output --no-require-summaries
```

The coordinate map is mandatory. It records the operational canonical key for
each residue and prevents the handoff from silently treating raw residue IDs as
cross-dataset alignment evidence. `coordinates.example.csv` shows the required
columns only; it is deliberately incomplete and cannot be used for a protein-
wide export.
