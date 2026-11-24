# STRIDE

STRIDE is a downstream, deterministic, replicate-aware descriptive accounting
framework for scalar run-item effect fields with marginal uncertainty. It does
not generate trajectories or upstream effect fields.

The software reports several quantities that should be kept separate:

- support-specific magnitude stability (`rho`);
- operational sign coherence;
- fitted between-run and propagated within-run variance components;
- provisional support-crossing summaries;
- optional conditional selected-support signed associations.

The public interface can accept other scalar effect fields, but interface
generality is not evidence that statistical operating characteristics have been
validated for every observable, run count, dependence structure, or hierarchy.

## Inputs

The aggregation layer consumes one table per run containing an item identifier,
a scalar effect estimate, and a marginal standard error. Support membership is
provided through a hierarchy configuration. The framework treats these inputs
as supplied; upstream trajectory construction and uncertainty generation are
outside STRIDE.

## Trajectory workflow status

The repository also contains a public trajectory-to-effect workflow based on
the route used for the frozen dengue application. It prepares a protein
trajectory, obtains a POVME pocket-volume trace, calculates one
residue-to-reference correlation per run, and then passes the resulting tables
to the replicate-aware layer. A new run is a separately declared analysis; it
is not represented as a regeneration of the frozen dengue outputs.

The supported public workflow is being made explicit in stages. The `stride`
command validates a versioned trajectory-workflow configuration, writes a run
plan and provenance manifest, and can run the declared POVME-to-per-residue
effect step. In particular, users must provide a topology, a trajectory, a
pocket/reference definition, and a compatible `POVME2.py` executable.

```bash
stride validate --config examples/dengue_pocket/stride.yaml
stride plan --config examples/dengue_pocket/stride.yaml --output workflow-plan
stride run --config examples/dengue_pocket/stride.yaml --output run-effects
```

The configuration captures explicit file paths and analysis choices, including
the alignment reference, reference-centre and time-mode definition, POVME grid
and cutoff settings, bootstrap settings, and provisional aggregation settings.
POVME points are fixed from the first aligned frame. The default distance
reference is also fixed; the historical DENV compatibility mode records any
declared moving-reference decision and its observed drift. The plan records
SHA-256 hashes of those inputs and the available runtime/tool metadata.
`stride run` writes one `item_effects.csv` file per declared run. These tables
contain the pocket-volume/residue-distance correlation field and its
within-run uncertainty columns. They are inputs to the separate
replicate-aware aggregation stage; they are not a completed cross-run STRIDE
analysis.

For the companion `STRIDE_downstream` repository, `stride export-downstream`
converts these files to its native Level-1 layout:

```bash
stride export-downstream --config path/to/stride.yaml \
  --run-effects run-effects --output downstream-input
python -m stride_analysis --data-root downstream-input \
  --output-dir canonical-output --no-require-summaries
```

This handoff requires a complete, user-supplied coordinate map. The map defines
the operational canonical coordinate identity; the export command never treats
raw residue numbers as evidence of correspondence across systems.

`export-downstream` and `summarize` accept only an intact directory produced by
`stride run`. They verify the declared run identities, configuration hash, and
hashes of each effect and volume table before writing a downstream dataset. Run,
export, and summarize require a new output directory, so a failed run cannot be
mistaken for a completed result.

When the workflow configuration also supplies a hierarchy and complete
coordinate map, one command can create both the Level-1 run inputs and the
Level-2 provisional STRIDE summaries expected by `STRIDE_downstream`:

```bash
stride summarize --config path/to/stride.yaml \
  --run-effects run-effects --output stride-dataset
python -m stride_analysis --data-root stride-dataset \
  --output-dir canonical-output
```

The summary uses STRIDE's existing support aggregation and its provisional
rho/coherence settings. It does not calibrate a threshold or establish a
cross-system coordinate alignment.

The test suite uses a synthetic PDB trajectory and a POVME-format test stub to
verify the input/output contract. It does not certify a specific installed
POVME release or replace an archived real-trajectory integration fixture.
The committed fixture and its limitations are documented in
`examples/synthetic_pocket_fixture/`. The requirements for a real-POVME
release fixture and the opt-in integration check are in
[`docs/REAL_POVME_FIXTURE.md`](docs/REAL_POVME_FIXTURE.md).
The complete public input and output contract is in
[`docs/TRAJECTORY_WORKFLOW.md`](docs/TRAJECTORY_WORKFLOW.md).

## Magnitude aggregation and rho

For support `R` and run `k`, STRIDE standardizes item effects by one pooled field
scale and computes the L2 magnitude

```
Theta_Rk = sqrt(sum_i z_ik^2).
```

Under the implemented diagonal-covariance approximation, item variances are
propagated through this operator. The run-support magnitudes are fitted with a
random-effects model. At small `K` the implementation uses its deterministic
Bayesian half-Normal heterogeneity path; heterogeneity is weakly identified at
`K=3`.

The rho numerator is the empirical bias-corrected magnitude term

```
B_R = max(mean(Theta_R)^2 - var(Theta_R, ddof=1) / K, 0),
rho_R = B_R / (B_R + tau_R^2 + sigma2_bar_R).
```

The stored fitted `beta` is not the squared numerator `B_R`. Rho is clipped to
`[0, 1]` by the implementation. It is a support-specific magnitude-stability
ratio, not a truth probability, a fraction of biologically reproducible signal,
sign agreement, item-pattern agreement, or carrier-identity agreement.

Different support sizes define different L2 aggregate objects. A larger rho at
a larger support is not automatically stronger evidence. V9A characterizes this
operator behavior under registered frozen-design synthetic assumptions and
finds support-size association even under zero-mean inputs.

## Operational references and signed outputs

The frozen descriptive application uses `rho = 0.5` only as a provisional,
uncalibrated operational reference. First crossing is bookkeeping over a
configured traversal; it is not a calibrated scale decision or a license for a
claim at one resolution.

Signed fitting follows data-dependent provisional support selection and an
operational coherence rule. Wald intervals, p-values, and within-dataset BH
flags are conditional descriptive outputs. Post-selection FDR control has not
been established.

## Validation scope

The repository contains mathematical unit tests, component-level synthetic
checks, and V9A. V9A uses the production estimator to characterize support-size
and operator behavior under a fixed `K=3`, zero-covariance synthetic design. It
conditions on registered support-alias classes; it does not discover them,
calibrate `0.5`, model the upstream process, or validate inferential or
biological claims.

V9B lives in the STRIDE_downstream repository and verifies deterministic
presence-aware recurrence arithmetic. Neither V9A nor V9B establishes arbitrary
statistical generality, homology, mechanism, selected-interval coverage, or
post-selection FDR control.

Frozen expected-value checks are consistency guards. They are not independent
statistical validation.

## Legacy terminology warning

Historical public names remain for compatibility. In particular, names that
contain `reproducibility`, `mechanism`, `licensed`, `exploratory`,
`significant_fdr`, `serotype`, or `conservation` must be interpreted through the
current definitions above and the relevant schema documentation. They do not,
by name alone, establish biological mechanism, evidential licensing,
post-selection FDR control, homology, or biological conservation.

## Installation and tests

```
python -m pip install -e ".[dev]"
pytest tests validation/tests
```

The V9A focused suite includes production-source integrity checks and Git
provenance checks for full, shallow, detached, and no-Git source layouts. A
source export can run without `.git` only when the frozen production source
hashes match. An unrelated full Git history or a modified production source is
rejected.

The repository's CI runs linting, core tests, validation tests, V9A tests, and a
focused `execute()` provenance-path test. Golden tests are reported separately
and skip when their optional external fixtures are absent.

## Repository map

- `src/mechanism/`: production implementation.
- `validation/`: synthetic validation and the V9A characterization.
- `tests/`: production and integration tests.
- `validation/tests/`: validation tests.
- `HIERARCHY.md`: hierarchy configuration notes.
- `VALIDATION_AND_BENCHMARKING.md`: historical validation record; interpret it
  within the bounded scope stated above.
- `MIGRATION_NOTES.md`: compatibility notes.

## License

MIT. See `LICENSE`.
