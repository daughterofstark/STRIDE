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
