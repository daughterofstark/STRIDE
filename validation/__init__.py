"""STRIDE validation & benchmarking framework (Phase 2).

This package is **completely separate** from the production ``mechanism`` package:

* it lives outside the packaged ``src/`` tree, so it is never part of the installed
  ``mechanism`` distribution;
* it imports ``mechanism`` only through its documented public API, and only from a
  single bridge module (``validation.adapters``); every other non-test module
  (notably ``validation.generate`` and ``validation.processes``) is ``mechanism``-free;
* ``mechanism`` never imports this package.

Deleting this directory leaves the production framework (M0-M6) fully functional and
golden-green. See ``VALIDATION_ROADMAP.md`` for the milestone plan (V0-V8).

Milestones:
* V0 — isolation, determinism, data contract.
* V1 — Tier-A field-level generator + the public-API adapter.
* V8 — publication: figures, manuscript tables, and the VALIDATION_AND_BENCHMARKING.md
  report + reproducibility package, all read from the frozen V7 store (no new numbers).
* V7 — orchestration: sweep runner, deterministic persistence, and a reproducible
  CLI (`python -m validation run/calibrate/sweep`) over >=2 abstract non-DENV systems;
  calibration is load-only inside a sweep. No new mathematics.
* V6 — Part VI baselines + comparative statistical tests: the Part VII demonstration
  that STRIDE refuses over-resolution single-trajectory/naive practice emit.
* V5 — empirical operating characteristics + empirical-vs-predicted check (Part IV):
  measures FPR/power/coverage/hierarchy-recovery at the calibrated rho* and pairs
  them with V3's predicted curves; prediction and calibration are fixed references.
* V4 — empirical rho* calibration via null surrogates (Part V step 11), producing a
  provenanced rho_star artifact; generator beta=0 draws independently validate it.
* V3 — closed-form predicted operating characteristics (Part IV): the untuned
  faithfulness anchor for the later empirical-vs-predicted check (V5).
* V2 — Tier-B series-level generator (autocorrelated V(t)/d_i(t)) + the §2.1
  effective-N / sampling-noise chain validated end-to-end through the production
  M1/M2 stack, including misspecified-noise stress processes.

The ``__init__`` intentionally does **not** import ``validation.adapters`` at package
import time (that would pull ``mechanism`` into every ``import validation``); the
adapter is imported explicitly where needed. This keeps ``import validation``
production-free, as the V0 separation guarantee requires.
"""
from ._seed import make_rng, spawn_seeds
from .types import RegionTruth, GroundTruthSystem, SimResult, SweepCell
from .generate import (  # [V1] pure Tier-A generator (imports no mechanism)
    SynChain,
    SynDomain,
    Driver,
    NullRegion,
    SyntheticSystemSpec,
    GeneratedSystem,
    generate_system,
    build_per_run_frame,
    region_path,
    frames_digest,
    # [V2] pure Tier-B series orchestration (imports no mechanism)
    SeriesResidueSpec,
    TierBSystemSpec,
    TierBReplicateSeries,
    generate_series_replicates,
    series_digest,
)
from .processes import (  # [V2] pure autocorrelated-process generators
    ar1_tau_int,
    ou_phi,
    ar1_series,
    ar2_series,
    gaussian_innovations,
    student_t_innovations,
    SeriesPair,
    coupled_ar1_pair,
    coupled_ou_pair,
    coupled_ar2_pair,
    coupled_heavy_tailed_pair,
    coupled_slow_mixing_pair,
)
from .predicted import (  # [V3] pure closed-form predicted operating characteristics
    lambda_snr,
    rho_from_lambda,
    rho_from_params,
    lambda_star,
    n_eff_from_T,
    sigma2_bar_from_neff,
    predicted_fpr,
    predicted_power,
    predicted_coverage,
    ScalePrediction,
    ell_min,
    over_resolution_bound,
    predicted_reference_table,
)
from .surrogates import (  # [V4] pure null-surrogate generators (no mechanism)
    permute_replicate_labels,
    phase_randomize,
    phase_randomize_pairs,
    power_spectrum,
)
from .calibrate import (  # [V4] surrogate-based rho* calibration (via adapters bridge)
    CalibrationResult,
    upper_alpha_quantile,
    surrogate_null_rho,
    ensemble_surrogate_null_rho,
    calibrate_rho_star,
    generator_null_rho,
    empirical_fpr,
    write_rho_star_yaml,
    load_rho_star_yaml,
)
from .metrics import (  # [V5] empirical operating characteristics + emp-vs-predicted
    OperatingPoint,
    MetricsReport,
    empirical_crossing_rate,
    empirical_rho_recovery,
    empirical_coverage,
    empirical_hierarchy_recovery,
    empirical_over_resolution_rate,
    check_I2_upward_closed,
    check_I3_standardization_invariance,
    roc_auc,
    operating_point,
    ell_min_grid,
    write_metrics_report,
    load_metrics_report,
)
from .baselines import (  # [V6] Part VI baselines + comparison pipeline (pure)
    SingleTrajClaim,
    NaiveEnsembleClaim,
    single_trajectory_claim,
    single_trajectory_over_resolves,
    naive_ensemble_claim,
    naive_ensemble_over_resolves,
    naive_coverage,
    residue_ranking_claim,
    gtheory_coefficient,
    baseline_over_resolution_rates,
    build_method_comparison,
)
from .stats_tests import (  # [V6] paired comparative statistical tests (pure)
    McNemarResult,
    WilcoxonResult,
    DeLongResult,
    PairedBootstrapResult,
    BHResult,
    mcnemar_test,
    wilcoxon_signed_rank,
    delong_auc_test,
    paired_bootstrap_diff,
    benjamini_hochberg,
)
from .systems import (  # [V7] abstract topology-named synthetic systems
    SystemDef,
    SYSTEMS,
    get_system,
    non_denv_systems,
)
from .experiments import (  # [V7] sweep orchestration + persistence (load-only calib)
    CalibrationMissingError,
    CellRecord,
    ResultStore,
    sweep_grid,
    run_cell,
    run_sweep,
    hierarchy_sensitivity,
    load_calibrated_rho_star,
    rho_star_artifact_path,
    results_digest,
    build_manifest,
)
from . import tables  # [V8] manuscript tables (frozen-artifact readers)
from . import figures  # [V8] publication figures (frozen-artifact readers)
from . import report  # [V8] report assembler + reproducibility package

__version__ = "0.9.0+v8"

__all__ = [
    "make_rng",
    "spawn_seeds",
    "RegionTruth",
    "GroundTruthSystem",
    "SimResult",
    "SweepCell",
    # [V1]
    "SynChain",
    "SynDomain",
    "Driver",
    "NullRegion",
    "SyntheticSystemSpec",
    "GeneratedSystem",
    "generate_system",
    "build_per_run_frame",
    "region_path",
    "frames_digest",
    # [V2]
    "SeriesResidueSpec",
    "TierBSystemSpec",
    "TierBReplicateSeries",
    "generate_series_replicates",
    "series_digest",
    "ar1_tau_int",
    "ou_phi",
    "ar1_series",
    "ar2_series",
    "gaussian_innovations",
    "student_t_innovations",
    "SeriesPair",
    "coupled_ar1_pair",
    "coupled_ou_pair",
    "coupled_ar2_pair",
    "coupled_heavy_tailed_pair",
    "coupled_slow_mixing_pair",
    # [V3]
    "lambda_snr",
    "rho_from_lambda",
    "rho_from_params",
    "lambda_star",
    "n_eff_from_T",
    "sigma2_bar_from_neff",
    "predicted_fpr",
    "predicted_power",
    "predicted_coverage",
    "ScalePrediction",
    "ell_min",
    "over_resolution_bound",
    "predicted_reference_table",
    # [V4]
    "permute_replicate_labels",
    "phase_randomize",
    "phase_randomize_pairs",
    "power_spectrum",
    "CalibrationResult",
    "upper_alpha_quantile",
    "surrogate_null_rho",
    "ensemble_surrogate_null_rho",
    "calibrate_rho_star",
    "generator_null_rho",
    "empirical_fpr",
    "write_rho_star_yaml",
    "load_rho_star_yaml",
    # [V5]
    "OperatingPoint",
    "MetricsReport",
    "empirical_crossing_rate",
    "empirical_rho_recovery",
    "empirical_coverage",
    "empirical_hierarchy_recovery",
    "empirical_over_resolution_rate",
    "check_I2_upward_closed",
    "check_I3_standardization_invariance",
    "roc_auc",
    "operating_point",
    "ell_min_grid",
    "write_metrics_report",
    "load_metrics_report",
    # [V6]
    "SingleTrajClaim",
    "NaiveEnsembleClaim",
    "single_trajectory_claim",
    "single_trajectory_over_resolves",
    "naive_ensemble_claim",
    "naive_ensemble_over_resolves",
    "naive_coverage",
    "residue_ranking_claim",
    "gtheory_coefficient",
    "baseline_over_resolution_rates",
    "build_method_comparison",
    "McNemarResult",
    "WilcoxonResult",
    "DeLongResult",
    "PairedBootstrapResult",
    "BHResult",
    "mcnemar_test",
    "wilcoxon_signed_rank",
    "delong_auc_test",
    "paired_bootstrap_diff",
    "benjamini_hochberg",
    # [V7]
    "SystemDef",
    "SYSTEMS",
    "get_system",
    "non_denv_systems",
    "CalibrationMissingError",
    "CellRecord",
    "ResultStore",
    "sweep_grid",
    "run_cell",
    "run_sweep",
    "hierarchy_sensitivity",
    "load_calibrated_rho_star",
    "rho_star_artifact_path",
    "results_digest",
    "build_manifest",
    # [V8]
    "tables",
    "figures",
    "report",
    "__version__",
]
