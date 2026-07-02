"""V2 Tier-B integration tests: series-level systems recovered through the
production §2.1 stack, then aggregated through the frozen M4/M5 estimator.

These validate the *end-to-end chain*: synthetic V(t)/d_i(t) -> production
correlation (theta) -> production N_eff/tau_int -> Fisher SE -> block bootstrap ->
production-schema frames -> M4 aggregation. Everything numeric comes from the
production functions; the validation code only synthesises series and assembles
the frames (reusing the V1 build_per_run_frame).
"""
import numpy as np
import pandas as pd
import pytest

from validation.generate import (
    SynChain, SynDomain, SeriesResidueSpec, TierBSystemSpec,
    generate_series_replicates, series_digest, region_path,
)
from validation.adapters import (
    tierb_hierarchy_config, recover_frames_from_series, recover_residue_effect,
)
from validation._seed import make_rng

from mechanism.statistics import effective_sample_size, pearson_both
from mechanism.statistics.reproducibility import aggregate_reproducibility
from mechanism.replicate.aggregator import run_aggregation, GateConfig

_LEVELS = ("complex", "protein", "chain", "domain", "residue")
_PROD_COLUMNS = ["file_resid", "canon_resid", "name", "chain",
                 "r", "abs_r", "theta_se", "theta_bootstrap_se"]


def _denv_b_spec(*, seed=0, K=3, T=3000, driver_r=0.6, null_r=0.05,
                 process="ar1", phi=0.7, offset=1000):
    triad = (51, 75, 135)
    oxy = (152, 153, 154)
    residues = tuple(
        SeriesResidueSpec(c, target_r=driver_r, process=process, phi=phi)
        for c in triad
    ) + tuple(
        SeriesResidueSpec(c, target_r=null_r, process=process, phi=phi)
        for c in oxy
    )
    return TierBSystemSpec(
        name="DENVB", levels=_LEVELS,
        chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"),
                 SynDomain("Oxy", oxy, "NS3")),
        residues=residues, K=K, T=T, v_phi=phi, offset=offset, seed=seed)


def _rho_of(out, level, label):
    sub = out[(out.scale_level == level) & (out.label == label)]
    return float(sub["rho"].iloc[0]) if len(sub) else float("nan")


# ── determinism ──────────────────────────────────────────────────────────────
def test_series_generation_deterministic():
    a = generate_series_replicates(_denv_b_spec(seed=5))
    b = generate_series_replicates(_denv_b_spec(seed=5))
    assert series_digest(a) == series_digest(b)


def test_series_generation_seed_sensitive():
    a = generate_series_replicates(_denv_b_spec(seed=1))
    b = generate_series_replicates(_denv_b_spec(seed=2))
    assert series_digest(a) != series_digest(b)


def test_recovered_frames_deterministic():
    spec = _denv_b_spec(seed=0)
    reps = generate_series_replicates(spec)
    f1, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200, seed=1)
    f2, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200, seed=1)
    for a, b in zip(f1, f2):
        pd.testing.assert_frame_equal(a, b)


# ── schema & shape ───────────────────────────────────────────────────────────
def test_recovered_frames_have_production_schema():
    spec = _denv_b_spec()
    reps = generate_series_replicates(spec)
    frames, records = recover_frames_from_series(spec, reps, bootstrap=True, B=200)
    assert len(frames) == spec.K
    for df in frames:
        assert list(df.columns) == _PROD_COLUMNS
        assert len(df) == len(spec.residues)
        assert (df["file_resid"] - df["canon_resid"] == spec.offset).all()
        assert np.allclose(df["abs_r"], np.abs(df["r"]))
        # uncertainty columns are finite and positive
        assert np.all(np.isfinite(df["theta_se"])) and np.all(df["theta_se"] > 0)
        assert np.all(np.isfinite(df["theta_bootstrap_se"]))


def test_series_length_matches_T():
    spec = _denv_b_spec(T=1500)
    reps = generate_series_replicates(spec)
    assert reps[0].V.size == 1500
    for cid, d in reps[0].d_by_canon.items():
        assert d.size == 1500


# ── the §2.1 recovery is done by PRODUCTION functions ────────────────────────
def test_recovered_theta_matches_production_correlation():
    # the r written to the frame must equal what pearson_both returns on the series
    spec = _denv_b_spec(seed=3)
    reps = generate_series_replicates(spec)
    frames, records = recover_frames_from_series(spec, reps, bootstrap=False)
    for k, rep in enumerate(reps):
        for cid in spec.canonical_ids:
            expected, *_ = pearson_both(rep.V, rep.d_by_canon[cid], 1)
            got = records[k][cid]["r"]
            assert got == pytest.approx(expected, abs=1e-9)


def test_recovered_neff_matches_production():
    spec = _denv_b_spec(seed=4)
    reps = generate_series_replicates(spec)
    _, records = recover_frames_from_series(spec, reps, bootstrap=False)
    for k, rep in enumerate(reps):
        for cid in spec.canonical_ids:
            exp = effective_sample_size(rep.V, rep.d_by_canon[cid])
            assert records[k][cid]["n_eff"] == pytest.approx(exp.n_eff, abs=1e-6)
            assert records[k][cid]["tau_int"] == pytest.approx(exp.tau_int, abs=1e-6)


def test_driver_residues_recover_planted_r():
    spec = _denv_b_spec(seed=0, T=5000, driver_r=0.6)
    reps = generate_series_replicates(spec)
    _, records = recover_frames_from_series(spec, reps, bootstrap=False)
    triad = (51, 75, 135)
    recovered = [records[k][c]["r"] for k in range(spec.K) for c in triad]
    assert float(np.mean(recovered)) == pytest.approx(0.6, abs=0.05)


# ── end-to-end: Tier-B frames drive the M4/M5 aggregation ────────────────────
def test_tierb_driver_domain_reproducible():
    spec = _denv_b_spec(seed=0, T=4000, driver_r=0.6, null_r=0.02)
    reps = generate_series_replicates(spec)
    frames, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200)
    cfg = tierb_hierarchy_config(spec)
    out = aggregate_reproducibility(frames, cfg, protein="DENVB")
    assert _rho_of(out, "domain", "Triad") > 0.8
    # the driver domain separates from the near-null domain
    assert _rho_of(out, "domain", "Triad") > _rho_of(out, "domain", "Oxy")


def test_tierb_gate_emits_mechanism():
    spec = _denv_b_spec(seed=1, T=4000, driver_r=0.7, null_r=0.02)
    reps = generate_series_replicates(spec)
    frames, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200)
    cfg = tierb_hierarchy_config(spec)
    _, mechs, _, meta = run_aggregation(frames, cfg, GateConfig(rho_star=0.5),
                                        protein="DENVB")
    triad = [m for m in mechs if "Triad" in m.region_id]
    assert triad, "expected a Triad-region mechanism"
    assert meta["calibrated"] is False   # production stays uncalibrated


# ── misspecified processes flow through the whole chain ──────────────────────
def test_tierb_heavy_tailed_system_round_trips():
    spec = _denv_b_spec(seed=0, T=4000, driver_r=0.6, null_r=0.02,
                        process="heavy_tailed", phi=0.7)
    reps = generate_series_replicates(spec)
    frames, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200)
    cfg = tierb_hierarchy_config(spec)
    out = aggregate_reproducibility(frames, cfg, protein="DENVB")
    # heavy tails do not break the chain; the driver is still reproducible
    assert _rho_of(out, "domain", "Triad") > 0.8


def test_tierb_ar2_system_round_trips():
    triad = (51, 75, 135)
    oxy = (152, 153, 154)
    residues = tuple(SeriesResidueSpec(c, target_r=0.6, process="ar2",
                                       a1=0.6, a2=0.3) for c in triad) + \
        tuple(SeriesResidueSpec(c, target_r=0.02, process="ar2",
                                a1=0.6, a2=0.3) for c in oxy)
    spec = TierBSystemSpec(
        name="AR2SYS", levels=_LEVELS,
        chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", triad, "NS3"),
                 SynDomain("Oxy", oxy, "NS3")),
        residues=residues, K=3, T=4000, v_phi=0.7, offset=0, seed=0)
    reps = generate_series_replicates(spec)
    frames, _ = recover_frames_from_series(spec, reps, bootstrap=True, B=200)
    cfg = tierb_hierarchy_config(spec)
    out = aggregate_reproducibility(frames, cfg, protein="AR2SYS")
    assert _rho_of(out, "domain", "Triad") > 0.8


# ── validation ───────────────────────────────────────────────────────────────
def test_duplicate_canonical_ids_rejected():
    with pytest.raises(ValueError):
        generate_series_replicates(TierBSystemSpec(
            name="DUP", levels=_LEVELS,
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1, 2), "A"),),
            residues=(SeriesResidueSpec(1, 0.5), SeriesResidueSpec(1, 0.5)),
            seed=0))


def test_levels_must_end_in_residue():
    with pytest.raises(ValueError):
        generate_series_replicates(TierBSystemSpec(
            name="L", levels=("complex", "domain"),
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1,), "A"),),
            residues=(SeriesResidueSpec(1, 0.5),), seed=0))


def test_unknown_process_rejected():
    spec = TierBSystemSpec(
        name="BAD", levels=_LEVELS,
        chains=(SynChain("A", (1, 99)),),
        domains=(SynDomain("D", (1,), "A"),),
        residues=(SeriesResidueSpec(1, 0.5, process="quantum"),), seed=0)
    with pytest.raises(ValueError):
        generate_series_replicates(spec)


# ── recover_residue_effect standalone (unit) ─────────────────────────────────
def test_recover_residue_effect_bootstrap_fields():
    from validation.processes import coupled_ar1_pair
    sp = coupled_ar1_pair(3000, 0.6, 0.7, make_rng(0))
    rec = recover_residue_effect(sp.V, sp.d, bootstrap=True, B=200, seed=0)
    assert set(["r", "tau_int", "n_eff", "neff_status", "theta_se",
                "theta_bootstrap_se", "bootstrap_method"]).issubset(rec.keys())
    assert rec["r"] == pytest.approx(0.6, abs=0.05)
    assert rec["bootstrap_method"] in ("circular", "stationary", "fisher_neff")


def test_recover_residue_effect_no_bootstrap_minimal_fields():
    from validation.processes import coupled_ar1_pair
    sp = coupled_ar1_pair(3000, 0.5, 0.5, make_rng(0))
    rec = recover_residue_effect(sp.V, sp.d, bootstrap=False)
    assert "theta_bootstrap_se" not in rec
    assert rec["neff_status"] in ("ok", "white_noise")
