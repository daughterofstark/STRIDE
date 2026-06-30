"""M5 tests: resolution profile, gate, directional homogeneity, mechanism assembly.

Anchors: profile Pi (§2.6), gate ell* = min{ell: rho>=rho*} (§2.5/§1.3, first
up-crossing), I4 monotonicity of ell* in rho* (§III), signed effect via A_sgn gated
by directional homogeneity (§2.4/A4), CI from the M4 posterior se with no extra
widening (Part V(a)), and the uncalibrated-by-construction guarantee (roadmap R8).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from mechanism.config.hierarchy_schema import HierarchyConfig, ChainSpec, GroupSpec
from mechanism.replicate.aggregator import (
    GateConfig, build_profiles, gate_profile, run_aggregation, calibrate_rho_star,
)
from mechanism.statistics.reproducibility import aggregate_reproducibility


def _cfg(triad=(51, 75, 135)):
    return HierarchyConfig(
        name="t",
        chains=(ChainSpec("NS3", (1, 999)),),
        domains=(GroupSpec("Triad", tuple(triad), chain="NS3", order=0),),
    )


def _df(values, seed, canon, offset=47, noise=1e-3):
    rng = np.random.default_rng(seed)
    canon = np.asarray(canon)
    r = np.asarray(values, float) + rng.normal(0, noise, len(canon))
    return pd.DataFrame(dict(
        file_resid=canon + offset, canon_resid=canon, name=["ALA"] * len(canon),
        chain=["NS3"] * len(canon), r=r, abs_r=np.abs(r),
        theta_se=np.full(len(canon), 1e-2),
        theta_bootstrap_se=np.full(len(canon), 1e-2)))


def _permuting_same_sign(canon=(51, 75, 135), a=0.9):
    # single strong residue permutes its location across runs, always positive:
    # reproducible at the region scale, not at residue scale (I1), coherent.
    dfs = []
    for k, c in enumerate(canon):
        vals = [a if cc == c else 0.0 for cc in canon]
        dfs.append(_df(vals, seed=k, canon=canon))
    return dfs


# ── profile construction (§2.6) ──────────────────────────────────────────────
def test_build_profiles_covers_all_scales_nested():
    dfs = _permuting_same_sign()
    table = aggregate_reproducibility(dfs, _cfg())
    prof = build_profiles(table)
    assert not prof.empty
    # every locus has a strictly increasing scale ladder starting at residue (0)
    for locus, pl in prof.groupby("locus"):
        idx = pl.sort_values("scale_index")["scale_index"].to_numpy()
        assert idx[0] == 0
        assert np.all(np.diff(idx) > 0)
        # each coarser region id is a path-prefix of the locus id
        for rid in pl["region_id"]:
            assert locus == rid or locus.startswith(rid + "/")


# ── gate: first up-crossing & none-path (§2.5) ───────────────────────────────
def _profile_fixture(rhos):
    # rhos indexed by scale_index 0..L
    return pd.DataFrame(dict(scale_index=list(range(len(rhos))), rho=rhos,
                             scale_level=[f"L{i}" for i in range(len(rhos))],
                             region_id=[f"r{i}" for i in range(len(rhos))],
                             status=["ok"] * len(rhos), beta=[1.0] * len(rhos)))


def test_gate_returns_finest_passing_scale():
    pf = _profile_fixture([0.1, 0.2, 0.7, 0.9])  # first >=0.5 at index 2
    g = gate_profile(pf, 0.5)
    assert g is not None and g["scale_index"] == 2


def test_gate_none_when_never_crosses():
    pf = _profile_fixture([0.1, 0.2, 0.3, 0.4])
    assert gate_profile(pf, 0.5) is None


def test_gate_picks_min_even_if_coarser_fails():
    # non-monotone profile: passes at 1, fails at 2, passes at 3 -> gate = 1
    pf = _profile_fixture([0.1, 0.8, 0.2, 0.95])
    assert gate_profile(pf, 0.5)["scale_index"] == 1


def test_i4_monotonicity_in_rho_star():
    dfs = _permuting_same_sign()
    prof = build_profiles(aggregate_reproducibility(dfs, _cfg()))
    for _, pl in prof.groupby("locus"):
        prev = -1
        for rs in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]:
            g = gate_profile(pl, rs)
            ell = 99 if g is None else int(g["scale_index"])  # None = +inf
            assert ell >= prev
            prev = ell


# ── end-to-end: emitted mechanisms ───────────────────────────────────────────
def test_reproducible_region_emits_signed_mechanism():
    dfs = _permuting_same_sign()
    _, mechs, unres, meta = run_aggregation(dfs, _cfg(), GateConfig(rho_star=0.5))
    assert meta["n_mechanisms"] >= 1
    multi = [m for m in mechs if m.n_loci > 1]
    assert multi, "expected a multi-residue gated region (I1)"
    m = max(multi, key=lambda x: x.n_loci)
    assert m.direction in ("increase", "decrease")
    assert m.beta_signed is not None
    assert m.beta_ci_lower < m.beta_signed < m.beta_ci_upper
    assert m.coherence > 0.9  # same-sign signal


def test_mixed_region_reports_energy_only():
    canon = tuple(range(51, 57))
    # pair-permuting opposing signs: reproducible energy, incoherent direction
    dfs = []
    for k, (p, q) in enumerate([(51, 52), (53, 54), (55, 56)]):
        vals = [0.0] * 6
        vals[p - 51] = 0.8
        vals[q - 51] = -0.8
        dfs.append(_df(vals, seed=k, canon=canon))
    _, mechs, _, _ = run_aggregation(dfs, _cfg(canon), GateConfig(rho_star=0.5))
    multi = [m for m in mechs if m.n_loci > 1]
    assert multi
    m = max(multi, key=lambda x: x.n_loci)
    assert m.direction == "mixed"
    assert m.beta_signed is None
    assert m.beta_ci_lower is None and m.beta_ci_upper is None
    assert m.coherence < 0.2


def test_unresolved_when_rho_star_above_all():
    dfs = _permuting_same_sign()
    _, mechs, unres, meta = run_aggregation(dfs, _cfg(), GateConfig(rho_star=1.01))
    assert mechs == []
    assert meta["n_unresolved"] == meta["n_loci"] > 0


# ── calibration separation & uncalibrated flag (R8) ──────────────────────────
def test_every_mechanism_marked_uncalibrated():
    dfs = _permuting_same_sign()
    _, mechs, _, meta = run_aggregation(dfs, _cfg(), GateConfig(rho_star=0.5))
    assert meta["calibrated"] is False
    assert all(m.calibrated is False for m in mechs)


def test_calibrate_rho_star_seam_returns_provisional():
    dfs = _permuting_same_sign()
    gc = GateConfig(rho_star=0.42)
    assert calibrate_rho_star(dfs, _cfg(), gc) == 0.42  # no calibration performed


def test_rho_star_is_configurable():
    dfs = _permuting_same_sign()
    _, mechs_lo, _, _ = run_aggregation(dfs, _cfg(), GateConfig(rho_star=0.5))
    _, mechs_hi, unres_hi, _ = run_aggregation(dfs, _cfg(), GateConfig(rho_star=1.01))
    assert len(mechs_lo) >= 1 and len(mechs_hi) == 0


# ── uncertainty: posterior se, no extra widening (Part V(a)) ─────────────────
def test_ci_halfwidth_equals_z_times_posterior_se():
    dfs = _permuting_same_sign()
    _, mechs, _, _ = run_aggregation(dfs, _cfg(),
                                     GateConfig(rho_star=0.5, alpha=0.05))
    z = sps.norm.ppf(0.975)
    for m in mechs:
        if m.beta_signed is not None:
            half = (m.beta_ci_upper - m.beta_ci_lower) / 2.0
            assert half == pytest.approx(z * m.beta_se, rel=1e-9)


def test_gate_uncertain_flag_set_at_k3():
    dfs = _permuting_same_sign()  # K=3 -> Bayesian -> gate_uncertain
    _, mechs, _, meta = run_aggregation(dfs, _cfg(), GateConfig(rho_star=0.5))
    assert meta["n_gate_uncertain"] >= 1
    assert any(m.gate_uncertain for m in mechs)


def test_empty_input_is_safe():
    prof, mechs, unres, meta = run_aggregation([], _cfg(), GateConfig())
    assert mechs == [] and unres == [] and prof.empty
