"""V1 generator tests: determinism, schema, I1, ordering/separation, and the
pathological regimes — all asserting the behavior of the **current** production
estimator, not an idealized one.

Design note (faithfulness): the production estimator fits the random-effects model
on the folded energy ``A_en = sqrt(sum theta^2)``. Two consequences shape these
tests:

* A pure-null region of ``m >= 2`` residues with ordinary within-noise reads a
  positive ``rho_hat`` that grows with ``m`` (folded-energy positivity). We do NOT
  assert "nulls read as zero"; we assert the *achievable* property that a driver
  region separates from equal-size null regions, and we CHARACTERIZE the null
  inflation as an observed property of the current implementation.
* A driver region's recovered ``beta_hat`` estimates ``E|beta + gamma|``; exact
  recovery holds only in the low-tau/high-SNR corner, which is where the
  closure test operates.
"""
import numpy as np
import pandas as pd
import pytest

from validation.generate import (
    SynChain, SynDomain, Driver, NullRegion,
    SyntheticSystemSpec, GeneratedSystem, generate_system,
    build_per_run_frame, region_path, frames_digest,
)
from validation.types import GroundTruthSystem, RegionTruth

# round-trip needs the production estimator (public API) + the adapter
from validation.adapters import to_hierarchy_config
from mechanism.statistics.reproducibility import aggregate_reproducibility
from mechanism.replicate.aggregator import run_aggregation, GateConfig


# ── shared spec builders ─────────────────────────────────────────────────────
_LEVELS = ("complex", "protein", "chain", "domain", "residue")
# scale indices in a 5-level hierarchy: residue=0, domain=1, chain=2, protein=3, complex=4


def _denv_like_spec(*, seed=0, K=3, carrier_mode="permute", beta=1.0, tau2=0.0,
                    sigma2=0.01, direction="increase", offset=1000,
                    sign_pattern=None):
    """A DENV-shaped system: Triad driver (domain scale) + Oxy null + a few loose
    NS3 residues (an implicit 'unassigned' null domain)."""
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155, 156, 157, 158, 159)
    loose = (10, 20, 30)
    residues = triad + oxy + loose
    chains = (SynChain("NS2B", (-999, 0)), SynChain("NS3", (1, 9999)))
    domains = (SynDomain("Triad", triad, "NS3"), SynDomain("Oxy", oxy, "NS3"))
    triad_id = region_path(chain="NS3", domain="Triad")
    oxy_id = region_path(chain="NS3", domain="Oxy")
    drivers = (Driver(support=triad, scale_level="domain", region_id=triad_id,
                      beta=beta, tau2=tau2, direction=direction,
                      carrier_mode=carrier_mode, sign_pattern=sign_pattern),)
    nulls = (NullRegion(oxy, "domain", oxy_id),)
    return SyntheticSystemSpec(
        name="DENVX", levels=_LEVELS, chains=chains, domains=domains,
        residues=residues, drivers=drivers, nulls=nulls,
        K=K, sigma2=sigma2, offset=offset, seed=seed, true_ell_star=1,
        direction=direction)


def _rho_of(out, level, label):
    sub = out[(out.scale_level == level) & (out.label == label)]
    return float(sub["rho"].iloc[0]) if len(sub) else float("nan")


# ── determinism ──────────────────────────────────────────────────────────────
def test_same_seed_byte_identical_frames():
    a = generate_system(_denv_like_spec(seed=7))
    b = generate_system(_denv_like_spec(seed=7))
    assert frames_digest(a) == frames_digest(b)
    for da, db in zip(a.per_run_dfs, b.per_run_dfs):
        pd.testing.assert_frame_equal(da, db)


def test_different_seed_differs():
    a = generate_system(_denv_like_spec(seed=1))
    b = generate_system(_denv_like_spec(seed=2))
    assert frames_digest(a) != frames_digest(b)


def test_frames_digest_is_stable_string():
    g = generate_system(_denv_like_spec(seed=0))
    d1 = frames_digest(g)
    d2 = frames_digest(generate_system(_denv_like_spec(seed=0)))
    assert isinstance(d1, str) and len(d1) == 64  # sha256 hex
    assert d1 == d2


# ── schema: frames carry exactly the production-consumed columns ─────────────
def test_per_run_frames_have_production_schema():
    g = generate_system(_denv_like_spec())
    expected = ["file_resid", "canon_resid", "name", "chain",
                "r", "abs_r", "theta_se", "theta_bootstrap_se"]
    for df in g.per_run_dfs:
        assert list(df.columns) == expected
        assert len(df) == len(g.spec.residues)
        # offset relation holds exactly
        assert (df["file_resid"] - df["canon_resid"] == g.spec.offset).all()
        # abs_r is |r|
        assert np.allclose(df["abs_r"], np.abs(df["r"]))
        # both uncertainty columns present and equal (Tier A writes sigma to both)
        assert np.allclose(df["theta_se"], df["theta_bootstrap_se"])


def test_build_per_run_frame_length_validation():
    with pytest.raises(ValueError):
        build_per_run_frame([1, 2], [0.1], [0.1, 0.1], ["A", "A"], offset=0)


# ── closure: round-trip through the production estimator ─────────────────────
def test_closure_driver_reproducible_high_snr():
    # low tau, modest sigma: driver domain rho ~ 1, and it gates at the domain.
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-3))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    assert _rho_of(out, "domain", "Triad") > 0.8


def test_closure_recovered_ell_star_matches_planted():
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-3))
    cfg = to_hierarchy_config(g.spec)
    prof, mechs, unresolved, meta = run_aggregation(
        list(g.per_run_dfs), cfg, GateConfig(rho_star=0.5), protein="DENVX")
    triad_id = region_path(chain="NS3", domain="Triad")
    triad_mechs = [m for m in mechs if m.region_id == triad_id]
    assert triad_mechs, "Triad driver produced no mechanism"
    m = triad_mechs[0]
    # planted true_ell_star is the domain scale index (1 in this 5-level hierarchy)
    assert m.scale_index == g.spec.true_ell_star == 1
    assert m.scale_level == "domain"


# ── driver-vs-null SEPARATION (replaces the false 'nulls read as zero' test) ──
def test_driver_separates_from_equal_context_nulls():
    # Driver domain rho must exceed the null-domain rho at matched region size.
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=0.01))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    driver_rho = _rho_of(out, "domain", "Triad")
    null_rho = _rho_of(out, "domain", "Oxy")
    assert driver_rho > null_rho, (driver_rho, null_rho)
    assert driver_rho > 0.8


# ── I1: the load-bearing invariance (region high, its residues low) ──────────
def test_I1_permuting_driver_region_dominates_residues():
    # Strict I1 tolerance (roadmap: residue rho < 0.2, region rho > 0.8) is met by
    # the DISJOINT permutation (each residue carries the effect in exactly one
    # replicate) — the construction the production I1 unit test uses.
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-8, K=3,
                                        carrier_mode="permute_disjoint"))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    triad_domain_rho = _rho_of(out, "domain", "Triad")
    res = out[out.scale_level == "residue"]
    triad_res = res[res.region_id.str.contains("/Triad/")]
    assert triad_domain_rho > 0.8
    assert triad_res["rho"].max() < 0.2, triad_res[["region_id", "rho"]].to_string()


def test_iid_permute_residue_rho_is_moderate_not_zero_current_estimator():
    # OBSERVED PROPERTY: with i.i.d.-uniform carrier draws (residues can repeat as
    # carrier across replicates), the folded-energy estimator leaves residue-level
    # rho_hat well above 0 (~0.3-0.5), even though the region is fully reproducible.
    # We characterize this honestly rather than asserting an idealized ~0.
    # [KNOWN LIMITATION of the current estimator; the disjoint construction is what
    #  achieves the strict I1 tolerance.]
    maxima = []
    for s in range(5):
        g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-8, K=3,
                                            carrier_mode="permute", seed=s))
        out = aggregate_reproducibility(list(g.per_run_dfs),
                                        to_hierarchy_config(g.spec))
        assert _rho_of(out, "domain", "Triad") > 0.8   # region still reproducible
        res = out[out.scale_level == "residue"]
        maxima.append(res[res.region_id.str.contains("/Triad/")]["rho"].max())
    # residue rho does NOT collapse to ~0 under i.i.d. permutation
    assert np.mean(maxima) > 0.2


def test_fixed_carrier_makes_residue_reproducible():
    # a fixed carrier is residue-reproducible: max residue rho ~ 1.
    g = generate_system(_denv_like_spec(carrier_mode="fixed", beta=1.0,
                                        tau2=0.0, sigma2=1e-8))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    res = out[out.scale_level == "residue"]
    assert res["rho"].max() > 0.9


# ── observed folded-energy property (characterization, not a claim about STRIDE) ─
def test_null_region_rho_grows_with_size_current_estimator():
    # OBSERVED PROPERTY of the CURRENT estimator (not spec, not a theorem):
    # a pure-null region's rho_hat increases with the number of residues, because
    # A_en = sqrt(sum eps^2) has a positive mean that grows with support size.
    # Documented here so V4 calibration has an explicit target. [KNOWN LIMITATION]
    def null_only_spec(n, seed):
        ids = tuple(range(1, n + 1))
        chains = (SynChain("A", (-10**9, 10**9)),)
        domains = (SynDomain("Null", ids, "A"),)
        return SyntheticSystemSpec(
            name="NULL", levels=_LEVELS, chains=chains, domains=domains,
            residues=ids, drivers=(), nulls=(NullRegion(ids, "domain",
                region_path(chain="A", domain="Null")),),
            K=3, sigma2=0.09, offset=0, seed=seed)

    means = {}
    for n in (2, 8, 32):
        rhos = []
        for s in range(6):
            g = generate_system(null_only_spec(n, s))
            out = aggregate_reproducibility(list(g.per_run_dfs),
                                            to_hierarchy_config(g.spec))
            rhos.append(_rho_of(out, "domain", "Null"))
        means[n] = float(np.mean(rhos))
    # monotone non-decreasing in region size (the observed folded-energy bias)
    assert means[2] < means[8] < means[32], means


# ── pathological / edge regimes ──────────────────────────────────────────────
def test_large_tau_signed_direction_goes_mixed():
    # tau^2 >> beta^2 with a permuting carrier: the effect magnitude wanders a lot
    # between runs. OBSERVED behavior of the current estimator:
    #   * the ENERGY rho_hat stays HIGH (folded energy |beta+gamma| has a positive
    #     mean with modest relative spread) — the folded-energy limitation, NOT a
    #     bug and NOT a spec claim; and
    #   * the SIGNED direction is correctly reported "mixed" because between-run
    #     sign wandering destroys directional coherence (A4).
    # We assert the faithful, achievable property (direction -> mixed) and
    # characterize the energy-rho behavior. [KNOWN LIMITATION]
    g = generate_system(_denv_like_spec(beta=0.3, tau2=4.0, sigma2=1e-3, K=6))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    # energy rho stays high under folded aggregation (characterization)
    assert _rho_of(out, "domain", "Triad") > 0.5
    # but the signed claim is not coherent
    _, mechs, _, _ = run_aggregation(list(g.per_run_dfs), cfg,
                                     GateConfig(rho_star=0.5), protein="DENVX")
    triad = [m for m in mechs if "Triad" in m.region_id]
    assert triad, "expected a Triad-region mechanism"
    assert triad[0].direction == "mixed"


def test_sigma_to_zero_beta_positive_drives_rho_up():
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-10,
                                        carrier_mode="distributed"))
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    assert _rho_of(out, "domain", "Triad") > 0.95


def test_whole_system_driver_gates_at_coarsest_scale():
    # a driver spanning both NS3 domains (across-domain support) is reproducible
    # only at the chain scale or coarser, not at any single domain.
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155)
    residues = triad + oxy
    chains = (SynChain("NS3", (1, 9999)),)
    domains = (SynDomain("Triad", triad, "NS3"), SynDomain("Oxy", oxy, "NS3"))
    # carrier permutes across BOTH domains' residues -> reproducible at chain scale
    chain_support = triad + oxy
    chain_id = region_path(chain="NS3")  # complex/protein/NS3
    drivers = (Driver(support=chain_support, scale_level="chain",
                      region_id=chain_id, beta=1.0, tau2=0.0,
                      carrier_mode="permute"),)
    spec = SyntheticSystemSpec(
        name="CHAINDRV", levels=_LEVELS, chains=chains, domains=domains,
        residues=residues, drivers=drivers, K=3, sigma2=1e-8, offset=0,
        seed=0, true_ell_star=2)  # chain index = 2
    g = generate_system(spec)
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="CHAINDRV")
    # chain-level rho high; neither single domain individually reproducible
    chain_rho = _rho_of(out, "chain", "NS3")
    assert chain_rho > 0.8
    # each domain sees the carrier only ~half the runs -> lower than chain
    dom = out[out.scale_level == "domain"]
    assert dom["rho"].max() < chain_rho


def test_singleton_region_driver():
    # a residue-scale driver (single-residue support) is reproducible at residue
    # scale; the carrier cannot permute (support size 1).
    residues = (51, 75, 135)
    chains = (SynChain("NS3", (1, 9999)),)
    domains = (SynDomain("Triad", residues, "NS3"),)
    res_id = "complex/protein/NS3/Triad/NS3:51"  # residue key form chain:file_resid
    # With offset 0, file_resid == canonical, so key is "NS3:51".
    drivers = (Driver(support=(51,), scale_level="residue", region_id=res_id,
                      beta=1.0, tau2=0.0, carrier_mode="fixed"),)
    spec = SyntheticSystemSpec(
        name="SGL", levels=_LEVELS, chains=chains, domains=domains,
        residues=residues, drivers=drivers, K=3, sigma2=1e-8, offset=0,
        seed=0, true_ell_star=0)
    g = generate_system(spec)
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="SGL")
    res = out[out.scale_level == "residue"]
    assert res["rho"].max() > 0.9  # the single carrier residue is reproducible


# ── direction: coherent vs mixed through the M5 gate ─────────────────────────
def test_direction_increase_recovered():
    # distributed carrier makes every triad residue carry signal each replicate,
    # so the effect is residue-reproducible and gates at residue scale. The
    # emitted direction of the triad-region mechanism(s) must be "increase".
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-4,
                                        direction="increase",
                                        carrier_mode="distributed"))
    cfg = to_hierarchy_config(g.spec)
    _, mechs, _, _ = run_aggregation(list(g.per_run_dfs), cfg,
                                     GateConfig(rho_star=0.5), protein="DENVX")
    triad = [m for m in mechs if "Triad" in m.region_id]
    assert triad, "expected at least one Triad-region mechanism"
    assert all(m.direction == "increase" for m in triad), \
        [(m.region_id, m.direction) for m in triad]


def test_direction_mixed_recovered():
    # per-residue alternating signs -> directionally incoherent region -> "mixed".
    # Use the domain-scale energy path: aggregate to the Triad domain and check
    # the coherence gate reports mixed. With distributed carriers and alternating
    # signs, the region is directionally balanced.
    g = generate_system(_denv_like_spec(
        beta=1.0, tau2=0.0, sigma2=1e-4, direction="mixed",
        carrier_mode="distributed", sign_pattern=(1, -1, 1)))
    cfg = to_hierarchy_config(g.spec)
    _, mechs, _, _ = run_aggregation(list(g.per_run_dfs), cfg,
                                     GateConfig(rho_star=0.5), protein="DENVX")
    # the triad residues themselves are individually coherent (each carries a
    # single fixed sign), so residue-scale mechanisms are not "mixed"; the mixed
    # property is a REGION-level statement. Verify it at the domain via the
    # coherence statistic in the per-scale table.
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="DENVX")
    triad_dom = out[(out.scale_level == "domain") & (out.label == "Triad")]
    assert len(triad_dom) == 1
    # |sum theta| / sum|theta| over the region is low when signs alternate
    assert float(triad_dom["coherence"].iloc[0]) < 0.6


# ── spec validation: duplicate ids, bad support, sign mismatch ───────────────
def test_duplicate_residue_ids_rejected():
    with pytest.raises(ValueError):
        generate_system(SyntheticSystemSpec(
            name="DUP", levels=_LEVELS,
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1, 2), "A"),),
            residues=(1, 2, 2), seed=0))


def test_driver_support_outside_residues_rejected():
    with pytest.raises(ValueError):
        generate_system(SyntheticSystemSpec(
            name="BAD", levels=_LEVELS,
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1, 2), "A"),),
            residues=(1, 2),
            drivers=(Driver(support=(1, 5), scale_level="domain",
                            region_id="x", beta=1.0, tau2=0.0),),
            seed=0))


def test_mixed_direction_requires_sign_pattern():
    with pytest.raises(ValueError):
        generate_system(SyntheticSystemSpec(
            name="M", levels=_LEVELS,
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1, 2, 3), "A"),),
            residues=(1, 2, 3),
            drivers=(Driver(support=(1, 2, 3), scale_level="domain",
                            region_id="x", beta=1.0, tau2=0.0,
                            direction="mixed"),),  # no sign_pattern
            seed=0))


def test_levels_must_end_in_residue():
    with pytest.raises(ValueError):
        generate_system(SyntheticSystemSpec(
            name="L", levels=("complex", "domain"),
            chains=(SynChain("A", (1, 99)),),
            domains=(SynDomain("D", (1,), "A"),),
            residues=(1,), seed=0))


# ── ragged membership: a residue present in only some runs must not crash ────
def test_ragged_membership_round_trips():
    g = generate_system(_denv_like_spec(beta=1.0, tau2=0.0, sigma2=1e-3))
    cfg = to_hierarchy_config(g.spec)
    dfs = list(g.per_run_dfs)
    dfs[1] = dfs[1].iloc[:-1].copy()  # drop one residue from run 2
    out = aggregate_reproducibility(dfs, cfg, protein="DENVX")
    assert not out.empty and out["rho"].notna().any()


# ── multiple drivers in one system ───────────────────────────────────────────
def test_multiple_drivers_both_reproducible():
    triad = (51, 75, 135)
    oxy = (152, 153, 154, 155, 156)
    residues = triad + oxy + (10, 20)
    chains = (SynChain("NS3", (1, 9999)),)
    domains = (SynDomain("Triad", triad, "NS3"), SynDomain("Oxy", oxy, "NS3"))
    drivers = (
        Driver(support=triad, scale_level="domain",
               region_id=region_path(chain="NS3", domain="Triad"),
               beta=1.0, tau2=0.0, carrier_mode="permute"),
        Driver(support=oxy, scale_level="domain",
               region_id=region_path(chain="NS3", domain="Oxy"),
               beta=1.2, tau2=0.0, carrier_mode="permute"),
    )
    spec = SyntheticSystemSpec(
        name="MULTI", levels=_LEVELS, chains=chains, domains=domains,
        residues=residues, drivers=drivers, K=3, sigma2=1e-6, offset=0,
        seed=0, true_ell_star=1)
    g = generate_system(spec)
    cfg = to_hierarchy_config(g.spec)
    out = aggregate_reproducibility(list(g.per_run_dfs), cfg, protein="MULTI")
    assert _rho_of(out, "domain", "Triad") > 0.8
    assert _rho_of(out, "domain", "Oxy") > 0.8


# ── ground-truth self-consistency ────────────────────────────────────────────
def test_ground_truth_rho_matches_formula():
    g = generate_system(_denv_like_spec(beta=2.0, tau2=0.05, sigma2=0.01))
    for rt in g.truth.regions:
        denom = rt.beta ** 2 + rt.tau2 + rt.sigma2_bar
        expected = 0.0 if denom <= 0 else rt.beta ** 2 / denom
        assert rt.rho == pytest.approx(expected)


def test_ground_truth_records_drivers_and_nulls():
    g = generate_system(_denv_like_spec())
    drivers = [r for r in g.truth.regions if r.is_driver]
    nulls = [r for r in g.truth.regions if not r.is_driver]
    assert len(drivers) == 1 and drivers[0].beta > 0
    assert len(nulls) == 1 and nulls[0].beta == 0.0
    assert isinstance(g.truth, GroundTruthSystem)


def test_ground_truth_roundtrips_through_serialization():
    g = generate_system(_denv_like_spec())
    assert GroundTruthSystem.from_dict(g.truth.to_dict()) == g.truth
