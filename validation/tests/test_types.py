"""V0 data contract: construction, derived rho, immutability, and serialisation."""
from validation.types import RegionTruth, GroundTruthSystem, SimResult, SweepCell


def _system():
    regions = (
        RegionTruth("c/p/NS3/Triad", scale_index=3, beta=2.0, tau2=0.05,
                    sigma2_bar=0.01, is_driver=True),
        RegionTruth("c/p/NS3/Oxy", scale_index=1, beta=0.0, tau2=0.20,
                    sigma2_bar=0.02, is_driver=False),
    )
    return GroundTruthSystem(
        name="SYN1",
        levels=("complex", "protein", "chain", "domain", "residue"),
        regions=regions, true_ell_star=3, direction="increase",
        K=3, T=200, tau_int=5.0, seed=42)


def test_region_truth_rho_derivation():
    assert RegionTruth("x", 0, 2.0, 0.0, 0.0, True).rho == 1.0   # perfectly reproducible
    assert RegionTruth("y", 0, 0.0, 1.0, 0.0, False).rho == 0.0  # null
    r = RegionTruth("z", 0, 1.0, 1.0, 0.0, True)
    assert abs(r.rho - 0.5) < 1e-12                              # beta^2/(beta^2+tau^2)
    assert RegionTruth("q", 0, 0.0, 0.0, 0.0, False).rho == 0.0  # 0/0 guard


def test_groundtruth_roundtrip():
    s = _system()
    assert GroundTruthSystem.from_dict(s.to_dict()) == s


def test_region_truth_roundtrip():
    r = RegionTruth("c/p/NS3/Triad", 3, 2.0, 0.05, 0.01, True)
    assert RegionTruth.from_dict(r.to_dict()) == r


def test_simresult_roundtrip_including_none_and_extra():
    r = SimResult(system="SYN1", method="stride", K=3, T=200, seed=1,
                  gated_scale_index=3, gated_region_id="c/p/NS3/Triad",
                  rho_at_gate=0.9, beta_signed=2.0, beta_ci_lower=1.5,
                  beta_ci_upper=2.5, direction="increase",
                  extra={"coherence": 0.99})
    assert SimResult.from_dict(r.to_dict()) == r
    # unresolved locus (None gate) round-trips too
    u = SimResult(system="SYN1", method="single_traj", K=1, T=200, seed=2,
                  gated_scale_index=None, gated_region_id=None, rho_at_gate=None,
                  beta_signed=None, beta_ci_lower=None, beta_ci_upper=None,
                  direction=None)
    assert SimResult.from_dict(u.to_dict()) == u


def test_sweepcell_roundtrip():
    c = SweepCell(system="SYN1", K=3, T=200, tau2=0.05, beta2=4.0, seed=7)
    assert SweepCell.from_dict(c.to_dict()) == c


def test_frozen_immutability():
    c = SweepCell("S", 3, 200, 0.05, 4.0, 7)
    try:
        c.K = 5
        raised = False
    except Exception:
        raised = True
    assert raised
