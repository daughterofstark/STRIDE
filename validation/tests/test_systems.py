"""V7 tests: abstract synthetic system definitions (validation.systems).

Verify the registry is well-formed, systems build valid specs, and — crucially for
the roadmap — that >= 2 non-DENV systems exist with hierarchies distinct from DENV and
from each other. No empirical outcome is frozen here.
"""
import pytest

from validation.systems import SYSTEMS, get_system, non_denv_systems, SystemDef
from validation.generate import generate_system, SyntheticSystemSpec
from validation.adapters import to_hierarchy_config, aggregate_via_production


def test_registry_has_denv_and_two_non_denv():
    assert "DENV_NS2B_NS3" in SYSTEMS
    nd = non_denv_systems()
    # roadmap: ">= 2 systems beyond DENV"
    assert len(nd) >= 2
    assert all(not d.is_denv for d in nd)


def test_non_denv_hierarchies_are_distinct():
    nd = non_denv_systems()
    sigs = {d.hierarchy_signature() for d in nd}
    # each non-DENV system has a distinct hierarchy signature
    assert len(sigs) == len(nd)
    # and each differs from DENV's
    denv_sig = SYSTEMS["DENV_NS2B_NS3"].hierarchy_signature()
    assert all(d.hierarchy_signature() != denv_sig for d in nd)


def test_systems_are_topology_named_not_biological():
    # abstract systems must be named by topology, not biology (V7 scope)
    for d in non_denv_systems():
        assert d.topology  # has a topology description
        # names describe structure (levels / chains), not biological entities
        assert any(tok in d.name for tok in ("level", "chain", "domain"))


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_system_builds_valid_spec(name):
    d = get_system(name)
    spec = d.build(seed=1, K=5, T=0, tau2=0.0, beta2=0.36, driver=True)
    assert isinstance(spec, SyntheticSystemSpec)
    assert spec.levels[-1] == "residue"
    assert spec.name == name


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_system_driver_region_reproducible(name):
    # a driver system's true-scale driver region should read a high rho_hat (a
    # structural sanity check on the generator wiring, not a frozen metric value)
    d = get_system(name)
    spec = d.build(seed=1, K=5, T=0, tau2=0.0, beta2=1.0, driver=True)
    out = aggregate_via_production(
        list(generate_system(spec).per_run_dfs), to_hierarchy_config(spec),
        protein=name)
    sub = out[(out.scale_level == d.true_scale_level) & (out.label == d.driver_label)]
    assert len(sub) == 1
    assert float(sub["rho"].iloc[0]) > 0.5     # reproducible signal present


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_system_null_build_has_no_driver(name):
    d = get_system(name)
    spec = d.build(seed=1, K=5, T=0, tau2=0.0, beta2=0.0, driver=False)
    assert len(spec.drivers) == 0


def test_get_system_unknown_raises():
    with pytest.raises(KeyError):
        get_system("does_not_exist")


def test_system_build_deterministic():
    d = get_system("two_level_single_chain")
    a = generate_system(d.build(seed=3, K=5, T=0, tau2=0.0, beta2=0.36))
    b = generate_system(d.build(seed=3, K=5, T=0, tau2=0.0, beta2=0.36))
    from validation.generate import frames_digest
    assert frames_digest(a) == frames_digest(b)
