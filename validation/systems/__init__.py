"""Abstract synthetic systems for the V7 sweep — named by *hierarchy topology*.

These systems exist **only** to exercise the orchestration layer against **distinct
hierarchy structures**. They are deliberately *abstract*: their names describe the
hierarchy shape (chain/domain counts, driver scale), **not** any biological entity.
No biological realism is claimed or implied — the specification says nothing about
particular proteins, and V7 is an engineering milestone, so inventing biologically
named systems would imply realism that is not part of the method. [CHOICE, per V7
scope: abstract topology-named systems only]

Each system is a pure factory over the existing V1 Tier-A generator API
(``SyntheticSystemSpec`` etc.) — no generator logic is duplicated. A factory returns a
fully-specified :class:`~validation.generate.SyntheticSystemSpec` for a given
``(seed, K, T, tau2, beta2, driver)`` cell, plus static metadata describing the
hierarchy and the planted truth (true reproducible scale, driver label/support).

Registry
--------
``SYSTEMS`` maps a system name to a :class:`SystemDef`. The DENV system used by V4/V5
is included as the historical anchor (``is_denv = True``); the roadmap's "≥2 systems
beyond DENV" is satisfied by the two abstract, non-DENV topologies here
(``two_level_single_chain`` and ``three_level_two_chain``), which have hierarchies
distinct from DENV and from each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import math

from ..generate import (
    SyntheticSystemSpec, SynChain, SynDomain, Driver, NullRegion, region_path,
)


@dataclass(frozen=True)
class SystemDef:
    """A named synthetic system: a spec factory plus static hierarchy metadata."""

    name: str
    levels: tuple                     # coarse -> fine
    true_scale_level: str             # the planted reproducible scale (driver on)
    driver_label: str                 # region label of the planted driver
    driver_region_substr: str         # substring identifying the driver region id
    driver_support: tuple             # canonical ids carrying the effect
    build: Callable                   # (seed,K,T,tau2,beta2,driver=True)->Spec
    is_denv: bool = False
    topology: str = ""                # short human description of the hierarchy
    calibration_key: str = ""         # stem used for rho_star_<key>_K<K>.yaml

    @property
    def n_chains(self) -> int:
        return self._n_chains

    _n_chains: int = 1

    def hierarchy_signature(self) -> tuple:
        """A structural fingerprint used to assert systems are genuinely distinct."""
        return (tuple(self.levels), self._n_chains, self.true_scale_level)


# ── DENV anchor (reuses the exact spec used by V4/V5/V6 artifacts) ───────────
_DENV_LEVELS = ("complex", "protein", "chain", "domain", "residue")
_DENV_TRIAD = (51, 75, 135)
_DENV_OXY = (152, 153, 154, 155)


def _build_denv(seed: int, K: int, T: int, tau2: float, beta2: float,
                driver: bool = True) -> SyntheticSystemSpec:
    beta = math.sqrt(max(beta2, 0.0))
    drivers = ()
    if driver and beta > 0:
        drivers = (Driver(support=_DENV_TRIAD, scale_level="domain",
                          region_id=region_path(chain="NS3", domain="Triad"),
                          beta=beta, tau2=tau2, carrier_mode="distributed"),)
    nulls = (NullRegion(_DENV_TRIAD if not drivers else _DENV_OXY, "domain",
             region_path(chain="NS3", domain="Triad" if not drivers else "Oxy")),)
    return SyntheticSystemSpec(
        name="DENV_NS2B_NS3", levels=_DENV_LEVELS,
        chains=(SynChain("NS3", (1, 9999)),),
        domains=(SynDomain("Triad", _DENV_TRIAD, "NS3"),
                 SynDomain("Oxy", _DENV_OXY, "NS3")),
        residues=_DENV_TRIAD + _DENV_OXY, drivers=drivers, nulls=nulls,
        K=K, T=T, sigma2=0.04, offset=0, seed=seed, true_ell_star=1)


# ── abstract system A: two-level, single chain, domain-scale driver ──────────
# Hierarchy topology:  complex / chain / domain / residue   (single chain, 3 domains)
# Distinct from DENV (no "protein" level, 3 domains, different residue layout).
_A_LEVELS = ("complex", "chain", "domain", "residue")
_A_D1 = (10, 11, 12, 13)         # driver domain
_A_D2 = (20, 21, 22)             # null domain
_A_D3 = (30, 31, 32, 33, 34)     # null domain


def _build_two_level_single_chain(seed: int, K: int, T: int, tau2: float,
                                  beta2: float, driver: bool = True
                                  ) -> SyntheticSystemSpec:
    beta = math.sqrt(max(beta2, 0.0))
    drivers = ()
    if driver and beta > 0:
        drivers = (Driver(support=_A_D1, scale_level="domain",
                          region_id=region_path(chain="CH", domain="D1"),
                          beta=beta, tau2=tau2, carrier_mode="distributed"),)
    nulls = (NullRegion(_A_D2, "domain", region_path(chain="CH", domain="D2")),
             NullRegion(_A_D3, "domain", region_path(chain="CH", domain="D3")))
    return SyntheticSystemSpec(
        name="two_level_single_chain", levels=_A_LEVELS,
        chains=(SynChain("CH", (1, 9999)),),
        domains=(SynDomain("D1", _A_D1, "CH"), SynDomain("D2", _A_D2, "CH"),
                 SynDomain("D3", _A_D3, "CH")),
        residues=_A_D1 + _A_D2 + _A_D3, drivers=drivers, nulls=nulls,
        K=K, T=T, sigma2=0.04, offset=0, seed=seed, true_ell_star=1)


# ── abstract system B: three-level, two chains, chain-scale driver ───────────
# Hierarchy topology:  complex / chain / domain / residue   (TWO chains; driver at
# the CHAIN scale). Distinct top structure: the reproducible scale is chain, not
# domain, and canonical id space is partitioned across two chains.
_B_LEVELS = ("complex", "chain", "domain", "residue")
_B_CH1_D1 = (101, 102, 103)
_B_CH1_D2 = (104, 105, 106)
_B_CH2_D1 = (201, 202, 203)
_B_CH2_D2 = (204, 205, 206)


def _build_three_level_two_chain(seed: int, K: int, T: int, tau2: float,
                                 beta2: float, driver: bool = True
                                 ) -> SyntheticSystemSpec:
    beta = math.sqrt(max(beta2, 0.0))
    # driver spans an entire chain (chain-scale reproducible region)
    ch1_support = _B_CH1_D1 + _B_CH1_D2
    drivers = ()
    if driver and beta > 0:
        drivers = (Driver(support=ch1_support, scale_level="chain",
                          region_id=region_path(chain="CH1"),
                          beta=beta, tau2=tau2, carrier_mode="distributed"),)
    nulls = (NullRegion(_B_CH2_D1 + _B_CH2_D2, "chain",
                        region_path(chain="CH2")),)
    return SyntheticSystemSpec(
        name="three_level_two_chain", levels=_B_LEVELS,
        chains=(SynChain("CH1", (1, 199)), SynChain("CH2", (200, 9999))),
        domains=(SynDomain("CH1_D1", _B_CH1_D1, "CH1"),
                 SynDomain("CH1_D2", _B_CH1_D2, "CH1"),
                 SynDomain("CH2_D1", _B_CH2_D1, "CH2"),
                 SynDomain("CH2_D2", _B_CH2_D2, "CH2")),
        residues=_B_CH1_D1 + _B_CH1_D2 + _B_CH2_D1 + _B_CH2_D2,
        drivers=drivers, nulls=nulls,
        K=K, T=T, sigma2=0.04, offset=0, seed=seed, true_ell_star=2)


# ── registry ─────────────────────────────────────────────────────────────────
SYSTEMS = {
    "DENV_NS2B_NS3": SystemDef(
        name="DENV_NS2B_NS3", levels=_DENV_LEVELS, true_scale_level="domain",
        driver_label="Triad", driver_region_substr="Triad",
        driver_support=_DENV_TRIAD, build=_build_denv, is_denv=True, _n_chains=1,
        calibration_key="DENV",
        topology="complex/protein/chain/domain/residue; 1 chain, 2 domains; "
                 "domain-scale driver"),
    "two_level_single_chain": SystemDef(
        name="two_level_single_chain", levels=_A_LEVELS,
        true_scale_level="domain", driver_label="D1", driver_region_substr="D1",
        driver_support=_A_D1, build=_build_two_level_single_chain, _n_chains=1,
        topology="complex/chain/domain/residue; 1 chain, 3 domains; "
                 "domain-scale driver"),
    "three_level_two_chain": SystemDef(
        name="three_level_two_chain", levels=_B_LEVELS,
        true_scale_level="chain", driver_label="CH1", driver_region_substr="CH1",
        driver_support=_B_CH1_D1 + _B_CH1_D2, build=_build_three_level_two_chain,
        _n_chains=2,
        topology="complex/chain/domain/residue; 2 chains, 4 domains; "
                 "chain-scale driver"),
}


def get_system(name: str) -> SystemDef:
    """Look up a system definition by name (KeyError with a helpful message)."""
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; known: {sorted(SYSTEMS)}")
    return SYSTEMS[name]


def non_denv_systems() -> list:
    """The abstract, non-DENV systems (the roadmap's '>= 2 systems beyond DENV')."""
    return [d for d in SYSTEMS.values() if not d.is_denv]
