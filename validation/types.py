"""Ground-truth data contract for the validation framework (milestone V0).

These dataclasses fix the interfaces every later milestone shares, so V1–V8 never
renegotiate them. They encode the specification's own parameterization (§2.2 /
§2.4) as *given ground truth*, not as anything STRIDE estimates:

* ``RegionTruth`` / ``GroundTruthSystem`` — planted truth produced by the
  generators (V1/V2) and scored against by the metrics and baselines (V5/V6):
  per-region ``(beta, tau^2, sigma_bar^2)`` with the derived ``rho`` and the true
  reproducible scale ``ell*``.
* ``SimResult`` — one method's estimate on one cell (V5/V6), persisted by V7.
* ``SweepCell`` — a ``(system, K, T, tau^2, beta^2, seed)`` coordinate iterated by
  the V7 sweep runner and read back by V8.

All are JSON-serialisable via ``to_dict`` / ``from_dict`` for deterministic
persistence. Nothing here imports ``mechanism``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class RegionTruth:
    """Planted ground truth for a single region at a single scale (§2.2/§2.4)."""

    region_id: str
    scale_index: int          # residue = 0 (finest), matching production convention
    beta: float               # reproducible population effect beta_R
    tau2: float               # between-replicate variance tau_R^2
    sigma2_bar: float         # mean within-replicate sampling variance sigma_bar_R^2
    is_driver: bool           # True for a planted driver region, False for a null

    @property
    def rho(self) -> float:
        """Derived reproducibility coefficient rho = beta^2/(beta^2+tau^2+sigma_bar^2)."""
        denom = self.beta ** 2 + self.tau2 + self.sigma2_bar
        return 0.0 if denom <= 0.0 else (self.beta ** 2) / denom

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RegionTruth":
        return cls(region_id=d["region_id"], scale_index=int(d["scale_index"]),
                   beta=float(d["beta"]), tau2=float(d["tau2"]),
                   sigma2_bar=float(d["sigma2_bar"]), is_driver=bool(d["is_driver"]))


@dataclass(frozen=True)
class GroundTruthSystem:
    """A synthetic system with fully known truth (the V1/V2 generator output)."""

    name: str
    levels: tuple                     # hierarchy level names, coarse -> fine
    regions: tuple                    # tuple[RegionTruth, ...]
    true_ell_star: int                # true finest reproducible scale index
    direction: str                    # "increase" | "decrease" | "mixed"
    K: int                            # replicate count
    T: int                            # per-replicate length
    tau_int: float                    # integrated autocorrelation time (sets N_eff)
    seed: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "levels": list(self.levels),
            "regions": [r.to_dict() for r in self.regions],
            "true_ell_star": self.true_ell_star,
            "direction": self.direction,
            "K": self.K, "T": self.T, "tau_int": self.tau_int, "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroundTruthSystem":
        return cls(
            name=d["name"], levels=tuple(d["levels"]),
            regions=tuple(RegionTruth.from_dict(r) for r in d["regions"]),
            true_ell_star=int(d["true_ell_star"]), direction=d["direction"],
            K=int(d["K"]), T=int(d["T"]), tau_int=float(d["tau_int"]),
            seed=int(d["seed"]),
        )


@dataclass(frozen=True)
class SimResult:
    """One method's estimate on one simulated cell (V5/V6), persisted by V7."""

    system: str
    method: str                       # "stride" | "single_traj" | "naive" | ...
    K: int
    T: int
    seed: int
    gated_scale_index: Optional[int]  # ell_hat_star; None = "no reproducible scale"
    gated_region_id: Optional[str]
    rho_at_gate: Optional[float]
    beta_signed: Optional[float]
    beta_ci_lower: Optional[float]
    beta_ci_upper: Optional[float]
    direction: Optional[str]
    extra: dict = field(default_factory=dict)   # method-specific provenance

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimResult":
        return cls(
            system=d["system"], method=d["method"], K=int(d["K"]), T=int(d["T"]),
            seed=int(d["seed"]), gated_scale_index=d["gated_scale_index"],
            gated_region_id=d["gated_region_id"], rho_at_gate=d["rho_at_gate"],
            beta_signed=d["beta_signed"], beta_ci_lower=d["beta_ci_lower"],
            beta_ci_upper=d["beta_ci_upper"], direction=d["direction"],
            extra=dict(d.get("extra", {})),
        )


@dataclass(frozen=True)
class SweepCell:
    """A single coordinate of the experiment sweep (V7)."""

    system: str
    K: int
    T: int
    tau2: float
    beta2: float
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SweepCell":
        return cls(system=d["system"], K=int(d["K"]), T=int(d["T"]),
                   tau2=float(d["tau2"]), beta2=float(d["beta2"]),
                   seed=int(d["seed"]))
