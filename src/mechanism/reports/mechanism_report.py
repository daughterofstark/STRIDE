"""Writers for the M5 mechanism report: ``profile.csv`` and ``mechanism.json``.

Implements the output artefacts of §1.2 (the resolution profile, the gated
mechanism with signed direction and effect magnitude + CI, and the unresolved
flag). Every mechanism is marked ``"calibrated": false`` until the validation
phase calibrates ``rho*`` (roadmap risk R8).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Sequence

import pandas as pd


_PROFILE_COLS = [
    "protein", "locus", "canon_label", "scale_index", "scale_level",
    "region_id", "rho", "gated", "beta", "beta_se", "tau2", "sigma2_bar",
    "a_signed", "coherence", "method", "status",
]


def write_profile_csv(profile_df: pd.DataFrame, path: str) -> str:
    """Write the per-locus resolution profile ``Pi`` (§2.6) to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [c for c in _PROFILE_COLS if c in profile_df.columns]
    df = profile_df[cols] if not profile_df.empty else pd.DataFrame(columns=cols)
    df.to_csv(path, index=False)
    return path


def write_mechanism_json(mechanisms: Sequence, unresolved: Sequence,
                         meta: dict, path: str) -> str:
    """Write the gated mechanisms (§2.5) + metadata to ``mechanism.json``.

    The top-level ``calibrated`` flag and per-mechanism ``calibrated: false`` make
    the provisional, uncalibrated status explicit and machine-checkable.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "schema_version": "m5",
        "calibrated": False,
        "uncalibrated_note": (
            "rho_star is a provisional configured threshold; empirical "
            "calibration (Part IV null-surrogate quantile) is deferred to the "
            "validation phase. No quantitative gate claim should be made until "
            "then."
        ),
        "gate": {
            "rho_star": meta.get("rho_star"),
            "alpha": meta.get("alpha"),
            "coherence_threshold": meta.get("coherence_threshold"),
        },
        "summary": {
            "n_loci": meta.get("n_loci", 0),
            "n_mechanisms": meta.get("n_mechanisms", 0),
            "n_unresolved": meta.get("n_unresolved", 0),
            "n_gate_uncertain": meta.get("n_gate_uncertain", 0),
        },
        "mechanisms": [asdict(m) for m in mechanisms],
        "unresolved_loci": list(unresolved),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    # numpy scalar / array safety
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_reports(profile_df, mechanisms, unresolved, meta, out_dir: str,
                  prefix: str = "") -> dict:
    """Write both artefacts into ``out_dir``; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""
    ppath = os.path.join(out_dir, f"{tag}profile.csv")
    mpath = os.path.join(out_dir, f"{tag}mechanism.json")
    write_profile_csv(profile_df, ppath)
    write_mechanism_json(mechanisms, unresolved, meta, mpath)
    return {"profile": ppath, "mechanism": mpath}
