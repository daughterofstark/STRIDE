"""Shared loaders for the V8 publication layer — **read frozen artifacts only**.

Every V8 module (``figures``, ``tables``, ``report``) reads its numbers through this
helper, which loads the *already-persisted* V4/V5/V6/V7 artifacts. V8 introduces **no
new numbers**: it neither recomputes nor extends any store. If a requested view cannot
be built from these frozen artifacts, the caller documents the limitation rather than
generating new data. Nothing here imports ``mechanism``.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Optional

import yaml

_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def artifact_dir() -> str:
    return _ARTIFACT_DIR


# ── V7 sweep results store ───────────────────────────────────────────────────
def load_sweep_records(artifact_dir: Optional[str] = None) -> list:
    """Load the frozen V7 sweep store as a list of plain dict records."""
    d = artifact_dir or _ARTIFACT_DIR
    path = os.path.join(d, "sweep_results.jsonl")
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sweep_manifest(artifact_dir: Optional[str] = None) -> dict:
    d = artifact_dir or _ARTIFACT_DIR
    path = os.path.join(d, "sweep_results_manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── V5 metrics report (empirical-vs-predicted + ell_min grid) ────────────────
def load_metrics(artifact_dir: Optional[str] = None,
                 system: str = "DENV") -> dict:
    d = artifact_dir or _ARTIFACT_DIR
    path = os.path.join(d, f"metrics_{system}.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── V6 method comparison (over-resolution, naive coverage, BH) ───────────────
def load_method_comparison(artifact_dir: Optional[str] = None,
                           system: str = "DENV") -> dict:
    d = artifact_dir or _ARTIFACT_DIR
    path = os.path.join(d, f"method_comparison_{system}.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── V4 calibration artifacts (rho* by scale, per system/K) ───────────────────
def load_calibration(system_key: str, K: int,
                     artifact_dir: Optional[str] = None) -> dict:
    d = artifact_dir or _ARTIFACT_DIR
    path = os.path.join(d, f"rho_star_{system_key}_K{K}.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def available_calibrations(artifact_dir: Optional[str] = None) -> list:
    """List (system_key, K) pairs for which a rho* artifact exists."""
    d = artifact_dir or _ARTIFACT_DIR
    out = []
    for p in sorted(glob.glob(os.path.join(d, "rho_star_*.yaml"))):
        stem = os.path.basename(p)[len("rho_star_"):-len(".yaml")]
        if "_K" in stem:
            key, k = stem.rsplit("_K", 1)
            try:
                out.append((key, int(k)))
            except ValueError:
                continue
    return out


def results_digest_of(records: list) -> str:
    """Recompute the V7-style content digest of a record list (provenance echo)."""
    import hashlib
    h = hashlib.sha256()
    for r in records:
        h.update(json.dumps(r, sort_keys=True).encode("utf-8"))
    return h.hexdigest()
