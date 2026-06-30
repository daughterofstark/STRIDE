"""M5 tests: report writers (profile.csv, mechanism.json)."""
import json

import numpy as np
import pandas as pd
import pytest

from mechanism.config.hierarchy_schema import HierarchyConfig, ChainSpec, GroupSpec
from mechanism.replicate.aggregator import GateConfig, run_aggregation
from mechanism.reports.mechanism_report import (
    write_profile_csv, write_mechanism_json, write_reports,
)


def _cfg(triad=(51, 75, 135)):
    return HierarchyConfig(
        name="t", chains=(ChainSpec("NS3", (1, 999)),),
        domains=(GroupSpec("Triad", tuple(triad), chain="NS3", order=0),))


def _permuting(canon=(51, 75, 135), a=0.9):
    dfs = []
    for k, c in enumerate(canon):
        rng = np.random.default_rng(k)
        vals = np.array([a if cc == c else 0.0 for cc in canon]) + rng.normal(0, 1e-3, len(canon))
        ca = np.asarray(canon)
        dfs.append(pd.DataFrame(dict(
            file_resid=ca + 47, canon_resid=ca, name=["ALA"] * len(ca),
            chain=["NS3"] * len(ca), r=vals, abs_r=np.abs(vals),
            theta_se=np.full(len(ca), 1e-2),
            theta_bootstrap_se=np.full(len(ca), 1e-2))))
    return dfs


def _run():
    return run_aggregation(_permuting(), _cfg(), GateConfig(rho_star=0.5),
                           protein="DENV")


def test_write_profile_csv(tmp_path):
    prof, *_ = _run()
    path = write_profile_csv(prof, str(tmp_path / "profile.csv"))
    back = pd.read_csv(path)
    assert "gated" in back.columns and "rho" in back.columns
    assert len(back) == len(prof)
    assert back["gated"].sum() >= 1


def test_write_mechanism_json_schema_and_flags(tmp_path):
    prof, mechs, unres, meta = _run()
    path = write_mechanism_json(mechs, unres, meta, str(tmp_path / "mechanism.json"))
    payload = json.load(open(path))
    # uncalibrated guarantee, machine-checkable
    assert payload["calibrated"] is False
    assert "uncalibrated_note" in payload
    assert payload["gate"]["rho_star"] == 0.5
    assert len(payload["mechanisms"]) == len(mechs)
    for m in payload["mechanisms"]:
        assert m["calibrated"] is False
        assert m["direction"] in ("increase", "decrease", "mixed")
        assert "rho" in m and "scale_level" in m


def test_write_reports_creates_both(tmp_path):
    prof, mechs, unres, meta = _run()
    paths = write_reports(prof, mechs, unres, meta, str(tmp_path), prefix="DENV")
    assert paths["profile"].endswith("DENV_profile.csv")
    assert paths["mechanism"].endswith("DENV_mechanism.json")
    assert json.load(open(paths["mechanism"]))["schema_version"] == "m5"
    assert len(pd.read_csv(paths["profile"])) == len(prof)


def test_mixed_mechanism_serialises_nulls(tmp_path):
    canon = tuple(range(51, 57))
    dfs = []
    for k, (p, q) in enumerate([(51, 52), (53, 54), (55, 56)]):
        vals = np.zeros(6)
        vals[p - 51] = 0.8
        vals[q - 51] = -0.8
        ca = np.asarray(canon)
        dfs.append(pd.DataFrame(dict(
            file_resid=ca + 47, canon_resid=ca, name=["ALA"] * 6,
            chain=["NS3"] * 6, r=vals, abs_r=np.abs(vals),
            theta_se=np.full(6, 1e-2), theta_bootstrap_se=np.full(6, 1e-2))))
    prof, mechs, unres, meta = run_aggregation(dfs, _cfg(canon),
                                               GateConfig(rho_star=0.5))
    path = write_mechanism_json(mechs, unres, meta, str(tmp_path / "m.json"))
    payload = json.load(open(path))
    mixed = [m for m in payload["mechanisms"] if m["direction"] == "mixed"]
    assert mixed
    assert mixed[0]["beta_signed"] is None  # null in JSON


def test_json_is_numpy_safe(tmp_path):
    # ensure numpy scalars in meta/mechanisms don't break json.dump
    prof, mechs, unres, meta = _run()
    meta = dict(meta)
    meta["rho_star"] = np.float64(0.5)
    meta["n_loci"] = np.int64(meta["n_loci"])
    path = write_mechanism_json(mechs, unres, meta, str(tmp_path / "m.json"))
    assert json.load(open(path))["gate"]["rho_star"] == 0.5
