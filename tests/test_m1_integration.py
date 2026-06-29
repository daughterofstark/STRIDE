"""M1 integration: attach_effective_sample_size appends only, preserves existing."""
import numpy as np
import pandas as pd

from mechanism.statistics.neff import attach_effective_sample_size


class _Res:
    def __init__(self, resid):
        self.resid = resid


def _toy(n_frames=2000, n_res=12, seed=0):
    rng = np.random.default_rng(seed)
    volumes = rng.standard_normal(n_frames).cumsum()  # autocorrelated
    dm = rng.standard_normal((n_frames, n_res))
    residues = [_Res(100 + i) for i in range(n_res)]
    # df_res as the pipeline builds it (subset of real columns), out of resid order
    rows = []
    for i, res in enumerate(residues):
        r = float(np.corrcoef(volumes, dm[:, i])[0, 1])
        rows.append(dict(file_resid=res.resid, canon_resid=res.resid - 147,
                         name="ALA", r=r, abs_r=abs(r), label=f"ALA{res.resid-147}"))
    df = pd.DataFrame(rows).sort_values("canon_resid").reset_index(drop=True)
    return df, residues, dm, volumes


def test_existing_columns_unchanged():
    df, residues, dm, volumes = _toy()
    before = df.copy(deep=True)
    out = attach_effective_sample_size(df, residues, dm, volumes)
    # every original column identical, same order, same values
    for col in before.columns:
        pd.testing.assert_series_equal(out[col], before[col], check_names=True)


def test_new_columns_added_in_order():
    df, residues, dm, volumes = _toy()
    orig_cols = list(df.columns)
    out = attach_effective_sample_size(df, residues, dm, volumes)
    assert list(out.columns) == orig_cols + ["tau_int", "n_eff", "neff_status", "theta_se"]


def test_new_columns_are_valid():
    df, residues, dm, volumes = _toy()
    out = attach_effective_sample_size(df, residues, dm, volumes)
    assert (out["tau_int"] >= 0.5).all()
    assert (out["n_eff"] >= 2.0).all()
    assert (out["n_eff"] <= len(volumes)).all()
    assert out["neff_status"].isin(
        {"ok", "white_noise", "constant_signal", "short_trajectory",
         "undersampled_capped"}).all()
    # theta_se consistent with the Fisher form using the appended n_eff
    expect = np.sqrt(((1 - out["r"] ** 2) ** 2) / out["n_eff"])
    np.testing.assert_allclose(out["theta_se"].to_numpy(), expect.to_numpy(), rtol=1e-12)


def test_alignment_survives_resid_sort():
    # df_res is sorted by canon_resid; mapping must still align by file_resid
    df, residues, dm, volumes = _toy(seed=3)
    out = attach_effective_sample_size(df, residues, dm, volumes)
    # recompute n_eff independently for the residue at a known file_resid
    from mechanism.statistics.neff import effective_sample_size
    fid = int(out.iloc[5]["file_resid"])
    col = fid - 100  # _Res index = resid-100
    indep = effective_sample_size(volumes, dm[:, col]).n_eff
    assert out.iloc[5]["n_eff"] == indep
