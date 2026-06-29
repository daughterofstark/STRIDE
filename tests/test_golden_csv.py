"""End-to-end golden parity for CSV outputs. Requires fixtures captured from the
ORIGINAL v5 (scripts/capture_golden.py) plus a data dir + MDAnalysis + POVME.
Skips cleanly in CI where these are absent."""
import os, glob, shutil, tempfile, pytest
import numpy as np, pandas as pd

ATOL = 1e-9

def _frames_equal(a: pd.DataFrame, b: pd.DataFrame):
    # a = golden (original v5); b = produced (refactor + later additive columns).
    # Every golden column must be present and identical; new columns are allowed.
    assert set(a.columns).issubset(set(b.columns)), \
        f"golden columns missing from produced: {set(a.columns) - set(b.columns)}"
    assert len(a) == len(b), "row count differs"
    for col in a.columns:
        ca, cb = a[col], b[col]
        fa, fb = pd.to_numeric(ca, errors="coerce"), pd.to_numeric(cb, errors="coerce")
        if fa.notna().all() and fb.notna().all():
            assert np.allclose(fa.values, fb.values, atol=ATOL, equal_nan=True), f"{col} numeric mismatch"
        else:
            assert ca.astype(str).tolist() == cb.astype(str).tolist(), f"{col} string mismatch"

@pytest.mark.skipif(not os.environ.get("MECHANISM_DATA_DIR"),
                    reason="set MECHANISM_DATA_DIR to a copy of the data tree to run e2e parity")
def test_csv_parity(golden_dir):
    from mechanism import Config, run_pipeline
    data = os.environ["MECHANISM_DATA_DIR"]
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "data"); shutil.copytree(data, work)
        run_pipeline(Config(base_dir=work))
        goldens = glob.glob(os.path.join(golden_dir, "**", "*.csv"), recursive=True)
        assert goldens, "no golden CSVs found"
        for g in goldens:
            rel = os.path.relpath(g, golden_dir)
            produced = os.path.join(work, rel)
            assert os.path.exists(produced), f"missing produced CSV: {rel}"
            _frames_equal(pd.read_csv(g), pd.read_csv(produced))
