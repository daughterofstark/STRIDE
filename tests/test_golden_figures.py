"""End-to-end golden parity for figures (pixel tolerance). Skips without fixtures/PIL."""
import os, glob, shutil, tempfile, pytest

PIXEL_TOL = 2.0  # mean absolute per-channel difference (0-255); generous for backend/font drift

@pytest.mark.skipif(not os.environ.get("MECHANISM_DATA_DIR"),
                    reason="set MECHANISM_DATA_DIR to run figure parity")
def test_figure_parity(golden_dir):
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    import numpy as np
    from mechanism import Config, run_pipeline
    data = os.environ["MECHANISM_DATA_DIR"]
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "data"); shutil.copytree(data, work)
        run_pipeline(Config(base_dir=work))
        pngs = glob.glob(os.path.join(golden_dir, "**", "*.png"), recursive=True)
        assert pngs, "no golden PNGs"
        for g in pngs:
            rel = os.path.relpath(g, golden_dir)
            produced = os.path.join(work, rel)
            assert os.path.exists(produced), f"missing produced figure: {rel}"
            a = np.asarray(Image.open(g).convert("RGB"), dtype=float)
            b = np.asarray(Image.open(produced).convert("RGB"), dtype=float)
            assert a.shape == b.shape, f"{rel} size differs"
            assert float(np.abs(a-b).mean()) <= PIXEL_TOL, f"{rel} exceeds pixel tolerance"
