"""CLI mirroring the v5 argparse interface exactly (same flags/defaults)."""
import argparse, os
from .config import Config
from .pipeline import run_pipeline

def main(argv=None):
    ap = argparse.ArgumentParser(description="Mechanism pipeline (M0: v5-parity)")
    ap.add_argument("base_dir", nargs="?",
        default=os.path.expanduser("~/Desktop/actual_final_stuff_medha"))
    ap.add_argument("--proteins", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--frame-ps", type=float, default=10.0, dest="frame_ps")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--dynamic-triad", action="store_true", dest="dynamic_triad")
    ap.add_argument("--no-msa", action="store_true", dest="no_msa")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    cfg = Config(base_dir=a.base_dir, proteins=a.proteins, stride=a.stride,
                 frame_ps=a.frame_ps, sensitivity=a.sensitivity,
                 dynamic_triad=a.dynamic_triad, no_msa=a.no_msa, seed=a.seed)
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
