#!/usr/bin/env python3
"""Capture golden fixtures by running the ORIGINAL v5 script on a COPY of the data,
then snapshotting every produced CSV/PNG into tests/golden/ (mirroring relative paths).
Run this ONCE on a machine with the data tree, MDAnalysis and POVME available.

    python scripts/capture_golden.py --v5 reference/v5_final_code_piece.py \\
        --data /path/to/TRIPLICATES --out tests/golden
"""
import argparse, os, shutil, subprocess, sys, tempfile, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True, help="original final_code_piece.py")
    ap.add_argument("--data", required=True, help="data root (1st_run/ 2nd_run/ 3rd_run/)")
    ap.add_argument("--out", default="tests/golden")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        work = os.path.join(tmp, "data"); shutil.copytree(a.data, work)
        subprocess.run([sys.executable, a.v5, work], check=True)
        os.makedirs(a.out, exist_ok=True)
        n = 0
        for ext in ("*.csv", "*.png"):
            for f in glob.glob(os.path.join(work, "**", ext), recursive=True):
                rel = os.path.relpath(f, work)
                dst = os.path.join(a.out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst); n += 1
        print(f"captured {n} golden files into {a.out}")

if __name__ == "__main__":
    main()
