# Auto-extracted verbatim from v5 (final_code_piece.py) — DO NOT edit logic in M0.
import numpy as np
from scipy.signal import correlate
MAX_LAG = 50

def xcorr_fn(vols, dist_col, max_lag=MAX_LAG):
    n    = min(len(vols), len(dist_col))
    v, d = np.array(vols[:n]), np.array(dist_col[:n])
    v    = (v - v.mean()) / (v.std() + 1e-12)
    d    = (d - d.mean()) / (d.std() + 1e-12)
    xc   = correlate(v, d, mode='full') / n
    mid  = len(xc) // 2
    lags = np.arange(-mid, mid + 1)
    mask = np.abs(lags) <= max_lag
    lc, xcc = lags[mask], xc[mask]
    pk = int(np.argmax(np.abs(xcc)))
    return lc, xcc, int(lc[pk]), float(xcc[pk])

