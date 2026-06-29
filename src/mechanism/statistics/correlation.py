# Auto-extracted verbatim from v5 (final_code_piece.py) — DO NOT edit logic in M0.
import numpy as np
from scipy.stats import pearsonr
ALPHA = 0.05

def pearson_both(x, y, n_bonf):
    """
    Pearson r with both Bonferroni and BH-FDR p-values.
    Returns: r, abs_r, p_raw, p_bonf, p_fdr (raw), sig_bonf
    Note: sig_fdr is computed in batch after all p_raw are collected.
    """
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0, 0.0, 1.0, 1.0, 1.0, False
    r, p  = pearsonr(x, y)
    p_b   = min(float(p) * n_bonf, 1.0)
    return float(r), abs(float(r)), float(p), p_b, float(p), (p_b < ALPHA)

