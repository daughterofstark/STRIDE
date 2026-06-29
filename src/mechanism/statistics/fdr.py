# Auto-extracted verbatim from v5 (final_code_piece.py) — DO NOT edit logic in M0.
import numpy as np
ALPHA = 0.05

def fdr_bh(pvals, alpha=ALPHA):
    """
    Benjamini-Hochberg FDR correction.
    Returns (reject_array, adjusted_pval_array).
    No external dependencies required.
    """
    pvals = np.asarray(pvals, dtype=float)
    n     = len(pvals)
    order = np.argsort(pvals)
    ranked_p    = pvals[order]
    thresholds  = (np.arange(1, n+1) / n) * alpha
    below       = ranked_p <= thresholds
    if not below.any():
        reject_sorted = np.zeros(n, dtype=bool)
    else:
        max_idx       = np.where(below)[0].max()
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[:max_idx+1] = True
    reject    = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    adj = np.zeros(n)
    adj[order] = np.minimum.accumulate(
        (n / np.arange(1, n+1)) * ranked_p[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    return reject, adj

