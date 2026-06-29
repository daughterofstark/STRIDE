# Auto-extracted verbatim from v5 (final_code_piece.py) — DO NOT edit logic in M0.
import numpy as np
from scipy.stats import pearsonr
N_BLOCKS = 4
CONV_WARN = 0.12

def block_conv_sem(vols, dist_col, n=N_BLOCKS):
    """
    Compute per-block |r|, block SEM, and 95% CI width.  [R7]
    Returns (block_rs array, mean_r, sem_r, ci_half_width, converged_bool)
    """
    sz   = len(vols) // n
    brs  = []
    for b in range(n):
        sl_v = vols[b*sz:(b+1)*sz]
        sl_d = dist_col[b*sz:(b+1)*sz]
        if np.std(sl_d) > 1e-6:
            brs.append(abs(pearsonr(sl_v, sl_d)[0]))
        else:
            brs.append(0.0)
    brs  = np.array(brs)
    sem  = float(np.std(brs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    # t-CI at 95%, df = n-1
    from scipy.stats import t as t_dist
    ci_hw = float(t_dist.ppf(0.975, df=n-1) * sem) if n > 1 else np.nan
    return brs, float(brs.mean()), sem, ci_hw, (float(np.std(brs)) < CONV_WARN)

