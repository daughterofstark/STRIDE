from .fdr import fdr_bh
from .correlation import pearson_both
from .convergence import block_conv_sem
from .xcorr import xcorr_fn
from .events import detect_near_closure_events
from .neff import (
    NeffResult,
    autocorr_fft,
    integrated_autocorr_time,
    effective_sample_size,
    corrected_standard_error,
    attach_effective_sample_size,
)
from .bootstrap import (
    BootstrapResult,
    bootstrap_correlation,
    attach_bootstrap_ci,
    select_block_length,
    circular_block_indices,
    stationary_block_indices,
)
