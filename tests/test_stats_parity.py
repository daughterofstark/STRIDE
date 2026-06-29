"""M0 parity: extracted pure functions are byte-identical to v5 and functionally correct."""
import ast, numpy as np, pytest
from mechanism.statistics import fdr_bh, pearson_both, block_conv_sem, xcorr_fn, detect_near_closure_events
from mechanism.effects import canon_label
import mechanism.statistics.fdr, mechanism.statistics.correlation, mechanism.statistics.convergence
import mechanism.statistics.xcorr, mechanism.statistics.events, mechanism.effects.labels

PURE = {
 "fdr_bh": mechanism.statistics.fdr,
 "pearson_both": mechanism.statistics.correlation,
 "block_conv_sem": mechanism.statistics.convergence,
 "xcorr_fn": mechanism.statistics.xcorr,
 "detect_near_closure_events": mechanism.statistics.events,
 "canon_label": mechanism.effects.labels,
}

def _seg(src, name):
    t = ast.parse(src)
    for n in t.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise KeyError(name)

@pytest.mark.parametrize("name", list(PURE))
def test_source_identity(name, v5_source_path):
    """The extracted function body is exactly the v5 source (no transcription drift)."""
    v5 = open(v5_source_path).read()
    extracted = open(PURE[name].__file__).read()
    assert _seg(v5, name) == _seg(extracted, name), f"{name} drifted from v5"

def test_fdr_bh_known():
    rej, adj = fdr_bh(np.array([0.001, 0.01, 0.5, 0.9]))
    assert rej.tolist() == [True, True, False, False]
    assert np.all((adj >= 0) & (adj <= 1))
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]  # monotone after BH step-up

def test_pearson_both_perfect_corr():
    x = np.arange(50, dtype=float); y = 2*x + 1
    r, ar, p_raw, p_bonf, p_fdr, sig = pearson_both(x, y, n_bonf=10)
    assert abs(r - 1.0) < 1e-9 and ar == abs(r) and sig is True

def test_pearson_both_constant():
    x = np.zeros(20); y = np.arange(20, dtype=float)
    assert pearson_both(x, y, 10) == (0.0, 0.0, 1.0, 1.0, 1.0, False)

def test_xcorr_lag_recovers_shift():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(400)
    d = np.roll(v, 7)                      # d lags v by 7
    lags, xc, opt, peak = xcorr_fn(list(v), list(d))
    assert abs(opt) == 7

def test_block_conv_sem_shape():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(400); d = v + 0.1*rng.standard_normal(400)
    brs, mean_r, sem, ci, conv = block_conv_sem(v, d, n=4)
    assert len(brs) == 4 and 0 <= mean_r <= 1 and isinstance(conv, bool)

def test_detect_near_closure_events():
    v = np.array([10,10,1,1,10,1,10], dtype=float)
    thr, events = detect_near_closure_events(v, pct=30)
    assert all(e[2] == e[1]-e[0] for e in events)

class _Res:
    def __init__(self, name, rid): self.resname=name; self.resid=rid
def test_canon_label():
    assert canon_label(_Res("HIS", 98), off=47) == "HIS51"
