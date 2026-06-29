# Auto-extracted verbatim from v5 (final_code_piece.py) — DO NOT edit logic in M0.
import numpy as np

def detect_near_closure_events(volumes, pct=10):
    thresh  = np.percentile(volumes, pct)
    near_cl = volumes < thresh
    events, in_ev, start_f = [], False, 0
    for fi in range(len(volumes)):
        if near_cl[fi] and not in_ev:
            start_f, in_ev = fi, True
        elif not near_cl[fi] and in_ev:
            events.append((start_f, fi, fi - start_f))
            in_ev = False
    if in_ev:
        events.append((start_f, len(volumes), len(volumes) - start_f))
    return thresh, events

