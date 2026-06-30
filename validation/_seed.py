"""Determinism layer for the validation framework (milestone V0).

All randomness in the validation framework flows through this module so every
draw, surrogate, and resample is reproducible from a recorded integer seed
(implementation-roadmap R5 / validation-roadmap VR4). No global RNG state is used,
and nothing here imports ``mechanism``.
"""
from __future__ import annotations

from typing import List

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh PCG64 ``Generator`` seeded by ``seed``.

    Uses an explicit ``SeedSequence`` so the stream depends only on ``seed`` and
    never on global interpreter state; two calls with the same ``seed`` yield
    byte-identical draws.
    """
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))


def spawn_seeds(seed: int, n: int) -> List[int]:
    """Deterministically derive ``n`` independent child seeds from ``seed``.

    Built on ``SeedSequence.spawn`` so the children are reproducible and mutually
    independent. Used later (V4) to keep calibration and evaluation draws on
    separate, non-interfering streams. Returns plain ``int`` seeds so they can be
    recorded in result provenance.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    parent = np.random.SeedSequence(int(seed))
    return [int(child.generate_state(1, dtype=np.uint64)[0])
            for child in parent.spawn(n)]
