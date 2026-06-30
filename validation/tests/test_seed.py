"""V0 determinism: seeded RNG reproducibility and independent child streams."""
import numpy as np

from validation import make_rng, spawn_seeds


def test_make_rng_same_seed_byte_identical():
    a = make_rng(123).random(8)
    b = make_rng(123).random(8)
    assert np.array_equal(a, b)


def test_make_rng_different_seeds_differ():
    a = make_rng(1).random(8)
    b = make_rng(2).random(8)
    assert not np.array_equal(a, b)


def test_make_rng_uses_no_global_state():
    # interleaving draws from two generators must not affect reproducibility
    g1 = make_rng(7)
    _ = np.random.random(5)          # perturb global RNG
    g2 = make_rng(7)
    assert np.array_equal(g1.random(4), g2.random(4))


def test_spawn_seeds_reproducible():
    assert spawn_seeds(42, 5) == spawn_seeds(42, 5)


def test_spawn_seeds_distinct_and_independent():
    seeds = spawn_seeds(42, 6)
    assert len(set(seeds)) == 6                       # distinct
    first_draws = [make_rng(s).random() for s in seeds]
    assert len(set(first_draws)) == 6                 # independent streams


def test_spawn_seeds_zero_and_negative():
    assert spawn_seeds(5, 0) == []
    try:
        spawn_seeds(5, -1)
        raised = False
    except ValueError:
        raised = True
    assert raised
