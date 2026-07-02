"""V8 tests: publication figures (validation.figures) render from frozen artifacts.

Verify each figure renders to a file without recomputation, and that the figure module
reads persisted artifacts only (no generator/estimator import path). Images are checked
for successful production and non-emptiness — NOT pixel-hashed (fragile across
matplotlib builds); the data behind figures is validated in test_tables.
"""
import os

import pytest

from validation import figures as F


def test_all_figures_render(tmp_path):
    paths = F.build_all_figures(str(tmp_path))
    assert len(paths) == 7
    for p in paths:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 1000     # a real PNG, not empty


def test_individual_figures_render(tmp_path):
    for fn, name in [
        (F.fig_calibration_curve, "cal.png"),
        (F.fig_empirical_vs_predicted, "evp.png"),
        (F.fig_coverage, "cov.png"),
        (F.fig_ell_min_heatmap, "ell.png"),
        (F.fig_over_resolution_comparison, "over.png"),
        (F.fig_hierarchy_sensitivity, "hier.png"),
    ]:
        p = fn(str(tmp_path / name))
        assert os.path.exists(p) and os.path.getsize(p) > 1000


def test_profile_schematic_renders(tmp_path):
    p = F.fig_profile_schematic(str(tmp_path / "prof.png"))
    assert os.path.exists(p) and os.path.getsize(p) > 1000


def test_figures_module_has_no_mechanism_or_generator_import():
    # figures must read frozen artifacts, not recompute; assert it doesn't pull in
    # the generators / estimator bridge at import time
    import sys
    import importlib
    # fresh import check: figures should not import generate/adapters/experiments
    src = importlib.util.find_spec("validation.figures").origin
    with open(src) as fh:
        text = fh.read()
    for forbidden in ("from .generate", "from .adapters", "from .experiments",
                      "import mechanism", "from mechanism"):
        assert forbidden not in text, f"figures.py must not use {forbidden!r}"


def test_figures_render_twice_produce_files(tmp_path):
    # determinism at the file level: rendering twice yields the same file set
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    p1 = F.build_all_figures(str(d1))
    p2 = F.build_all_figures(str(d2))
    assert [os.path.basename(p) for p in p1] == [os.path.basename(p) for p in p2]
