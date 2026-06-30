"""V0 separation guards: prove complete isolation between `validation` and `mechanism`.

These tests enforce the package's central invariant — production has no dependency
on validation, and (at V0) validation has none on production — by both static
source scanning and dynamic import accounting.
"""
import importlib
import os
import pkgutil
import re
import sys

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
_MECH = os.path.join(_SRC, "mechanism")
_VALIDATION = os.path.join(_REPO_ROOT, "validation")

_IMPORTS_VALIDATION = re.compile(r"^\s*(?:import\s+validation\b|from\s+validation\b)", re.M)
_IMPORTS_MECHANISM = re.compile(r"^\s*(?:import\s+mechanism\b|from\s+mechanism\b)", re.M)


def _py_files(root):
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


# ── static guarantees ────────────────────────────────────────────────────────
def test_production_source_never_imports_validation():
    offenders = [p for p in _py_files(_MECH)
                 if _IMPORTS_VALIDATION.search(open(p, encoding="utf-8").read())]
    assert not offenders, f"production imports validation: {offenders}"


def test_v0_validation_modules_do_not_import_mechanism():
    # At V0, no validation module imports mechanism. Tests are exempt (the
    # separation test below deliberately imports mechanism to inspect it).
    offenders = []
    for p in _py_files(_VALIDATION):
        if os.path.basename(os.path.dirname(p)) == "tests":
            continue
        if _IMPORTS_MECHANISM.search(open(p, encoding="utf-8").read()):
            offenders.append(p)
    assert not offenders, f"V0 validation modules import mechanism: {offenders}"


# ── dynamic guarantees ───────────────────────────────────────────────────────
def test_importing_mechanism_pulls_in_no_validation_module():
    before = {m for m in sys.modules if m == "validation" or m.startswith("validation.")}
    mechanism = importlib.import_module("mechanism")
    for _finder, name, _ispkg in pkgutil.walk_packages(
            mechanism.__path__, mechanism.__name__ + "."):
        try:
            importlib.import_module(name)
        except Exception:
            # optional heavy deps (e.g. MDAnalysis) may be absent in this env;
            # that is irrelevant to the validation-coupling question.
            pass
    after = {m for m in sys.modules if m == "validation" or m.startswith("validation.")}
    assert after == before, f"importing mechanism pulled in validation modules: {after - before}"


def test_public_mechanism_api_is_available_for_later_milestones():
    mechanism = importlib.import_module("mechanism")
    assert hasattr(mechanism, "Config")
    assert hasattr(mechanism, "run_pipeline")


def test_validation_package_imports_without_mechanism_on_path():
    # validation must be usable on its own; importing it must not require mechanism.
    saved_path = list(sys.path)
    saved_modules = dict(sys.modules)
    try:
        for m in list(sys.modules):
            if m == "validation" or m.startswith("validation.") \
                    or m == "mechanism" or m.startswith("mechanism."):
                del sys.modules[m]
        sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _SRC]
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import validation  # noqa: F401  (should succeed with src/ removed)
        assert "mechanism" not in sys.modules
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)
