"""Separation guards: prove the controlled isolation between `validation` and
`mechanism` (V1 boundary — tightened from V0).

At V0 no validation module imported ``mechanism`` at all. V1 opens a **single,
narrow** edge: the generator round-trips through the production estimator, which
requires translating a synthetic spec into a production ``HierarchyConfig``. That
translation is confined to exactly one module, ``validation/adapters.py``, and it
imports only the **public** hierarchy schema — never underscore-prefixed internals.

These tests enforce, by static source scanning and dynamic import accounting:

* production never imports validation (unchanged from V0);
* importing ``mechanism`` pulls in no validation module (unchanged from V0);
* the set of validation non-test modules importing ``mechanism`` is a **subset of
  {adapters.py}** (tightened: the coupling cannot spread to other modules);
* any validation import of ``mechanism`` uses only public names (no
  ``mechanism._something`` or ``from mechanism.x import _y``);
* ``validation.generate`` in particular is ``mechanism``-free;
* ``import validation`` does not require ``mechanism`` on the path.
"""
import ast
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

# The whitelist of validation non-test modules allowed to import mechanism.
_BRIDGE_MODULES = {"adapters.py"}


def _py_files(root):
    for dirpath, _dirs, files in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(dirpath, f)


def _is_test_module(path):
    return os.path.basename(os.path.dirname(path)) == "tests"


# ── static guarantees ────────────────────────────────────────────────────────
def test_production_source_never_imports_validation():
    offenders = [p for p in _py_files(_MECH)
                 if _IMPORTS_VALIDATION.search(open(p, encoding="utf-8").read())]
    assert not offenders, f"production imports validation: {offenders}"


def test_only_the_bridge_module_imports_mechanism():
    # Every validation non-test module that imports mechanism must be in the
    # bridge whitelist. This prevents the validation->production coupling from
    # spreading beyond adapters.py as later milestones grow the package.
    offenders = []
    for p in _py_files(_VALIDATION):
        if _is_test_module(p):
            continue
        if _IMPORTS_MECHANISM.search(open(p, encoding="utf-8").read()):
            if os.path.basename(p) not in _BRIDGE_MODULES:
                offenders.append(p)
    assert not offenders, (
        f"only {_BRIDGE_MODULES} may import mechanism; offenders: {offenders}")


def test_generate_module_is_mechanism_free():
    gen = os.path.join(_VALIDATION, "generate.py")
    assert os.path.exists(gen)
    assert not _IMPORTS_MECHANISM.search(open(gen, encoding="utf-8").read()), \
        "validation.generate must not import mechanism (it is the pure generator)"


def test_bridge_imports_only_public_mechanism_names():
    # adapters.py may import mechanism, but only public (non-underscore) names,
    # and only from the public hierarchy schema. Parse the AST to check both the
    # module path and every imported symbol.
    adapters = os.path.join(_VALIDATION, "adapters.py")
    tree = ast.parse(open(adapters, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("mechanism"):
            # module path components must all be public
            for comp in node.module.split("."):
                assert not comp.startswith("_"), \
                    f"bridge imports underscore module component: {node.module}"
            # imported names must be public
            for alias in node.names:
                assert not alias.name.startswith("_"), \
                    f"bridge imports underscore name: {alias.name}"
            # confine to the documented public schema module
            assert node.module == "mechanism.config.hierarchy_schema", \
                (f"bridge must import only mechanism.config.hierarchy_schema, "
                 f"got {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("mechanism") or \
                    "._" not in alias.name, \
                    f"bridge uses underscore import: {alias.name}"


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
    # validation (and its pure generator) must be usable on its own; importing it
    # must not require mechanism. The adapter is imported lazily, not at package
    # import, so `import validation` stays production-free.
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
        import validation.generate  # noqa: F401  (the pure generator, too)
        assert "mechanism" not in sys.modules
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)
