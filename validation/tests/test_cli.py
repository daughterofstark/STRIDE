"""V7 tests: the command-line runner (validation.cli / python -m validation).

Verify the CLI parses, reports its version, runs a tiny deterministic sweep, and
re-runs reproducibly. No empirical value is frozen.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

from validation.cli import build_parser, main
from validation.systems import non_denv_systems

_NONDENV = [d.name for d in non_denv_systems()]


def test_parser_has_v7_subcommands():
    parser = build_parser()
    # subcommands run / calibrate / sweep are all present
    sub = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    assert sub, "no subparsers registered"
    choices = set(sub[0].choices)
    assert {"run", "calibrate", "sweep"} <= choices


def test_main_no_command_prints_help_returns_zero(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "validation" in out


def test_cli_sweep_writes_store_and_manifest():
    path = tempfile.mktemp(suffix=".jsonl")
    try:
        rc = main(["sweep", "--systems", *_NONDENV, "--K", "5",
                   "--beta2", "0.36", "--n-eval", "15", "--out", path])
        assert rc == 0
        assert os.path.exists(path)
        # one JSON record per cell, each with provenance
        with open(path) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == len(_NONDENV)
        rec = json.loads(lines[0])
        assert "provenance" in rec and "rho_star" in rec
        man_path = os.path.splitext(path)[0] + "_manifest.json"
        assert os.path.exists(man_path)
        with open(man_path) as fh:
            man = json.load(fh)
        assert man["n_non_denv_systems"] >= 2
    finally:
        for p in (path, os.path.splitext(path)[0] + "_manifest.json"):
            if os.path.exists(p):
                os.remove(p)


def test_cli_sweep_rerun_is_byte_identical():
    p1 = tempfile.mktemp(suffix=".jsonl")
    p2 = tempfile.mktemp(suffix=".jsonl")
    try:
        args = ["sweep", "--systems", _NONDENV[0], "--K", "5", "--beta2", "0.36",
                "--n-eval", "15", "--out", None]
        main(args[:-1] + [p1])
        main(args[:-1] + [p2])
        with open(p1) as a, open(p2) as b:
            assert a.read() == b.read()
    finally:
        for p in (p1, p2, os.path.splitext(p1)[0] + "_manifest.json",
                  os.path.splitext(p2)[0] + "_manifest.json"):
            if os.path.exists(p):
                os.remove(p)


def test_cli_run_single_cell(capsys):
    rc = main(["run", "--system", _NONDENV[0], "--K", "5", "--beta2", "0.36",
               "--n-eval", "15"])
    assert rc == 0
    out = capsys.readouterr().out
    rec = json.loads(out)
    assert rec["system"] == _NONDENV[0]


def test_python_m_validation_version():
    # the __main__ module must make `python -m validation --version` work
    proc = subprocess.run([sys.executable, "-m", "validation", "--version"],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    assert "validation" in (proc.stdout + proc.stderr)


# ── V8: report subcommand ────────────────────────────────────────────────────
def test_cli_has_report_subcommand():
    parser = build_parser()
    sub = [a for a in parser._actions if hasattr(a, "choices") and a.choices][0]
    assert "report" in sub.choices


def test_cli_report_builds_package(tmp_path):
    rc = main(["report", "--out", str(tmp_path / "pkg")])
    assert rc == 0
    assert os.path.exists(tmp_path / "pkg" / "VALIDATION_AND_BENCHMARKING.md")
    assert len(list((tmp_path / "pkg" / "figures").glob("*.png"))) == 7
    assert len(list((tmp_path / "pkg" / "tables").glob("*.md"))) == 5
