"""Skeleton command-line entry point for the validation framework (milestone V0).

Only argument parsing and a placeholder subcommand registry exist at V0; the real
subcommands are implemented in later milestones (calibrate -> V4, sweep -> V7,
report -> V8). Nothing here imports ``mechanism``.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__

_PLACEHOLDERS = [
    ("calibrate", "(V4) empirically calibrate rho* by the null-surrogate quantile"),
    ("sweep", "(V7) run the experiment sweep and persist results"),
    ("report", "(V8) build the publication figures and report"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validation",
        description="STRIDE validation & benchmarking framework (Phase 2).")
    parser.add_argument("--version", action="version",
                        version=f"validation {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in _PLACEHOLDERS:
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    print(f"validation: subcommand '{args.command}' is not implemented yet "
          f"(scheduled for a later milestone).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
