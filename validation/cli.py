"""Command-line entry point for the validation framework (completed at V7).

Subcommands:

* ``run``       — execute a single sweep cell and print its record (orchestration).
* ``calibrate`` — the explicit V4 calibration step: compute and persist a
  ``rho_star_<system>_K<K>.yaml`` artifact (this is the *only* place a sweep's
  calibration inputs are produced; sweeps never calibrate).
* ``sweep``     — run a deterministic grid and persist a results store.

Reproducibility: every subcommand is deterministic in its seeds. ``sweep`` and
``run`` are **load-only** for calibration — a missing artifact is a clear error, not
an implicit calibration. Nothing here imports ``mechanism`` (production is reached
only via the lazily-imported ``adapters`` bridge inside ``experiments``/``calibrate``).
"""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validation",
        description="STRIDE validation & benchmarking framework (Phase 2).")
    parser.add_argument("--version", action="version",
                        version=f"validation {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="run a single sweep cell (orchestration)")
    p_run.add_argument("--system", required=True)
    p_run.add_argument("--K", type=int, default=5)
    p_run.add_argument("--T", type=int, default=0)
    p_run.add_argument("--tau2", type=float, default=0.0)
    p_run.add_argument("--beta2", type=float, default=0.36)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--n-eval", type=int, default=60)

    p_cal = sub.add_parser("calibrate",
                           help="(V4) calibrate rho* and persist the artifact")
    p_cal.add_argument("--system", required=True)
    p_cal.add_argument("--K", type=int, required=True)
    p_cal.add_argument("--alpha", type=float, default=0.05)
    p_cal.add_argument("--base-seeds", type=int, default=40)
    p_cal.add_argument("--surr-per-base", type=int, default=10)
    p_cal.add_argument("--seed", type=int, default=1234)
    p_cal.add_argument("--out", default=None)

    p_sw = sub.add_parser("sweep", help="run a deterministic grid and persist results")
    p_sw.add_argument("--systems", nargs="+", required=True)
    p_sw.add_argument("--K", type=int, nargs="+", default=[5])
    p_sw.add_argument("--T", type=int, nargs="+", default=[0])
    p_sw.add_argument("--tau2", type=float, nargs="+", default=[0.0])
    p_sw.add_argument("--beta2", type=float, nargs="+", default=[0.36])
    p_sw.add_argument("--seed", type=int, default=0)
    p_sw.add_argument("--n-seeds", type=int, default=1)
    p_sw.add_argument("--n-eval", type=int, default=60)
    p_sw.add_argument("--out", required=True)

    p_rep = sub.add_parser(
        "report", help="(V8) build publication figures, tables, and the report "
                       "from the frozen results store")
    p_rep.add_argument("--out", required=True,
                       help="output directory for the reproducibility package")
    return parser


def _cmd_run(args) -> int:
    from .experiments import run_cell
    from .types import SweepCell
    cell = SweepCell(system=args.system, K=args.K, T=args.T, tau2=args.tau2,
                     beta2=args.beta2, seed=args.seed)
    rec = run_cell(cell, n_eval=args.n_eval)
    print(json.dumps(rec.to_dict(), sort_keys=True, indent=2))
    return 0


def _cmd_calibrate(args) -> int:
    from .systems import get_system
    from .generate import generate_system
    from .adapters import to_hierarchy_config
    from ._seed import spawn_seeds
    from .calibrate import calibrate_rho_star, write_rho_star_yaml
    from .experiments import rho_star_artifact_path

    sysdef = get_system(args.system)

    def mb(s):
        return list(generate_system(
            sysdef.build(seed=s, K=args.K, T=0, tau2=0.0, beta2=0.0,
                         driver=False)).per_run_dfs)

    cfg = to_hierarchy_config(
        sysdef.build(seed=1, K=args.K, T=0, tau2=0.0, beta2=0.0, driver=False))
    res = calibrate_rho_star(
        mb, cfg, system=args.system, K=args.K, T=0,
        base_seeds=spawn_seeds(args.seed, args.base_seeds),
        surr_per_base=args.surr_per_base, alpha=args.alpha, seed=args.seed,
        protein=args.system)
    out = args.out or rho_star_artifact_path(args.system, args.K)
    write_rho_star_yaml(res, out)
    print(f"wrote calibrated rho* -> {out}")
    print(json.dumps({k: round(v, 4) for k, v in res.rho_star.items()},
                     sort_keys=True))
    return 0


def _cmd_sweep(args) -> int:
    from .experiments import (sweep_grid, run_sweep, ResultStore, build_manifest,
                              results_digest)
    cells = sweep_grid(args.systems, Ks=args.K, Ts=args.T, tau2s=args.tau2,
                       beta2s=args.beta2, seed=args.seed, n_seeds=args.n_seeds)
    records = run_sweep(cells, n_eval=args.n_eval)
    store = ResultStore(args.out)
    store.write(records, manifest=build_manifest(cells, seed=args.seed,
                                                 n_eval=args.n_eval))
    print(f"wrote {len(records)} records -> {args.out}")
    print(f"results_digest: {results_digest(records)}")
    return 0


def _cmd_report(args) -> int:
    from .report import build_package
    man = build_package(args.out)
    print(f"wrote report -> {man['report']}")
    print(f"  figures: {len(man['figures'])}, tables: {len(man['tables'])}")
    print(f"  results_digest: {man['results_digest']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    if args.command == "sweep":
        return _cmd_sweep(args)
    if args.command == "report":
        return _cmd_report(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
