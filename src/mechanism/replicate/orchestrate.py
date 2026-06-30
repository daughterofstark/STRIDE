"""M6 orchestration: invoke the M5 aggregation after the existing pipeline runs.

This module is pure wiring. It introduces **no new mathematics**: it reads the
per-run ``{proj}_correlations_v5.csv`` files that the (unmodified) v5 engine wrote
to disk, groups them per protein across replicate run directories, and hands them
to the already-implemented M5 ``run_aggregation`` + report writers. The validated
per-run engine in ``_legacy.py`` is not touched and no existing output is altered;
the only artefacts produced here are the M5 report files (``{proj}_profile.csv``
and ``{proj}_mechanism.json``), written alongside the existing triplicate outputs.

Design constraints honoured (see milestone M6):
* reads the per-run CSVs from disk rather than any in-memory engine state, so the
  engine stays byte-for-byte frozen;
* additive only — never renames or rewrites an existing file;
* degrades to a warning (never raises) so a metadata/aggregation problem cannot
  break a validated run.
"""
from __future__ import annotations

import glob
import os
from typing import Optional, Sequence

import pandas as pd

from .aggregator import GateConfig, run_aggregation
from ..reports.mechanism_report import write_reports
from ..hierarchy.config_resolver import resolve_hierarchy_config

_CSV_SUFFIX = "_correlations_v5.csv"


def _collect_per_protein(
    base_dir: str, run_dirs: Sequence[str],
    proteins: Optional[Sequence[str]],
) -> dict[str, list[pd.DataFrame]]:
    """Group per-run effect tables by protein, in run order.

    Reads ``{base_dir}/{run}/**/{proj}{_CSV_SUFFIX}`` for each run directory. The
    protein name is recovered from the file name; ``proteins`` (when given) acts as
    a whitelist.
    """
    per_proj: dict[str, list[pd.DataFrame]] = {}
    for run in run_dirs:
        run_path = os.path.join(base_dir, run)
        if not os.path.isdir(run_path):
            continue
        matches = sorted(glob.glob(
            os.path.join(run_path, "**", f"*{_CSV_SUFFIX}"), recursive=True))
        for csv in matches:
            proj = os.path.basename(csv)[: -len(_CSV_SUFFIX)]
            if proteins and proj not in proteins:
                continue
            try:
                df = pd.read_csv(csv)
            except Exception as exc:   # unreadable CSV: skip, keep going
                print(f"  WARN [M6] could not read {csv}: {exc}")
                continue
            per_proj.setdefault(proj, []).append(df)
    return per_proj


def gate_config_from(config) -> GateConfig:
    """Build a :class:`GateConfig` from the pipeline ``Config`` (provisional values).

    ``rho_star`` is provisional and uncalibrated; the resulting mechanisms are
    stamped ``calibrated = False`` by the M5 writers.
    """
    return GateConfig(
        rho_star=getattr(config, "rho_star", 0.5),
        alpha=getattr(config, "alpha", 0.05),
        coherence_threshold=getattr(config, "coherence_threshold", 0.6),
    )


def aggregate_from_rundirs(
    base_dir: str,
    run_dirs: Sequence[str],
    proteins: Optional[Sequence[str]] = None,
    *,
    gate_config: Optional[GateConfig] = None,
    hierarchy_config_path: Optional[str] = None,
) -> dict[str, dict]:
    """Read per-run CSVs and emit the M5 mechanism reports per protein.

    Returns a mapping ``proj -> {"profile": path, "mechanism": path}`` for each
    protein for which reports were written. Proteins with fewer than two
    replicates (variance components need ``K >= 2``) are skipped with a warning.
    """
    gate_config = gate_config or GateConfig()
    per_proj = _collect_per_protein(base_dir, run_dirs, proteins)
    written: dict[str, dict] = {}

    for proj, dfs in sorted(per_proj.items()):
        if len(dfs) < 2:
            print(f"  WARN [M6] {proj}: only {len(dfs)} replicate(s); "
                  f"need K>=2 for variance components — skipping aggregation")
            continue
        try:
            hcfg = resolve_hierarchy_config(proj, hierarchy_config_path)
            profile_df, mechanisms, unresolved, meta = run_aggregation(
                dfs, hcfg, gate_config, protein=proj)
            if profile_df.empty:
                print(f"  WARN [M6] {proj}: empty profile — no report written")
                continue
            paths = write_reports(profile_df, mechanisms, unresolved, meta,
                                  base_dir, prefix=proj)
            written[proj] = paths
            print(f"  OK [M6] {proj}: {meta['n_mechanisms']} mechanism(s), "
                  f"{meta['n_unresolved']} unresolved locus/loci "
                  f"(rho*={meta['rho_star']}, uncalibrated) -> "
                  f"{os.path.basename(paths['mechanism'])}, "
                  f"{os.path.basename(paths['profile'])}")
        except Exception as exc:   # never break the validated run
            print(f"  WARN [M6] {proj}: aggregation skipped: {exc}")
            continue
    return written


def run_aggregation_tail(config) -> dict[str, dict]:
    """Entry point used by ``pipeline.run_pipeline`` after the engine completes.

    Pulls run layout and (provisional) gate settings from the pipeline ``Config``
    and aggregates from the on-disk per-run CSVs.
    """
    return aggregate_from_rundirs(
        os.path.abspath(config.base_dir),
        list(config.run_dirs),
        config.proteins,
        gate_config=gate_config_from(config),
        hierarchy_config_path=getattr(config, "hierarchy_config", None),
    )
