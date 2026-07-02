"""Manuscript tables for the V8 publication layer (reads frozen artifacts only).

Each function returns a :class:`Table` (header + rows of stringified cells) built from
the persisted V4/V5/V6/V7 artifacts, and can render to Markdown or CSV. **No numbers
are recomputed** — every cell is a value already present in a frozen store. Nothing
here imports ``mechanism``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ._artifacts import (
    load_sweep_records, load_metrics, load_method_comparison, load_calibration,
    available_calibrations,
)


@dataclass(frozen=True)
class Table:
    """A simple titled table: a header row and stringified data rows."""

    title: str
    header: tuple
    rows: tuple

    def to_markdown(self) -> str:
        head = "| " + " | ".join(self.header) + " |"
        sep = "| " + " | ".join("---" for _ in self.header) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in self.rows)
        return f"**{self.title}**\n\n{head}\n{sep}\n{body}\n"

    def to_csv(self) -> str:
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.header)
        for r in self.rows:
            w.writerow(r)
        return buf.getvalue()


def _f(x, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# ── Table 1: calibrated rho* by system / K / scale ───────────────────────────
def table_calibration(artifact_dir: Optional[str] = None) -> Table:
    """Calibrated rho* (at each system's true scale) by system and K."""
    from .systems import SYSTEMS
    # map calibration-file key -> (system name, true scale) via the registry
    key_to_system = {}
    for name, d in SYSTEMS.items():
        key = d.calibration_key or name
        key_to_system[key] = (name, d.true_scale_level)
    rows = []
    for (key, K) in available_calibrations(artifact_dir):
        cal = load_calibration(key, K, artifact_dir)
        system, scale = key_to_system.get(key, (key, "domain"))
        rho = cal["rho_star"].get(scale)
        ci = cal.get("rho_star_ci", {}).get(scale, [None, None])
        rows.append((system, str(K), scale, _f(rho),
                     f"[{_f(ci[0])}, {_f(ci[1])}]"))
    rows.sort(key=lambda t: (t[0], int(t[1])))
    return Table("Table 1. Calibrated reproducibility threshold rho* "
                 "(true scale) by system and replicate count K",
                 ("system", "K", "true scale", "rho*", "95% CI"), tuple(rows))


# ── Table 2: empirical vs predicted operating characteristics (DENV) ─────────
def table_empirical_vs_predicted(artifact_dir: Optional[str] = None) -> Table:
    m = load_metrics(artifact_dir)
    rows = []
    for op in m["operating_points"]:
        rows.append((str(op["K"]), _f(op["beta2"], 2), _f(op["rho_true"]),
                     _f(op["empirical_power"]), _f(op["predicted_power"]),
                     _f(op["power_diff"]), _f(op["empirical_fpr"]),
                     _f(op["predicted_fpr"])))
    rows.sort(key=lambda t: (int(t[0]), float(t[1])))
    return Table("Table 2. Empirical vs predicted operating characteristics "
                 "(DENV, domain scale, at calibrated rho*)",
                 ("K", "beta^2", "rho_true", "emp power", "pred power",
                  "power diff", "emp FPR", "pred FPR"), tuple(rows))


# ── Table 3: over-resolution comparison + paired tests (Part VII) ────────────
def table_over_resolution(artifact_dir: Optional[str] = None) -> Table:
    c = load_method_comparison(artifact_dir)
    rows = []
    for K in sorted(c["comparisons_by_K"], key=int):
        cell = c["comparisons_by_K"][K]
        stride = cell["stride_over_resolution_rate"]
        for name, comp in sorted(cell["comparisons"].items()):
            rows.append((str(K), name, _f(stride), _f(comp["baseline_rate"]),
                         f"{comp['mcnemar_p']:.1e}",
                         _f(comp["paired_bootstrap_diff"])))
    return Table("Table 3. Over-resolution on planted nulls: STRIDE vs baselines "
                 "(Part VII), with paired McNemar / bootstrap",
                 ("K", "baseline", "STRIDE rate", "baseline rate",
                  "McNemar p", "paired diff"), tuple(rows))


# ── Table 4: naive SD/sqrt(K) coverage by K (Part IV coverage bullet) ────────
def table_coverage(artifact_dir: Optional[str] = None) -> Table:
    c = load_method_comparison(artifact_dir)
    rows = []
    for K in sorted(c["naive_coverage_by_K"], key=int):
        rows.append((str(K), _f(c["naive_coverage_by_K"][K]), "0.950"))
    return Table("Table 4. Naive ensemble SD/sqrt(K) interval coverage by K "
                 "(nominal 0.95)", ("K", "naive coverage", "nominal"),
                 tuple(rows))


# ── Table 5: hierarchy sensitivity across systems (>=2 non-DENV) ─────────────
def table_hierarchy_sensitivity(artifact_dir: Optional[str] = None) -> Table:
    recs = load_sweep_records(artifact_dir)
    # summarize each system at a representative moderate SNR cell per K
    rows = []
    for r in sorted(recs, key=lambda x: (x["system"], x["K"], x["beta2"])):
        rows.append((r["system"], r["scale_level"], str(r["K"]),
                     _f(r["beta2"], 2), _f(r["empirical_power"]),
                     _f(r["empirical_fpr"]),
                     _f(r["stride_over_resolution_rate"])))
    return Table("Table 5. Hierarchy sensitivity: per-system operating "
                 "characteristics across the sweep grid",
                 ("system", "true scale", "K", "beta^2", "emp power",
                  "emp FPR", "over-res rate"), tuple(rows))


def all_tables(artifact_dir: Optional[str] = None) -> list:
    """Return every manuscript table in report order."""
    return [
        table_calibration(artifact_dir),
        table_empirical_vs_predicted(artifact_dir),
        table_over_resolution(artifact_dir),
        table_coverage(artifact_dir),
        table_hierarchy_sensitivity(artifact_dir),
    ]


def write_tables(out_dir: str, artifact_dir: Optional[str] = None) -> list:
    """Write each table as a Markdown file; return the list of paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, tbl in enumerate(all_tables(artifact_dir), start=1):
        p = os.path.join(out_dir, f"table_{i}.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(tbl.to_markdown())
        paths.append(p)
    return paths
