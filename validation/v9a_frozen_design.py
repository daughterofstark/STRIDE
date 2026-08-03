"""V9A frozen-design descriptive characterization.

This validation-only module reads a neutral snapshot extracted from frozen
per-run/profile tables, generates the two pre-registered K=3 independent-error
benchmarks, and evaluates magnitude rho through :mod:`validation.adapters`.
It does not calibrate a cutoff, modify production code, or write frozen inputs.

Inputs
------
``outputs/replicate_table.parquet`` supplies the three per-item effects and the
marginal uncertainty (finite bootstrap SE, otherwise finite fallback SE).
``outputs/stride_table.parquet`` supplies the exact seven-level support paths.

Outputs
-------
The CLI writes only the eight pre-registered V9A files: this module and its test
live in source control; the neutral snapshot, results, three CSV summaries, and
manifest live below ``validation/artifacts/v9a``.

Assumptions
-----------
Both regimes use K=3, independent item/run errors, zero off-diagonal covariance,
fixed known marginal variances, fixed support membership, and no mapping
uncertainty. They characterize the implemented descriptive statistic; neither
regime models the real upstream process or calibrates the provisional 0.5
reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .generate import (
    SynChain,
    SynDomain,
    SyntheticSystemSpec,
    build_per_run_frame,
)


STRIDE_COMMIT = "b06f7d1d40b5d860915cf1e1f046a6392f71a09b"
DOWNSTREAM_COMMIT = "4388dbcefaaa1672b63cc7bbce357e1b4ec40b71"
MASTER_SEED = 2026072801
N_DRAWS = 500
K = 3
RHO_STAR = 0.5
LEVELS_COARSE_TO_FINE = (
    "complex",
    "protein",
    "chain",
    "domain",
    "motif",
    "secondary_structure",
    "residue",
)
LEVELS_FINE_TO_COARSE = tuple(reversed(LEVELS_COARSE_TO_FINE))
REGIMES = ("zero_mean", "frozen_scale_fixed_mean")
DISPLAY_STRATA = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129+", 129, math.inf),
)
REPLICATE_REQUIRED = (
    "serotype",
    "replicate_index",
    "canon_label",
    "r",
    "theta_bootstrap_se",
    "theta_se",
    "chain",
    "domain",
)
STRIDE_REQUIRED = (
    "serotype",
    "canon_label",
    "scale_level",
    "scale_index",
    "region_id",
    "h_complex",
    "h_protein",
    "h_chain",
    "h_domain",
    "h_motif",
    "h_secondary_structure",
    "h_residue",
)
SNAPSHOT_FILENAME = "v9a_frozen_design_input.json"
RESULTS_FILENAME = "v9a_frozen_design_results.json"
SUPPORT_SIZE_FILENAME = "v9a_frozen_design_support_size.csv"
FIRST_CROSSING_FILENAME = "v9a_frozen_design_first_crossing.csv"
ALIAS_FILENAME = "v9a_frozen_design_alias_effects.csv"
MANIFEST_FILENAME = "v9a_frozen_design_manifest.json"


class SnapshotError(RuntimeError):
    """Frozen snapshot does not satisfy the pre-registered design."""


class AliasInvariantError(RuntimeError):
    """Collapsing exact aliases changed selected membership or rho."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n").encode("ascii")


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_bytes(_json_bytes(value))


def _neutral_sort(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["chain", "canon_label"], kind="mergesort"
    ).reset_index(drop=True)


def _support_key(members: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(str(x) for x in members))


def _t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    mean = float(vals.mean())
    if vals.size == 1:
        return (mean, mean)
    sem = float(stats.sem(vals))
    half = float(stats.t.ppf((1.0 + confidence) / 2.0, vals.size - 1) * sem)
    return (mean - half, mean + half)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    return None if vals.size == 0 else float(np.quantile(vals, q))


def deterministic_child_seeds(
    master_seed: int, dataset_ids: Sequence[str], regimes: Sequence[str],
    draws: int,
) -> dict[tuple[str, str], list[int]]:
    """Spawn deterministic independent uint64 seeds for every cell and draw."""
    cells = [(d, r) for d in sorted(dataset_ids) for r in regimes]
    cell_seqs = np.random.SeedSequence(int(master_seed)).spawn(len(cells))
    out: dict[tuple[str, str], list[int]] = {}
    for cell, seq in zip(cells, cell_seqs):
        children = seq.spawn(int(draws))
        out[cell] = [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in children
        ]
    return out


def _resolve_uncertainty(bootstrap_se: float, fallback_se: float) -> tuple[float, str]:
    """Apply the frozen uncertainty preference rule to one run-item row."""
    bootstrap = float(bootstrap_se)
    fallback = float(fallback_se)
    if np.isfinite(bootstrap):
        se = bootstrap
        source = "theta_bootstrap_se"
    elif np.isfinite(fallback):
        se = fallback
        source = "theta_se"
    else:
        raise SnapshotError("no finite uncertainty")
    if se <= 0:
        raise SnapshotError("nonpositive uncertainty")
    return float(se), source


def extract_snapshot(frozen_root: str | Path) -> dict:
    """Extract and verify the neutral frozen-design snapshot.

    Raw labels and coordinates are replaced by deterministic neutral identifiers.
    The snapshot retains exact support membership, K=3 marginal variances, and
    each item's frozen three-run arithmetic mean.
    """
    frozen_root = Path(frozen_root).resolve()
    replicate_path = frozen_root / "outputs" / "replicate_table.parquet"
    stride_path = frozen_root / "outputs" / "stride_table.parquet"
    if not replicate_path.is_file() or not stride_path.is_file():
        raise SnapshotError(
            "expected outputs/replicate_table.parquet and "
            "outputs/stride_table.parquet under frozen root"
        )
    rep = pd.read_parquet(replicate_path)
    profile = pd.read_parquet(stride_path)
    missing_rep = sorted(set(REPLICATE_REQUIRED) - set(rep.columns))
    missing_profile = sorted(set(STRIDE_REQUIRED) - set(profile.columns))
    if missing_rep or missing_profile:
        raise SnapshotError(
            f"missing required columns: replicate={missing_rep}, "
            f"profile={missing_profile}"
        )
    if rep.duplicated(["serotype", "replicate_index", "canon_label"]).any():
        raise SnapshotError("duplicate dataset/run/item rows in replicate table")
    if profile.duplicated(["serotype", "canon_label", "scale_level"]).any():
        raise SnapshotError("duplicate dataset/item/level rows in profile table")

    source_datasets = sorted(rep["serotype"].astype(str).unique())
    if len(source_datasets) != 4:
        raise SnapshotError(f"expected four datasets, observed {source_datasets}")
    datasets = []
    global_source_counts = Counter()

    for dataset_number, source_dataset in enumerate(source_datasets, start=1):
        dataset_id = f"dataset_{dataset_number}"
        rd = rep[rep["serotype"].astype(str) == source_dataset].copy()
        pdataset = profile[
            profile["serotype"].astype(str) == source_dataset
        ].copy()
        base = _neutral_sort(
            rd.drop_duplicates(["canon_label"])[
                ["canon_label", "chain", "domain"]
            ]
        )
        item_count = len(base)
        expected_counts = (236, 247, 242, 199)
        if item_count != expected_counts[dataset_number - 1]:
            raise SnapshotError(
                f"{dataset_id}: expected {expected_counts[dataset_number - 1]} "
                f"items, observed {item_count}"
            )

        chain_values = sorted(base["chain"].astype(str).unique())
        chain_map = {value: f"chain_{i + 1}" for i, value in enumerate(chain_values)}
        item_map = {}
        neutral_items = []
        chain_offsets = {
            value: (i + 1) * 1000 for i, value in enumerate(chain_values)
        }
        chain_positions = Counter()
        for item_number, (_, row) in enumerate(base.iterrows(), start=1):
            source_item = str(row["canon_label"])
            chain_source = str(row["chain"])
            chain_positions[chain_source] += 1
            canon_resid = chain_offsets[chain_source] + chain_positions[chain_source]
            neutral_id = f"item_{item_number:04d}"
            item_map[source_item] = {
                "item_id": neutral_id,
                "canon_resid": int(canon_resid),
                "chain_id": chain_map[chain_source],
                "source_chain": chain_source,
                "source_domain": str(row["domain"]),
            }

        group_keys = sorted({
            (item_map[str(row.canon_label)]["chain_id"], str(row.domain))
            for row in base.itertuples(index=False)
        })
        group_map = {
            key: f"group_{i + 1:02d}" for i, key in enumerate(group_keys)
        }
        run_indices = sorted(int(x) for x in rd["replicate_index"].unique())
        if run_indices != [1, 2, 3]:
            raise SnapshotError(
                f"{dataset_id}: expected run indices [1,2,3], observed {run_indices}"
            )

        for source_item in base["canon_label"].astype(str):
            meta = item_map[source_item]
            rows = rd[rd["canon_label"].astype(str) == source_item].sort_values(
                "replicate_index"
            )
            if len(rows) != K:
                raise SnapshotError(
                    f"{dataset_id}/{meta['item_id']}: expected K=3, observed {len(rows)}"
                )
            resolved_se = []
            sources = []
            effects = []
            for row in rows.itertuples(index=False):
                try:
                    se, source = _resolve_uncertainty(
                        row.theta_bootstrap_se, row.theta_se
                    )
                except SnapshotError as exc:
                    raise SnapshotError(
                        f"{dataset_id}/{meta['item_id']}: {exc}"
                    ) from exc
                resolved_se.append(float(se))
                sources.append(source)
                effects.append(float(row.r))
                global_source_counts[source] += 1
            group_key = (meta["chain_id"], meta["source_domain"])
            neutral_items.append({
                "item_id": meta["item_id"],
                "canon_resid": meta["canon_resid"],
                "chain_id": meta["chain_id"],
                "group_id": group_map[group_key],
                "mu_frozen": float(np.mean(effects)),
                "sigma2_by_run": [float(x * x) for x in resolved_se],
                "uncertainty_source_by_run": sources,
            })

        item_by_source = {
            source: item_map[source]["item_id"] for source in item_map
        }
        item_rows = pdataset[
            pdataset["scale_level"].astype(str) == "residue"
        ][["canon_label"]].copy()
        if set(item_rows["canon_label"].astype(str)) != set(item_by_source):
            raise SnapshotError(f"{dataset_id}: profile and replicate item sets differ")

        items_by_chain = defaultdict(list)
        items_by_group = defaultdict(list)
        for item in neutral_items:
            items_by_chain[item["chain_id"]].append(item["item_id"])
            items_by_group[item["group_id"]].append(item["item_id"])
        all_items = tuple(sorted(item["item_id"] for item in neutral_items))

        named_objects = []
        support_to_id: dict[tuple[str, ...], str] = {}
        support_catalog: dict[str, dict] = {}

        def register(
            level: str, region_id: str, label: str, members: Iterable[str]
        ) -> str:
            key = _support_key(members)
            if key not in support_to_id:
                support_id = f"support_{len(support_to_id) + 1:04d}"
                support_to_id[key] = support_id
                support_catalog[support_id] = {
                    "support_id": support_id,
                    "members": list(key),
                    "support_size": len(key),
                    "named_representations": [],
                }
            support_id = support_to_id[key]
            object_id = f"object_{len(named_objects) + 1:04d}"
            obj = {
                "object_id": object_id,
                "level": level,
                "region_id": region_id,
                "label": label,
                "support_id": support_id,
            }
            named_objects.append(obj)
            support_catalog[support_id]["named_representations"].append({
                "object_id": object_id,
                "level": level,
                "region_id": region_id,
            })
            return object_id

        root_complex = register("complex", "complex", "complex", all_items)
        root_protein = register(
            "protein", "complex/protein", "protein", all_items
        )
        chain_objects = {}
        group_objects = {}
        motif_objects = {}
        ss_objects = {}
        residue_objects = {}
        for chain_id in sorted(items_by_chain):
            chain_objects[chain_id] = register(
                "chain",
                f"complex/protein/{chain_id}",
                chain_id,
                items_by_chain[chain_id],
            )
        for group_id in sorted(items_by_group):
            members = items_by_group[group_id]
            chain_id = next(
                item["chain_id"] for item in neutral_items
                if item["group_id"] == group_id
            )
            base_path = f"complex/protein/{chain_id}/{group_id}"
            group_objects[group_id] = register(
                "domain", base_path, group_id, members
            )
            motif_objects[group_id] = register(
                "motif", f"{base_path}/none", "none", members
            )
            ss_objects[group_id] = register(
                "secondary_structure",
                f"{base_path}/none/unknown",
                "unknown",
                members,
            )
        for item in neutral_items:
            path = (
                f"complex/protein/{item['chain_id']}/{item['group_id']}"
                f"/none/unknown/{item['chain_id']}:{item['canon_resid']}"
            )
            residue_objects[item["item_id"]] = register(
                "residue", path, f"{item['chain_id']}:{item['canon_resid']}",
                [item["item_id"]],
            )

        object_by_id = {obj["object_id"]: obj for obj in named_objects}
        item_paths = []
        for item in neutral_items:
            ids = {
                "complex": root_complex,
                "protein": root_protein,
                "chain": chain_objects[item["chain_id"]],
                "domain": group_objects[item["group_id"]],
                "motif": motif_objects[item["group_id"]],
                "secondary_structure": ss_objects[item["group_id"]],
                "residue": residue_objects[item["item_id"]],
            }
            item_paths.append({
                "item_id": item["item_id"],
                "named_object_by_level": ids,
                "support_by_level": {
                    level: object_by_id[object_id]["support_id"]
                    for level, object_id in ids.items()
                },
            })

        multi_name_classes = []
        for support in support_catalog.values():
            reps = support["named_representations"]
            levels = sorted(
                (rep["level"] for rep in reps),
                key=LEVELS_COARSE_TO_FINE.index,
            )
            if len(levels) > 1:
                multi_name_classes.append({
                    "support_id": support["support_id"],
                    "levels": levels,
                    "support_size": support["support_size"],
                })

        chains = []
        for chain_id in sorted(items_by_chain):
            canons = [
                item["canon_resid"] for item in neutral_items
                if item["chain_id"] == chain_id
            ]
            chains.append({
                "chain_id": chain_id,
                "canonical_range": [min(canons), max(canons)],
                "members": sorted(items_by_chain[chain_id]),
            })
        groups = []
        for group_id in sorted(items_by_group):
            members = sorted(items_by_group[group_id])
            member_records = [
                item for item in neutral_items if item["item_id"] in set(members)
            ]
            groups.append({
                "group_id": group_id,
                "chain_id": member_records[0]["chain_id"],
                "canon_resids": sorted(x["canon_resid"] for x in member_records),
                "members": members,
            })

        datasets.append({
            "dataset_id": dataset_id,
            "item_count": item_count,
            "K": K,
            "items": neutral_items,
            "chains": chains,
            "groups": groups,
            "support_catalog": sorted(
                support_catalog.values(), key=lambda x: x["support_id"]
            ),
            "named_support_objects": named_objects,
            "item_paths": item_paths,
            "verification": {
                "named_support_object_count": len(named_objects),
                "unique_support_count": len(support_catalog),
                "multi_name_equivalence_class_count": len(multi_name_classes),
                "multi_name_equivalence_classes": multi_name_classes,
            },
        })

    snapshot = {
        "schema_version": "v9a.snapshot.1",
        "purpose": "descriptive_synthetic_characterization_not_calibration",
        "assumptions": {
            "K": K,
            "independent_item_errors": True,
            "independent_run_errors": True,
            "zero_off_diagonal_covariance": True,
            "marginal_variances_fixed_known": True,
            "fixed_support_membership": True,
            "mapping_uncertainty": False,
            "additional_between_run_heterogeneity": False,
            "models_real_upstream_process": False,
        },
        "source": {
            "stride_commit": STRIDE_COMMIT,
            "downstream_commit": DOWNSTREAM_COMMIT,
            "files": [
                {
                    "relative_path": "outputs/replicate_table.parquet",
                    "sha256": sha256_file(replicate_path),
                    "rows": int(len(rep)),
                    "columns": list(rep.columns),
                    "required_columns": list(REPLICATE_REQUIRED),
                },
                {
                    "relative_path": "outputs/stride_table.parquet",
                    "sha256": sha256_file(stride_path),
                    "rows": int(len(profile)),
                    "columns": list(profile.columns),
                    "required_columns": list(STRIDE_REQUIRED),
                },
            ],
        },
        "uncertainty_resolution": {
            "rule": "finite theta_bootstrap_se else finite theta_se",
            "source_counts": dict(sorted(global_source_counts.items())),
        },
        "dataset_count": len(datasets),
        "dataset_item_count_total": int(sum(x["item_count"] for x in datasets)),
        "dataset_run_item_count_total": int(len(rep)),
        "levels_named": list(LEVELS_COARSE_TO_FINE),
        "datasets": datasets,
    }
    verify_snapshot(snapshot)
    return snapshot


def verify_snapshot(snapshot: dict) -> dict:
    """Verify pre-registered frozen design counts and alias equivalence classes."""
    if snapshot.get("dataset_count") != 4:
        raise SnapshotError("snapshot must contain four datasets")
    datasets = snapshot.get("datasets", [])
    observed_counts = [int(x["item_count"]) for x in datasets]
    if observed_counts != [236, 247, 242, 199]:
        raise SnapshotError(f"unexpected item counts: {observed_counts}")
    if snapshot.get("dataset_item_count_total") != 924:
        raise SnapshotError("snapshot must contain 924 dataset-item profiles")
    if snapshot.get("dataset_run_item_count_total") != 2772:
        raise SnapshotError("snapshot must contain 2,772 run-item rows")
    named = sum(
        int(x["verification"]["named_support_object_count"]) for x in datasets
    )
    unique = sum(
        int(x["verification"]["unique_support_count"]) for x in datasets
    )
    multi = sum(
        int(x["verification"]["multi_name_equivalence_class_count"])
        for x in datasets
    )
    if (named, unique, multi) != (1036, 964, 36):
        raise SnapshotError(
            "support equivalence mismatch: expected "
            f"(1036,964,36), observed {(named, unique,multi)}"
        )
    class_counts = Counter()
    item_pattern_counts = Counter()
    for dataset in datasets:
        item_ids = {x["item_id"] for x in dataset["items"]}
        if len(item_ids) != dataset["item_count"]:
            raise SnapshotError(f"{dataset['dataset_id']}: duplicate neutral item")
        for item in dataset["items"]:
            if len(item["sigma2_by_run"]) != K:
                raise SnapshotError("each item must have exactly K=3 variances")
            if not all(np.isfinite(item["sigma2_by_run"])) \
                    or not all(x > 0 for x in item["sigma2_by_run"]):
                raise SnapshotError("all marginal variances must be finite and positive")
        catalog = {x["support_id"]: x for x in dataset["support_catalog"]}
        for support in catalog.values():
            if set(support["members"]) - item_ids:
                raise SnapshotError("support contains unknown item")
            if support["support_size"] != len(support["members"]):
                raise SnapshotError("support size does not match membership")
            levels = tuple(
                rep["level"] for rep in support["named_representations"]
            )
            if len(levels) > 1:
                key = "=".join(sorted(levels, key=LEVELS_FINE_TO_COARSE.index))
                class_counts[key] += 1
        for path in dataset["item_paths"]:
            supports = path["support_by_level"]
            group_alias = (
                supports["domain"] == supports["motif"]
                == supports["secondary_structure"]
            )
            root_alias = supports["complex"] == supports["protein"]
            chain_alias = supports["chain"] == supports["domain"]
            if not group_alias or not root_alias:
                raise SnapshotError("expected universal group/root aliases")
            item_pattern_counts["group_chain_alias" if chain_alias else "distinct_chain"] += 1
    expected_classes = {
        "secondary_structure=motif=domain": 28,
        "secondary_structure=motif=domain=chain": 4,
        "protein=complex": 4,
    }
    if dict(class_counts) != expected_classes:
        raise SnapshotError(
            f"alias classes differ: expected {expected_classes}, "
            f"observed {dict(class_counts)}"
        )
    if item_pattern_counts != Counter(
        {"distinct_chain": 721, "group_chain_alias": 203}
    ):
        raise SnapshotError(f"item alias patterns differ: {item_pattern_counts}")
    return {
        "dataset_item_counts": observed_counts,
        "dataset_items_total": 924,
        "run_items_total": 2772,
        "named_support_objects": named,
        "unique_supports": unique,
        "multi_name_equivalence_classes": multi,
        "equivalence_class_counts": expected_classes,
        "item_alias_pattern_counts": dict(item_pattern_counts),
    }


def _dataset_spec(dataset: dict) -> SyntheticSystemSpec:
    items = dataset["items"]
    chains = tuple(
        SynChain(
            chain["chain_id"],
            (int(chain["canonical_range"][0]), int(chain["canonical_range"][1])),
        )
        for chain in dataset["chains"]
    )
    groups = tuple(
        SynDomain(
            group["group_id"],
            tuple(int(x) for x in group["canon_resids"]),
            group["chain_id"],
        )
        for group in dataset["groups"]
    )
    return SyntheticSystemSpec(
        name=dataset["dataset_id"],
        levels=LEVELS_COARSE_TO_FINE,
        chains=chains,
        domains=groups,
        residues=tuple(int(x["canon_resid"]) for x in items),
        resnames=tuple("ALA" for _ in items),
        K=K,
        T=0,
        sigma2=1.0,
        offset=0,
        seed=0,
    )


def _draw_frames(dataset: dict, regime: str, seed: int) -> list[pd.DataFrame]:
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    rng = np.random.default_rng(int(seed))
    items = dataset["items"]
    canon = np.asarray([x["canon_resid"] for x in items], dtype=int)
    chains = [x["chain_id"] for x in items]
    mu = np.asarray([
        0.0 if regime == "zero_mean" else x["mu_frozen"] for x in items
    ], dtype=float)
    sigma2 = np.asarray([x["sigma2_by_run"] for x in items], dtype=float)
    frames = []
    for run_index in range(K):
        se = np.sqrt(sigma2[:, run_index])
        theta = rng.normal(mu, se)
        frames.append(build_per_run_frame(
            canon, theta, se, chains, offset=0,
            names=["ALA"] * len(items),
        ))
    return frames


def _collapsed_paths(dataset: dict) -> dict[str, list[dict]]:
    catalog = {x["support_id"]: x for x in dataset["support_catalog"]}
    out = {}
    for path in dataset["item_paths"]:
        seen = set()
        collapsed = []
        for level in LEVELS_FINE_TO_COARSE:
            support_id = path["support_by_level"][level]
            if support_id in seen:
                continue
            seen.add(support_id)
            if level == "residue":
                collapsed_level = "item"
            elif level in ("secondary_structure", "motif", "domain"):
                collapsed_level = "configured_group"
            elif level == "chain":
                collapsed_level = "chain"
            else:
                collapsed_level = "whole_system_root"
            collapsed.append({
                "collapsed_level": collapsed_level,
                "first_named_level": level,
                "support_id": support_id,
                "support_size": catalog[support_id]["support_size"],
            })
        out[path["item_id"]] = collapsed
    return out


def _named_profiles(dataset: dict, rho_table: pd.DataFrame) -> dict[str, list[dict]]:
    by_key = {
        (str(row.scale_level), str(row.region_id)): row
        for row in rho_table.itertuples(index=False)
    }
    objects = {x["object_id"]: x for x in dataset["named_support_objects"]}
    profiles = {}
    for path in dataset["item_paths"]:
        rows = []
        for level in LEVELS_FINE_TO_COARSE:
            obj = objects[path["named_object_by_level"][level]]
            key = (level, obj["region_id"])
            if key not in by_key:
                raise SnapshotError(
                    f"production output missing named support {key}"
                )
            prod = by_key[key]
            rows.append({
                "named_level": level,
                "support_id": obj["support_id"],
                "rho": float(prod.rho),
                "status": str(prod.status),
                "support_size": int(prod.n_residues),
            })
        profiles[path["item_id"]] = rows
    return profiles


def _prepare_evaluation_context(dataset: dict) -> dict:
    """Precompute immutable hierarchy lookups once per dataset/regime cell."""
    from .adapters import to_hierarchy_config

    objects = {x["object_id"]: x for x in dataset["named_support_objects"]}
    named_paths = {}
    for path in dataset["item_paths"]:
        named_paths[path["item_id"]] = [
            {
                "named_level": level,
                "support_id": objects[
                    path["named_object_by_level"][level]
                ]["support_id"],
            }
            for level in LEVELS_FINE_TO_COARSE
        ]
    return {
        "config": to_hierarchy_config(_dataset_spec(dataset)),
        "named_objects": tuple(dataset["named_support_objects"]),
        "named_paths": named_paths,
        "collapsed_paths": _collapsed_paths(dataset),
        "catalog": {
            x["support_id"]: x for x in dataset["support_catalog"]
        },
    }


def _support_results_from_output(
    context: dict, rho_table: pd.DataFrame
) -> tuple[dict[str, float], dict[str, str]]:
    """Resolve production rows to distinct supports, checking every alias."""
    by_key = {
        (str(row.scale_level), str(row.region_id)): row
        for row in rho_table.itertuples(index=False)
    }
    rho_by_support: dict[str, float] = {}
    status_by_support: dict[str, str] = {}
    for obj in context["named_objects"]:
        key = (obj["level"], obj["region_id"])
        if key not in by_key:
            raise SnapshotError(f"production output missing named support {key}")
        prod = by_key[key]
        support_id = obj["support_id"]
        rho = float(prod.rho)
        if support_id in rho_by_support:
            previous = rho_by_support[support_id]
            if (np.isfinite(previous) != np.isfinite(rho)) or (
                np.isfinite(previous) and abs(previous - rho) > 1e-12
            ):
                raise AliasInvariantError(
                    f"duplicate named support {support_id} has unequal rho"
                )
        else:
            expected_size = int(context["catalog"][support_id]["support_size"])
            if int(prod.n_residues) != expected_size:
                raise SnapshotError(
                    f"support-size mismatch for {support_id}: "
                    f"{prod.n_residues} != {expected_size}"
                )
            rho_by_support[support_id] = rho
            status_by_support[support_id] = str(prod.status)
    return rho_by_support, status_by_support


def _first_crossing(rows: Sequence[dict], level_key: str) -> dict | None:
    for row in rows:
        if np.isfinite(row["rho"]) and row["rho"] >= RHO_STAR:
            return {
                "level": row[level_key],
                "support_id": row["support_id"],
                "rho": float(row["rho"]),
            }
    return None


def _alias_block(dataset: dict, support_id: str | None) -> str:
    if support_id is None:
        return "unresolved"
    support = next(
        x for x in dataset["support_catalog"] if x["support_id"] == support_id
    )
    levels = [
        x["level"] for x in support["named_representations"]
    ]
    if len(levels) <= 1:
        return "not_aliased"
    return "=".join(sorted(levels, key=LEVELS_FINE_TO_COARSE.index))


def evaluate_realization(
    dataset: dict, regime: str, seed: int, draw_index: int,
    *, context: dict | None = None,
) -> dict:
    """Evaluate one K=3 synthetic realization through production magnitude rho."""
    try:
        from .adapters import aggregate_via_production

        context = context or _prepare_evaluation_context(dataset)
        frames = _draw_frames(dataset, regime, seed)
        rho_table = aggregate_via_production(
            frames, context["config"],
            protein=dataset["dataset_id"], standardize=True
        )
        rho_by_support, status_by_support = _support_results_from_output(
            context, rho_table
        )

        size_rows = [
            {
                "support_id": sid,
                "support_size": int(
                    context["catalog"][sid]["support_size"]
                ),
                "rho": rho,
                "status": status_by_support[sid],
            }
            for sid, rho in sorted(rho_by_support.items())
        ]
        finite_size_rows = [x for x in size_rows if np.isfinite(x["rho"])]
        if len(finite_size_rows) >= 2:
            association = float(stats.spearmanr(
                [x["support_size"] for x in finite_size_rows],
                [x["rho"] for x in finite_size_rows],
            ).statistic)
        else:
            association = float("nan")

        delta_values = []
        named_crossings = Counter()
        collapsed_crossings = Counter()
        alias_blocks = Counter()
        label_changes = membership_changes = rho_changes = 0
        for item_id in sorted(context["named_paths"]):
            named_rows = [
                {**row, "rho": rho_by_support[row["support_id"]]}
                for row in context["named_paths"][item_id]
            ]
            collapsed_rows = []
            for row in context["collapsed_paths"][item_id]:
                collapsed_rows.append({
                    **row,
                    "rho": rho_by_support[row["support_id"]],
                })
            named = _first_crossing(named_rows, "named_level")
            collapsed = _first_crossing(collapsed_rows, "collapsed_level")
            named_crossings[
                "unresolved" if named is None else named["level"]
            ] += 1
            collapsed_crossings[
                "unresolved" if collapsed is None else collapsed["level"]
            ] += 1
            named_sid = None if named is None else named["support_id"]
            collapsed_sid = None if collapsed is None else collapsed["support_id"]
            if named_sid != collapsed_sid:
                membership_changes += 1
            named_rho = None if named is None else named["rho"]
            collapsed_rho = None if collapsed is None else collapsed["rho"]
            rho_diff = (
                (named_rho is None) != (collapsed_rho is None)
                or (
                    named_rho is not None and collapsed_rho is not None
                    and abs(named_rho - collapsed_rho) > 1e-12
                )
            )
            if rho_diff:
                rho_changes += 1
            block = _alias_block(dataset, named_sid)
            alias_blocks[block] += 1
            if block not in ("not_aliased", "unresolved"):
                label_changes += 1

            item_support = named_rows[0]["support_id"]
            group_support = next(
                x["support_id"] for x in collapsed_rows
                if x["collapsed_level"] == "configured_group"
            )
            item_rho = rho_by_support[item_support]
            group_rho = rho_by_support[group_support]
            delta_values.append(float(group_rho - item_rho))

        if membership_changes or rho_changes:
            raise AliasInvariantError(
                f"alias collapse changed membership for {membership_changes} or "
                f"rho for {rho_changes} item decisions"
            )
        deltas = np.asarray(delta_values, dtype=float)
        finite_delta = np.isfinite(deltas)
        n_items = int(dataset["item_count"])
        if finite_delta.sum() != n_items:
            raise RuntimeError(
                f"non-finite item-to-aggregate delta for "
                f"{n_items - int(finite_delta.sum())} items"
            )
        increased = float(np.mean(deltas > 1e-12))
        equal = float(np.mean(np.abs(deltas) <= 1e-12))
        decreased = float(np.mean(deltas < -1e-12))

        model_status_counts = Counter(x["status"] for x in size_rows)
        return {
            "dataset_id": dataset["dataset_id"],
            "regime": regime,
            "draw_index": int(draw_index),
            "seed": int(seed),
            "success": True,
            "error_type": None,
            "error_message": None,
            "support_rows": size_rows,
            "association": association,
            "supports_per_coefficient": len(finite_size_rows),
            "item_comparisons": n_items,
            "fraction_increased": increased,
            "fraction_equal": equal,
            "fraction_decreased": decreased,
            "median_delta_rho": float(np.median(deltas)),
            "named_crossings": dict(sorted(named_crossings.items())),
            "collapsed_crossings": dict(sorted(collapsed_crossings.items())),
            "alias_label_changes": int(label_changes),
            "alias_membership_changes": int(membership_changes),
            "alias_rho_changes": int(rho_changes),
            "alias_block_counts": dict(sorted(alias_blocks.items())),
            "model_status_counts": dict(sorted(model_status_counts.items())),
        }
    except AliasInvariantError:
        raise
    except Exception as exc:
        return {
            "dataset_id": dataset["dataset_id"],
            "regime": regime,
            "draw_index": int(draw_index),
            "seed": int(seed),
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "support_rows": [],
            "association": None,
            "supports_per_coefficient": 0,
            "item_comparisons": int(dataset["item_count"]),
            "fraction_increased": None,
            "fraction_equal": None,
            "fraction_decreased": None,
            "median_delta_rho": None,
            "named_crossings": {},
            "collapsed_crossings": {},
            "alias_label_changes": 0,
            "alias_membership_changes": 0,
            "alias_rho_changes": 0,
            "alias_block_counts": {},
            "model_status_counts": {},
        }


def _run_cell(args: tuple[dict, str, list[int]]) -> list[dict]:
    dataset, regime, seeds = args
    started = time.perf_counter()
    context = _prepare_evaluation_context(dataset)
    durations = []
    rows = []
    total = len(seeds)
    print(
        f"V9A cell start dataset={dataset['dataset_id']} regime={regime} "
        f"realizations={total}",
        file=sys.stderr, flush=True,
    )
    for draw_index, seed in enumerate(seeds):
        draw_started = time.perf_counter()
        rows.append(evaluate_realization(
            dataset, regime, seed, draw_index, context=context
        ))
        durations.append(time.perf_counter() - draw_started)
        completed = draw_index + 1
        if completed % 100 == 0 or completed == total:
            print(
                f"V9A progress dataset={dataset['dataset_id']} regime={regime} "
                f"completed={completed}/{total}",
                file=sys.stderr, flush=True,
            )
    elapsed = time.perf_counter() - started
    print(
        f"V9A cell complete dataset={dataset['dataset_id']} regime={regime} "
        f"total_seconds={elapsed:.6f} "
        f"mean_seconds={float(np.mean(durations)):.6f} "
        f"median_seconds={float(np.median(durations)):.6f}",
        file=sys.stderr, flush=True,
    )
    return rows


def build_failure_ledger(realization_rows: Sequence[dict]) -> list[dict]:
    """Return deterministic failure records without removing failed denominators."""
    return [
        {
            "dataset_id": row["dataset_id"],
            "regime": row["regime"],
            "draw_index": row["draw_index"],
            "seed": row["seed"],
            "error_type": row["error_type"],
            "error_message": row["error_message"],
        }
        for row in realization_rows if not row["success"]
    ]


def _stratum(size: int) -> str:
    for label, low, high in DISPLAY_STRATA:
        if low <= size <= high:
            return label
    raise ValueError(f"support size outside strata: {size}")


def _summarize_support_size(draws: list[dict], snapshot: dict) -> pd.DataFrame:
    dataset_lookup = {x["dataset_id"]: x for x in snapshot["datasets"]}
    rows = []
    for dataset_id in sorted(dataset_lookup):
        for regime in REGIMES:
            cell = [
                x for x in draws
                if x["dataset_id"] == dataset_id and x["regime"] == regime
            ]
            successful = [x for x in cell if x["success"]]
            support_catalog = dataset_lookup[dataset_id]["support_catalog"]
            support_size_counts = Counter(
                int(x["support_size"]) for x in support_catalog
            )
            for group_type in ("exact", "stratum"):
                groups: dict[str, list[int]] = defaultdict(list)
                for support in support_catalog:
                    size = int(support["support_size"])
                    key = str(size) if group_type == "exact" else _stratum(size)
                    groups[key].append(size)
                for key in sorted(
                    groups,
                    key=lambda x: (
                        int(x) if group_type == "exact"
                        else [z[0] for z in DISPLAY_STRATA].index(x)
                    ),
                ):
                    allowed_sizes = set(groups[key])
                    distinct_supports = sum(
                        count for size, count in support_size_counts.items()
                        if size in allowed_sizes
                    )
                    pooled = []
                    draw_means = []
                    for draw in successful:
                        vals = [
                            float(x["rho"]) for x in draw["support_rows"]
                            if int(x["support_size"]) in allowed_sizes
                            and np.isfinite(x["rho"])
                        ]
                        pooled.extend(vals)
                        draw_means.append(
                            float(np.mean(vals)) if vals else float("nan")
                        )
                    finite = np.asarray(pooled, dtype=float)
                    ci = _t_interval(draw_means)
                    rows.append({
                        "dataset_id": dataset_id,
                        "regime": regime,
                        "group_type": group_type,
                        "support_size_group": key,
                        "distinct_supports": int(distinct_supports),
                        "expected_fit_count": int(N_DRAWS * distinct_supports),
                        "finite_fit_count": int(np.isfinite(finite).sum()),
                        "mean_rho": (
                            float(np.mean(finite)) if finite.size else np.nan
                        ),
                        "median_rho": (
                            float(np.median(finite)) if finite.size else np.nan
                        ),
                        "across_draw_mean_rho": (
                            float(np.nanmean(draw_means)) if draw_means else np.nan
                        ),
                        "mc_t95_lower": ci[0],
                        "mc_t95_upper": ci[1],
                        "distribution_q025": _quantile(finite, 0.025),
                        "distribution_q25": _quantile(finite, 0.25),
                        "distribution_q50": _quantile(finite, 0.50),
                        "distribution_q75": _quantile(finite, 0.75),
                        "distribution_q975": _quantile(finite, 0.975),
                    })
    return pd.DataFrame(rows)


def _summarize_association(draws: list[dict]) -> list[dict]:
    rows = []
    for dataset_id in sorted({x["dataset_id"] for x in draws}):
        for regime in REGIMES:
            cell = [
                x for x in draws
                if x["dataset_id"] == dataset_id and x["regime"] == regime
            ]
            vals = np.asarray([
                x["association"] for x in cell
                if x["success"] and x["association"] is not None
            ], dtype=float)
            supports = [
                int(x["supports_per_coefficient"]) for x in cell if x["success"]
            ]
            ci = _t_interval(vals)
            rows.append({
                "dataset_id": dataset_id,
                "regime": regime,
                "expected_coefficients": N_DRAWS,
                "n_coefficients": int(np.isfinite(vals).sum()),
                "supports_per_coefficient_min": min(supports) if supports else 0,
                "supports_per_coefficient_max": max(supports) if supports else 0,
                "mean_coefficient": (
                    float(np.nanmean(vals)) if vals.size else None
                ),
                "mc_t95_lower": _finite_or_none(ci[0]),
                "mc_t95_upper": _finite_or_none(ci[1]),
                "median": _quantile(vals, 0.5),
                "distribution_q025": _quantile(vals, 0.025),
                "distribution_q975": _quantile(vals, 0.975),
            })
    return rows


def _summarize_delta(draws: list[dict]) -> list[dict]:
    rows = []
    for dataset_id in sorted({x["dataset_id"] for x in draws}):
        for regime in REGIMES:
            cell = [
                x for x in draws
                if x["dataset_id"] == dataset_id and x["regime"] == regime
            ]
            successful = [x for x in cell if x["success"]]
            metrics = {}
            for key in (
                "fraction_increased",
                "fraction_equal",
                "fraction_decreased",
                "median_delta_rho",
            ):
                vals = [float(x[key]) for x in successful]
                ci = _t_interval(vals)
                metrics[key] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "mc_t95_lower": _finite_or_none(ci[0]),
                    "mc_t95_upper": _finite_or_none(ci[1]),
                }
            rows.append({
                "dataset_id": dataset_id,
                "regime": regime,
                "expected_draws": N_DRAWS,
                "successful_draws": len(successful),
                "failed_draws": len(cell) - len(successful),
                "item_comparisons_per_draw": (
                    int(cell[0]["item_comparisons"]) if cell else 0
                ),
                "total_expected_comparisons": (
                    int(N_DRAWS * cell[0]["item_comparisons"]) if cell else 0
                ),
                "draw_level": [
                    {
                        "draw_index": int(x["draw_index"]),
                        "success": bool(x["success"]),
                        "fraction_increased": x["fraction_increased"],
                        "fraction_equal": x["fraction_equal"],
                        "fraction_decreased": x["fraction_decreased"],
                        "median_delta_rho": x["median_delta_rho"],
                    }
                    for x in cell
                ],
                "summary": metrics,
            })
    return rows


def _summarize_crossing(draws: list[dict], snapshot: dict) -> pd.DataFrame:
    item_counts = {x["dataset_id"]: x["item_count"] for x in snapshot["datasets"]}
    rows = []
    for dataset_id in sorted(item_counts):
        for regime in REGIMES:
            cell = [
                x for x in draws
                if x["dataset_id"] == dataset_id and x["regime"] == regime
            ]
            successful = [x for x in cell if x["success"]]
            for variant, field, ordered_levels in (
                (
                    "named_duplicate",
                    "named_crossings",
                    list(LEVELS_FINE_TO_COARSE) + ["unresolved"],
                ),
                (
                    "collapsed_unique_support",
                    "collapsed_crossings",
                    [
                        "item", "configured_group", "chain",
                        "whole_system_root", "unresolved",
                    ],
                ),
            ):
                for level in ordered_levels:
                    counts = [
                        int(x[field].get(level, 0)) for x in successful
                    ]
                    fractions = [
                        count / item_counts[dataset_id] for count in counts
                    ]
                    ci = _t_interval(fractions)
                    rows.append({
                        "dataset_id": dataset_id,
                        "regime": regime,
                        "hierarchy_variant": variant,
                        "crossing_level": level,
                        "expected_draws": N_DRAWS,
                        "successful_draws": len(successful),
                        "failed_draws": len(cell) - len(successful),
                        "item_profiles_per_draw": item_counts[dataset_id],
                        "expected_item_decisions": (
                            N_DRAWS * item_counts[dataset_id]
                        ),
                        "total_crossing_count_successful_draws": sum(counts),
                        "mean_draw_count": (
                            float(np.mean(counts)) if counts else np.nan
                        ),
                        "mean_draw_fraction": (
                            float(np.mean(fractions)) if fractions else np.nan
                        ),
                        "mc_t95_lower": ci[0],
                        "mc_t95_upper": ci[1],
                    })
                item_key = (
                    "residue" if variant == "named_duplicate" else "item"
                )
                item_cross = sum(
                    int(x[field].get(item_key, 0)) for x in successful
                )
                unresolved = sum(
                    int(x[field].get("unresolved", 0)) for x in successful
                )
                total_success_decisions = len(successful) * item_counts[dataset_id]
                larger_counts = [
                    (
                        item_counts[dataset_id]
                        - int(x[field].get(item_key, 0))
                        - int(x[field].get("unresolved", 0))
                    )
                    for x in successful
                ]
                larger_fractions = [
                    count / item_counts[dataset_id] for count in larger_counts
                ]
                larger_ci = _t_interval(larger_fractions)
                rows.append({
                    "dataset_id": dataset_id,
                    "regime": regime,
                    "hierarchy_variant": variant,
                    "crossing_level": "__requiring_larger_unique_support__",
                    "expected_draws": N_DRAWS,
                    "successful_draws": len(successful),
                    "failed_draws": len(cell) - len(successful),
                    "item_profiles_per_draw": item_counts[dataset_id],
                    "expected_item_decisions": (
                        N_DRAWS * item_counts[dataset_id]
                    ),
                    "total_crossing_count_successful_draws": (
                        total_success_decisions - item_cross - unresolved
                    ),
                    "mean_draw_count": (
                        float(np.mean(larger_counts))
                        if larger_counts else np.nan
                    ),
                    "mean_draw_fraction": (
                        float(np.mean(larger_fractions))
                        if larger_fractions else np.nan
                    ),
                    "mc_t95_lower": larger_ci[0],
                    "mc_t95_upper": larger_ci[1],
                })
    return pd.DataFrame(rows)


def _summarize_alias(draws: list[dict], snapshot: dict) -> pd.DataFrame:
    item_counts = {x["dataset_id"]: x["item_count"] for x in snapshot["datasets"]}
    rows = []
    for dataset_id in sorted(item_counts):
        for regime in REGIMES:
            cell = [
                x for x in draws
                if x["dataset_id"] == dataset_id and x["regime"] == regime
            ]
            successful = [x for x in cell if x["success"]]
            denominator = len(successful) * item_counts[dataset_id]
            label_changes = sum(x["alias_label_changes"] for x in successful)
            membership_changes = sum(
                x["alias_membership_changes"] for x in successful
            )
            rho_changes = sum(x["alias_rho_changes"] for x in successful)
            block_counts = Counter()
            for draw in successful:
                block_counts.update(draw["alias_block_counts"])
            for metric, count in (
                ("reported_label_changed", label_changes),
                ("selected_membership_changed", membership_changes),
                ("selected_rho_changed_gt_1e-12", rho_changes),
            ):
                rows.append({
                    "dataset_id": dataset_id,
                    "regime": regime,
                    "record_type": "paired_metric",
                    "metric_or_alias_block": metric,
                    "expected_item_decisions": N_DRAWS * item_counts[dataset_id],
                    "successful_item_decisions": denominator,
                    "count": int(count),
                    "fraction_of_successful_decisions": (
                        count / denominator if denominator else np.nan
                    ),
                })
            for block, count in sorted(block_counts.items()):
                rows.append({
                    "dataset_id": dataset_id,
                    "regime": regime,
                    "record_type": "selected_alias_block",
                    "metric_or_alias_block": block,
                    "expected_item_decisions": N_DRAWS * item_counts[dataset_id],
                    "successful_item_decisions": denominator,
                    "count": int(count),
                    "fraction_of_successful_decisions": (
                        count / denominator if denominator else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def run_v9a(
    snapshot: dict, *, master_seed: int = MASTER_SEED,
    draws: int = N_DRAWS, workers: int = 1,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all pre-registered cells and return deterministic summaries."""
    if int(master_seed) != MASTER_SEED:
        raise ValueError(f"V9A master seed is fixed at {MASTER_SEED}")
    if int(draws) != N_DRAWS:
        raise ValueError(f"V9A repetitions are fixed at {N_DRAWS}")
    verification = verify_snapshot(snapshot)
    dataset_ids = [x["dataset_id"] for x in snapshot["datasets"]]
    seeds = deterministic_child_seeds(master_seed, dataset_ids, REGIMES, draws)
    tasks = [
        (dataset, regime, seeds[(dataset["dataset_id"], regime)])
        for dataset in snapshot["datasets"]
        for regime in REGIMES
    ]
    run_started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
            cells = list(pool.map(_run_cell, tasks))
    else:
        cells = [_run_cell(task) for task in tasks]
    print(
        f"V9A total realization_seconds="
        f"{time.perf_counter() - run_started:.6f}",
        file=sys.stderr, flush=True,
    )
    realization_rows = [
        row for cell in cells for row in cell
    ]
    realization_rows.sort(
        key=lambda x: (x["dataset_id"], REGIMES.index(x["regime"]), x["draw_index"])
    )
    if len(realization_rows) != 4000:
        raise RuntimeError(
            f"expected 4,000 realization records, observed {len(realization_rows)}"
        )
    failures = build_failure_ledger(realization_rows)
    support_size = _summarize_support_size(realization_rows, snapshot)
    first_crossing = _summarize_crossing(realization_rows, snapshot)
    alias = _summarize_alias(realization_rows, snapshot)
    association = _summarize_association(realization_rows)
    delta = _summarize_delta(realization_rows)
    status_counts = Counter()
    for row in realization_rows:
        status_counts.update(row["model_status_counts"])
    results = {
        "schema_version": "v9a.results.1",
        "purpose": "descriptive_synthetic_characterization_not_calibration",
        "master_seed": MASTER_SEED,
        "seed_method": "numpy.random.SeedSequence child streams",
        "draws_per_dataset_regime": N_DRAWS,
        "dataset_count": 4,
        "regime_count": 2,
        "expected_realizations": 4000,
        "successful_realizations": 4000 - len(failures),
        "failed_realizations": len(failures),
        "failure_rate": len(failures) / 4000,
        "failure_ledger": failures,
        "snapshot_verification": verification,
        "assumptions": snapshot["assumptions"],
        "regimes": {
            "zero_mean": "theta_ik ~ Normal(0, sigma2_ik)",
            "frozen_scale_fixed_mean": (
                "theta_ik ~ Normal(mu_i, sigma2_ik), "
                "mu_i=frozen arithmetic mean over K=3 effects"
            ),
        },
        "support_size_association": association,
        "item_to_first_unique_aggregate": delta,
        "model_status_counts": dict(sorted(status_counts.items())),
        "interpretation_boundaries": [
            "does_not_calibrate_0.5",
            "does_not_establish_false_positive_or_fdr_control",
            "does_not_validate_a_finest_supported_scale",
            "does_not_model_real_upstream_dependence",
            "does_not_recover_unavailable_covariance",
            "does_not_validate_biological_truth",
        ],
    }
    return results, support_size, first_crossing, alias


def _git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def _package_versions() -> dict[str, str]:
    from importlib.metadata import version
    names = ("numpy", "pandas", "scipy", "pyarrow", "PyYAML", "pytest")
    return {name: version(name) for name in names}


def write_artifacts(
    output_dir: str | Path,
    snapshot: dict,
    results: dict,
    support_size: pd.DataFrame,
    first_crossing: pd.DataFrame,
    alias: pd.DataFrame,
    *,
    repo_root: str | Path,
) -> dict[str, str]:
    """Write deterministic artifacts and a manifest hashing the other files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "snapshot": output_dir / SNAPSHOT_FILENAME,
        "results": output_dir / RESULTS_FILENAME,
        "support_size": output_dir / SUPPORT_SIZE_FILENAME,
        "first_crossing": output_dir / FIRST_CROSSING_FILENAME,
        "alias_effects": output_dir / ALIAS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    write_json(paths["snapshot"], snapshot)
    write_json(paths["results"], results)
    for key, frame in (
        ("support_size", support_size),
        ("first_crossing", first_crossing),
        ("alias_effects", alias),
    ):
        frame.to_csv(
            paths[key], index=False, lineterminator="\n",
            float_format="%.12g",
        )
    repo_root = Path(repo_root).resolve()
    manifest = {
        "schema_version": "v9a.manifest.1",
        "purpose": "descriptive_synthetic_characterization_not_calibration",
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": _package_versions(),
        },
        "source": {
            "stride_commit_required": STRIDE_COMMIT,
            "stride_commit_observed": _git_value(repo_root, "rev-parse", "HEAD"),
            "downstream_commit": DOWNSTREAM_COMMIT,
        },
        "design": {
            "K": K,
            "master_seed": MASTER_SEED,
            "draws_per_dataset_regime": N_DRAWS,
            "expected_realizations": 4000,
            "rho_star_provisional": RHO_STAR,
        },
        "artifact_sha256": {
            key: sha256_file(path)
            for key, path in paths.items() if key != "manifest"
        },
    }
    write_json(paths["manifest"], manifest)
    return {key: sha256_file(path) for key, path in paths.items()}


def execute(
    frozen_root: str | Path,
    output_dir: str | Path,
    *,
    master_seed: int = MASTER_SEED,
    draws: int = N_DRAWS,
    workers: int = 1,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    """Extract the snapshot, run V9A, and write the six data artifacts."""
    repo_root = (
        Path(repo_root).resolve()
        if repo_root is not None else Path(__file__).resolve().parents[1]
    )
    observed_commit = _git_value(repo_root, "rev-parse", "HEAD")
    if observed_commit != STRIDE_COMMIT:
        raise RuntimeError(
            f"V9A requires STRIDE {STRIDE_COMMIT}, observed {observed_commit}"
        )
    snapshot = extract_snapshot(frozen_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist the preregistered design snapshot before simulation begins.
    write_json(output_dir / SNAPSHOT_FILENAME, snapshot)
    results, support_size, first_crossing, alias = run_v9a(
        snapshot, master_seed=master_seed, draws=draws, workers=workers
    )
    return write_artifacts(
        output_dir, snapshot, results, support_size, first_crossing, alias,
        repo_root=repo_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pre-registered V9A frozen-design characterization"
    )
    parser.add_argument("--frozen-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument(
        "--workers", type=int, default=max(1, min(8, os.cpu_count() or 1))
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hashes = execute(
        args.frozen_root,
        args.output_dir,
        master_seed=args.seed,
        draws=args.draws,
        workers=args.workers,
    )
    print(json.dumps(hashes, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
