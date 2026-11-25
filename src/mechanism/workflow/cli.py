"""CLI for validating and auditing explicit trajectory-workflow inputs."""

from __future__ import annotations

import argparse

from .downstream import DownstreamExportError, export_downstream_layout
from .executor import WorkflowExecutionError, execute_workflow, resume_workflow
from .provenance import build_manifest, write_manifest
from .schema import WorkflowConfigError, load_workflow_config
from .summary import WorkflowSummaryError, summarize_to_downstream_dataset
from .validation import validate_workflow_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stride",
        description="Validate and record explicit STRIDE trajectory-workflow inputs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan", "run", "resume", "export-downstream", "summarize"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", required=True, help="Path to a workflow YAML file")
        if command in {"plan", "run", "resume", "export-downstream", "summarize"}:
            subparser.add_argument("--output", required=True, help="Output directory")
        if command in {"export-downstream", "summarize"}:
            subparser.add_argument(
                "--run-effects",
                required=True,
                help="Directory previously written by 'stride run'",
            )
        if command == "resume":
            subparser.add_argument(
                "--staging",
                required=True,
                help="Interrupted staging directory created by 'stride run'",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_workflow_config(args.config)
    except WorkflowConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    validation = validate_workflow_config(config)
    for message in validation.warnings:
        print(f"WARNING: {message}")
    for message in validation.errors:
        print(f"ERROR: {message}")
    if not validation.valid:
        return 1

    if args.command == "validate":
        print(f"VALID: {config.system_id} ({len(config.runs)} declared runs)")
        return 0

    if args.command == "run":
        execution_validation = validate_workflow_config(config, require_povme=True)
        if not execution_validation.valid:
            for message in execution_validation.errors:
                print(f"ERROR: {message}")
            return 1
        try:
            completed = execute_workflow(config, args.output)
        except WorkflowExecutionError as exc:
            print(f"EXECUTION ERROR: {exc}")
            return 1
        print(f"COMPLETE: {config.system_id} ({len(completed)} per-run effect fields)")
        return 0

    if args.command == "resume":
        execution_validation = validate_workflow_config(config, require_povme=True)
        if not execution_validation.valid:
            for message in execution_validation.errors:
                print(f"ERROR: {message}")
            return 1
        try:
            completed = resume_workflow(config, args.staging, args.output)
        except WorkflowExecutionError as exc:
            print(f"RESUME ERROR: {exc}")
            return 1
        print(f"COMPLETE: {config.system_id} ({len(completed)} per-run effect fields)")
        return 0

    if args.command == "export-downstream":
        try:
            destination = export_downstream_layout(config, args.run_effects, args.output)
        except DownstreamExportError as exc:
            print(f"EXPORT ERROR: {exc}")
            return 1
        print(destination)
        return 0

    if args.command == "summarize":
        try:
            destination = summarize_to_downstream_dataset(config, args.run_effects, args.output)
        except WorkflowSummaryError as exc:
            print(f"SUMMARY ERROR: {exc}")
            return 1
        print(destination)
        return 0

    manifest = build_manifest(config, validation)
    output_path = write_manifest(manifest, args.output)
    print(output_path)
    return 0
