"""Opt-in integration test for an externally supplied real POVME installation.

This test is intentionally skipped in ordinary CI. It exists to certify a
release fixture once its topology, trajectory, and POVME installation are
available under an explicit configuration file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mechanism.workflow.executor import execute_workflow
from mechanism.workflow.provenance import sha256
from mechanism.workflow.schema import load_workflow_config

pytestmark = pytest.mark.real_povme


def test_external_povme_fixture_produces_per_run_effect_fields(tmp_path):
    config_path = os.environ.get("STRIDE_REAL_POVME_CONFIG")
    if not config_path:
        pytest.skip("set STRIDE_REAL_POVME_CONFIG to run an archived real-POVME fixture")

    config = load_workflow_config(config_path)
    if config.pocket.povme_executable is None:
        pytest.fail("the real-POVME configuration must declare pocket.povme_executable")
    expected_hash = os.environ.get("STRIDE_REAL_POVME_EXECUTABLE_SHA256")
    if expected_hash and sha256(config.pocket.povme_executable) != expected_hash:
        pytest.fail("the configured POVME executable does not match the declared SHA-256")

    expected_path = os.environ.get("STRIDE_REAL_POVME_EXPECTED_OUTPUTS")
    if not expected_path:
        pytest.fail(
            "set STRIDE_REAL_POVME_EXPECTED_OUTPUTS to an archived expected-output hash manifest"
        )
    expected = json.loads(Path(expected_path).read_text())
    if expected.get("workflow_config_sha256") != sha256(config.source_path):
        pytest.fail("expected-output manifest does not match the declared real-POVME configuration")
    expected_runs = expected.get("runs")
    if not isinstance(expected_runs, dict):
        pytest.fail("expected-output manifest must contain a run-id keyed 'runs' mapping")

    completed = execute_workflow(config, tmp_path / "run-effects")
    assert len(completed) >= 2
    for result in completed:
        expected_run = expected_runs.get(result.run_id)
        assert isinstance(expected_run, dict), f"missing expected hashes for run {result.run_id}"
        assert result.effects_path.is_file()
        assert result.volume_path.is_file()
        assert result.frame_count > 1
        assert sha256(result.effects_path) == expected_run["effects_sha256"]
        assert sha256(result.volume_path) == expected_run["volume_sha256"]
        assert result.frame_count == expected_run["frame_count"]
