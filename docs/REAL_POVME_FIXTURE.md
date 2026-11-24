# Real POVME integration fixture

The committed synthetic fixture verifies STRIDE's workflow interface. It does
not verify a POVME release, a force field, or a molecular-dynamics analysis.

To certify the external dependency before a public release, deposit a small
licensed fixture outside the Git repository and record its immutable archive
identifier in the release notes. The fixture should contain:

1. One topology and at least two declared runs for the same system. Each run may
   contain one trajectory file or an ordered list of trajectory segments.
2. A `stride.yaml` with explicit selections, frame stride, reference selection,
   hierarchy configuration, and complete coordinate map.
3. The exact POVME executable or a citable installation record, including its
   SHA-256, version, license, and invocation method.
4. A manifest with SHA-256 values and frame counts for every source file.
5. An expected-output JSON with the SHA-256 of the workflow YAML and, for each
   run, the expected `effects_sha256`, `volume_sha256`, and `frame_count`.
   Produce it once from the archived inputs with the recorded software versions.

The expected-output file has this shape:

```json
{
  "workflow_config_sha256": "<sha256-of-stride.yaml>",
  "runs": {
    "run_1": {
      "effects_sha256": "<sha256>",
      "volume_sha256": "<sha256>",
      "frame_count": 100
    },
    "run_2": {
      "effects_sha256": "<sha256>",
      "volume_sha256": "<sha256>",
      "frame_count": 100
    }
  }
}
```

Do not use the frozen dengue application as this fixture unless its missing
upstream provenance and redistribution permissions have been resolved.

## Running the opt-in check

Set the configuration path and, when available, the executable hash:

```bash
export STRIDE_REAL_POVME_CONFIG=/archive/fixture/stride.yaml
export STRIDE_REAL_POVME_EXECUTABLE_SHA256=<sha256-of-POVME2.py>
export STRIDE_REAL_POVME_EXPECTED_OUTPUTS=/archive/fixture/expected_outputs.json
pytest -m real_povme tests/test_real_povme_integration.py -q
```

The test creates output in pytest's temporary directory. It runs the declared
POVME executable and verifies that each configured run reproduces the archived
volume-table and per-run-effect hashes. It is intentionally excluded from
ordinary CI until a public, versioned fixture can be downloaded by the runner.
