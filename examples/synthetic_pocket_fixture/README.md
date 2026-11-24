# Synthetic trajectory-workflow fixture

This small fixture verifies the public STRIDE workflow contract in a clean
checkout. It contains two declared runs, a tiny multi-frame PDB trajectory, an
explicit coordinate map, and a simple hierarchy.

`povme_stub.py` emits a POVME-format volume table so the repository can test
the command interface without bundling POVME or making claims about a POVME
release. It is not scientific validation of POVME or an MD workflow.

Run from the repository root:

```bash
stride validate --config examples/synthetic_pocket_fixture/stride.yaml
stride run --config examples/synthetic_pocket_fixture/stride.yaml --output /tmp/stride-effects
stride summarize --config examples/synthetic_pocket_fixture/stride.yaml \
  --run-effects /tmp/stride-effects --output /tmp/stride-dataset
```
