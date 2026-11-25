"""Preparation-only checks for explicit dengue dataset identities."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / 'examples/dengue_pocket/prepare_legacy_denv1_pilot.py'


@pytest.mark.parametrize('dataset', ['DENV1', 'DENV2', 'DENV3', 'DENV4'])
def test_preparation_uses_requested_dataset_and_mapping(tmp_path, dataset):
    repo = Path(__file__).resolve().parents[1]
    for run in ('1st_run', '2nd_run', '3rd_run'):
        folder = tmp_path / run / dataset
        (folder / 'analysis_output').mkdir(parents=True)
        (folder / '33-md.gro').write_text('topology placeholder')
        (folder / '36-md-fit.xtc').write_bytes(b'trajectory placeholder')
        (folder / 'analysis_output' / f'{dataset}_correlations_v5.csv').write_text(
            'file_resid,canon_resid\n151,51\n175,75\n235,135\n'
        )
    povme = tmp_path / 'POVME2.py'
    povme.write_text('')
    out = tmp_path / 'prepared'
    result = subprocess.run([
        sys.executable, str(SCRIPT), '--legacy-root', str(tmp_path),
        '--workflow-root', str(repo), '--povme-executable', str(povme),
        '--output-dir', str(out), '--frame-interval-ps', '10', '--dataset', dataset,
    ], capture_output=True, text=True, check=True)
    config = yaml.safe_load((out / f'{dataset}_stride.yaml').read_text())
    assert config['downstream_export']['dataset_id'] == dataset
    assert config['pocket']['reference']['selection'] == 'protein and resid 151 175 235'
    assert config['preprocessing']['alignment_mode'] == 'use_supplied_fitted_trajectory'
    assert config['preprocessing']['frame_stride'] == 20
    assert '[151, 175, 235]' in result.stdout
