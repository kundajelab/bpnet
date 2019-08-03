"""Test modisco
"""
import os
import numpy as np
from bpnet.cli.modisco import bpnet_modisco_run
from pytest import fixture
import gin

@fixture
def expected_modisco_files():
    return [
        'modisco.h5',
        'modisco-run.config.gin',
        'modisco-run.config.gin.json',
        'modisco-run.input-config.gin',
        'modisco-run.kwargs.json',
        'modisco-run.subset-contrib-file.npy',
        'log'
    ]

def test_modisco_run(tmp_path, contrib_score_grad, modisco_config_gin,
                     expected_modisco_files):
    gin.clear_config()
    bpnet_modisco_run(contrib_file=str(contrib_score_grad),
                      output_dir=tmp_path,
                      config=str(modisco_config_gin),
                      )
    output_files = os.listdir(tmp_path)
    for f in expected_modisco_files:
        assert f in output_files
    assert np.all(np.load(tmp_path / 'modisco-run.subset-contrib-file.npy') == 1)


def test_modisco_run_null(tmp_path, contrib_score_grad, contrib_score_grad_null,
                          modisco_config_gin, expected_modisco_files):
    gin.clear_config()
    bpnet_modisco_run(contrib_file=str(contrib_score_grad),
                      output_dir=str(tmp_path),
                      null_contrib_file=str(contrib_score_grad_null),
                      config=str(modisco_config_gin),
                      )
    output_files = os.listdir(tmp_path)
    for f in expected_modisco_files:
        assert f in output_files
    assert np.all(np.load(tmp_path / 'modisco-run.subset-contrib-file.npy') == 1)
