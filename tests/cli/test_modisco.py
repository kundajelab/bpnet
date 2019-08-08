"""Test modisco
"""
import os
import pandas as pd
import numpy as np
from bpnet.cli.modisco import bpnet_modisco_run, cwm_scan
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


def test_cwm_scan(tmp_path, modisco_dir, contrib_file):
    output_file = str(tmp_path / 'instances.csv.gz')
    cwm_scan(modisco_dir=str(modisco_dir),
             output_file=output_file,
             contrib_file=None,
             add_profile_features=False)
    df = pd.read_csv(output_file)
    assert list(df.columns[:7]) == ['example_chrom', 'pattern_start_abs', 'pattern_end_abs',
                                    'pattern', 'contrib_weighted_p', 'strand', 'match_weighted_p']
    cm_path = modisco_dir / f'cwm-scan-seqlets.trim-frac=0.08.csv.gz'
    assert os.path.exists(cm_path)


def test_cwm_scan_new_file(tmp_path, modisco_dir, contrib_file):
    output_file = str(tmp_path / 'instances.csv.gz')
    trim_frac = 0.05
    cwm_scan(modisco_dir=str(modisco_dir),
             output_file=output_file,
             contrib_file=contrib_file,
             trim_frac=trim_frac,
             add_profile_features=True)

    cm_path = modisco_dir / f'cwm-scan-seqlets.trim-frac={trim_frac:.2f}.csv.gz'
    assert os.path.exists(cm_path)

    # make sure the normalized file exists
    df = pd.read_csv(output_file)

    assert 'Oct4/profile_counts' in df
    assert 'Oct4/profile_match_p' in df
