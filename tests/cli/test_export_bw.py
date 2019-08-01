"""Test bpnet_export_bw
"""
import os
from bpnet.cli.export_bw import bpnet_export_bw
import keras.backend as K

EXPECTED_FILES = ['Task1.contrib.counts.bw',
                  'Task1.contrib.profile.bw',
                  'Task1.preds.pos.bw',
                  'Task1.preds.neg.bw',
                  'log'
                  ]


def test_export_bw_w_bias(tmp_path, trained_model_w_bias):
    K.clear_session()
    bpnet_export_bw(str(trained_model_w_bias),
                    str(tmp_path),
                    contrib_method='grad',
                    scale_contribution=True)

    output_files = os.listdir(tmp_path)
    for f in EXPECTED_FILES:
        assert f in output_files


def test_export_bw(tmp_path, trained_model, regions):
    K.clear_session()
    bpnet_export_bw(str(trained_model),
                    str(tmp_path),
                    contrib_method='grad',
                    regions=str(regions))

    output_files = os.listdir(tmp_path)
    for f in EXPECTED_FILES:
        assert f in output_files
