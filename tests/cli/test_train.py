"""Test bpnet train
"""
import os
import pytest
from bpnet.cli.train import bpnet_train
from pathlib import Path
from bpnet.seqmodel import SeqModel
from concise.preprocessing import encodeDNA
import gin

def test_output_files(trained_model):
    output_files = os.listdir(str(trained_model))
    expected_files = [
        'config.gin',
        'config.gin.json',
        'bpnet-train.kwargs.json',
        'dataspec.yml',
        'evaluate.ipynb',
        'evaluate.html',
        'evaluation.valid.json',
        'history.csv',
        'model.h5',
        'seq_model.pkl',
        'note_params.json',
    ]
    for f in expected_files:
        assert f in output_files

    m = SeqModel(trained_model / 'seq_model.pkl')
    m.predict(encodeDNA(["A" * 200]))


def test_output_files_model_w_bias(trained_model_w_bias):
    output_files = os.listdir(str(trained_model_w_bias))
    expected_files = [
        'config.gin',
        'config.gin.json',
        'bpnet-train.kwargs.json',
        'dataspec.yml',
        'evaluate.ipynb',
        'evaluate.html',
        'evaluation.valid.json',
        'history.csv',
        'model.h5',
        'seq_model.pkl',
        'note_params.json',
    ]
    for f in expected_files:
        assert f in output_files

    m = SeqModel(trained_model_w_bias / 'seq_model.pkl')
    m.predict(encodeDNA(["A" * 200]))


def test_trained_model_override_in_memory(tmp_path, data_dir, config_gin, dataspec_bias):
    gin.clear_config()
    bpnet_train(dataspec=dataspec_bias,
                output_dir=tmp_path,
                premade='bpnet9',
                config=config_gin,
                in_memory=True,
                override='seq_width=201;train.batch_size=8',
                num_workers=2
                )


def test_trained_model_premade_pyspec(tmp_path, data_dir, config_gin, dataspec_bias):
    gin.clear_config()
    bpnet_train(dataspec=dataspec_bias,
                output_dir=tmp_path,
                premade='bpnet9-pyspec.gin',
                config=config_gin,
                num_workers=2
                )


def test_trained_model_vmtouch(tmp_path, data_dir, config_gin, dataspec_bias):
    gin.clear_config()
    bpnet_train(dataspec=dataspec_bias,
                output_dir=tmp_path,
                premade='bpnet9',
                config=config_gin,
                vmtouch=True,
                num_workers=1
                )
