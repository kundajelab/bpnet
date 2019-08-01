"""Test bpnet train
"""
import os
import pytest
from bpnet.cli.train import bpnet_train
from pathlib import Path
from bpnet.seqmodel import SeqModel
from concise.preprocessing import encodeDNA
import gin
import keras.backend as K


def test_output_files(trained_model):
    K.clear_session()
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

    m = SeqModel.load(trained_model / 'seq_model.pkl')
    m.predict(encodeDNA(["A" * 200]))


def test_output_files_model_w_bias(trained_model_w_bias):
    K.clear_session()
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

    m = SeqModel.load(trained_model_w_bias / 'seq_model.pkl')
    m.predict(encodeDNA(["A" * 200]))


def test_trained_model_override_in_memory(tmp_path, data_dir, config_gin, dataspec_bias):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=str(dataspec_bias),
                output_dir=str(tmp_path),
                premade='bpnet9',
                config=str(config_gin),
                in_memory=True,
                override='seq_width=190;train.batch_size=8',
                num_workers=2
                )


def test_train_regions(tmp_path, data_dir, config_gin, dataspec_bias, regions):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=str(dataspec_bias),
                output_dir=str(tmp_path),
                premade='bpnet9',
                config=str(config_gin),
                override=f'bpnet_data.intervals_file="{regions}"',
                num_workers=2
                )


def test_trained_model_premade_pyspec(tmp_path, data_dir, config_gin, dataspec_bias):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=str(dataspec_bias),
                output_dir=str(tmp_path),
                premade='bpnet9-pyspec',
                config=str(config_gin),
                num_workers=2
                )


def test_trained_model_vmtouch(tmp_path, data_dir, config_gin, dataspec_bias):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=str(dataspec_bias),
                output_dir=str(tmp_path),
                premade='bpnet9',
                config=str(config_gin),
                vmtouch=True,
                num_workers=1
                )
