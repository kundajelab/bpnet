"""
"""
import os
from pytest import fixture
from pathlib import Path
from bpnet.cli.train import bpnet_train
from bpnet.cli.contrib import bpnet_contrib
import gin


@fixture
def test_dir():
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return Path(os.path.dirname(os.path.abspath(filename)))


@fixture
def data_dir(test_dir):
    return test_dir / 'data'


@fixture
def fasta_file(data_dir):
    return data_dir / 'dummy/dummy.fa'


@fixture
def regions(data_dir):
    return data_dir / 'dummy/peaks.bed'


@fixture
def genome_file(data_dir):
    return data_dir / 'dummy/dummy.genome'


@fixture
def dataspec_task1(data_dir):
    return data_dir / 'dataspec.task1.yml'


@fixture
def dataspec_bias(data_dir):
    return data_dir / 'dataspec.w-bias.yml'


@fixture
def config_gin(data_dir):
    return data_dir / 'config.gin'


@fixture(scope="session")
def trained_model(data_dir, dataspec_task1):
    gin.clear_config()
    bpnet_train(dataspec=dataspec_task1,
                output_dir=data_dir,
                run_id='trained_model',
                premate='bpnet9',
                config=config_gin,
                num_workers=1
                )
    return data_dir / 'trained_model'


@fixture(scope="session")
def trained_model_w_bias(config_gin, data_dir, dataspec_bias):
    gin.clear_config()
    bpnet_train(dataspec=dataspec_bias,
                output_dir=data_dir,
                run_id='trained_model_w_bias',
                premate='bpnet9',
                config=config_gin,
                num_workers=1
                )
    return data_dir / 'trained_model_w_bias'


@fixture(scope='session')
def contrib_score_grad(trained_model):
    fpath = trained_model / 'imp-score.grad.fixture.h5'
    bpnet_contrib(trained_model,
                  fpath,
                  method='grad')
    return fpath
