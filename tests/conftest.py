"""
"""
import os
from pytest import fixture
from pathlib import Path
from bpnet.cli.train import bpnet_train
from bpnet.cli.contrib import bpnet_contrib
import gin
import keras.backend as K


@fixture(scope='session')
def test_dir():
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return Path(os.path.dirname(os.path.abspath(filename)))


@fixture(scope='session')
def data_dir(test_dir):
    return test_dir / 'data'


@fixture(scope='session')
def fasta_file(data_dir):
    return data_dir / 'dummy/dummy.fa'


@fixture(scope='session')
def regions(data_dir):
    return data_dir / 'dummy/peaks.bed'


@fixture(scope='session')
def genome_file(data_dir):
    return data_dir / 'dummy/dummy.genome'


@fixture(scope='session')
def dataspec_task1(data_dir):
    return data_dir / 'dataspec.task1.yml'


@fixture(scope='session')
def dataspec_bias(data_dir):
    return data_dir / 'dataspec.w-bias.yml'


@fixture(scope='session')
def config_gin(data_dir):
    return data_dir / 'config.gin'

@fixture(scope='session')
def modisco_config_gin(data_dir):
    return data_dir / 'modisco-config.gin'


@fixture(scope="session")
def trained_model(data_dir, dataspec_task1, config_gin):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=dataspec_task1,
                output_dir=data_dir,
                run_id='trained_model',
                premade='bpnet9',
                config=str(config_gin),
                num_workers=1,
                overwrite=True
                )
    return data_dir / 'trained_model'


@fixture(scope="session")
def trained_model_w_bias(config_gin, data_dir, dataspec_bias):
    K.clear_session()
    gin.clear_config()
    bpnet_train(dataspec=str(dataspec_bias),
                output_dir=str(data_dir),
                run_id='trained_model_w_bias',
                premade='bpnet9',
                config=str(config_gin),
                num_workers=1,
                overwrite=True,
                )
    return data_dir / 'trained_model_w_bias'


@fixture(scope='session')
def contrib_score_grad(trained_model):
    K.clear_session()
    fpath = trained_model / 'imp-score.grad.fixture.h5'
    bpnet_contrib(str(trained_model),
                  str(fpath),
                  method='grad',
                  overwrite=True)
    return fpath


@fixture(scope='session')
def contrib_score_grad_null(trained_model):
    K.clear_session()
    fpath = trained_model / 'imp-score.grad.null.fixture.h5'
    bpnet_contrib(str(trained_model),
                  str(fpath),
                  method='grad',
                  shuffle_seq=True,
                  max_regions=16,
                  overwrite=True)
    return fpath
