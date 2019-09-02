"""
"""
import os
from pytest import fixture
from pathlib import Path
import gdown
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
def dataspec_bed6(data_dir):
    return data_dir / 'dataspec.bed6.yml'


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


@fixture(scope='session')
def download_dir():
    dir_path = '/tmp/bpnet'
    os.makedirs(dir_path, exist_ok=True)
    return Path(dir_path)


@fixture(scope='session')
def contrib_file(download_dir):
    """Download the contributon file
    """
    contrib_file = download_dir / 'contrib.deeplift.h5'
    url = 'https://drive.google.com/uc?id=1-70VlFvcOCwwt4SrEXoqkaXyBQPnlQGZ'
    md5 = '56e456f0d1aeffc9d3fcdfead0520c17'
    gdown.cached_download(url, str(contrib_file), md5=md5)
    return contrib_file


@fixture(scope='session')
def modisco_dir(download_dir):
    """Download the contributon file
    """
    _modisco_dir = download_dir / 'Oct4'
    _modisco_dir.mkdir(exist_ok=True)

    path = _modisco_dir / 'modisco-run.subset-contrib-file.npy'
    url = 'https://drive.google.com/uc?id=11uW8WaJ2EZuXQXUPq9g_TqrmWYTF-59V'
    md5 = '5b1425618cf87127f5a02cf54b2e361a'
    gdown.cached_download(url, str(path), md5=md5)

    path = _modisco_dir / 'modisco-run.kwargs.json'
    url = 'https://drive.google.com/uc?id=1zExhfQZ0-3irlpK6RrG1M-FBXvjHuQdm'
    md5 = '6bb33a8a0a2d0745ea9bf1ab2f5d241d'
    gdown.cached_download(url, str(path), md5=md5)

    path = _modisco_dir / 'modisco.h5'
    url = 'https://drive.google.com/uc?id=10owbBEB3PasIBSnMJ6KZQmJjbsyzhEgG'
    md5 = '8132fdbe7095748e8b229e81db45a6c9'
    gdown.cached_download(url, str(path), md5=md5)

    return _modisco_dir


@fixture(scope='session')
def mf(modisco_dir):
    """ModiscoFile
    """
    from bpnet.modisco.files import ModiscoFile
    mf = ModiscoFile(modisco_dir / 'modisco.h5')
    return mf


@fixture(scope='session')
def mfg(mf):
    """ModiscoFile
    """
    from bpnet.modisco.files import ModiscoFileGroup
    return ModiscoFileGroup({"Oct4": mf, "Sox2": mf})
