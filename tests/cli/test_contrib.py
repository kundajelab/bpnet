"""Test bpnet contrib
"""
import pytest
from bpnet.cli.contrib import bpnet_contrib, ContribFile


@pytest.mark.parametrize("method", ['deeplift', 'grad'])
def test_contrib_bias_model(tmp_path, method, trained_model_w_bias):
    """Test whether we can compute differnet contribution scores
    """
    fpath = tmp_path / 'imp-score.h5'
    bpnet_contrib(trained_model_w_bias,
                  fpath,
                  method=method)

    cf = ContribFile(fpath)
    assert cf.get_contrib()['Task1'].shape[-1] == 4


@pytest.mark.parametrize("method", ['deeplift', 'grad'])
def test_contrib_default_model(tmp_path, method, trained_model):
    """Test whether we can compute differnet contribution scores
    """
    fpath = tmp_path / 'imp-score.h5'
    bpnet_contrib(trained_model,
                  fpath,
                  method=method)

    cf = ContribFile(fpath)
    assert cf.get_contrib()['Task1'].shape[-1] == 4


@pytest.mark.parametrize("method", ['grad'])
def test_contrib_regions(tmp_path, method, trained_model, regions):
    """Test different scenarios regarding subsetting
    """
    bpnet_contrib(trained_model,
                  tmp_path / 'imp-score.h5',
                  method=method,
                  regions=regions)


@pytest.mark.parametrize("method", ['grad'])
def test_contrib_dataspec(tmp_path, method, trained_model, dataspec_bias, regions):
    """Test different scenarios regarding subsetting
    """
    bpnet_contrib(trained_model,
                  tmp_path / 'imp-score.h5',
                  method=method,
                  dataspec=dataspec_bias,
                  regions=regions)


@pytest.mark.parametrize("method", ['grad'])
def test_contrib_fasta_file(tmp_path, method, trained_model, fasta_file, regions):
    """Test different scenarios regarding subsetting
    """
    bpnet_contrib(trained_model,
                  tmp_path / 'imp-score.h5',
                  method=method,
                  fasta_file=fasta_file,
                  regions=regions)


@pytest.mark.parametrize("method", ['grad'])
def test_contrib_dataspec_fasta_file(tmp_path, method, trained_model, dataspec_bias, fasta_file, regions):
    """Test different scenarios regarding subsetting
    """
    with pytest.raises(ValueError):
        bpnet_contrib(trained_model,
                      tmp_path / 'imp-score.h5',
                      method=method,
                      fasta_file=fasta_file,
                      dataspec=dataspec_bias,
                      regions=regions)
