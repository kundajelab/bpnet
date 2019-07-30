"""Test gin config string parsing
"""


from bpnet.cli.train import gin2dict
GIN_STR = """
import bpnet
import bpnet.datasets
import bpnet.heads
import bpnet.layers
import bpnet.losses
import bpnet.metrics
import bpnet.models
import bpnet.seqmodel
import bpnet.trainers

# Macros:
# ==============================================================================
augment_interval = True
batchnorm = False
dataspec = 'dataspec.task1.yml'
exclude_chr = ['chr1', 'chr2']
filters = 64
lambda = 10
lr = 0.004
n_bias_tracks = 0
n_dil_layers = 1
seq_width = 200
tasks = ['Task1']
tconv_kernel_size = 25
test_chr = []
use_bias = False
valid_chr = ['chr2']

# Parameters for bpnet_data:
# ==============================================================================
bpnet_data.augment_interval = %augment_interval
bpnet_data.dataspec = %dataspec
bpnet_data.exclude_chr = %exclude_chr
bpnet_data.include_metadata = False
bpnet_data.interval_augmentation_shift = 100
bpnet_data.intervals_file = None
bpnet_data.peak_width = %seq_width
bpnet_data.seq_width = %seq_width
bpnet_data.shuffle = True
"""


def test_gin2dict():
    d = gin2dict(GIN_STR)
    assert d['bpnet_data.dataspec'] == 'dataspec.task1.yml'
    assert d['bpnet_data.peak_width'] == 200
    assert d['bpnet_data.seq_width'] == 200
    assert d['bpnet_data.exclude_chr'] == ['chr1', 'chr2']
