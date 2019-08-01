"""Test sequence model
"""
from bpnet.seqmodel import SeqModel
from bpnet.heads import ScalarHead, BinaryClassificationHead, ProfileHead
import numpy as np
import keras.layers as kl


class TopDense:
    """Class to be used as functional model interpretation
    """

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def __call__(self, inp):
        x = kl.GlobalAvgPool1D()(inp)
        return kl.Dense(1)(x)


class TopConv:
    """Class to be used as functional model interpretation
    """

    def __init__(self, n_output=2):
        self.n_output = n_output

    def __call__(self, inp):
        return kl.Conv1D(self.n_output, 1)(inp)


class BaseNet:
    """Class to be used as functional model interpretation
    """

    def __init__(self, activation='relu'):
        self.activation = activation

    def __call__(self, inp):
        x = kl.Conv1D(16, kernel_size=3, activation=self.activation, padding='same')(inp)
        return x


def test_interpret_wo_bias():
    from bpnet.metrics import RegressionMetrics, ClassificationMetrics, PeakPredictionProfileMetric
    from concise.preprocessing import encodeDNA
    # test the model
    seqs = encodeDNA(['ACAGA'] * 100)

    inputs = {"seq": seqs,
              "bias/a/profile": np.random.randn(100, 5, 2)}

    # Let's use regression
    targets = {"a/class": np.random.randint(low=0, high=2, size=(100, 1)).astype(float),
               "a/counts": 1 + np.ceil(np.abs(np.random.randn(100))),
               "a/profile": 1 + np.ceil(np.abs(np.random.randn(100, 5, 2))),
               }

    import keras.backend as K
    # K.clear_session()
    # use bias
    m = SeqModel(
        body=BaseNet('relu'),
        heads=[BinaryClassificationHead('{task}/class',
                                        net=TopDense(pool_size=2),
                                        use_bias=False),
               ScalarHead('{task}/counts',
                          loss='mse',
                          metric=RegressionMetrics(),
                          net=TopDense(pool_size=2),
                          use_bias=False),
               ProfileHead('{task}/profile',
                           loss='mse',
                           metric=PeakPredictionProfileMetric(neg_max_threshold=0.05,
                                                              required_min_pos_counts=0),
                           net=TopConv(n_output=2),
                           use_bias=True,
                           bias_shape=(5, 2)),  # NOTE: the shape currently has to be hard-coded to the sequence length
               ],
        tasks=['a']
    )
    m.model.fit(inputs, targets)

    o = m.contrib_score_all(seqs)
    assert 'a/profile/wn' in o
    assert o['a/profile/wn'].shape == seqs.shape
    assert 'a/profile/wn' in o
    assert o['a/profile/wn'].shape == seqs.shape

    # evaluate the dataset -> setup an array dataset (NumpyDataset) -> convert to
    from bpnet.data import NumpyDataset
    ds = NumpyDataset({"inputs": inputs, "targets": targets})
    o = m.evaluate(ds)
    assert 'avg/counts/mad' in o
