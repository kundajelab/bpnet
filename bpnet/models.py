import numpy as np
import keras.layers as kl
from keras.optimizers import Adam
from keras.models import Model
from concise.utils.helper import get_from_module
import bpnet
import bpnet.losses as blosses
from bpnet.losses import twochannel_multinomial_nll, MultichannelMultinomialNLL
import gin
import keras


# TODO - setup the following model as a simple bpnet (?)

@gin.configurable
def multihead_seq_model(tasks,
                        filters,
                        n_dil_layers,
                        conv1_kernel_size,
                        tconv_kernel_size,
                        b_loss_weight=1,
                        c_loss_weight=1,
                        p_loss_weight=1,
                        c_splines=20,
                        p_splines=0,
                        merge_profile_reg=False,
                        lr=0.004,
                        padding='same',
                        batchnorm=False,
                        use_bias=False,
                        n_profile_bias_tracks=2,
                        n_bias_tracks=2,
                        seqlen=None,
                        skip_type='residual'):
    from bpnet.seqmodel import SeqModel
    from bpnet.layers import DilatedConv1D, DeConv1D, GlobalAvgPoolFCN
    from bpnet.metrics import BPNetMetricSingleProfile, default_peak_pred_metric
    from bpnet.heads import ScalarHead, ProfileHead
    from bpnet.metrics import ClassificationMetrics, RegressionMetrics
    from bpnet.losses import mc_multinomial_nll_2, CountsMultinomialNLL
    from bpnet.activations import clipped_exp
    from bpnet.functions import softmax

    assert p_loss_weight >= 0
    assert c_loss_weight >= 0
    assert b_loss_weight >= 0

    # Heads -------------------------------------------------
    heads = []
    # Profile prediction
    if p_loss_weight > 0:
        if not merge_profile_reg:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=2,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=0,
                                                  batchnorm=batchnorm
                                                  ),
                                     loss=mc_multinomial_nll_2,
                                     loss_weight=p_loss_weight,
                                     postproc_fn=softmax,
                                     use_bias=use_bias,
                                     bias_input='bias/{task}/profile',
                                     bias_shape=(None, n_profile_bias_tracks),
                                     metric=default_peak_pred_metric
                                     ))
        else:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=2,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=1,  # use 1 hidden layer in that case
                                                  batchnorm=batchnorm
                                                  ),
                                     activation=clipped_exp,
                                     loss=CountsMultinomialNLL(2, c_task_weight=c_loss_weight),
                                     loss_weight=p_loss_weight,
                                     bias_input='bias/{task}/profile',
                                     use_bias=use_bias,
                                     bias_shape=(None, n_profile_bias_tracks),
                                     metric=BPNetMetricSingleProfile(count_metric=RegressionMetrics(),
                                                                     profile_metric=default_peak_pred_metric)
                                     ))
            c_loss_weight = 0  # don't need to use the other count loss

    # Count regression
    if c_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/counts',
                                net=GlobalAvgPoolFCN(n_tasks=2,
                                                     n_splines=p_splines,
                                                     batchnorm=batchnorm),
                                activation=None,
                                loss='mse',
                                loss_weight=c_loss_weight,
                                bias_input='bias/{task}/counts',
                                use_bias=use_bias,
                                bias_shape=(n_bias_tracks, ),
                                metric=RegressionMetrics(),
                                ))

    # Binary classification
    if b_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/class',
                                net=GlobalAvgPoolFCN(n_tasks=1,
                                                     n_splines=c_splines,
                                                     batchnorm=batchnorm),
                                activation='sigmoid',
                                loss='binary_crossentropy',
                                loss_weight=b_loss_weight,
                                metric=ClassificationMetrics(),
                                ))
    # -------------------------------------------------
    m = SeqModel(
        body=DilatedConv1D(filters=filters,
                           conv1_kernel_size=conv1_kernel_size,
                           n_dil_layers=n_dil_layers,
                           padding=padding,
                           batchnorm=batchnorm,
                           skip_type=skip_type),
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


@gin.configurable
def binary_seq_model(tasks,
                     net_body,
                     net_head,
                     lr=0.004,
                     seqlen=None):
    """NOTE: This doesn't work with gin-train since
    the classes injected by gin-config can't be pickled.

    Instead, I created `basset_seq_model`

    ```
    Can't pickle <class 'bpnet.layers.BassetConv'>: it's not the same
    object as bpnet.layers.BassetConv
    ```

    """
    from bpnet.seqmodel import SeqModel
    from bpnet.heads import ScalarHead, ProfileHead
    from bpnet.metrics import ClassificationMetrics
    # Heads -------------------------------------------------
    heads = [ScalarHead(target_name='{task}/class',
                        net=net_head,
                        activation='sigmoid',
                        loss='binary_crossentropy',
                        metric=ClassificationMetrics(),
                        )]
    # -------------------------------------------------
    m = SeqModel(
        body=net_body,
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


def get(name):
    return get_from_module(name, globals())
