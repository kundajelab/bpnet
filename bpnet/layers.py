import keras.layers as kl
from keras.engine.topology import Layer
import tensorflow as tf
from concise.utils.helper import get_from_module
from concise.layers import SplineWeight1D
from keras.models import Model, Sequential
import numpy as np
import gin


@gin.configurable
class GlobalAvgPoolFCN:

    def __init__(self,
                 n_tasks=1,
                 dropout=0,
                 hidden=None,
                 dropout_hidden=0,
                 n_splines=0,
                 batchnorm=False):
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.n_splines = n_splines
        self.hidden = hidden if hidden is not None else []
        assert self.n_splines >= 0

    def __call__(self, x):
        if self.n_splines == 0:
            x = kl.GlobalAvgPool1D()(x)
        else:
            # Spline-transformation for the position aggregation
            # This allows to up-weight positions in the middle
            x = SplineWeight1D(n_bases=self.n_splines,
                               share_splines=True)(x)
            x = kl.GlobalAvgPool1D()(x)

        if self.dropout:
            x = kl.Dropout(self.dropout)(x)

        # Hidden units (not used by default)
        for h in self.hidden:
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Dense(h, activation='relu')(x)
            if self.dropout_hidden:
                x = kl.Dropout(self.dropout_hidden)(x)

        # Final dense layer
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Dense(self.n_tasks)(x)
        return x


@gin.configurable
class FCN:

    def __init__(self,
                 n_tasks=1,
                 hidden=None,
                 dropout=0,
                 dropout_hidden=0,
                 batchnorm=False):
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.hidden = hidden if hidden is not None else []

    def __call__(self, x):
        if self.dropout:
            x = kl.Dropout(self.dropout)(x)

        # Hidden units (not used by default)
        for h in self.hidden:
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Dense(h, activation='relu')(x)
            if self.dropout_hidden:
                x = kl.Dropout(self.dropout_hidden)(x)

        # Final dense layer
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Dense(self.n_tasks)(x)
        return x


@gin.configurable
class DilatedConv1D:
    """Dillated convolutional layers

    - add_pointwise -> if True, add a 1by1 conv right after the first conv
    """

    def __init__(self, filters=21,
                 conv1_kernel_size=25,
                 n_dil_layers=6,
                 skip_type='residual',  # or 'dense', None
                 padding='same',
                 batchnorm=False,
                 add_pointwise=False):
        self.filters = filters
        self.conv1_kernel_size = conv1_kernel_size
        self.n_dil_layers = n_dil_layers
        self.skip_type = skip_type
        self.padding = padding
        self.batchnorm = batchnorm
        self.add_pointwise = add_pointwise

    def __call__(self, inp):
        """inp = (None, 4)
        """
        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               padding='same',
                               activation='relu')(inp)
        if self.add_pointwise:
            if self.batchnorm:
                first_conv = kl.BatchNormalization()(first_conv)
            first_conv = kl.Conv1D(self.filters,
                                   kernel_size=1,
                                   padding='same',
                                   activation='relu')(first_conv)

        prev_layer = first_conv
        for i in range(1, self.n_dil_layers + 1):
            if self.batchnorm:
                x = kl.BatchNormalization()(prev_layer)
            else:
                x = prev_layer
            conv_output = kl.Conv1D(self.filters, kernel_size=3, padding='same',
                                    activation='relu', dilation_rate=2**i)(x)

            # skip connections
            if self.skip_type is None:
                prev_layer = conv_output
            elif self.skip_type == 'residual':
                prev_layer = kl.add([prev_layer, conv_output])
            elif self.skip_type == 'dense':
                prev_layer = kl.concatenate([prev_layer, conv_output])
            else:
                raise ValueError("skip_type needs to be 'add' or 'concat' or None")

        combined_conv = prev_layer

        if self.padding == 'valid':
            # Trim the output to only valid sizes
            # (e.g. essentially doing valid padding with skip-connections)
            combined_conv = kl.Cropping1D(cropping=-self.get_len_change() // 2)(combined_conv)

        # add one more layer in between for densly connected layers to reduce the
        # spatial dimension
        if self.skip_type == 'dense':
            combined_conv = kl.Conv1D(self.filters,
                                      kernel_size=1,
                                      padding='same',
                                      activation='relu')(combined_conv)
        return combined_conv

    def get_len_change(self):
        """How much will the length change
        """
        if self.padding == 'same':
            return 0
        else:
            d = 0
            # conv
            d -= 2 * (self.conv1_kernel_size // 2)
            for i in range(1, self.n_dil_layers + 1):
                dillation = 2**i
                d -= 2 * dillation
            return d


@gin.configurable
class DeConv1D:
    def __init__(self, filters, n_tasks,
                 tconv_kernel_size=25,
                 padding='same',
                 n_hidden=0,
                 batchnorm=False):
        self.filters = filters
        self.n_tasks = n_tasks
        self.tconv_kernel_size = tconv_kernel_size
        self.n_hidden = n_hidden
        self.batchnorm = batchnorm
        self.padding = padding

    def __call__(self, x):

        # `hidden` conv layers
        for i in range(self.n_hidden):
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Conv1D(self.filters,
                          kernel_size=1,
                          padding='same',  # anyway doesn't matter
                          activation='relu')(x)

        # single de-conv layer
        x = kl.Reshape((-1, 1, self.filters))(x)
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Conv2DTranspose(self.n_tasks, kernel_size=(self.tconv_kernel_size, 1), padding='same')(x)
        x = kl.Reshape((-1, self.n_tasks))(x)

        # TODO - allow multiple de-conv layers

        if self.padding == 'valid':
            # crop to make it translationally invariant
            x = kl.Cropping1D(cropping=-self.get_len_change() // 2)(x)
        return x

    def get_len_change(self):
        """How much will the length change
        """
        if self.padding == 'same':
            return 0
        else:
            return - 2 * (self.tconv_kernel_size // 2)


@gin.configurable
class MovingAverages:
    """Layer to compute moving averages at multiple resolutions
    followed by a conv layer
    """

    def __init__(self, window_sizes):
        self.window_sizes = window_sizes

    def __call__(self, x):
        # x.shape = (batch, seqlen, features)
        out = []
        for window_size in self.window_sizes:
            if window_size == 1:
                # no need to perform the convolution
                out.append(x)
            else:
                conv = kl.SeparableConv1D(1,
                                          kernel_size=window_size,
                                          padding='same',
                                          depthwise_initializer='ones',
                                          pointwise_initializer='ones',
                                          use_bias=False,
                                          trainable=False)
                out.append(conv(x))
        # (batch, seqlen, len(window_sizes))
        binp = kl.concatenate(out)
        return kl.Conv1D(1, kernel_size=1, use_bias=False)(binp)


AVAILABLE = []


def get(name):
    return get_from_module(name, globals())
