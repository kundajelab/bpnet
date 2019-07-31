"""Head modules
"""
import numpy as np
from bpnet.utils import dict_prefix_key
from bpnet.metrics import ClassificationMetrics, RegressionMetrics
import keras.backend as K
import tensorflow as tf
import keras.layers as kl
import gin
import os
import abc


class BaseHead:

    # loss
    # weight -> loss weight (1 by default)
    # kwargs -> kwargs for the model
    # name -> name of the module
    # _model -> gets setup in the init

    @abc.abstractmethod
    def get_target(self, task):
        pass

    @abc.abstractmethod
    def __call__(self, inp, task):
        """Useful for writing together the model
        Returns the output tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_preact_tensor(self, graph=None):
        """Return the single pre-activation tensors
        """
        pass

    @abc.abstractmethod
    def intp_tensors(self, preact_only=False, graph=None):
        """Dictionary of all available interpretation tensors
        for `get_interpretation_node`
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def get_intp_tensor(self, which=None):
    #     """Returns a target tensor which is a scalar
    #     w.r.t. to which to compute the outputs

    #     Args:
    #       which [string]: If None, use the default
    #       **kwargs: optional kwargs for the interpretation method

    #     Returns:
    #       scalar tensor
    #     """
    #     raise NotImplementedError

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


class BaseHeadWBias(BaseHead):

    @abc.abstractmethod
    def get_bias_input(self, task):
        pass

    @abc.abstractmethod
    def neutral_bias_input(self, task, length, seqlen):
        pass


def id_fn(x):
    return x


def named_tensor(x, name):
    return kl.Lambda(id_fn, name=name)(x)


# --------------------------------------------
# Head implementations

@gin.configurable
class ScalarHead(BaseHeadWBias):

    def __init__(self, target_name,  # "{task}/scalar"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation=None,
                 loss='mse',
                 loss_weight=1,
                 metric=RegressionMetrics(),
                 postproc_fn=None,  # post-processing to apply so that we are in the right scale
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/scalar',
                 bias_shape=(1,),
                 ):
        self.net = net
        self.loss = loss
        self.loss_weight = loss_weight
        self.metric = metric
        self.postproc_fn = postproc_fn
        self.target_name = target_name
        self.activation = activation
        self.bias_input = bias_input
        self.bias_net = bias_net
        self.use_bias = use_bias
        self.bias_shape = bias_shape

    def get_target(self, task):
        return self.target_name.format(task=task)

    def __call__(self, inp, task):
        o = self.net(inp)

        # remember the tensors useful for interpretation (referred by name)
        self.pre_act = o.name

        # add the target bias
        if self.use_bias:
            binp = kl.Input(self.bias_shape, name=self.get_bias_input(task))
            bias_inputs = [binp]

            # add the bias term
            if self.bias_net is not None:
                bias_x = self.bias_net(binp)
                # This allows to normalize the bias data first
                # (e.g. when we have profile counts to aggregate it first)
            else:
                # Don't use the nn 'bias' so that when the measurement bias = 0,
                # this term vanishes
                bias_x = kl.Dense(1, use_bias=False)(binp)
            o = kl.add([o, bias_x])
        else:
            bias_inputs = []

        if self.activation is not None:
            if isinstance(self.activation, str):
                o = kl.Activation(self.activation)(o)
            else:
                o = self.activation(o)

        self.post_act = o.name

        # label the target op so that we can use a dictionary of targets
        # to train the model
        return named_tensor(o, name=self.get_target(task)), bias_inputs

    def get_preact_tensor(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        return graph.get_tensor_by_name(self.pre_act)

    def intp_tensors(self, preact_only=False, graph=None):
        """Return the required interpretation tensors
        """
        if graph is None:
            graph = tf.get_default_graph()

        if self.activation is None:
            # the post-activation doesn't
            # have any specific meaning when
            # we don't use any activation function
            return {"pre-act": graph.get_tensor_by_name(self.pre_act)}

        if preact_only:
            return {"pre-act": graph.get_tensor_by_name(self.pre_act)}
        else:
            return {"pre-act": graph.get_tensor_by_name(self.pre_act),
                    "output": graph.get_tensor_by_name(self.post_act)}

    # def get_intp_tensor(self, which='pre-act'):
    #     return self.intp_tensors()[which]

    def get_bias_input(self, task):
        return self.bias_input.format(task=task)

    def neutral_bias_input(self, task, length, seqlen):
        """Create dummy bias input

        Return: (k, v) tuple
        """
        shape = tuple([x if x is not None else seqlen
                       for x in self.bias_shape])
        return (self.get_bias_input(task), np.zeros((length, ) + shape))


@gin.configurable
class BinaryClassificationHead(ScalarHead):

    def __init__(self, target_name,  # "{task}/scalar"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation='sigmoid',
                 loss='binary_crossentropy',
                 loss_weight=1,
                 metric=ClassificationMetrics(),
                 postproc_fn=None,
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/scalar',
                 bias_shape=(1,),
                 ):
        # override the default
        super().__init__(target_name,
                         net,
                         activation=activation,
                         loss=loss,
                         metric=metric,
                         postproc_fn=postproc_fn,
                         use_bias=use_bias,
                         bias_net=bias_net,
                         bias_input=bias_input,
                         bias_shape=bias_shape)

        # TODO - mabye override the way we call outputs?


@gin.configurable
class ProfileHead(BaseHeadWBias):
    """Deals with the case where the output are multiple tracks of
    total shape (L, C) (L = sequence length, C = number of channels)

    Note: Since the contribution score will be a single scalar, the
    interpretation method will have to aggregate both across channels
    as well as positions
    """

    def __init__(self, target_name,  # "{task}/profile"
                 net,  # function that takes a keras tensor and returns a keras tensor
                 activation=None,
                 loss='mse',
                 loss_weight=1,
                 metric=RegressionMetrics(),
                 postproc_fn=None,
                 # bias input
                 use_bias=False,
                 bias_net=None,
                 bias_input='bias/{task}/profile',
                 bias_shape=(None, 1),
                 ):
        self.net = net
        self.loss = loss
        self.loss_weight = loss_weight
        self.metric = metric
        self.postproc_fn = postproc_fn
        self.target_name = target_name
        self.activation = activation
        self.bias_input = bias_input
        self.bias_net = bias_net
        self.use_bias = use_bias
        self.bias_shape = bias_shape

    def get_target(self, task):
        return self.target_name.format(task=task)

    def __call__(self, inp, task):
        o = self.net(inp)

        # remember the tensors useful for interpretation (referred by name)
        self.pre_act = o.name

        # add the target bias
        if self.use_bias:
            binp = kl.Input(self.bias_shape, name=self.get_bias_input(task))
            bias_inputs = [binp]

            # add the bias term
            if self.bias_net is not None:
                bias_x = self.bias_net(binp)
                # This allows to normalize the bias data first
                # (e.g. when we have profile counts to aggregate it first)
            else:
                # Don't use the nn 'bias' so that when the measurement bias = 0,
                # this term vanishes
                bias_x = kl.Conv1D(1, kernel_size=1, use_bias=False)(binp)
            o = kl.add([o, bias_x])
        else:
            bias_inputs = []

        if self.activation is not None:
            if isinstance(self.activation, str):
                o = kl.Activation(self.activation)(o)
            else:
                o = self.activation(o)

        self.post_act = o.name

        # label the target op so that we can use a dictionary of targets
        # to train the model
        return named_tensor(o, name=self.get_target(task)), bias_inputs

    def get_preact_tensor(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        return graph.get_tensor_by_name(self.pre_act)

    @staticmethod
    def profile_contrib(p):
        """Summarizing the profile for the contribution scores

        wn: Normalized contribution (weighted sum of the contribution scores)
          where the weighted sum uses softmax(p) to weight it
        w2: Simple sum (p**2)
        w1: sum(p)
        winf: max(p)
        """
        # Note: unfortunately we have to use the kl.Lambda boiler-plate
        # to be able to do Model(inp, outputs) in deep-explain code

        # Normalized contribution  - # TODO - update with tensorflow
        wn = kl.Lambda(lambda p:
                       K.mean(K.sum(K.stop_gradient(tf.nn.softmax(p, dim=-2)) * p, axis=-2), axis=-1)
                       )(p)

        # Squared weight
        w2 = kl.Lambda(lambda p:
                       K.mean(K.sum(p * p, axis=-2), axis=-1)
                       )(p)

        # W1 weight
        w1 = kl.Lambda(lambda preact_m:
                       K.mean(K.sum(preact_m, axis=-2), axis=-1)
                       )(p)

        # Winf
        # 1. max across the positional axis, average the strands
        winf = kl.Lambda(lambda p:
                         K.mean(K.max(p, axis=-2), axis=-1)
                         )(p)

        return {"wn": wn,
                "w1": w1,
                "w2": w2,
                "winf": winf
                }

    def intp_tensors(self, preact_only=False, graph=None):
        """Return the required interpretation tensors (scalars)

        Note: Since we are predicting a track,
            we should return a single scalar here
        """
        if graph is None:
            graph = tf.get_default_graph()

        preact = graph.get_tensor_by_name(self.pre_act)
        postact = graph.get_tensor_by_name(self.post_act)

        # Contruct the profile summary ops
        preact_tensors = self.profile_contrib(preact)
        postact_tensors = dict_prefix_key(self.profile_contrib(postact), 'output_')

        if self.activation is None:
            # the post-activation doesn't
            # have any specific meaning when
            # we don't use any activation function
            return preact_tensors

        if preact_only:
            return preact_tensors
        else:
            return {**preact_tensors, **postact_tensors}

    # def get_intp_tensor(self, which='wn'):
    #     return self.intp_tensors()[which]

    def get_bias_input(self, task):
        return self.bias_input.format(task=task)

    def neutral_bias_input(self, task, length, seqlen):
        """Create dummy bias input

        Return: (k, v) tuple
        """
        shape = tuple([x if x is not None else seqlen
                       for x in self.bias_shape])
        return (self.get_bias_input(task), np.zeros((length, ) + shape))
