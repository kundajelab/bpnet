import keras
import keras.backend as K
import keras.layers as kl
from concise.utils.helper import get_from_module
from keras.activations import softmax
import tensorflow as tf
import gin


@gin.configurable
def clipped_exp(x, min_value=-50, max_value=50):
    return kl.Lambda(lambda x, min_value, max_value: K.exp(K.clip(x, min_value, max_value)),
                     arguments={"min_value": min_value, "max_value": max_value})(x)


@gin.configurable
def softmax_2(x):
    """
    Softmax along the second-last axis
    """
    return kl.Lambda(lambda x: tf.nn.softmax(x, axis=-2))(x)


AVAILABLE = ["clipped_exp", "softmax_2", 'tf']


def get(name):
    return get_from_module(name, globals())
