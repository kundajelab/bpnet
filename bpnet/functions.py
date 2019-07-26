import numpy as np
import gin


@gin.configurable
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-2, keepdims=True))
    return e_x / e_x.sum(axis=-2, keepdims=True)


def mean(x):
    return sum(x) / len(x)
