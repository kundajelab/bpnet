"""

NOTE: Always pass either a list or a dictionary of loss functions
when using model.compile(). Otherwise, the dict obtained by get_config()
during serialization will be confused with a dictionary of loss functions

In `keras.training.engine.py`

```
112        # Prepare loss functions.
113        if isinstance(loss, dict):
```
"""
import tensorflow as tf
import keras.losses as kloss
from concise.utils.helper import get_from_module
import keras.backend as K
import gin
from gin import config


@gin.configurable
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood

    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tf.contrib.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits)
    return -tf.reduce_sum(dist.log_prob(true_counts)) / tf.to_float(tf.shape(true_counts)[0])


@gin.configurable
class CountsMultinomialNLL:

    def __init__(self, n, c_task_weight=1):
        self.n = n
        self.c_task_weight = c_task_weight
        self.multinomial_nll = MultichannelMultinomialNLL(n)

    def __call__(self, true_counts, preds):
        probs = preds / K.sum(preds, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = self.multinomial_nll(true_counts, logits)

        mse_loss = kloss.mse(K.log(1 + K.sum(true_counts, axis=(-2, -1))),
                             K.log(1 + K.sum(preds, axis=(-2, -1))))

        return multinomial_loss + self.c_task_weight * mse_loss

    def get_config(self):
        return {"n": self.n,
                "c_task_weight": self.c_task_weight}


@gin.configurable
class PoissonMultinomialNLL:

    def __init__(self, n, c_task_weight=1):
        self.n = n
        self.c_task_weight = c_task_weight
        self.multinomial_nll = MultichannelMultinomialNLL(n)

    def __call__(self, true_counts, preds):
        probs = preds / K.sum(preds, axis=-2, keepdims=True)
        logits = K.log(probs / (1 - probs))

        # multinomial loss
        multinomial_loss = self.multinomial_nll(true_counts, logits)

        poisson_loss = kloss.poisson(K.sum(true_counts, axis=(-2, -1)),
                                     K.sum(preds, axis=(-2, -1)))

        return multinomial_loss + self.c_task_weight * poisson_loss

    def get_config(self):
        return {"n": self.n,
                "c_task_weight": self.c_task_weight}


@gin.configurable
class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}


mc_multinomial_nll_1 = MultichannelMultinomialNLL(1)
mc_multinomial_nll_1.__name__ = "mc_multinomial_nll_1"
config.external_configurable(mc_multinomial_nll_1)

mc_multinomial_nll_2 = MultichannelMultinomialNLL(2)
mc_multinomial_nll_2.__name__ = "mc_multinomial_nll_2"
config.external_configurable(mc_multinomial_nll_2)

twochannel_multinomial_nll = mc_multinomial_nll_2
twochannel_multinomial_nll.__name__ = "twochannel_multinomial_nll"


AVAILABLE = ["multinomial_nll",
             "twochannel_multinomial_nll",
             "mc_multinomial_nll_1",
             "mc_multinomial_nll_2",
             "MultichannelMultinomialNLL",
             "CountsMultinomialNLL",
             "PoissonMultinomialNLL"]


def get(name):
    try:
        return kloss.get(name)
    except ValueError:
        return get_from_module(name, globals())
