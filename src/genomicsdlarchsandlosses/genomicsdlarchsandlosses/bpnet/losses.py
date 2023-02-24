import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as kb


class CustomMeanSquaredError(object):
    """ Custom class to compute mean squared error
        but ignore the incoming sample weights
    
    """
    
    def __init__(self):
        self.__name__ = "CustomMeanSquaredError"

    def __call__(self, y_true, y_pred):
        
        mse = tf.keras.losses.MeanSquaredError()
        
        return mse(y_true, y_pred)

    def get_config(self):
        return {}


def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logits values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
    """ Class to compute combined loss from 'n' tasks
    
        Args:
            n (int): the number of channels / tasks 
    """
    
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        total = 0
        
        # only keep those samples with non zero weight,
        # here we assume non-zero is 1
        
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}

