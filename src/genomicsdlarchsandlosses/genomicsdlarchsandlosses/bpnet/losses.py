import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as kb

from genomicsdlarchsandlosses.bpnet.attribution_prior_utils import \
    smooth_tensor_1d



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


def fourier_att_prior_loss(
    status, input_grads, freq_limit, limit_softness,
    att_prior_grad_smooth_sigma):
    
    """
    Computes an attribution prior loss for some given training 
    examples, using a Fourier transform form.
    
    Args:
        status (tensor): a B-tensor, where B is the batch size; each
            entry is 1 if that example is to be treated as a positive
            example, and 0 otherwise
        input_grads (tensor): a B x L x 4 tensor, where B is the batch
            size, L is the length of the input; this needs to be the
            gradients of the input with respect to the output; this
            should be *gradient times input*
        freq_limit (int): the maximum integer frequency index, k, to
            consider for the loss; this corresponds to a frequency
            cut-off of pi * k / L; k should be less than L / 2
        limit_softness (float): amount to soften the limit by, using
            a hill function; None means no softness
        att_prior_grad_smooth_sigma (float): amount to smooth the
            gradient before computing the loss
            
    Returns:
        tensor: a single scalar Tensor consisting of the attribution
        loss for the batch.
    
    """
    abs_grads = kb.sum(kb.abs(input_grads), axis=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )

    # Only do the positives
    pos_grads = grads_smooth[status == 1]

    def _zero_constant():
        return kb.constant(0)
    
    def _fourier_att_prior_loss(pos_grads):
        pos_fft = tf.signal.rfft(pos_grads)
        pos_mags = tf.abs(pos_fft)
        pos_mag_sum = kb.sum(pos_mags, axis=1, keepdims=True)
        zero_mask = tf.cast(pos_mag_sum == 0, tf.float32)
        # Keep 0s when the sum is 0  
        pos_mag_sum = pos_mag_sum + zero_mask  
        pos_mags = pos_mags / pos_mag_sum

        # Cut off DC
        pos_mags = pos_mags[:, 1:]

        # Construct weight vector
        if limit_softness is None:
            weights = tf.sequence_mask(
                [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32)
        else:
            weights = tf.sequence_mask(
                [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32)
            # Take absolute value of negatives just to avoid NaN;
            # they'll be removed
            x = tf.abs(tf.range(
                -freq_limit + 1, tf.shape(pos_mags)[1] - freq_limit + 1, 
                dtype=tf.float32))  
            decay = 1 / (1 + tf.pow(x, limit_softness))
            weights = weights + ((1.0 - weights) * decay)

        # Multiply frequency magnitudes by weights
        pos_weighted_mags = pos_mags * weights

        # Add up along frequency axis to get score
        pos_score = tf.reduce_sum(pos_weighted_mags, axis=1)
        pos_loss = 1 - pos_score
        return kb.mean(pos_loss)
    
    
    # if size is 0 then return 0, else compute attribution prior loss
    return tf.cond(tf.equal(tf.size(pos_grads), 0), 
                   _zero_constant,
                   lambda:  _fourier_att_prior_loss(pos_grads))
