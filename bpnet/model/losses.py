import tensorflow as tf
import tensorflow_probability as tfp


def mse_loss_function(y_log_true, y_log_pred):
    # logcounts mse loss without sample weights
    mse_loss = tf.keras.losses.mean_squared_error(
        y_log_true, y_log_pred)
    return mse_loss

def poisson_loss_function(y_log_true, y_log_pred):
    # we can use the Possion PMF from TensorFlow as well
    # dist = tf.contrib.distributions
    # return -tf.reduce_mean(dist.Poisson(y_pred).log_pmf(y_true))

    # last term can be avoided since it doesn't depend on y_pred
    # however keeping it gives a nice lower bound to zero
    
    y_true = tf.math.exp(y_log_true) # -1? 
    y_pred = tf.math.exp(y_log_pred)
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)
    loss = y_pred - y_true*tf.math.log(y_pred+1e-8) + tf.math.lgamma(y_true+1.0)

    return loss

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
