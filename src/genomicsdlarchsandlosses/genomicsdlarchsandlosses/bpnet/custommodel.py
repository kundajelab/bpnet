import tensorflow as tf
import tensorflow.keras.backend as kb

from genomicsdlarchsandlosses.bpnet.losses import multinomial_nll

from tensorflow import keras
from tensorflow.keras import Model
import tensorflow_probability as tfp

class CustomModel(Model):

    def __init__(self, num_tasks, num_output_tracks, tracks_for_each_task, output_profile_len, loss_weights,counts_loss, orig_multi_loss=False, **kwargs):

        # call the base class with inputs and outputs
        super(CustomModel, self).__init__(**kwargs)
        
        # number of tasks
        self.num_tasks = num_tasks
        
        # number of output tracks used for original multinomial loss
        self.num_output_tracks = num_output_tracks
        
        self.orig_multi_loss = orig_multi_loss
        
        # number of tracks for each task
        self.tracks_for_each_task = tracks_for_each_task
        
        # output profile length
        self.output_profile_len = output_profile_len
        
        # weights for the profile mnll and logcounts losses
        self.loss_weights = loss_weights
        
        # logcounts loss funtion
        self.counts_loss = counts_loss
        
        # object to track overall mean loss per epoch
        self.loss_tracker = keras.metrics.Mean(name="loss")


    def _get_loss(self, x, y, sample_weights, training=True):
        # boolean mask for sample weights != 0
        
                
        y_pred = self(x, training=training)  # Forward pass
        
        
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
        
        def _poisson_loss_function(_y_log_true,_y_log_pred):
            total_poisson_loss = 0
            track_count_cuml = 0
            if self.orig_multi_loss:
                for i in range(self.num_output_tracks):
                    y_log_true = _y_log_true[:,i:(i+1)][:,-1]
                    y_log_pred = _y_log_pred[:,i:(i+1)][:,-1]
                    loss = poisson_loss_function(y_log_true, y_log_pred)
                    total_poisson_loss += loss               
            else:
                for i in range(self.num_tasks):
                    num_of_tracks = self.tracks_for_each_task[i]
                    y_log_true = tf.reduce_logsumexp(_y_log_true[:,track_count_cuml:(track_count_cuml+num_of_tracks)],axis=1)
                    y_log_pred = _y_log_pred[:,i:(i+1)][:,-1]
                    loss = poisson_loss_function(y_log_true, y_log_pred)
                    track_count_cuml += num_of_tracks
                    total_poisson_loss += loss
            return total_poisson_loss
    
        def mse_loss_function(y_log_true, y_log_pred):
            # logcounts mse loss without sample weights
            mse_loss = keras.losses.mean_squared_error(
                y_log_true, y_log_pred)
            return mse_loss
        
        def _mse_loss_function(_y_log_true,_y_log_pred):
            total_mse_loss = 0
            track_count_cuml = 0
            num_tasks_count_cuml = 0
            
            if self.orig_multi_loss:
                for i in range(self.num_output_tracks):
                    y_log_true = _y_log_true[:,i:(i+1)][:,-1]
                    y_log_pred = _y_log_pred[:,i:(i+1)][:,-1]
                    loss = mse_loss_function(y_log_true, y_log_pred)
                    total_mse_loss += loss
            else:

                for i in range(self.num_tasks):
                    num_of_tracks = self.tracks_for_each_task[i]
                    y_log_true = tf.reduce_logsumexp(_y_log_true[:,track_count_cuml:(track_count_cuml+num_of_tracks)],axis=1)
                    y_log_pred = _y_log_pred[:,i:(i+1)][:,-1]

                    loss = mse_loss_function(y_log_true, y_log_pred)
                    track_count_cuml += num_of_tracks
                    num_tasks_count_cuml += 1
                    total_mse_loss += loss
            return total_mse_loss
        
        if self.counts_loss == "MSE":
            total_counts_loss = _mse_loss_function(y['logcounts_predictions'],y_pred[1])
        
        elif self.counts_loss == "POISSON":
        
            total_counts_loss = _poisson_loss_function(y['logcounts_predictions'],y_pred[1])
            
        else:
            raise Exception("Sorry, unknown loss funtion")
        

        # for mnll loss we mask out samples with weight == 0.0
        
        boolean_mask = tf.math.greater_equal(sample_weights, 1.0)
        
        _y = tf.boolean_mask(y['profile_predictions'], boolean_mask)
        _y_pred = tf.boolean_mask(y_pred[0], boolean_mask)

        def _zero_constant():
            return kb.constant(0)
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
    
        def _multinomial_nll(_y,_y_pred):
            total_mnll_loss = 0
            track_count_cuml = 0
            
            if self.orig_multi_loss:
                for i in range(self.num_output_tracks):
                    loss = multinomial_nll(_y[..., i], _y_pred[..., i])
                    total_mnll_loss += loss
                
            else:
                for i in range(self.num_tasks):
                    num_of_tracks = self.tracks_for_each_task[i]
                    _y_reshape = tf.reshape(\
                                            _y[:,:,track_count_cuml:(track_count_cuml+num_of_tracks)],\
                                            [-1,(num_of_tracks)*(self.output_profile_len)]\
                                           )
                    _y_pred_reshape = tf.reshape(\
                                                 _y_pred[:,:,track_count_cuml:(track_count_cuml+num_of_tracks)],\
                                                 [-1,(num_of_tracks)*(self.output_profile_len)]\
                                                )

                    loss = multinomial_nll(_y_reshape, _y_pred_reshape)
                    track_count_cuml = track_count_cuml+num_of_tracks
                    total_mnll_loss += loss
            return total_mnll_loss
                    
        total_mnll_loss = tf.cond(tf.equal(tf.size(_y), 0), 
                  _zero_constant,
                  lambda:  _multinomial_nll(_y,_y_pred))
        
        if self.counts_loss == "MSE":
            loss =  (self.loss_weights[0] * total_mnll_loss) + \
                (self.loss_weights[1] * total_counts_loss)   
        elif self.counts_loss == "POISSON":
        
            loss =  total_mnll_loss + total_counts_loss            
        else:
            raise Exception("Sorry, unknown loss funtion")

        return loss, total_mnll_loss, total_counts_loss
            
    def train_step(self, data):
        x, y, sample_weights = data       
    
        with tf.GradientTape() as tape:
            loss, total_mnll_loss, total_counts_loss = \
                self._get_loss(x, y, sample_weights)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "batch_loss": loss,
                "profile_predictions_loss": total_mnll_loss, 
                "logcounts_predictions_loss": total_counts_loss}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]
    
    
    def test_step(self, data):
        # Unpack the data
        x, y, sample_weights = data
        
        loss, total_mnll_loss, total_counts_loss = \
            self._get_loss(x, y, sample_weights, training=False)
            
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "batch_loss": loss,
                "profile_predictions_loss": total_mnll_loss, 
                "logcounts_predictions_loss": total_counts_loss}