import genomicsdlarchsandlosses.bpnet.bpnetdefaults as bpnetdefaults
import tensorflow as tf
import tensorflow.keras.backend as kb

from genomicsdlarchsandlosses.bpnet.losses import fourier_att_prior_loss
from tensorflow.keras import Model
from tensorflow.nn import log_softmax

class AttributionPriorModel(Model):

    def __init__(
        self, 
        frequency_limit=\
            bpnetdefaults.ATTRIBUTION_PRIOR_PARAMS['frequency_limit'],
        limit_softness=\
            bpnetdefaults.ATTRIBUTION_PRIOR_PARAMS['limit_softness'],
        grad_smooth_sigma=\
            bpnetdefaults.ATTRIBUTION_PRIOR_PARAMS['grad_smooth_sigma'],
        profile_grad_loss_weight=\
            bpnetdefaults.ATTRIBUTION_PRIOR_PARAMS['profile_grad_loss_weight'],
        counts_grad_loss_weight=\
            bpnetdefaults.ATTRIBUTION_PRIOR_PARAMS['counts_grad_loss_weight'], 
        **kwargs):
        
        # call the base class with inputs and outputs
        super(AttributionPriorModel, self).__init__(**kwargs)
        
        self.freq_limit = frequency_limit
        self.limit_softness = limit_softness
        self.grad_smooth_sigma = grad_smooth_sigma
        self.profile_grad_loss_weight = profile_grad_loss_weight
        self.counts_grad_loss_weight = counts_grad_loss_weight
 

    def _get_attribution_prior_loss(self, x, y_pred, input_grad_tape):
        # mean-normalize the profile logits output & weight by
        # post-softmax probabilities
        y_pred_profile = y_pred[0] - kb.mean(y_pred[0])
        y_pred_profile *= log_softmax(y_pred_profile)

        # gradients of profile output w.r.t to input
        input_grads_profile = input_grad_tape.gradient(
            y_pred[0], x['sequence'])
        # gradients of counts output w.r.t to input
        input_grads_counts = input_grad_tape.gradient(
            y_pred[1], x['sequence'])

        # Gradient * input
        input_grads_profile = input_grads_profile * x['sequence']  
        input_grads_counts = input_grads_counts * x['sequence']  

        # attribution prior loss of profile
        batch_attr_prior_loss_profile = fourier_att_prior_loss(
            x['status'], input_grads_profile, self.freq_limit, 
            self.limit_softness, self.grad_smooth_sigma)

        # attribution prior loss of counts
        batch_attr_prior_loss_counts = fourier_att_prior_loss(
            x['status'], input_grads_counts, self.freq_limit, 
            self.limit_softness, self.grad_smooth_sigma)

        # weighted loss
        batch_attr_prior_loss = \
            (self.profile_grad_loss_weight * batch_attr_prior_loss_profile) + \
            (self.counts_grad_loss_weight * batch_attr_prior_loss_counts)

        return batch_attr_prior_loss
    

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as input_grad_tape:
                input_grad_tape.watch(x)
                
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                batch_loss = self.compiled_loss(
                    y, y_pred, regularization_losses=self.losses)
            
            batch_attr_prior_loss = self._get_attribution_prior_loss(
                x, y_pred, input_grad_tape)
            batch_loss = batch_loss + batch_attr_prior_loss

            # delete the gradient tape because we made it persistent 
            # to compute gradients more than once 
            del input_grad_tape

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(batch_loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
                
        # the metrics dict has the compiled loss in 'loss', so we
        # add the batch_attr_prior_loss
        metrics['loss'] = metrics['loss'] + batch_attr_prior_loss
        
        # add the attribution prior loss as a separate key in to the
        # dict
        metrics['attribution_prior_loss'] = batch_attr_prior_loss
        
        return metrics

    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            with tf.GradientTape(persistent=True) as input_grad_tape:
                input_grad_tape.watch(x)
                
                y_pred = self(x, training=False)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                batch_loss = self.compiled_loss(
                    y, y_pred, regularization_losses=self.losses)
            
            batch_attr_prior_loss = self._get_attribution_prior_loss(
                x, y_pred, input_grad_tape)
            batch_loss = batch_loss + batch_attr_prior_loss

            # delete the gradient tape because we made it persistent 
            # to compute gradients more than once 
            del input_grad_tape

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        
        # the metrics dict has the compiled loss in 'loss', so we
        # add the batch_attr_prior_loss
        metrics['loss'] = metrics['loss'] + batch_attr_prior_loss
        
        # add the attribution prior loss as a separate key in to the
        # dict
        metrics['attribution_prior_loss'] = batch_attr_prior_loss
        
        return metrics
