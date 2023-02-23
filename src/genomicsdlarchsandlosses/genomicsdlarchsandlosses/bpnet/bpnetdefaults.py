"""
    Default parameters for BPNet architecture defintion - single
    unstranded task
"""

# length of one hot encoded input sequence
INPUT_LEN = 2114

# length of profile predictions
OUTPUT_PROFILE_LEN = 1000

# parameters to the motif module that has one or more 
# regular convolutional layers
MOTIF_MODULE_PARAMS = {
    # 'filters' and 'kernel_sizes' are lists in case we have more
    # than one convolutional layer. In the default case we have a
    # single convolutional layer for the motif module
    'filters': [64],
    'kernel_sizes': [21],
    'padding': 'valid',
}

# parameters to the syntax module that has many dilated 
# convolutions
SYNTAX_MODULE_PARAMS = {
    'num_dilation_layers': 8,
    'filters': 64,
    'kernel_size': 3,
    'padding': 'valid',
    'pre_activation_residual_unit': True
}

# parameters to the profile head (the pre-bias profile predictions)
PROFILE_HEAD_PARAMS = {
    # the default for a single unstranded task, one filter resulting 
    # in one profile head output
    'filters': 1,
    'kernel_size':  75,
    'padding': 'valid'
}

# parameters to the counts head (the pre-bias logcounts predictions)
COUNTS_HEAD_PARAMS = {
    # number of Dense layer units, the default for a single 
    # unstranded task
    'units': [1],
    'dropouts': [0.0],
    # 'linear', 'relu', 'leakyrelu', 'sigmoid', 'softmax', 'softplus',
    # 'softsign', 'tanh', 'selu', 'elu', 'exponential'
    'activations': ['linear']     
}

# parameters to the profile bias module
PROFILE_BIAS_MODULE_PARAMS = {
    # 'kernel_sizes' is a list in case we have a multitask scenario,
    # each task could have a different kernel size.
    # Using a kernel_size of 1 results in a 1x1 convolution
    # Note: its possible that not all tasks in the multitask scenario
    # have bias, for e.g. in a 4-task scenarios wherein the 3rd task
    # doesn't have bias, you can specify 'kernel_sizes': [1, 3, None, 1]
    'kernel_sizes': [1]
}

# parameters to the counts bias module
COUNTS_BIAS_MODULE_PARAMS = {
}

# enable attribution prior loss
USE_ATTRIBUTION_PRIOR = False

# attribution prior parameter defaults
ATTRIBUTION_PRIOR_PARAMS = {
    'frequency_limit': 150,
    'limit_softness': 0.2,
    'grad_smooth_sigma': 3,
    'profile_grad_loss_weight': 200,
    'counts_grad_loss_weight': 100
}

LOSS_WEIGHTS = [1.0, 1.0]
COUNTS_LOSS = "MSE"