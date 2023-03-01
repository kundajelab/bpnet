"""
    This module contains all the functions that define various
    bpnet network architectures
"""

from numpy.random import seed
seed(1234)
from tensorflow.random import set_seed 
set_seed(1234)

from bpnet.model.custommodel import CustomModel
from bpnet.utils.exceptionhandler \
    import NoTracebackException
from tensorflow.keras import layers
from tensorflow.keras.backend import int_shape

import tensorflow as tf
from bpnet.model import bpnetdefaults

def _crop_layer(layer, new_size):
    """
        Crop a layer's first dimension
        
        Args:
            layer (tensorflow.keras.layers.*) - layer to crop
            new_size (int) - new size for the layer's first dimension
                
        Returns:
            N-D tensor with shape: (batch_size, new_size, ...)
    """
    crop_size = int_shape(layer)[1] // 2 - new_size // 2
    return layers.Cropping1D(
        crop_size, name=layer.name.split('/')[0] + '_cr')(layer)


def _get_num_bias_tracks_for_task(task):
    """
        Get total number of bias tracks for the task including
        original and smoothed bias tracks
        
        Args:
            task_info (dict): task specific dictionary with 'signal',
                'loci', and 'bias' info
                
        Returns:
            int: total number of bias tracks for this task
    """
    # get number of original bias tracks for the task
    num_bias_tracks = len(task['bias']['source'])

    # if no bias tracks are found for this task
    if num_bias_tracks == 0:
        return 0

    # the length of the 'bias_smoothing' list should be the same
    # as the 'source' list
    if len(task['bias']['smoothing']) != num_bias_tracks:
        raise NoTracebackException(
            "RuntimeError 'bias': Length mismatch 'source' vs 'smoothing'")

    # count the number of 'smoothed' bias tracks that will be
    # added on
    for i in range(num_bias_tracks):
        if task['bias']['smoothing'][i] is not None:
            # add 1 for every smoothed track, not all bias tracks
            # may have their corresponding smoothed versions
            num_bias_tracks += 1

    return num_bias_tracks


def _slice(dimension, start, end, name):
    """
        Slices a Tensor on a given dimension from start to end
        example : to crop tensor x[:, :, 5:10]
        call slice(2, 5, 10) as you want to crop on the second 
        dimension
    
        Args:
            dimension (int): dimension on which to crop
            start (int): start index of the required slice
            end (int): end index of the required slice
            name (str): name for the Lambda layer
            
        Returns:
            N-D tensor with shape: (batch_size, ..., end-start)
    """
    
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    
    return layers.Lambda(func, name=name)


def motif_module(
    one_hot_input,
    filters=bpnetdefaults.MOTIF_MODULE_PARAMS['filters'], 
    kernel_sizes=bpnetdefaults.MOTIF_MODULE_PARAMS['kernel_sizes'], 
    padding=bpnetdefaults.MOTIF_MODULE_PARAMS['padding'], 
    name_prefix=None):
    
    """
        One or more regular convolutional layers that operate on
        BPNet input 
        
        Args:
            one_hot_input (tensorflow.keras.layers.Input): one hot
                encoded input sequence
            filters (list): list of number of filters for each
                convolutional layer
            kernel_sizes (list): list of filter sizes for each
                convolutional layer
            padding (str): padding to use for the convolutional
                layers. Either 'valid' or 'same'
            naem_prefix (str): prefix for layer name
        Returns:
            N-D tensor with shape: (batch_size, ..., filters[-1])
    """
    
    motif_module_out = one_hot_input
    # n-1 conv layers with activation (n = len(kernel_sizes))
    for i in range(len(kernel_sizes)-1):
        motif_module_out = layers.Conv1D(
            filters[i], kernel_size=kernel_sizes[i], padding=padding, 
            activation='relu', 
            name='{}_conv_{}'.format(name_prefix, i))(motif_module_out)
        
    # The activation of the last conv in the motif module is done
    # in the syntax module. The reason for this is to accommodate 
    # the 'pre_activation_residual_unit' option in the syntax module
    return layers.Conv1D(
        filters[-1], kernel_size=kernel_sizes[-1], padding=padding, 
        name='{}_conv_{}'.format(
            name_prefix, len(filters) - 1))(motif_module_out)
    

def syntax_module(
    motif_module_output,
    num_dilation_layers=\
        bpnetdefaults.SYNTAX_MODULE_PARAMS['num_dilation_layers'], 
    filters=bpnetdefaults.SYNTAX_MODULE_PARAMS['filters'], 
    kernel_size=bpnetdefaults.SYNTAX_MODULE_PARAMS['kernel_size'],
    padding=bpnetdefaults.SYNTAX_MODULE_PARAMS['padding'], 
    pre_activation_residual_unit=\
        bpnetdefaults.SYNTAX_MODULE_PARAMS['pre_activation_residual_unit'], 
    name_prefix=None):
    
    """
        Dilated convolutions with resnet-style additions. Each layer 
        receives the sum of feature maps (from previous two layers or
        all previous layers)
    
        Args:
            motif_module_output (tensorflow.keras.layers.Conv1D): 
                output of the BPNet motif module 
            num_dilation_layers (int): number of dilation layers
            filters (int): number of filters for each of the dilation
                layers
            kernel_size (int): kernel size for each of the dilation
                layers
            padding (str): padding to use for the dilated convolutional
                layers. Either 'valid' or 'same'
            pre_activation_residual_unit (boolean): True, if you want
                activations to be passed only to the dilated  conv 
                layer and not to the Add layer. Please refer to Fig 5. 
                in this paper https://arxiv.org/pdf/1603.05027.pdf. 
                The True condition is highlighted in 5(c) and False 
                in 5(a). Default is True
            name_prefix (str): prefix to use for layer name
        Returns:
            N-D tensor with shape: (batch_size, ..., filters)
        
    """
                           
    # first layer for the residual connections
    x = motif_module_output

    for i in range(1, num_dilation_layers + 1):     
        # apply relu to 'x' before applying dilated conv
        # (activation before the weights layer in the residual unit)
        x_activated = layers.ReLU(name=x.name.split('/')[0]+'_relu')(x)
            
        # dilated convolution
        conv_output_without_activation = layers.Conv1D(
            filters=filters, kernel_size=kernel_size, padding=padding, 
            dilation_rate=2**i,
            name='{}_dil_conv_{}'.format(name_prefix, i))(x_activated)

        if pre_activation_residual_unit:
            # If padding is 'valid' we need to crop layer to match 
            # size before we add
            if padding == 'valid':
                x = _crop_layer(
                    x, int_shape(conv_output_without_activation)[1]) 
            
            x = layers.add([conv_output_without_activation, x], 
                           name='{}_add_{}'.format(name_prefix, i))
        else:
            # If padding is 'valid' we need to crop layer to match 
            # size before we add
            if padding == 'valid':
                x_activated = _crop_layer(
                    x_activated, int_shape(conv_output_without_activation)[1]) 
                
            x = layers.add([conv_output_without_activation, x_activated], 
                           name='{}_add_{}'.format(name_prefix, i))
            
    x_activated = layers.ReLU(name=x.name.split('/')[0]+'_relu')(x)
            
    # the final output from the dilated convolutions with 
    # resnet-style connections
    return x_activated


def profile_head(
    syntax_module_out, 
    filters=bpnetdefaults.PROFILE_HEAD_PARAMS['filters'], 
    kernel_size=bpnetdefaults.PROFILE_HEAD_PARAMS['kernel_size'], 
    padding=bpnetdefaults.PROFILE_HEAD_PARAMS['padding'], 
    name_prefix=None):
    
    """
        Pre-bias profile output
        
        Args:
            syntax_module_out (tensorflow.keras.layers.Conv1D): output
                of the BPNet syntax module
            filters (int): number of filters for the convolutional 
                layer (same as number of tracks across all tasks)
            kernel_size (int): size of convolutional kernel 
            padding (str): padding to use for the convolutional layer
            name_prefix (str): prefix to use for layer names
            
        Returns:
            N-D tensor with shape: (batch_size, ..., filters)

    """
    
    return layers.Conv1D(
        filters=filters, kernel_size=kernel_size, padding=padding, 
        name='{}_profile_head'.format(name_prefix))(syntax_module_out)


def counts_head(
    syntax_module_out, name, units=bpnetdefaults.COUNTS_HEAD_PARAMS['units'], 
    dropouts=bpnetdefaults.COUNTS_HEAD_PARAMS['dropouts'],
    activations=bpnetdefaults.COUNTS_HEAD_PARAMS['activations'],    
    name_prefix=None):
    
    """
        Pre-bias counts output
    
        Args:
            syntax_module_out (tensorflow.keras.layers.Conv1D): output
                of the BPNet syntax module
            name (str): name for the counts head layer
            units (list): list of Dense layer sizes with the last layer
                specifying the dimensionality of the counts output space
                (same as number of tasks)
            name_prefix (str): prefix to use for layer names
        Returns:
            N-D tensor with shape: (batch_size, ..., units)
    """
    
    if len(units) == 1:
        # Step 1: average all the filter outputs of the syntax module
        x = layers.GlobalAveragePooling1D(
            name='{}_global_avg_pooling'.format(name_prefix))(syntax_module_out)
    else:
        x = layers.Flatten(
            name='{}_flatten'.format(name_prefix))(syntax_module_out)
        x = layers.Dropout(0.25, name="counts_flatten_dropout")(x)
        
    # Step 2: Connect the Flattened layer to zero or more 
    # intermediate Dense layers before the final Dense layer
    for i in range(len(units) - 1):
        
        if activations[i] != 'leakyrelu':
            # we can use this formulation for all activations that 
            # have a string name representation
            x = layers.Dense(units[i], name='counts_dense_{}'.format(i),
                             activation=activations[i])(x)
        else:
            # add a separate leaky relu layer since keras doesnt 
            # have a string name for leaky relu
            x = layers.Dense(units[i], name='counts_dense_{}'.format(i))(x)
            x = layers.LeakyReLU(name='counts_dense_LReLU_{}'.format(i))(x)
        
        #add dropout layer
        if dropouts[i] > 0.0:
            x = layers.Dropout(dropouts[i],
                               name='counts_dropout_{}'.format(i))(x)
            
    # the final Dense layer with linear activation and no dropout
    return layers.Dense(units[-1], name=name)(x)


def profile_bias_module(
    profile_head, profile_bias_inputs, tasks_info,
    kernel_sizes=bpnetdefaults.PROFILE_BIAS_MODULE_PARAMS['kernel_sizes'], 
    name_prefix=None):
    
    """
        Apply bias correction to profile head
        
        Args:
            profile_head (tensorflow.keras.layers.Conv1D): pre-bias
                profile output
            profile_bias_inputs (list): list of 
                tensorflow.keras.layers.Input layers with each element
                in the list being the profile bias for the ith task
            kernel_sizes (list): list of kernel sizes, one for each 
                task, to apply bias correction using a conv layer
            name_prefix (str): prefix to use for layer names
       
        Returns:
            N-D tensor with shape: (batch_size, ..., #tasks)
    
    """
    # number of tasks
    num_tasks = len(profile_bias_inputs)
    
    # list of all profile outputs so we can concatentate at the end
    profile_outputs = []

    # start idx for the ith task in the profile head
    task_offset = 0
    
    # iterate through each task and get corresponding profile output
    for i in range(num_tasks): 
        # number of profile output tracks for this task
        num_task_tracks = len(tasks_info[i]['signal']['source'])
        
        if num_tasks == 1:
            # no need to slice for a single task scenario
            _profile_head = profile_head
        else:            
            #  get the slice of profile head for this task
            _profile_head = _slice(
                2, task_offset, task_offset + num_task_tracks, 
                name="{}_prof_head_{}".format(name_prefix, i))(profile_head)
        
        # increment the offset 
        task_offset += num_task_tracks
        
        # if no bias tracks are found for this task, we directly append
        # the slice of the profile_head corresponding to this task to
        # profile_outputs
        if profile_bias_inputs[i] is None:
            profile_outputs.append(_profile_head)
        else:
            # concatenate profile head with corresponding profile bias
            # input
            concat_with_profile_bias_input = layers.concatenate(
                [_profile_head, profile_bias_inputs[i]], 
                name="{}_concat_with_prof_bias_{}".format(name_prefix, i), 
                axis=-1)

            # conv layer to yield the profile output prediction
            # for this task. If kernel size is 1 this is a 1x1 convolution
            if num_tasks == 1:
                name = "profile_predictions"
            else:
                name = "profile_predictions_{}".format(i)
            profile_outputs.append(layers.Conv1D(
                filters=num_task_tracks, kernel_size=kernel_sizes[i], 
                name=name)(concat_with_profile_bias_input))

    # profile output
    if len(profile_outputs) == 1:
        return profile_outputs[0]
    else:
        return layers.concatenate(
            profile_outputs, name="profile_predictions", axis=-1)

    
def counts_bias_module(counts_head, counts_bias_inputs, tasks_info, 
                       name_prefix=None,orig_multi_loss=False):
    """
        Apply bias correction to counts head
        
        Args:
            counts_head (tensorflow.keras.layers.Dense): pre-bias
                counts output
            counts_bias_inputs (list): list of 
                tensorflow.keras.layers.Input layers with each element
                in the list being the counts bias for the ith task
            name_prefix (str): prefix to use for layer names
            
        Returns:
            N-D tensor with shape: (batch_size, #tasks)
    
    """
    # number of tasks
    num_tasks = len(counts_bias_inputs)

    # list of all counts outputs so we can concatentate at the end
    counts_outputs = []

    # start idx for the ith task in the counts head
    task_offset = 0

    # iterate through each task and get corresponding counts output
    for i in range(num_tasks):  
        # number of counts output tracks for this task
        num_task_tracks = len(tasks_info[i]['signal']['source'])

        if num_tasks == 1:
            # no need to slice for a single task scenario
            _counts_head = counts_head
        else:            
            #  get the slice of profile head for this task
            _counts_head = _slice(
                1, task_offset, task_offset + num_task_tracks, 
                name="{}_counts_head_{}".format(name_prefix, i))(counts_head)

        
        # increment the offset 
        task_offset += num_task_tracks
        
        # if no bias tracks are found for this task, we directly append
        # the slice of the counts_head corresponding to this task to
        # counts_outputs
        if counts_bias_inputs[i] is None:
            counts_outputs.append(_counts_head)
        else:
            # concatenate counts head with slice of counts bias input
            # for this task
            if orig_multi_loss:
                concat_with_counts_bias_input = layers.concatenate(
                    [_counts_head, counts_bias_inputs[i]], 
                    name="{}_concat_with_counts_bias_{}".format(name_prefix, i),
                    axis=-1)
            else:
                # summing over strands
                counts_bias_input_out_logcounts = layers.Lambda(
                    lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True),
                    name="{}_logsumexp_counts_bias_{}".format(name_prefix, i))(counts_bias_inputs[i])

                concat_with_counts_bias_input = layers.concatenate(
                    [_counts_head, counts_bias_input_out_logcounts], 
                    name="{}_concat_with_counts_bias_{}".format(name_prefix, i),
                    axis=-1)

            # single unit Dense layer to yield the counts output 
            # prediction for this task
            if num_tasks == 1:
                name = "logcounts_predictions"
            else:
                name = "logcounts_predictions_{}".format(i)
            #     name = "logcounts_predictions_{}".format(i)
            # counts_outputs.append(layers.Dense(
            #     units=num_task_tracks, 
            #     name=name)(concat_with_counts_bias_input))
            if orig_multi_loss:
                counts_outputs.append(layers.Dense(
                    units=num_task_tracks, 
                    name=name)(concat_with_counts_bias_input))
            else: 
                counts_outputs.append(layers.Dense(
                    units=num_tasks, 
                    name=name)(concat_with_counts_bias_input))

    # counts output
    if len(counts_outputs) == 1:
        return counts_outputs[0]
    else:
        return layers.concatenate(
            counts_outputs, 
            name="logcounts_predictions", axis=-1)


def load_params(params):
    """
        Load BPNet parameters from dictionary, override defaults for
        keys that are present in the dictionary
        
        Args: 
            params (dict): parameters to the BPNet architecture
        
        Returns:
            tuple - all parameters to BPNet
    """
        
    # initialize all params from defaults and then run through
    # all the override values from the params json and replace
    # default values with the user defined values
    input_len = bpnetdefaults.INPUT_LEN
    if 'input_len' in params:
        input_len = params['input_len']

    output_profile_len = bpnetdefaults.OUTPUT_PROFILE_LEN
    if 'output_profile_len' in params:
        output_profile_len = params['output_profile_len']

    motif_module_params = bpnetdefaults.MOTIF_MODULE_PARAMS
    if 'motif_module_params' in params:
        for key in params['motif_module_params']:
            motif_module_params[key] = params['motif_module_params'][key]

    syntax_module_params = bpnetdefaults.SYNTAX_MODULE_PARAMS
    if 'syntax_module_params' in params:
        for key in params['syntax_module_params']:
            syntax_module_params[key] = params['syntax_module_params'][key]
            
    profile_head_params = bpnetdefaults.PROFILE_HEAD_PARAMS
    if 'profile_head_params' in params:
        for key in params['profile_head_params']:
            profile_head_params[key] = params['profile_head_params'][key]
            
    counts_head_params = bpnetdefaults.COUNTS_HEAD_PARAMS
    if 'counts_head_params' in params:
        for key in params['counts_head_params']:
            counts_head_params[key] = params['counts_head_params'][key]
        
    profile_bias_module_params = bpnetdefaults.PROFILE_BIAS_MODULE_PARAMS
    if 'profile_bias_module_params' in params:
        for key in params['profile_bias_module_params']:
            profile_bias_module_params[key] = \
                params['profile_bias_module_params'][key]

    counts_bias_module_params = bpnetdefaults.COUNTS_BIAS_MODULE_PARAMS
    if 'counts_bias_module_params' in params:
        for key in params['counts_bias_module_params']:
            counts_bias_module_params[key] = \
                params['counts_bias_module_params'][key]
            
    loss_weights = bpnetdefaults.LOSS_WEIGHTS
    if 'loss_weights' in params:
        loss_weights = params['loss_weights']
        
    counts_loss = bpnetdefaults.COUNTS_LOSS
    if 'counts_loss' in params:
        counts_loss = params['counts_loss']
        
    return (input_len, output_profile_len, motif_module_params, 
            syntax_module_params, profile_head_params, counts_head_params,
            profile_bias_module_params, counts_bias_module_params,
            loss_weights, counts_loss)

    
def BPNet(
    tasks, bpnet_params, orig_multi_loss=False,
    name_prefix=None):

    """
        BPNet architecture definition
    
        Args:
            tasks (dict): dictionary of tasks info specifying
                'signal', 'loci', and 'bias' for each task
            bpnet_params (dict): parameters to the BPNet architecture
                The keys include (all are optional)- 
                'input_len': (int)
                'output_profile_len': (int), 
                'motif_module_params': (dict) - 
                    'filters' (list)
                    'kernel_sizes' (list)
                    'padding' (str) 
                'syntax_module_params': (dict) -     
                    'num_dilation_layers' (int)
                    'filters' (int)
                    'kernel_size' (int)
                    'padding': (str)
                    'pre_activation_residual_unit' (boolean)
                'profile_head_params': (dict) -
                    'filters' (int)
                    'kernel_size' (int)
                    'padding' (str)
                'counts_head_params': (dict) -
                    'units' (int)
                'profile_bias_module_params': (dict) - 
                    'kernel_sizes' (list)
                'counts_bias_module_params': (dict) - N/A
                'loss_weights': (list)
                'counts_loss': (str)            
            name_prefix (str): prefix to use for layer names
                
        Returns:
            tensorflow.keras.layers.Model
    """
    
    # load params, override defaults
    (input_len, 
     output_profile_len, 
     motif_module_params, 
     syntax_module_params, 
     profile_head_params, 
     counts_head_params,
     profile_bias_module_params,
     counts_bias_module_params,
     loss_weights,
     counts_loss) = load_params(bpnet_params)    

    # Step 1 - sequence input
    one_hot_input = layers.Input(shape=(input_len, 4), name='sequence')
    
    # Step 2 - Motif module (one or more conv layers)
    motif_module_out = motif_module(
        one_hot_input, motif_module_params['filters'], 
        motif_module_params['kernel_sizes'], motif_module_params['padding'], 
        name_prefix=name_prefix)
    
    # Step 3 - Syntax module (all dilation layers)
    syntax_module_out = syntax_module(
        motif_module_out, syntax_module_params['num_dilation_layers'], 
        syntax_module_params['filters'], syntax_module_params['kernel_size'],
        syntax_module_params['padding'], 
        syntax_module_params['pre_activation_residual_unit'], 
        name_prefix=name_prefix)

    # Step 4.1 - Profile head (large conv kernel)
    # Step 4.1.1 - get total number of output tracks across all tasks
    num_tasks = len(list(tasks.keys()))
    total_tracks = 0
    tracks_for_each_task = []
    for i in range(num_tasks):
        total_tracks += len(tasks[i]['signal']['source'])
        tracks_for_each_task.append(total_tracks)
    
    # Step 4.1.2 - conv layer to get pre bias profile prediction
    profile_head_out = profile_head(
        syntax_module_out, total_tracks, 
        profile_head_params['kernel_size'], profile_head_params['padding'], 
        name_prefix=name_prefix)
    
    # first let's figure out if bias input is required based on 
    # tasks info, this also affects the naming of the profile head
    # and counts head layers
    # total number of bias tasks in the tasks_info dictionary
    total_bias_tracks = 0
    # number of bias tracks in each task
    task_bias_tracks = {}
    for i in range(num_tasks):
        task_bias_tracks[i] = _get_num_bias_tracks_for_task(tasks[i])
        total_bias_tracks += task_bias_tracks[i]

    # Step 4.1.3 crop profile head to match output_len
    if total_bias_tracks == 0:
        profile_head_name = 'profile_predictions'
    else:
        profile_head_name = '{}_profile_head_cropped'.format(name_prefix)
        
        
    crop_size = int_shape(profile_head_out)[1] // 2 - output_profile_len // 2
    profile_head_out = layers.Cropping1D(
        crop_size, name=profile_head_name)(profile_head_out)
    
    # Step 4.2 - Counts head (global average pooling)
    if total_bias_tracks == 0:
        counts_head_name = 'logcounts_predictions'
    else:
        counts_head_name = '{}_counts_head'.format(name_prefix)
    # the units for the Dense layers
    units = counts_head_params["units"]
    
    print("units:",units)
    print("num_tasks:",num_tasks)
    print("total_tracks:",total_tracks)
    
    # the last Dense layer's units are set to total tracks
    if units[-1]==-1:
        if orig_multi_loss:
            units[-1] = total_tracks
        else:
            units[-1] = num_tasks
            
    counts_head_out = counts_head(
        syntax_module_out, counts_head_name, units, 
        counts_head_params['dropouts'], counts_head_params['activations'],
        name_prefix=name_prefix)
    
    # Step 5 - Bias Input
    # if the tasks have no bias tracks then profile_head and 
    # counts_head are the outputs of the model
    inputs = [one_hot_input]
    print("total_bias_tracks:",total_bias_tracks)
    if total_bias_tracks == 0:
        profile_outputs = profile_head_out
        logcounts_outputs = counts_head_out  
        
    else:        
        if num_tasks != len(profile_bias_module_params['kernel_sizes']):
            raise NoTracebackException(
                "Length on 'kernel_sizes' in profile_bias_module_params "
                "must match #tasks")
        
        # Step 5.1 - Define the bias input layers 
        profile_bias_inputs = []
        counts_bias_inputs = []
        for i in range(num_tasks):
            if task_bias_tracks[i] > 0:
                # profile bias input for task i
                profile_bias_inputs.append(layers.Input(
                    shape=(output_profile_len, task_bias_tracks[i]),
                    name="profile_bias_input_{}".format(i)))

                # counts bias input for task i
                counts_bias_inputs.append(layers.Input(
                    shape=(task_bias_tracks[i]), 
                    name="counts_bias_input_{}".format(i)))
                
                # append to inputs
                inputs.append(profile_bias_inputs[i])
                inputs.append(counts_bias_inputs[i])
            else:
                profile_bias_inputs.append(None)    
                counts_bias_inputs.append(None)
            
        # Step 5.2 - account for profile bias
        profile_outputs = profile_bias_module(
            profile_head_out, profile_bias_inputs, tasks, 
            kernel_sizes=profile_bias_module_params['kernel_sizes'], 
            name_prefix=name_prefix)
        
    
        # Step 5.3 - account for counts bias
        logcounts_outputs = counts_bias_module(
            counts_head_out, counts_bias_inputs, tasks, 
            name_prefix=name_prefix,orig_multi_loss=orig_multi_loss)
    
    # instantiate keras Model with inputs and outputs
    # print({'num_tasks':num_tasks,\
    #        'tracks_for_each_task':tracks_for_each_task,\
    #        'output_profile_len':output_profile_len,\
    #        'loss_weights':loss_weights,\
    #        'inputs':inputs, 'outputs':[profile_outputs, logcounts_outputs]})
    return CustomModel(num_tasks,
                        total_tracks, 
                        tracks_for_each_task, 
                        output_profile_len, 
                        loss_weights, 
                        counts_loss,
                        orig_multi_loss,
                        inputs=inputs, 
                        outputs=[profile_outputs, logcounts_outputs])


