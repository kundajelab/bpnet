import tensorflow as tf
import tensorflow.keras.backend as kb
import numpy as np

from scipy.ndimage import gaussian_filter

def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    
    Args:
        input_tensor (tensor): a A x B tensor to smooth along the 
            second dimension
        smooth_sigma (float): width of the Gaussian to use for 
            smoothing; this is the standard deviation of the Gaussian
            to use, and the Gaussian will be truncated after 1 sigma
            (i.e. the smoothing window is 1 + (2 * sigma); sigma of 0
            means no smoothing
    
    Returns:
        tensor: an array the same shape as the input tensor, with
            the dimension of `B` smoothed.
    """

    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    # Center of window is 1 everywhere else is 0
    base[sigma] = 1  
    kernel = gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = kb.constant(kernel)

    # Expand the input and kernel to 3D, with channels of 1
    # Shape: A x B x 1
    input_tensor = kb.expand_dims(input_tensor, axis=2)  
    # Shape: (1 + 2s) x 1 x 1
    kernel = kb.expand_dims(kb.expand_dims(kernel, axis=1), axis=2)  

    smoothed = tf.nn.conv1d(
        input_tensor, kernel, stride=1, padding="SAME", data_format="NWC")

    return kb.squeeze(smoothed, axis=2)
