""" 
    General utitlity functions

    IGNORE_FOR_SPHINX_DOCS:
    
    List of functions:
    
        gaussian1D_smoothing: Function to smooth input array using 
            1D gaussian smoothing
    
    License:
    
    MIT License

    Copyright (c) 2021 Kundaje Lab

    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    IGNORE_FOR_SPHINX_DOCS

"""

from scipy.ndimage import gaussian_filter1d


def gaussian1D_smoothing(input_array, sigma, window_size):
    """
        Function to smooth input array using 1D gaussian smoothing
        
        Args:
            input_array (numpy.array): input array of values
            
            sigma (float): sigma value for gaussian smoothing
            
            window_size (int): window size for gaussian smoothing
            
        Returns:
            numpy.array: smoothed output array
        
    """

    # compute truncate value (#standard_deviations)
    truncate = (((window_size - 1) / 2) - 0.5) / sigma
    
    return gaussian_filter1d(input_array, sigma=sigma, truncate=truncate)
