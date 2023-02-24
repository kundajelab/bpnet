"""
    Python script to compute the counts loss weight

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

"""
from basepairmodels.cli.argparsers import counts_loss_weight_argsparser
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.common import stats

import json
import pandas as pd
import os
import sys

def counts_loss_weight_main():
    """
        main function for counts loss weight computation
    """
    
    # parse the command line arguments
    parser = counts_loss_weight_argsparser()
    args = parser.parse_args()

    # check if the input data file exists
    if not os.path.exists(args.input_data):
        # output the default value to stdout
        print(args.default)

        raise NoTracebackException(
            "Input data file {} does not exist. Using default weight "
            "{}".format(args.input_data, args.default))
            
    with open(args.input_data, 'r') as inp_json:
        try:
            input_data = json.loads(inp_json.read())
        except Exception as e:
            # output the default value to stdout
            print(args.default)
        
            exc_type, exc_value, exc_traceback = sys.exc_info()
            raise NoTracebackException(
                "{} {}. Using default weight {}".format(
                    exc_type.__name__, str(exc_value), args.default))
    
    # get all the bigWigs and peaks from the input_data
    bigWigs = []
    peaks = []
    for task in input_data:
        if 'signal' in input_data[task].keys():
            bigWigs.extend(input_data[task]['signal']["source"])

        if 'loci' in input_data[task].keys():
            peaks.extend(input_data[task]['loci']['source'])
            
    # if no bigWigs found
    if len(bigWigs) == 0:
        # output the default value to stdout
        print(args.default)

        raise NoTracebackException(
            "No 'signal' bigWigs found. Using default weight {}".format(
                args.default))
    else:
        # check to see if all are valid paths
        for bigWig in bigWigs:
            if not os.path.exists(bigWig):
                raise NoTracebackException(
                    "File {} does not exist. Using default weight "
                    "{}".format(bigWig, args.default))
        
    # if no peaks found
    if len(peaks) == 0:
        raise NoTracebackException(
            "No 'peaks' files found. Using default weight {}".format(
                args.default))
    else:
        # check to see if all are valid paths
        for peak_file in peaks:
            if not os.path.exists(peak_file):
                raise NoTracebackException(
                    "File {} does not exist. Using default weight "
                    "{}".format(peak_file, args.default))

    # list of all peaks dataframes to be passed to stats function
    peaks_dfs = []
    
    # load each peak file and compute the correct 'start' and 'end'
    # intervals
    for peak_file in peaks:   
        peaks_df = pd.read_csv(peak_file, sep='\t', header=None, 
                               names=['chrom', 'st', 'e', 'name', 'score',
                                      'strand', 'signal', 'p', 'q', 'summit'])

        # create new column for peak start
        peaks_df['start'] = peaks_df['st'] + \
                            peaks_df['summit'] - \
                            args.peak_width//2

        # create new column for peak end
        peaks_df['end'] = peaks_df['st'] + \
                            peaks_df['summit'] + \
                            args.peak_width//2

        # append to the list of peaks dataframes
        peaks_dfs.append(peaks_df[['chrom', 'start', 'end']])

    peaks_df = pd.concat(peaks_dfs)
    
    # compute the counts loss weight using the stats module function
    clw = stats.get_recommended_counts_loss_weight(
        bigWigs, peaks_df, args.alpha, args.orig_multi_loss)
    
    print(clw)

if __name__ == '__main__':
    counts_loss_weight_main()

