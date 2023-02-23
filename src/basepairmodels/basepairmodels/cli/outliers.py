import json
import logging
import numpy as np
import os
import pandas as pd
import pyBigWig

from basepairmodels.cli.argparsers import outliers_argsparser
from basepairmodels.cli.exceptionhandler import NoTracebackException
from basepairmodels.cli import logger
from tqdm import tqdm

def getPeakPositions(task, chroms, chrom_sizes, flank, drop_duplicates=False):
    """ 
        Peak positions for given task filtered based on required
        chromosomes and other qc filters. 
        
        Args:
            tasks (dict): A python dictionary containing the task
                information for a single task
            chroms (list): The list of required chromosomes
            chrom_sizes (pandas.Dataframe): dataframe of chromosome 
                sizes with 'chrom' and 'size' columns
            flank (int): half of sequence length
            drop_duplicates (boolean): True if duplicates should be
                dropped from returned dataframe. 
            
        Returns:
            pandas.DataFrame: dataframe of peak positions 
            
    """

    # necessary for dataframe apply operation below --->>>
    chrom_size_dict = dict(chrom_sizes.to_records(index=False))

    # initialize an empty dataframe
    allPeaks = pd.DataFrame()

    # we concatenate all the peaks from list of peaks files
    for peaks_file in task['loci']['source']:

        peaks_df = pd.read_csv(
            peaks_file, sep='\t', header=None,
            names=['chrom', 'st', 'e', 'name', 'weight', 'strand', 
                   'signal', 'p', 'q', 'summit'])

        # keep only those rows corresponding to the required 
        # chromosomes
        peaks_df = peaks_df[peaks_df['chrom'].isin(chroms)]

        # create new column for peak pos
        peaks_df['pos'] = peaks_df['st'] + peaks_df['summit']

        # compute left flank coordinates of the input sequences 
        # (including the allowed jitter)
        peaks_df['start'] = (peaks_df['pos'] - flank).astype(int)

        # compute right flank coordinates of the input sequences 
        # (including the allowed jitter)
        peaks_df['end'] = (peaks_df['pos'] + flank).astype(int)

        # filter out rows where the left flank coordinate is < 0
        peaks_df = peaks_df[peaks_df['start'] >= 0]

        # --->>> create a new column for chrom size
        peaks_df["chrom_size"] = peaks_df['chrom'].apply(
            lambda chrom: chrom_size_dict[chrom])

        # filter out rows where the right flank coordinate goes beyond
        # chromosome size
        peaks_df = peaks_df[
            peaks_df['end'] <= peaks_df['chrom_size']]

        # sort based on chromosome number and right flank coordinate
        peaks_df = peaks_df.sort_values(
            ['chrom', 'end']).reset_index(drop=True)

        # append to all peaks data frame
        allPeaks = allPeaks.append(peaks_df[
            ['chrom', 'st', 'e', 'start', 'end', 'name', 'weight', 'strand', 
             'signal', 'p', 'q', 'summit']])

        allPeaks = allPeaks.reset_index(drop=True)
    
    # drop the duplicate rows, i.e. the peaks that get duplicated
    # for the plus and minus strand tasks
    if drop_duplicates:
        allPeaks = allPeaks.drop_duplicates(ignore_index=True)
    
    return allPeaks


def remove_blacklist_peaks(peaks_df, blacklist_df):
    """
        Function to remove peaks that overlap with blacklist regions
        
        Args:
            peaks_df (pandas.Dataframe): 10 column dataframe of peaks
            blacklist_df (pandas.Dataframe): 3 column dataframe of 
                blacklist regions
                
        Returns:
            pandas.Dataframe: 10 column peaks dataframe 
            
    """
    
    def check_blacklist(chrom, start, end, blacklist_df):
        """
            Check if a given chromosome region overlaps with the 
            blacklist regions
            
            Args:
                chrom (str): chromosome name
                start (int): start coordinate
                end (int): end coordinate
                blacklist_df (pandas.Dataframe): 3 column dataframe of 
                    blacklist regions
            
            Returns:
                boolean: True if chromosome region overlaps with 
                    blacklist regions
        """

        # check if either the start or the end coordinate lies within
        # the blacklist region
        in_blacklist =  (blacklist_df['chrom'] == chrom) & \
            (((blacklist_df['st'] <= start) & (blacklist_df['e'] >= start)) | \
             ((blacklist_df['st'] <= end) & (blacklist_df['e'] >= end)))
        
        return (sum(in_blacklist) > 0)

    # for each peak check if it overlaps with any blacklist region
    peaks_df['in_blacklist'] = peaks_df.apply(
        lambda x: check_blacklist(x.chrom, x.start, x.end, blacklist_df),
        axis=1)

    # return peaks that dont overlap with blacklist regions
    return peaks_df[peaks_df['in_blacklist'] == False]
    

def outliers_main():
    
    # parse the command line arguments
    parser = outliers_argsparser()
    args = parser.parse_args()
    
    # filename to write debug logs
    logfname = "outliers.log"
    
    # set up the loggers
    logger.init_logger(logfname)
    
    # check if the input json file exists
    if not os.path.exists(args.input_data):
        raise NoTracebackException(
            "File {} does not exist".format(args.input_data))

    # check if the chrom sizes file exists
    if not os.path.exists(args.chrom_sizes):
        raise NoTracebackException(
            "File {} does not exist".format(args.chrom_sizes))

    # load the chrom sizes into a dataframe
    chrom_sizes_df = pd.read_csv(
            args.chrom_sizes, sep='\t', header=None, names=['chrom', 'size'])

    # load the tasks json file
    with open(args.input_data, 'r') as inp_json:
        try:
            tasks = json.loads(inp_json.read())
        except json.decoder.JSONDecodeError:
            raise NoTracebackException(
                "Unable to load json file {}. Valid json expected. "
                "Check the file for syntax errors.".format(
                    tasks_json))
    
    # get all peaks for a given task in 10 column ENCODE narrowPeak
    # format dataframe
    peaks_df = getPeakPositions(
        tasks[args.task], args.chroms, chrom_sizes_df, args.sequence_len // 2, 
        drop_duplicates=True)
    
    # if a global sample weight is specified set it here
    if args.global_sample_weight is not None:   
        peaks_df['weight'] = args.global_sample_weight
        
    # remove peaks that fall within blacklist regions
    if args.blacklist != None:
        # check if the blacklist file exists
        if not os.path.exists(args.blacklist):
            raise NoTracebackException(
                "Blacklist file {} does not exist".format(args.blacklist))

        blacklist_df = pd.read_csv(args.blacklist, sep='\t', 
                                   names=['chrom', 'st', 'e'])
        
        logging.info("Filtering blacklist peaks ...")
        logging.info("old size {}".format(len(peaks_df)))
        peaks_df = remove_blacklist_peaks(peaks_df, blacklist_df)
        logging.info("new size {}".format(len(peaks_df)))
    
    # open all the signal bigWigs for reading
    signal_files = []
    for signal_file in tasks[args.task]['signal']['source']:
        # check if the bigWig file exists
        if not os.path.exists(signal_file):
            raise NoTracebackException(
                "BigWig file {} does not exist".format(signal_file))
                
        signal_files.append(pyBigWig.open(signal_file))

    # counts dictionary maintains a list of counts for each 
    # peak for each of the signal files
    # we will use this list to create a counts column for each of the
    # signal files
    counts = {}
    for signal_file in signal_files:
        counts[signal_file] = []
    
    # iterate through all peaks and read values from the bigWig files
    logging.info("Computing counts for each peak")
    for _, row in tqdm(peaks_df.iterrows(), desc='peaks', total=len(peaks_df)):
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        
        for signal_file in signal_files:
            counts[signal_file].append(
                np.sum(np.nan_to_num(signal_file.values(chrom, start, end))))
                
    # add a new counts column to the peaks dataframe for each
    # signal file
    for signal_file in signal_files:
        peaks_df[signal_file] = counts[signal_file]
    
    # average the counts across the signal files
    peaks_df['avg_counts'] = peaks_df[signal_files].mean(axis=1)
    
    # sort the dataframe in ascending order of counts
    peaks_df = peaks_df.sort_values(by=['avg_counts'])
    
    # compute the quantile value
    counts = peaks_df['avg_counts'].values
    nth_quantile = np.quantile(counts, args.quantile)
    logging.info("{} quantile {}".format(args.quantile, nth_quantile))
    
    # get index of quantile value 
    quantile_idx = abs(counts - nth_quantile).argmin()
    logging.info("quantile idx {}".format(quantile_idx))
                
    # scale value at quantile index
    scaled_value = counts[quantile_idx] * args.quantile_value_scale_factor
    logging.info("scaled_value {}".format(scaled_value))

    # check if any of the counts are above the scaled_value
    if np.sum(counts > scaled_value) > 0:
        # index of values greater than scaled_value
        max_idx = np.argmax(counts > scaled_value)
        logging.info("max_idx {}".format(max_idx))    

        # trimmed data frame with outliers removed
        logging.info("original size {}".format(len(peaks_df)))    
        peaks_df = peaks_df[:max_idx]
        logging.info("new size {}".format(len(peaks_df)))    
    else:
        logging.info("No outliers found based on criteria. "
                     "Keeping original loci.")

    # save the new dataframe
    logging.info("Saving output bed file ... {}".format(args.output_bed))
    peaks_df = peaks_df[['chrom', 'st', 'e', 'name', 'weight', 
                         'strand', 'signal', 'p', 'q', 'summit']]
    peaks_df.to_csv(args.output_bed, header=None, sep='\t', index=False)
                
if __name__ == '__main__':
    outliers_main()
