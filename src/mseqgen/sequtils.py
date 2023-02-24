""" 
    Utitlity functions that will be used by the sequence data 
    generators

    IGNORE_FOR_SPHINX_DOCS:
    
    List of functions:
    
        getChromPositions - returns two column dataframe of chromosome
            positions spanning the entire chromosome at 
            a) regular intervals or b) random locations
        
        getPeakPositions - returns two column dataframe of chromosome
            positions
        
        getInputTasks - when input data is fed as a path to a directory,
            that contains files (single task) or sub directories (multi
            tasl) that follow a strict naming convention, this function
            returns a nested python dictionary of tasks, specifying the
            'signal' and/or 'control' bigWigs, 'peaks' file, 'task_id'
            & 'strand
                
        roundToMultiple - Return the largest multiple of y < x
        
        one_hot_encode - returns a 3-dimension numpy array of one hot
            encoding of a list of DNA sequences
        
        reverse_complement_of_sequences - returns the reverse 
            complement of a list of sequences
        
        reverse_complement_of_profiles - returns the reverse 
            complement of the assay signal
        


    
    License:
    
    MIT License

    Copyright (c) 2020 Kundaje Lab

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

import random
random.seed(1234)
from numpy.random import seed
seed(1234)

import pandas as pd
import hashlib
import glob
import logging
import numpy as np
import os
import pandas as pd

from collections import OrderedDict
from mseqgen.exceptionhandler import NoTracebackException



def getChromPositions(chroms, chrom_sizes, flank, mode='sequential',
                      num_positions=-1, step=50):
    """
        Chromosome positions spanning the entire chromosome at 
        a) regular intervals or b) random locations
        
        Args:
            chroms (list): The list of required chromosomes 
            chrom_sizes (pandas.Dataframe): dataframe of chromosome 
                sizes with 'chrom' and 'size' columns
            flank (int): Buffer size before & after the position to  
                ensure we dont fetch values at index < 0 & > chrom size
            mode (str): mode of returned position 'sequential' (from
                the beginning) or 'random'
            num_positions (int): number of chromosome positions
                to return on each chromosome, use -1 to return 
                positions across the entrire chromosome for all given
                chromosomes in `chroms`. mode='random' cannot be used
                with num_positions=-1
            step (int): the interval between consecutive chromosome
                positions in 'sequential' mode
            
        Returns:
            pandas.DataFrame: 
                two column dataframe of chromosome positions (chrom, pos)
            
    """
    
    if mode == 'random' and num_positions == -1:
        raise NoTracebackException(
            "Incompatible parameter pairing: 'mode' = random, "
            "'num_positions' = -1")

    # check if chrom_sizes has a column called 'chrom'
    if 'chrom' not in chrom_sizes.columns:
        logging.error("Expected column 'chrom' not found in chrom_sizes")
        return None

    chrom_sizes = chrom_sizes.set_index('chrom')
    
    # initialize an empty dataframe with 'chrom' and 'pos' columns
    positions = pd.DataFrame(columns=['chrom', 'pos'])

    # for each chromosome in the list
    for i in range(len(chroms)):
        chrom_size = chrom_sizes.at[chroms[i], 'size']
        
        # keep start & end within bounds
        start = flank
        end = chrom_size - flank + 1
                
        if mode == 'random':
            # randomly sample positions
            pos_array = np.random.randint(start, end, num_positions)

        if mode == 'sequential':
            _end = end
            if num_positions != -1:
                # change the last positon based on the number of 
                # required positions
                _end = start + step * num_positions
                
                # if the newly computed 'end' goes beyond the 
                # chromosome end (we could throw an error here)
                if _end > end:
                    _end = end
        
            # positions at regular intervals
            pos_array = list(range(start, _end, step))    

        # construct a dataframe for this chromosome
        chrom_df = pd.DataFrame({'chrom': [chroms[i]] * len(pos_array), 
                                 'pos': pos_array})
        
        # concatenate to existing df
        positions = pd.concat([positions, chrom_df])
        
    return positions    
    

def getPeakPositions(tasks, chrom_sizes, flank,  
                     chroms=None,  
                     loci_indices=None,
                     background_loci_indices=None, mode='train',
                     loci_keys=['loci', 'background_loci'], 
                     drop_duplicates=False, background_only=False, 
                     foreground_weight=1, background_weight=0):
    """ 
        Peak positions for all the tasks filtered based on required
        chromosomes and other qc filters. Since 'task' here refers 
        one strand of input/output, if the data is stranded the peaks
        will be duplicated for the plus and minus strand.
        
        
        Args:
            tasks (dict): A python dictionary containing the task
                information. Each task in tasks should have atleast 
                the key 'loci' that has the path to he peaks file,
                optional key is 'background_loci'
            chrom_sizes (pandas.Dataframe): dataframe of chromosome 
                sizes with 'chrom' and 'size' columns
            flank (int): Buffer size before & after the position to  
                ensure we dont fetch values at index < 0 & > chrom size
            chroms (list): The list of chromosomes to include, takes
                 precedence over loci_indices. Applies to both loci
                 and background_loci
            loci_indices (list): list of indices to filter loci peaks.
            background_loci_indices (list): list of indices to filter 
                background loci peaks.
            loci_keys (list): list of keys that specify the loci to
                select from the input json for training/testing              
            drop_duplicates (boolean): True if duplicates should be
                dropped from returned dataframe. 
            background_only (boolean): True if only 'background_loci'
                have to be selected. This option can be set when you
                need to train a background/bias model
            foreground_weight (int): The value to set for the weight 
                of 'loci'
            background_weight (int): The value to set for the weight 
                of 'background_loci'
            
        Returns:
            pandas.DataFrame: 
                two column dataframe of peak positions (chrom, pos)
            
    """

    # necessary for dataframe apply operation below --->>>
    chrom_size_dict = dict(chrom_sizes.to_records(index=False))

    # initialize an empty dataframe
    allPeaks = pd.DataFrame()

    # check to see if background model training has been opted
    if background_only:
        if 'background_loci' not in loci_keys:
            raise NoTracebackException("Key not specified: 'background_loci'")            
        loci_keys = ['background_loci']
        background_weight = 1
    
    # we concatenate all the peaks from across all the tasks
    for task in tasks:
        for loci_key in loci_keys:
            if loci_key not in tasks[task].keys():
                continue
                
            # each task could have multiple peaks files in the 'source' list
            idx = 0
            for peaks_file in tasks[task][loci_key]['source']:

                peaks_df = pd.read_csv(
                    peaks_file, sep='\t', header=None, 
                    names=['chrom', 'st', 'end', 'name', 'weight', 'strand', 
                           'signal', 'p', 'q', 'summit'])
                               

                # keep only those rows corresponding to the required 
                # chromosomes
                
                if chroms != None:
                    # keep only those rows corresponding to the required 
                    # chromosomes
                    # applies both to loci and background_loci
                    peaks_df = peaks_df[peaks_df['chrom'].isin(chroms)]
                elif loci_key == 'loci' and loci_indices != None:
                    peaks_df = peaks_df.loc[peaks_df.index[loci_indices]]
                elif loci_key == 'background_loci' and background_loci_indices != None:
                    peaks_df = peaks_df.loc[peaks_df.index[background_loci_indices]]

                    
                # create new column for peak pos
                peaks_df['pos'] = peaks_df['st'] + peaks_df['summit']

                # compute left flank coordinates of the input sequences 
                # (including the allowed jitter)
                peaks_df['start_coord'] = (peaks_df['pos'] - flank).astype(int)

                # compute right flank coordinates of the input sequences 
                # (including the allowed jitter)
                peaks_df['end_coord'] = (peaks_df['pos'] + flank).astype(int)

                # filter out rows where the left flank coordinate is < 0
                peaks_df = peaks_df[peaks_df['start_coord'] >= 0]

                # --->>> create a new column for chrom size
                peaks_df["chrom_size"] = peaks_df['chrom'].apply(
                    lambda chrom: chrom_size_dict[chrom])

                # filter out rows where the right flank coordinate goes beyond
                # chromosome size
                peaks_df = peaks_df[
                    peaks_df['end_coord'] <= peaks_df['chrom_size']]

                # sort based on chromosome number and right flank coordinate
                peaks_df = peaks_df.sort_values(
                    ['chrom', 'end_coord']).reset_index(drop=True)

                # set num_foreground in case of foreground
                if loci_key == 'loci':
                    num_foreground = len(peaks_df)
                
                # select random samples if its background
                elif loci_key == 'background_loci' and not background_only:
                    
                    # the number of samples to randomly select
                    num_samples = int(num_foreground * \
                        tasks[task][loci_key]['ratio'][idx])
                    
                    
                    print("before sample hash:",
                          int(hashlib.sha256(str(peaks_df).encode('utf-8')).hexdigest(), 16))
  
                    if num_samples > len(peaks_df):                    
                        raise NoTracebackException(
                            "The number of background loci supplied is "
                            "insufficent for the ratio specified") 
                                           
                    else:
                        if mode == "train":
                            peaks_df = peaks_df.sample(
                                n=num_samples, replace=False)
                        else:
                            peaks_df = peaks_df.sample(
                                n=num_samples, replace=False,random_state=1)
                            
                    print("after sample hash:",
                          int(hashlib.sha256(str(peaks_df).encode('utf-8')).hexdigest(), 16))                
                # set weight of sample based on loci_key
                if loci_key == 'loci':
                    peaks_df['weight'] = foreground_weight
                else:
                    peaks_df['weight'] = background_weight
                
                # append to all peaks data frame
                allPeaks = allPeaks.append(peaks_df[
                    ['chrom', 'start_coord', 'end_coord', 'pos', 'weight']])

                allPeaks = allPeaks.reset_index(drop=True)
                
                idx += 1
    
    # drop the duplicate rows
    if drop_duplicates:
        allPeaks = allPeaks.drop_duplicates(ignore_index=True)
        
    return allPeaks


def roundToMultiple(x, y): 
    """Return the largest multiple of y < x
        
        Args:
            x (int): the number to round
            y (int): the multiplier
        
        Returns:
            int: largest multiple of y <= x
            
    """

    r = (x + int(y / 2)) & ~(y - 1)
    if r > x:
        r = r - y
    return r


def round_to_multiple(x, y, smallest=False): 
    """
        Return the largest multiple of y <= x or 
        smallest multiple of y >= x
        
        Args:
            x (int): the number to round up to
            y (int): the multiplier
            smallest (boolean): set to True to return smallest 
                multiple of y >= x
            
        Returns:
            int: if 'smallest' is False then largest multiple of y <= x, 
                else smallest multiple of y >= x

    """
    # remainder
    val = x % y

    # x is a multiple of y
    if val == 0:
        return x
    
    if smallest:
        # subtract remainder and the multiplier
        return (x - val) + y
    else:
        # subtract remainder
        return (x - val)


def fix_sequence_length(sequence, length):
    """
        Function to check if length of sequence matches specified
        length and then return a sequence that's either padded or
        truncated to match the given length

        Args:
            sequence (str): the input sequence
            length (int): expected length

        Returns:
            str: string of length 'length'
    """

    # check if the sequence is smaller than expected length
    if len(sequence) < length:
        # pad the sequence with 'N's
        sequence += 'N' * (length - len(sequence))
    # check if the sequence is larger than expected length
    elif len(sequence) > length:
        # truncate to expected length
        sequence = sequence[:length]

    return sequence


def one_hot_encode(sequences, seq_length):
    """
    
       One hot encoding of a list of DNA sequences 
       
       Args:
           sequences (list): python list of strings of equal length
           seq_length (int): expected length of each sequence in the 
               list
           
       Returns:
           numpy.ndarray: 
               3-dimension numpy array with shape 
               (len(sequences), len(list_item), 4)

    """
    
    if len(sequences) == 0:
        logging.error("'sequences' is empty")
        return None
    
    # First, let's make sure all sequences are of equal length
    sequences = list(map(
        fix_sequence_length, sequences, [seq_length] * len(sequences)))

    # Step 1. convert sequence list into a single string
    _sequences = ''.join(sequences)
    
    # Step 2. translate the alphabet to a string of digits
    transtab = str.maketrans('ACGTNYRMSWK', '01234444444')    
    sequences_trans = _sequences.translate(transtab)
    
    # Step 3. convert to list of ints
    int_sequences = list(map(int, sequences_trans))
    
    # Step 4. one hot encode using int_sequences to index 
    # into an 'encoder' array
    encoder = np.vstack([np.eye(4), np.zeros(4)])
    X = encoder[int_sequences]

    # Step 5. reshape 
    return X.reshape(len(sequences), len(sequences[0]), 4)

            
def reverse_complement_of_sequences(sequences):
    """
    
        Reverse complement of DNA sequences
       
        Args:
           sequences (list): python list of strings of DNA sequence of
               arbitraty length
    
        Returns:
            list: python list of strings
            
    """

    if len(sequences) == 0:
        logging.error("'sequences' is empty")
    
    # reverse complement translation table
    rev_comp_tab = str.maketrans("ACTG", "TGAC")

    # translate and reverse ([::-1] <= [start:end:step])
    return [seq.translate(rev_comp_tab)[::-1] for seq in sequences]


def reverse_complement_of_profiles(profiles, stranded=True):
    r"""
    
        Reverse complement of an genomics assay signal profile 

        Args:
            profiles (numpy.ndarray): 3-dimensional numpy array, a 
                batch of multitask profiles of shape 
                (#examples, seq_len, #assays) if unstranded and 
                (#examples, seq_len, #assays*2) if stranded. In the
                stranded case the assumption is: the postive & negative
                strands occur in pairs on axis=2(i.e. 3rd dimension) 
                e.g. 0th & 1st index, 2nd & 3rd...

        Returns:
            numpy.ndarray: 3-dimensional numpy array 
            
            
        IGNORE_FOR_SPHINX_DOCS:
        
        CONVERT (Stranded profile) 
                 ______ 
                |      | 
                |      | 
         _______|      |___________________________________________ 
         acgggttttccaaagggtttttaaaacccggttgtgtgtccacacacagtgtgtcaca 
         ---------------------------------------------------------- 
         ---------------------------------------------------------- 
         ʇƃɔɔɔɐɐɐɐƃƃʇʇʇɔɔɔɐɐɐɐɐʇʇʇʇƃƃƃɔɔɐɐɔɐɔɐɔɐƃƃʇƃʇƃʇƃʇɔɐɔɐɔɐƃʇƃʇ 
         ____________________________________________      ________ 
                                                     \    /         
                                                      \  /           
                                                       \/            
                                                                      
         TO                                                                      
         
                   /\
                  /  \      
         ________/    \____________________________________________
         tgtgacacactgtgtgtggacacacaaccgggttttaaaaaccctttggaaaacccgt
         ----------------------------------------------------------
         ----------------------------------------------------------
         ɐɔɐɔʇƃʇƃʇƃɐɔɐɔɐɔɐɔɔʇƃʇƃʇƃʇʇƃƃɔɔɔɐɐɐɐʇʇʇʇʇƃƃƃɐɐɐɔɔʇʇʇʇƃƃƃɔɐ
         ___________________________________________        _______
                                                    |      |
                                                    |      | 
                                                    |______|                                                          
         
         
         OR 
         
         CONVERT (unstranded profile)
         
                 ______
                |      |
                |      |       
         _______|      |___________________________________________
         acgggttttccaaagggtttttaaaacccggttgtgtgtccacacacagtgtgtcaca
        
         TO                                          
                                                     ______
                                                    |      |
                                                    |      |
         ___________________________________________|      |_______
         tgtgacacactgtgtgtggacacacaaccgggttttaaaaaccctttggaaaacccgt
        
        IGNORE_FOR_SPHINX_DOCS
       
    """
    
    # check if profiles is 3-dimensional
    if profiles.ndim != 3:
        logging.error("'profiles' should be a 3-dimensional array. "
                      "Found {}".format(profiles.ndim))

    # check if the 3rd dimension is an even number if profiles are stranded
    if stranded and (profiles.shape[2] % 2) != 0:
        logging.error("3rd dimension of stranded 'profiles' should be even. "
                      "Found {}".format(profiles.shape))

    if stranded:
    
        # get reshaped version of profiles array
        # axis = 2 becomes #assays
        tmp_prof = profiles.reshape(
            (profiles.shape[0], profiles.shape[1], -1, 2))

        # get reverse complement by flipping along axis 1 & 3
        # axis 1 is the sequence length axis & axis 3 is the 
        # +/- strand axis after reshaping
        rev_comp_profile = np.flip(tmp_prof, axis=(1, 3))
        
        # reshape back to match shape of the input
        return rev_comp_profile.reshape(profiles.shape)

    else:
        
        return np.flip(profiles, axis=1)
