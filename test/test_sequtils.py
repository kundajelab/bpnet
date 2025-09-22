"""
test cases for sequtils
"""

from collections import OrderedDict
from bpnet.generators import sequtils

import numpy as np
import pandas as pd
import unittest

import os

DATA_PATH = os.path.join(os.path.dirname(__file__)) + "/data/"

class TestSeqUtils(unittest.TestCase):
    def test_getPeakPositions(self):
        """
            test getPeakPositions function that returns a pandas 
            dataframe. Check for shape of dataframe and column names
        """
    
        chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 
                  'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 
                  'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 
                  'chrX', 'chrY']
        
        tasks = {
            0: {
                'signal': {
                    'source': ["UNUSED/plus.bw",
                               "UNUSED/minus.bw"]
                },
                'loci': {
                    'source': [f"{DATA_PATH}/single_task/stranded_with_controls/task0/peaks.bed", 
                               f"{DATA_PATH}/single_task/stranded_with_controls/task0/peaks.bed"],
                },
                'bias': {
                    'source': ["UNUSED/control_plus.bw",
                               "UNUSED/control_minus.bw"],
                    'smoothing': [None]
                }            
            }
        }
    
        # read the chrom sizes into a dataframe and filter rows from
        # unwanted chromosomes
        chrom_sizes = pd.read_csv(f"{DATA_PATH}/GRCh38_EBV.chrom.sizes", sep='\t', 
                                  header=None, names=['chrom', 'size']) 
        chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]
    
        # get peak positions for each task as one dataframe
        peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128,
                                             chroms=chroms,
                                             drop_duplicates=False)
        
        # check if columns match
        columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight']    
        self.assertTrue(all([a == b for a, b in zip(columns, peaks_df.columns)]))
        
        # check if the shape matches
        self.assertTrue(peaks_df.shape == (48, 5))
        
        # get peak positions for each task as one dataframe, this time
        # drop duplicates. Since we are using the same peaks.bed file
        # the total number of peak position should be reduced by half
        peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128,
                                             chroms=chroms,
                                             drop_duplicates=True)
        
        # check if columns match
        columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight'] 
        self.assertTrue(all([a == b for a, b in zip(columns, peaks_df.columns)]))
        
        # check if the shape matches
        self.assertTrue(peaks_df.shape == (24, 5))
    
    def test_getPeakPositions_background(self):
        """
            test getPeakPositions function that returns a pandas 
            dataframe. Check for shape of dataframe and column names
        """
    
        chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 
                  'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 
                  'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 
                  'chrX', 'chrY']
        
        tasks = {
                    "0": {
                        "signal": {
                            "source": ["UNUSED/plus.bigWig", 
                                       "UNUSED/_minus.bigWig"]
                        },
                        "loci": {
                            "source": [f"{DATA_PATH}/loci.bed"]
                        },
                        "background_loci": {
                            "source": [f"{DATA_PATH}/background.bed"],
                            "ratio": [3]
                        },
                        "bias": {
                            "source": ["UNUSED/control_plus.bigWig",
                                       "UNUSED/control_minus.bigWig"],
                            "smoothing": [None, None]
                        }
                    }
                }
    
        # read the chrom sizes into a dataframe and filter rows from
        # unwanted chromosomes
        chrom_sizes = pd.read_csv(f"{DATA_PATH}/GRCh38_EBV.chrom.sizes", sep='\t', 
                                  header=None, names=['chrom', 'size']) 
        chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]
    
        # get peak positions for each task as one dataframe
        peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128, 
                                             chroms=chroms,
                                             drop_duplicates=False)
        
        # check if columns match
        columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight']    
        self.assertTrue(all([a == b for a, b in zip(columns, peaks_df.columns)]))
        
        # check if the shape matches
        self.assertTrue(peaks_df.shape == (200, 5))
    
    
    def test_roundToMultiple(self):
        """
            test roundToMultiple function
        """
        
        x = 500
        y = 64
        
        # y * 7 = 448, y * 8 = 512, 448 is the largest multiple of 64 < 500
        expected_res = 448
        
        self.assertTrue(sequtils.round_to_multiple(x, y) == expected_res)
    
        
    def test_one_hot_encode(self):
        """
            test once hot encoding of dna sequences
        """
    
        # list of same length sequences
        sequences = ['ACGN', 'AAGG', 'CTCT', 'NNNN', 'CCCC']
        
        # the expected one hot encoding
        expected_res = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                        [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                        [[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1]],
                        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]]
        
        res = sequtils.one_hot_encode(sequences, 4)
    
        self.assertTrue(np.all(np.equal(res, np.array(expected_res))))
        
        # list of unequal length sequences
        sequences = ['ACGN', 'AAGG', 'CTCTF', 'NNNN', 'CCCC']
        
        # this will truncate the 3rd sequence
        res = sequtils.one_hot_encode(sequences, 4)
    
        self.assertTrue(np.all(np.equal(res, expected_res)))
    
        
    def test_reverse_complement_of_sequences(self):
        """
            test reverse complement of one hot encoded sequences
        """
        
        # list of arbitrary length sequences
        sequences = ['ACGN', 'AAGGTTCC', 'CTCTGG', 'NNNN', 'CCCCAAA']
        
        # the expected reverse complement sequences3
        expected_res = ['NCGT', 'GGAACCTT', 'CCAGAG', 'NNNN', 'TTTGGGG']
        
        res = sequtils.reverse_complement_of_sequences(sequences)
    
        self.assertTrue(res == expected_res)
    
    
    def test_reverse_complement_of_profiles(self):
        """
            test reverse complement of genomic assay signal profiles
        """
        
        # stranded profile 
        # examples, #seq_len, #assays*2) = (5, 3, 2*2)
        stranded_profile = [[[1, 0, 0, 0], 
                             [0, 1, 0, 0], 
                             [0, 0, 1, 0]], 
                            [[1, 0, 0, 0], 
                             [1, 0, 0, 0], 
                             [0, 0, 1, 0]], 
                            [[0, 1, 0, 0], 
                             [0, 0, 0, 1], 
                             [0, 1, 0, 0]], 
                            [[0, 1, 1, 0], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0]], 
                            [[0, 1, 0, 0], 
                             [0, 1, 0, 0], 
                             [0, 1, 0, 1]]]
    
        # reverese complement of stranded profile
        expected_res = [[[0, 0, 0, 1], 
                         [1, 0, 0, 0], 
                         [0, 1, 0, 0]],
                        [[0, 0, 0, 1], 
                         [0, 1, 0, 0], 
                         [0, 1, 0, 0]],
                        [[1, 0, 0, 0], 
                         [0, 0, 1, 0], 
                         [1, 0, 0, 0]],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0], 
                         [1, 0, 0, 1]],
                        [[1, 0, 1, 0], 
                         [1, 0, 0, 0], 
                         [1, 0, 0, 0]]]        
    
        # get reverese complement of stranded profile
        res = sequtils.reverse_complement_of_profiles(np.array(stranded_profile),
                                                      stranded=True)    
        self.assertTrue(np.all(np.equal(res, np.array(expected_res))))
    
        # examples, #seq_len, #assays) = (5, 3, 4)
        unstranded_profile = [[[1, 0, 0, 0], 
                               [0, 1, 0, 0], 
                               [0, 0, 1, 0]], 
                              [[1, 0, 0, 0], 
                               [1, 0, 0, 0], 
                               [0, 0, 1, 0]], 
                              [[0, 1, 0, 0], 
                               [0, 0, 0, 1], 
                               [0, 1, 0, 0]], 
                              [[0, 1, 1, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]], 
                              [[0, 1, 0, 0], 
                               [0, 1, 0, 0], 
                               [0, 1, 0, 1]]]
        
        # reverese complement of unstranded profile
        expected_res = [[[0, 0, 1, 0], 
                         [0, 1, 0, 0], 
                         [1, 0, 0, 0]],
                        [[0, 0, 1, 0], 
                         [1, 0, 0, 0], 
                         [1, 0, 0, 0]],
                        [[0, 1, 0, 0], 
                         [0, 0, 0, 1], 
                         [0, 1, 0, 0]],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0], 
                         [0, 1, 1, 0]],
                        [[0, 1, 0, 1], 
                         [0, 1, 0, 0], 
                         [0, 1, 0, 0]]]        
    
        # get reverese complement of stranded profile
        res = sequtils.reverse_complement_of_profiles(np.array(unstranded_profile),
                                                      stranded=False)    
        self.assertTrue(np.all(np.equal(res, np.array(expected_res))))
