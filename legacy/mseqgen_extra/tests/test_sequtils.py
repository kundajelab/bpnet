"""
test cases for sequtils
"""

from collections import OrderedDict
from mseqgen import sequtils

import numpy as np
import pandas as pd


def test_getChromPositions_sequential():
    """
        test getChromPositions in 'sequential' mode. The function 
        returns a pandas dataframe with sequentially sampled 
        positions from each chromosome. Check for shape of dataframe
        and column names
    """
    
    chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 
              'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 
              'chrX', 'chrY']
    
    # read the chrom sizes into a dataframe and filter rows from
    # unwanted chromosomes
    chrom_sizes = pd.read_csv('tests/GRCh38_EBV.chrom.sizes', sep='\t', 
                              header=None, names=['chrom', 'size']) 
    
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]

    peaks_df = sequtils.getChromPositions(chroms, chrom_sizes, flank=128, 
                                          mode='sequential', num_positions=100,
                                          step=50)
    
    # check if return value is not None
    assert peaks_df is not None

    # check if the columns match
    columns = ['chrom', 'pos']    
    assert all([a == b for a, b in zip(columns, peaks_df.columns)])
    
    # check if the shape matches
    assert peaks_df.shape == (24 * 100, 2)  


def test_getChromPositions_random():
    """
        test getChromPositions in 'random' mode. The function 
        returns a pandas dataframe with randomly sampled positions
        from each chromosome. Check for shape of dataframe and 
        column names
    """
    
    chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 
              'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 
              'chrX', 'chrY']
    
    # read the chrom sizes into a dataframe and filter rows from
    # unwanted chromosomes
    chrom_sizes = pd.read_csv('tests/GRCh38_EBV.chrom.sizes', sep='\t', 
                              header=None, names=['chrom', 'size']) 
    
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]

    peaks_df = sequtils.getChromPositions(chroms, chrom_sizes, flank=128, 
                                          mode='random', num_positions=100,
                                          step=50)
    
    # check if return value is not None
    assert peaks_df is not None

    # check if the columns match
    columns = ['chrom', 'pos']    
    assert all([a == b for a, b in zip(columns, peaks_df.columns)])
    
    # check if the shape matches
    assert peaks_df.shape == (24 * 100, 2)  


def test_getPeakPositions():
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
                'source': ["tests/test_data/single_task/"
                           "stranded_with_controls/task0/plus.bw", 
                           "tests/test_data/single_task/"
                           "stranded_with_controls/task0/minus.bw"]
            },
            'loci': {
                'source': ["tests/test_data/single_task/"
                           "stranded_with_controls/task0/peaks.bed", 
                           "tests/test_data/single_task/"
                           "stranded_with_controls/task0/peaks.bed"],
                'samples_per_epoch': [-1, -1]
            },
            'bias': {
                'source': ["tests/test_data/single_task/"
                           "stranded_with_controls/task0/control_plus.bw", 
                           "tests/test_data/single_task/"
                           "stranded_with_controls/task0/control_minus.bw"],
                'smoothing': [None]
            }            
        }
    }

    # read the chrom sizes into a dataframe and filter rows from
    # unwanted chromosomes
    chrom_sizes = pd.read_csv('tests/GRCh38_EBV.chrom.sizes', sep='\t', 
                              header=None, names=['chrom', 'size']) 
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]

    # get peak positions for each task as one dataframe
    peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128,
                                         chroms=chroms,
                                         drop_duplicates=False)
    
    # check if columns match
    columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight']    
    assert all([a == b for a, b in zip(columns, peaks_df.columns)])
    
    # check if the shape matches
    assert peaks_df.shape == (48, 5)
    
    # get peak positions for each task as one dataframe, this time
    # drop duplicates. Since we are using the same peaks.bed file
    # the total number of peak position should be reduced by half
    peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128,
                                         chroms=chroms,
                                         drop_duplicates=True)
    
    # check if columns match
    columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight'] 
    assert all([a == b for a, b in zip(columns, peaks_df.columns)])
    
    # check if the shape matches
    assert peaks_df.shape == (24, 5)

def test_getPeakPositions_background():
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
                        "source": ["/users/zahoor/lab_data3/TF-Atlas/test_TF/data/ENCSR362NWP_plus.bigWig", 
                                   "/users/zahoor/lab_data3/TF-Atlas/test_TF/data/ENCSR362NWP_minus.bigWig"]
                    },
                    "loci": {
                        "source": ["/users/zahoor/mseqgen/tests/test_data/loci.bed"]
                    },
                    "background_loci": {
                        "source": ["/users/zahoor/mseqgen/tests/test_data/background.bed"],
                        "ratio": [3]
                    },
                    "bias": {
                        "source": ["/users/zahoor/lab_data3/TF-Atlas/test_TF/data/ENCSR362NWP_control_plus.bigWig",
                                   "/users/zahoor/lab_data3/TF-Atlas/test_TF/data/ENCSR362NWP_control_minus.bigWig"],
                        "smoothing": [None, None]
                    }
                }
            }

    # read the chrom sizes into a dataframe and filter rows from
    # unwanted chromosomes
    chrom_sizes = pd.read_csv('tests/GRCh38_EBV.chrom.sizes', sep='\t', 
                              header=None, names=['chrom', 'size']) 
    chrom_sizes = chrom_sizes[chrom_sizes['chrom'].isin(chroms)]

    # get peak positions for each task as one dataframe
    peaks_df = sequtils.getPeakPositions(tasks, chrom_sizes, flank=128, 
                                         chroms=chroms,
                                         drop_duplicates=False)
    
    # check if columns match
    columns = ['chrom', 'start_coord', 'end_coord', 'pos', 'weight']    
    assert all([a == b for a, b in zip(columns, peaks_df.columns)])
    
    # check if the shape matches
    assert peaks_df.shape == (200, 5)


def test_roundToMultiple():
    """
        test roundToMultiple function
    """
    
    x = 500
    y = 64
    
    # y * 7 = 448, y * 8 = 512, 448 is the largest multiple of 64 < 500
    expected_res = 448
    
    assert sequtils.roundToMultiple(x, y) == expected_res

    
def test_one_hot_encode():
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

    np.testing.assert_array_equal(res, np.array(expected_res))
    
    # list of unequal length sequences
    sequences = ['ACGN', 'AAGG', 'CTCTF', 'NNNN', 'CCCC']
    
    # this will truncate the 3rd sequence
    res = sequtils.one_hot_encode(sequences, 4)

    np.testing.assert_array_equal(res, expected_res)

    
def test_reverse_complement_of_sequences():
    """
        test reverse complement of one hot encoded sequences
    """
    
    # list of arbitrary length sequences
    sequences = ['ACGN', 'AAGGTTCC', 'CTCTGG', 'NNNN', 'CCCCAAA']
    
    # the expected reverse complement sequences3
    expected_res = ['NCGT', 'GGAACCTT', 'CCAGAG', 'NNNN', 'TTTGGGG']
    
    res = sequtils.reverse_complement_of_sequences(sequences)

    assert res == expected_res


def test_reverse_complement_of_profiles():
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
    np.testing.assert_array_equal(res, np.array(expected_res))

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
    np.testing.assert_array_equal(res, np.array(expected_res))
