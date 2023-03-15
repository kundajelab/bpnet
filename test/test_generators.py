"""
test cases for generators
"""

from bpnet.generators import generators
from bpnet.generators.sequtils import one_hot_encode

import numpy as np
import pandas as pd
import unittest
import tempfile
import pyfaidx
import pyBigWig
import json

import os

DATA_PATH = os.path.join(os.path.dirname(__file__)) + "/data/"

def one_hot_to_dna(one_hot):
    bases = np.array(["A", "C", "G", "T", "N"])
    # Create N x L array of all 5s
    one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

    # Get indices of where the 1s are
    batch_inds, seq_inds, base_inds = np.where(one_hot)

    # In each of the locations in the N x L array, fill in the location of the 1
    one_hot_inds[batch_inds, seq_inds] = base_inds

    # Fetch the corresponding base for each position using indexing
    seq_array = bases[one_hot_inds]
    return ["".join(seq) for seq in seq_array]

def write_bigwig_file(fname, data):
    bw = pyBigWig.open(fname, 'w')
    bw.addHeader([("chr1", 50), ("chr2", 50)], maxZooms=0)
    chroms = ["chr1"]*50 + ["chr2"]*50
    starts = list(range(50)) + list(range(50))
    ends = list(range(1,51)) + list(range(1,51))
    bw.addEntries(chroms, starts=starts, ends=ends, values=data)
    bw.close()

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.genome_params = {}
        self.genome_params['reference_genome'] = f"{DATA_PATH}/mini_genome/mini.genome.fa"
        self.genome_params['chrom_sizes'] = f"{DATA_PATH}/mini_genome/mini.genome.chrom.sizes"
 
        self.batch_gen_params = {}
        self.batch_gen_params['sequence_generator_name'] = "BPNet"
        self.batch_gen_params['input_seq_len'] = 6
        self.batch_gen_params['output_len'] = 4
        self.batch_gen_params['shuffle'] = False

        # dots (floats) are for pyBigWig
        self.plus_data = list(np.arange(50.)) + list(np.arange(100.,50.,-1.))
        self.minus_data = list(np.arange(0.,100.,2.)) + list(np.arange(200.,100.,-2.))

        _, self.plus_bw = tempfile.mkstemp()
        _, self.minus_bw = tempfile.mkstemp()
 
        write_bigwig_file(self.plus_bw, self.plus_data)
        write_bigwig_file(self.minus_bw, self.minus_data)

        # using the same for signal and bias
        tasks = {
            0: {
                'signal': {
                    'source': [self.plus_bw,
                               self.minus_bw]
                },
                'loci': {
                    'source': [f"{DATA_PATH}/mini_genome/peaks.bed"]
                },
                'background_loci': {
                    'source': [f"{DATA_PATH}/mini_genome/background.bed"],
                    'ratio': [1./3]
                },
                'bias': {
                    'source': [self.plus_bw,
                               self.minus_bw],
                    'smoothing': [None, None]
                }            
            }
        }
        
        _, self.input_json = tempfile.mkstemp()
        with open(self.input_json, 'w') as f:
            json.dump(tasks, f)

        # manually extracted from files 
        self.peak_seqs = ["ATATAT", "ACACAC", "CCAACC", 
                       "ATTTTT", "CGGGGG", "CGGAAG"]
        self.peak_seqs_one_hot = one_hot_encode(self.peak_seqs, 6)

        self.background_seqs = ["TGCGCG", "GTGTGT", "GAAAAA", "TCCCCC"]
        self.background_seqs_one_hot = one_hot_encode(self.background_seqs, 6)
        
        # load plus and minus bw values corresponding to peaks and background
        SCHEMA = ['chr','start', 'end', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'summit']

        self.plus_vals, self.minus_vals = {}, {}

        peak_coords = pd.read_csv(f"{DATA_PATH}/mini_genome/peaks.bed", sep='\t', names=SCHEMA)
        for i,x in peak_coords.iterrows():
            offset = 50 if x['chr']=="chr2" else 0
            st = offset+x['start']+x['summit']-2
            en = offset+x['start']+x['summit']+2
            self.plus_vals[self.peak_seqs[i]] = self.plus_data[st:en]
            self.minus_vals[self.peak_seqs[i]] = self.minus_data[st:en]

        bg_coords = pd.read_csv(f"{DATA_PATH}/mini_genome/background.bed", sep='\t', names=SCHEMA)
        for i,x in bg_coords.iterrows():
            offset = 50 if x['chr']=="chr2" else 0
            st = offset+x['start']+x['summit']-2
            en = offset+x['start']+x['summit']+2
            self.plus_vals[self.background_seqs[i]] = self.plus_data[st:en]
            self.minus_vals[self.background_seqs[i]] = self.minus_data[st:en]


    def tearDown(self):
        os.remove(self.plus_bw)
        os.remove(self.minus_bw)
        os.remove(self.input_json)

    def test_test_gen(self):
        """
        Test data generation from test set. 
        """
        self.batch_gen_params['rev_comp_aug'] = False
        self.batch_gen_params['max_jitter'] =  0
        self.batch_gen_params['mode'] = 'test'
        
        gen = generators.MBPNetSequenceGenerator(self.input_json, 
                 self.batch_gen_params, 
                 self.genome_params['reference_genome'], 
                 self.genome_params['chrom_sizes'], 
                 ['chr1', 'chr2'], num_threads=1, batch_size=2)
    
        ep1_gen = gen.gen()

        batches = [x for x in ep1_gen]

        seqs = np.vstack([x['sequence'] for x in batches])

        self.assertTrue(seqs.shape == (6, 6, 4))

        self.assertTrue(np.all(np.equal(seqs, self.peak_seqs_one_hot)))

    def test_val_gen(self):
        """ 
        Test data generation from val set
        """
        self.batch_gen_params['rev_comp_aug'] = False
        self.batch_gen_params['max_jitter'] =  0
        self.batch_gen_params['mode'] = 'val'

        gen = generators.MBPNetSequenceGenerator(self.input_json,
                 self.batch_gen_params,
                 self.genome_params['reference_genome'],
                 self.genome_params['chrom_sizes'],
                 ['chr1', 'chr2'], num_threads=1, batch_size=2)

        ep1_gen = gen.gen()
        seqs1 = np.vstack([x[0]['sequence'] for x in ep1_gen])  
        
        ep2_gen = gen.gen()
        batches = [x for x in ep2_gen]
        seqs2 = np.vstack([x[0]['sequence'] for x in batches])

        # check equivalence of generator for different epochs
        self.assertTrue(np.all(np.equal(seqs1,seqs2)))

        # check it contains all positive sequences and 2 bg sequences
        # not assuming order is maintained
        seqs = one_hot_to_dna(seqs1)

        self.assertTrue(len(set(seqs).intersection(self.peak_seqs)) == 6)
        self.assertTrue(len(set(seqs).intersection(self.background_seqs)) == 2)

        # check values
        profile_bias = np.vstack([x[0]['profile_bias_input_0'] for x in batches])
        profile_labels = np.vstack([x[1]['profile_predictions'] for x in batches])

        # since we used the same file for bias and signal
        self.assertTrue(np.all(np.equal(profile_bias, profile_labels)))
        
        # check against expected values
        ref_plus_vals = np.array([self.plus_vals[x] for x in seqs])
        ref_minus_vals = np.array([self.minus_vals[x] for x in seqs])

        self.assertTrue(np.all(np.equal(profile_bias[:,:,0], ref_plus_vals)))
        self.assertTrue(np.all(np.equal(profile_bias[:,:,1], ref_minus_vals)))
