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

# length of both the chromosomes
CHR_LEN = 80

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
    bw.addHeader([("chr1", CHR_LEN), ("chr2", CHR_LEN)], maxZooms=0)
    chroms = ["chr1"]*CHR_LEN + ["chr2"]*CHR_LEN
    starts = list(range(CHR_LEN)) + list(range(CHR_LEN))
    ends = list(range(1,CHR_LEN+1)) + list(range(1,CHR_LEN+1))
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
        self.plus_data = list(np.arange(0., CHR_LEN, 1.)) + list(np.arange(2*CHR_LEN,CHR_LEN,-1.))
        self.minus_data = list(np.arange(0.,2*CHR_LEN,2.)) + list(np.arange(4*CHR_LEN,2*CHR_LEN,-2.))

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
                    'ratio': [0.5]
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
        self.peak_seqs = ["ATATAT", "ACACAC", "CCAACC", "GGGGGG", 
                          "ATTTTT", "CGGGGG", "CGGAAG", "TTTTTT"]
        self.peak_seqs_one_hot = one_hot_encode(self.peak_seqs, 6)

        self.background_seqs = ["TGCGCG", "GTGTGT", "AAAAAA", "TAAATT", 
                                "GAAAAA", "TCCCCC", "CCCCCC", "GTATCT"]
        self.background_seqs_one_hot = one_hot_encode(self.background_seqs, 6)
        
        # load plus and minus bw values corresponding to peaks and background
        SCHEMA = ['chr','start', 'end', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'summit']

        self.plus_vals, self.minus_vals = {}, {}

        peak_coords = pd.read_csv(f"{DATA_PATH}/mini_genome/peaks.bed", sep='\t', names=SCHEMA)
        for i,x in peak_coords.iterrows():
            offset = CHR_LEN if x['chr']=="chr2" else 0
            st = offset+x['start']+x['summit']-2
            en = offset+x['start']+x['summit']+2
            self.plus_vals[self.peak_seqs[i]] = self.plus_data[st:en]
            self.minus_vals[self.peak_seqs[i]] = self.minus_data[st:en]

        bg_coords = pd.read_csv(f"{DATA_PATH}/mini_genome/background.bed", sep='\t', names=SCHEMA)
        for i,x in bg_coords.iterrows():
            offset = CHR_LEN if x['chr']=="chr2" else 0
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
                 ['chr1', 'chr2'], num_threads=2, batch_size=2)
    
        ep1_gen = gen.gen()

        batches = [x for x in ep1_gen]

        seqs = np.vstack([x['sequence'] for x in batches])

        # no background loci should be used for test set
        self.assertTrue(seqs.shape == (8, 6, 4))

        self.assertTrue(set(one_hot_to_dna(seqs)) == set(self.peak_seqs))
 
        # not sure if this always holds
        #self.assertTrue(np.all(np.equal(seqs, self.peak_seqs_one_hot)))

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
                 ['chr1', 'chr2'], num_threads=2, batch_size=2)

        ep1_gen = gen.gen()
        seqs1 = np.vstack([x[0]['sequence'] for x in ep1_gen])  
        
        ep2_gen = gen.gen()
        seqs2 = np.vstack([x[0]['sequence'] for x in ep2_gen])

        ep3_gen = gen.gen()
        batches = [x for x in ep3_gen]
        seqs3 = np.vstack([x[0]['sequence'] for x in batches])
        
        # check equivalence of generator for different epochs
        self.assertTrue(set(one_hot_to_dna(seqs1)) == set(one_hot_to_dna(seqs2)) == set(one_hot_to_dna(seqs3)))

        # stricter doesn't hold 
        # self.assertTrue(np.all(np.equal(seqs1,seqs2)))
        # self.assertTrue(np.all(np.equal(seqs2,seqs3)))

        # check it contains all positive sequences and 4 bg sequences
        # not assuming order is maintained
        seqs = one_hot_to_dna(seqs3)

        self.assertTrue(len(set(seqs).intersection(self.peak_seqs)) == 8)
        self.assertTrue(len(set(seqs).intersection(self.background_seqs)) == 4)

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

        counts_labels = np.vstack([x[1]['logcounts_predictions'] for x in batches])

        self.assertTrue(np.all(np.equal(counts_labels, np.log(1+profile_labels.sum(1)))))

    def test_train_gen(self):
        """
        Test data generation from train set
        """
        self.batch_gen_params['rev_comp_aug'] = True
        self.batch_gen_params['max_jitter'] =  0
        self.batch_gen_params['mode'] = 'train'

        gen = generators.MBPNetSequenceGenerator(self.input_json,
                 self.batch_gen_params,
                 self.genome_params['reference_genome'],
                 self.genome_params['chrom_sizes'],
                 ['chr1', 'chr2'], num_threads=2, batch_size=2)

        ep1_gen = gen.gen()
        batches = [x for x in ep1_gen]

        # each batch should contain in first half the sequences
        # and second half their rev comps
        self.assertTrue(batches[0][0]['sequence'].shape == (4,6,4))

        self.bg_plus, self.bg_minus = {}, {}

        seqs_fwd = np.vstack([x[0]['sequence'][:2] for x in batches])
        seqs_rev = np.vstack([x[0]['sequence'][2:] for x in batches])

        seqs_dna_fwd = one_hot_to_dna(seqs_fwd)
        self.assertTrue(len(set(seqs_dna_fwd).intersection(self.peak_seqs)) == 8)
        self.assertTrue(len(set(seqs_dna_fwd).intersection(self.background_seqs)) == 4)

        # check that weights of peak is 1 and bg is 0
        weights_fwd = np.array([y for x in batches for y in x[2][:2]])
        weights_rev = np.array([y for x in batches for y in x[2][2:]])

        # rev seqs (2nd half of each batch) should have same weights as fwd
        self.assertTrue(np.all(np.equal(weights_fwd, weights_rev)))

        self.assertTrue(len(set(one_hot_to_dna(seqs_fwd[weights_fwd == 1])).intersection(self.peak_seqs)) == 8)
        self.assertTrue(len(set(one_hot_to_dna(seqs_fwd[weights_fwd == 0])).intersection(self.background_seqs)) == 4)

        # check rev comp-ing of sequence
        self.assertTrue(np.all(np.equal(seqs_fwd, seqs_rev[:,::-1,::-1])))

        profile_bias_fwd = np.vstack([x[0]['profile_bias_input_0'][:2] for x in batches])
        profile_bias_rev = np.vstack([x[0]['profile_bias_input_0'][2:] for x in batches])
        profile_labels_fwd = np.vstack([x[1]['profile_predictions'][:2] for x in batches])
        profile_labels_rev = np.vstack([x[1]['profile_predictions'][2:] for x in batches])

        # since we used the same file for bias and signal
        self.assertTrue(np.all(np.equal(profile_bias_fwd, profile_labels_fwd)))
        self.assertTrue(np.all(np.equal(profile_bias_rev, profile_labels_rev)))

        # check rev-comping of profiles
        self.assertTrue(np.all(np.equal(profile_bias_fwd, profile_bias_rev[:, ::-1, ::-1])))

        logcts_preds_fwd = np.vstack([x[1]['logcounts_predictions'][:2] for x in batches])
        logcts_preds_rev = np.vstack([x[1]['logcounts_predictions'][2:] for x in batches])

        self.assertTrue(np.all(np.equal(logcts_preds_fwd, np.log(1+profile_labels_fwd.sum(1)))))
        self.assertTrue(np.all(np.equal(logcts_preds_rev, np.log(1+profile_labels_rev.sum(1)))))
