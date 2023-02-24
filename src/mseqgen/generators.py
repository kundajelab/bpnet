"""
    This module contains classes for all the sequence data generators

    Classes
    
    MSequenceGenerator - The main base class for all generators.
     
    Multi task batch data generation for training deep nural networks
    on high-throughput sequencing data of various geonmics assays
    

    MBPNetSequenceGenerator - Derives from MSequenceGenerator.
    
    Multi task batch data generation for training BPNet on
    high-throughput sequencing data of various geonmics assays
         
    
    IGNORE_FOR_SPHINX_DOCS:
    
    License
    
    MIT License
    
    Copyright (c) 2020 Kundaje Lab
    
    Permission is hereby granted, free of charge, to any person 
    obtaining a copy of this software and associated documentation 
    files (the "Software"), to deal in the Software without 
    restriction, including without limitation the rights to use, 
    copy, modify, merge, publish, distribute, sublicense, and/or 
    sell copiesof the Software, and to permit persons to whom the 
    Software is furnished to do so, subject to the following 
    conditions: 
    
    The above copyright notice and this permission notice shall be 
    included in all copies or substantial portions of the 
    Software. 
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY 
    KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE  
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR  
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS   
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR  
    OTHERWISE, ARISING FROM, OUT OF OR IN  CONNECTION WITH THE 
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
    
    
    IGNORE_FOR_SPHINX_DOCS:

"""
import random
random.seed(1234)
from numpy.random import seed
seed(1234)

import hashlib
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd 
import pyBigWig
import pyfaidx

import re

from mseqgen import sequtils
from mseqgen.exceptionhandler import NoTracebackException
from mseqgen import utils
from queue import Queue
from threading import Thread





class MSequenceGenerator:
    
    """ Multi task batch data generation for training deep neural
        networks on high-throughput sequencing data of various
        geonmics assays
          
        Args:
            tasks_json (str): path to the json file containing task
                information, required keys - 'signal', 'loci', 'bias'
                
            batch_gen_params (dictionary): python dictionary with batch
                generation parameters. Contains the following keys - 
            
                *input_seq_len (int)*
                    length of input DNA sequence
                
                *output_len (int)*
                    length of output profile
                
                *max_jitter (int)*
                    maximum value for randomized jitter to offset the 
                    peaks from the exact center of the input
                
                *rev_comp_aug (boolean)*
                    enable reverse complement augmentation
                
                *negative_sampling_rate (float)*
                    the fraction of batch_size that determines how many 
                    negative samples are added to each batch

                *shuffle (boolean)*
                    specify whether input data is shuffled at the 
                    begininning of each epoch
                
                *mode (str)*
                    'train', 'val' or 'test'
                 
            reference_genome (str): the path to the reference genome 
                fasta file
                
            chrom_sizes (str): path to the chromosome sizes file
            
            chroms (list): the list of chromosomes that will be sampled
                for batch generation
            
            loci_indices (list): list of indices to filter rows from
                 the 'loci' peaks file
            
            background_loci_indices (list): list of indices to filter rows from
                 the 'loci' backgrounds file

            num_threads (int): number of parallel threads for batch
                generation, default = 10
                
            batch_size (int): size of each generated batch of data, 
                default = 64
                
            epochs (int): the number of epochs for which data has to 
                be generated
                
            background_only (boolean): True, if batches are to be 
                generated with background samples alone
                
            foreground_weight (float): sample weight for foreground
                samples
            
            background_weight (float): sample weight for background
                samples
                
            set_bias_as_zero (boolean): if True bias tracks will be zero. This can be
            used for testing without bias after training with bias tracks.
                
        **Members**
        
        IGNORE_FOR_SPHINX_DOCS:
            _mode (str): 'train', 'val' or 'test'

            _tasks (collections.OrderedDict): dictionary of input tasks
                taken from tasks_json
            
            _num_tasks (int): the number of tasks in '_tasks'
            
            _total_signal_tracks (int): the number of tracks across all 
                tasks

            _bias_tracks (dict): number of bias tracks for each task

            _reference (str): the path to the reference genome fasta
                file
            
            _chroms (list): the list of chromosomes that will be sampled
                for batch generation
            
            _chrom_sizes_df (pandas.Dataframe): dataframe of the 
                chromosomes and their corresponding sizes
            
            _num_threads (int): number of parallel threads for batch
                generation
            
            _batch_size (int): size of each generated batch of data
                        
            _input_flank (int): one half of input sequence length
            
            _output_flank (int): one half of output sequence length
            
            _max_jitter (int): the maximum absolute value of jitter to
                vary the position of the peak summit to left or right
                of the exact center of the input sequence. Range is
                -max_jitter to +max_jitter.
            
            _negative_sampling_rate (float): Use a positive value > 0.0
                to specify how many negative samples will be added to
                each batch. num_negative_samples = 
                negative_sampling_rate * batch_size. Ignored if 
                --mode is not 'train'
            
            _rev_comp_aug (boolean): specify whether reverse complement
                augmentation should be applied to each batch of data.
                If True, the size of the generated batch is doubled 
                (i.e batch_size*2 or if negative samples are added then
                (batch_size + num_negative_samples)*2). Ignored if 
                --mode is not 'train'
            
            _shuffle (boolean): if True input data will be shuffled at
                the begininning of each epoch
                
            _loci (pandas.DataFrame): pandas dataframe of aggregated 
                loci across all tasks

            _loci_size (int): size of the input loci dataframe
            
            _resized_loci (pandas.DataFrame): pandas dataframe of loci
                after resizing for optimal batch generation
                
            _resized_loci_size (int): size of the resized loci dataframe
            
            
            
        IGNORE_FOR_SPHINX_DOCS
    """

    def __init__(self, tasks_json, batch_gen_params, reference_genome, 
                 chrom_sizes, chroms=None, 
                 loci_indices=None,background_loci_indices=None, num_threads=10, batch_size=64, 
                 epochs=100, background_only=False, foreground_weight=1, 
                 background_weight=0, set_bias_as_zero=False):
        
        #: ML task mode 'train', 'val' or 'test'
        self._mode = batch_gen_params['mode']

        # make sure the input_data json file exists
        if not os.path.isfile(tasks_json):
            raise NoTracebackException(
                "File not found: {} ".format(tasks_json))
        
        # load the json file
        with open(tasks_json, 'r') as inp_json:
            try:
                tasks = json.loads(inp_json.read())
                # since the json has keys as strings, we convert the 
                # top level keys to int so we can used them later for
                # indexing
                #: dictionary of tasks for training
                self._tasks = {int(k): v for k, v in tasks.items()}
            except json.decoder.JSONDecodeError:
                raise NoTracebackException(
                    "Unable to load json file {}. Valid json expected. "
                    "Check the file for syntax errors.".format(
                        tasks_json))

        # check if the reference genome file exists
        if not os.path.isfile(reference_genome):
            raise NoTracebackException(
                "File not found: {} OR you may have accidentally "
                "specified a directory path.", reference_genome)
        
        # check if the chrom_sizes file exists
        if not os.path.isfile(chrom_sizes):
            raise NoTracebackException(
                "File not found: {} OR you may have accidentally "
                "specified a directory path.".format(chrom_sizes))

        #: the number of tasks in _tasks 
        self._num_tasks = len(list(self._tasks.keys()))
        
        #: the number of tracks across all tasks
        self._total_signal_tracks = 0
        for i in range(self._num_tasks):
            self._total_signal_tracks += \
                len(self._tasks[i]['signal']['source'])
        
        #: the number of bias tracks for each task
        self._bias_tracks = {}
        for i in range(self._num_tasks):
            self._bias_tracks[i] = len(self._tasks[i]['bias']['source'])
            
        #whether to set bias as zero
        self._set_bias_as_zero = set_bias_as_zero

        #: path to the reference genome
        self._reference = reference_genome

        #: dataframe of the chromosomes and their corresponding sizes
        self._chrom_sizes_df = pd.read_csv(
            chrom_sizes, sep='\t', header=None, names=['chrom', 'size']) 

        #: list of chromosomes that will be sampled for batch generation
        self._chroms = chroms
        
        #: list of indices to select rows from the 'loci' peaks file
        self._loci_indices = loci_indices
        
        #: list of indices to select rows from the 'background_loci' peaks file
        self._background_loci_indices = background_loci_indices
        
        # keep only those _chrom_sizes_df rows corresponding to the 
        # required chromosomes in _chroms
        if self._chroms != None:
            self._chrom_sizes_df = self._chrom_sizes_df[
                self._chrom_sizes_df['chrom'].isin(self._chroms)]

        # generate a new column for sampling weights of the chromosomes
        self._chrom_sizes_df['weights'] = \
            (self._chrom_sizes_df['size'] / self._chrom_sizes_df['size'].sum())

        #: number of parallel threads for batch generation 
        self._num_threads = num_threads
        
        #: size of each generated batch of data
        self._batch_size = batch_size

        # rest of batch generation parameters
        #: int:one half of input sequence length
        self._input_flank = batch_gen_params['input_seq_len'] // 2
        
        #: one half of input sequence length
        self._output_flank = batch_gen_params['output_len'] // 2        
        
        #: the maximum absolute value of jitter to vary the position
        #: of the peak summit to left or right of the exact center
        #: of the input sequence. Range is -max_jitter to +max_jitter.
        self._max_jitter = batch_gen_params['max_jitter']
        
        #: Use a positive value > 0.0 to specify how many negative
        #: samples will be added to each batch. num_negative_samples = 
        #: negative_sampling_rate * batch_size. Ignored if --mode is
        # not 'train'
        self._negative_sampling_rate = \
            batch_gen_params['negative_sampling_rate']
        
        #: if True, reverse complement augmentation will be applied to
        #: each batch of data. The size of the generated batch is 
        #: doubled (i.e batch_size*2 or if negative samples are added 
        #: then (batch_size + num_negative_samples)*2). Ignored if
        #: --mode is not 'train'
        self._rev_comp_aug = batch_gen_params['rev_comp_aug']
        
        #: if True, shuffle the data before the beginning of the epoch
        self._shuffle = batch_gen_params['shuffle']
        
        # list of loci dataframes for each epoch
        self._loci = []
        
        # size of each loci dataframe
        # we expect this to be the same for each epoch
        self._loci_size = [] 
        
        # resized loci for each epoch for multithreaded load balancing
        self._resized_loci = []
        
        # size of the resized loci dataframe for each epoch
        # we expect this to be the same for each epoch
        self._resized_loci_size = [] 

        # create a dataframe of samples for each epoch
        # In 'train' and 'val' If no background is specified then
        # this results in an identical dataframe of foreground loci for
        # all epochs, but when background is specified you will get the
        # foreground samples mixed with a random sampling of the  
        # background samples depending on the user speficied 
        # background:foreground ratio
        # In 'test' only foreground loci are used
        for i in range(epochs):
            # keys from input json to specify where to get samples from
            loci_keys = ['loci', 'background_loci']
            if self._mode == 'test':
                loci_keys = ['loci']                
            
            #: pandas dataframe of aggregated loci across all tasks
            peaks_df = sequtils.getPeakPositions(
                self._tasks,
                self._chrom_sizes_df[['chrom', 'size']], self._input_flank,
                self._chroms,mode=self._mode,
                loci_indices=self._loci_indices,
                background_loci_indices=self._background_loci_indices,
                loci_keys=loci_keys,
                drop_duplicates=True, background_only=background_only, 
                foreground_weight=foreground_weight, 
                background_weight=background_weight)
            self._loci.append(peaks_df)

            #: size of the input loci dataframe
            self._loci_size.append(len(self._loci[-1]))

            if self._mode == 'train' or self._mode == 'val':
                # trim samples so we can balance load across all threads
                # the resulting samples will be in self._resized_loci
                # Note: In 'train' or 'val' mode we want to make sure to
                # not to repeat samples to skew training or validation 
                # loss, so trimming and not padding is the better option
                self._trim_samples()

            elif self._mode == 'test':
                # pad samples so we can balance load across all threads
                # the resulting samples will be in self._resized_loci
                # Note: In 'test' mode we want to make sure not to miss
                # any samples, so we pad by repeating a few samples, but 
                # we need to make sure to not count those samples for 
                # the metrics computation
                self._pad_samples()
        
        #: the current epoch#
        self.curr_epoch = 0
            

    def _trim_samples(self):
        """
            trim self._loci dataframe so that the length of the 
            dataframe is an exact multiple of num_threads * batch_size.
            We do this so we can equally divide the batches across 
            several batch generation threads                 
        """
        
        # auto adjust num_threads and batch_size if the number of
        # samples is too low
        loop_cnt = 0
        while True:
            # adjust values from the second iteration onwards
            if loop_cnt > 0:      
                # we'll adjust batch_size first
                if self._batch_size > 1:
                    self._batch_size = self._batch_size // 2
                    
                elif self._num_threads > 1:
                    self._num_threads -= 1

            # largest multiple of num_threads * batch_size < sample_size
            largest_multiple = sequtils.round_to_multiple(
                self._loci_size[-1],
                self._num_threads * self._batch_size, 
                smallest=False)

            loop_cnt += 1
            
            # end condition, we cant have batch_size or num_threads < 1
            if self._batch_size == 1 and self._num_threads == 1:
                break
            
            # we keep adjusting if we cant find a good combination
            # of batch_size and num_threads
            if largest_multiple == 0:
                continue
            else:
                break
    
        # if we still can't find a good (batch_size, num_threads) pair,
        # which literally means that train or val set has 0 samples
        # then we raise an exception
        if largest_multiple == 0:
            raise NoTracebackException(
                "Your data does not have enough samples to generate batches "
                "of size {} across {} threads. Either reduce 'threads' or "
                "'batch_size'".format(self._batch_size, self._num_threads))

        #: pandas dataframe of loci after resizing for optimal
        #: batch generation
        if self._mode == "train":
            resized_loci_df = self._loci[-1].sample(largest_multiple, replace=False)
            self._resized_loci.append(resized_loci_df)
            print("after trim hash:",int(hashlib.sha256(str(resized_loci_df).encode('utf-8')).hexdigest(), 16))
        else:
            resized_loci_df = self._loci[-1].sample(largest_multiple, replace=False, random_state=1)
            self._resized_loci.append(resized_loci_df)
            print("after trim hash:",int(hashlib.sha256(str(resized_loci_df).encode('utf-8')).hexdigest(), 16))

        #: size of the loci dataframe after resizing
        self._resized_loci_size.append(len(self._resized_loci[-1]))
        
        # lets output the logging info only for the data for the first
        # epoch, this will be identical for the remaining epochs
        if len(self._resized_loci) == 1:
            logging.info(
                "mode '{}': batch size - {}".format(
                    self._mode, self._batch_size))
            logging.info(
                "mode '{}': #threads - {}".format(
                    self._mode, self._num_threads))
            logging.info(
                "mode '{}': Data size (after trimming {} samples) - {}".format(
                    self._mode, self._loci_size[-1] - largest_multiple, 
                    self._resized_loci_size[-1]))
 
    def _pad_samples(self):
        """
            pad self._loci dataframe so that the length of the 
            dataframe is an exact multiple of num_threads * batch_size.
            We do this so we can equally divide the batches across 
            several batch generation threads                 
        """
        
        # smallest multiple of num_threads * batch_size > loci_size
        smallest_multiple = sequtils.round_to_multiple(
            self._loci_size[-1], self._num_threads * self._batch_size, 
            smallest=True)

        #: pandas dataframe of loci after resizing for optimal
        #: batch generation
        self._resized_loci.append(pd.concat(
            [self._loci[-1], self._loci[-1].sample(
                smallest_multiple - self._loci_size[-1], replace=True)]))

        #: size of the loci dataframe after resizing
        self._resized_loci_size.append(len(self._resized_loci[-1]))
        
        # print the logging info only once
        if len( self._resized_loci) == 1:
            logging.info(
                "mode '{}': Data size (after padding {} samples) - {}".format(
                    self._mode, smallest_multiple - self._loci_size[-1], 
                    self._resized_loci_size[-1]))

    def _get_num_bias_tracks_for_task(self, task):
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

    def get_input_tasks(self):
        """
            The dictionary of tasks loaded from the json file
            input_config['data']
            
            Returns:
                
                dict: dictionary of input tasks
        """
        
        return self._tasks

    def get_samples(self):
        """
            The data samples before resizing
            
            Returns:
                
                pandas.Dataframe: dataframe of samples before resizing
        """
        
        return self._loci
    
    def get_resized_samples(self):
        """
            The data samples used in batch generation
            (after resizing, either trimming for train/val and 
            padding for test)
            
            Returns:
                
                pandas.Dataframe: dataframe of samples after resizing
        """
        
        return self._resized_loci
    
    def get_samples_len(self):
        """
            The number of data samples before resizing
            
            Returns:
                
                int: number of data samples before resizing
        """
        
        return self._loci_size[0]
    
    def get_resized_samples_len(self):
        """
            The number of data samples used in batch generation
            (after resizing, either trimming for train/val and 
            padding for test)
            
            Returns:
                
                int: number of data samples used in batch generation
        """
        
        return self._resized_loci_size[0]
    
    def len(self):
        """
            The number of batches per epoch
            
            Returns:
                int: number of batches of data generated in each epoch
        """
        
        return self._resized_loci_size[0] // self._batch_size
   
    def _generate_batch(self, coords):
        """ 
            Generate one batch of inputs and outputs
            
        """
        
        raise NotImplementedError("Method not implemented.")

    def get_name(self):
        """ 
            Name of the sequence generator
            
        """
        raise NotImplementedError("Method not implemented.")

    def _get_random_negative_batch(self):
        """
            Get chrom positions for the negative samples using
            uniform random sampling from across the all chromosomes
            in self._chroms
            
            Returns:
                pandas.DataFrame: 
                    two column dataframe of chromosome positions with
                    'chrom' & 'pos' columns

        """

        # Step 1: select chromosomes, using sampling weights 
        # according to sizes
        chrom_df = self._chrom_sizes_df.sample(
            n=int(self._batch_size * self._negative_sampling_rate),
            weights=self._chrom_sizes_df.weights, replace=True)

        # Step 2: generate 'n' random numbers where 'n' is the length
        # of chrom_df 
        r = [random.random() for _ in range(chrom_df.shape[0])]

        # Step 3. multiply the random numbers with the size column.
        # Additionally, factor in the flank size and jitter while 
        # computing the position
        chrom_df['pos'] = (
            (chrom_df['size'] 
             - ((self._input_flank + self._max_jitter) * 2)) * r 
            + self._input_flank + self._max_jitter).astype(int)

        return chrom_df[['chrom', 'pos']]

    def _proc_target(self, coords_df, mpq, proc_idx):
        """
            Function that will be executed in a separate process.
            Takes a dataframe of peak coordinates and parses them in 
            batches, to get one hot encoded sequences and corresponding
            outputs, and adds the batches to the multiprocessing queue.
            Optionally, samples negative locations and adds them to 
            each batch
            
            Args:
                coords_df (pandas.DataFrame): dataframe containing
                    the chrom & peak pos
                
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to hold the batches
        """
        
        # divide the coordinates dataframe into batches
        cnt = 0
        for i in range(0, coords_df.shape[0], self._batch_size):   
            # we need to make sure we dont try to fetch 
            # data beyond the length of the dataframe
            if (i + self._batch_size) > coords_df.shape[0]:
                break
                
            batch_df = coords_df.iloc[i:i + self._batch_size]
            batch_df = batch_df.copy()
            
            # positive samples
            batch_df['status'] = 1
            
            # there are implicit negative samples (possibly gc matched)
            # whose weights are zero, we set the status of those to -1
            batch_df.loc[batch_df['weight'] == 0, 'status'] = -1
            
            # index into the original loci .bed file
            batch_df['idx'] = batch_df.index.values
            
            # add random negative samples
            if self._mode == "train" and self._negative_sampling_rate > 0.0:
                    
                neg_batch = self._get_random_negative_batch()
                
                # negative sample
                neg_batch['status'] = -1
                
                # since these samples are not in the original loci
                neg_batch['idx'] = -1
                
                batch_df = pd.concat([batch_df, neg_batch])
            
            # generate a batch of one hot encoded sequences and 
            # corresponding outputs
            batch = self._generate_batch(batch_df)
            
            # add batch to the multiprocessing queue
            mpq.put(batch)
    
            cnt += 1
        
        logging.debug(
            "{} process {} put {} batches into mpq".format(
                self._mode, proc_idx, cnt))
            
    def _stealer(self, mpq, q, num_batches, thread_id):
        """
            Thread target function to "get" (steal) from the
            multiprocessing queue and "put" in the regular queue

            Args:
                mpq (multiprocessing.Queue): The multiprocessing queue
                    to steal from
                
                q (Queue): The regular queue to put the batch into
                
                num_batches (int): the number of batches to "steal"
                    from the mp queue
                
                thread_id (int): thread id for debugging purposes

        """
        for i in range(num_batches):            
            q.put(mpq.get())

        logging.debug(
            "{} stealer thread {} got {} batches from mpq".format(
                self._mode, thread_id, num_batches))

    def _epoch_run(self, data):
        """
            Manage batch generation processes & threads
            for one epoch

            Args:
                data (pandas.DataFrame): dataframe with 'chrom' &
                    'pos' columns
        """
        
        # list of processes that are spawned
        procs = []     
        
        # list of multiprocessing queues corresponding to each 
        # process
        mp_queues = [] 

        # list of stealer threads (that steal the items out of 
        # the mp queues)
        threads = []   
                       
        # the regular queue
        q = Queue()    

        # to make sure we dont flood the user with warning messages
        warning_dispatched = False
        
        # number of data samples to assign to each processor
        # (since we have already resized loci, len(data) is directly
        # divisible by num_threads)
        samples_per_process = int(len(data) / self._num_threads)

        # batches that will be generated by each process thread
        num_batches = []
        
        # spawn processes that will generate batches of data and "put"
        # into the multiprocessing queues
        for i in range(self._num_threads):
            mpq = mp.Queue()

            # give each process a  copy of the slice of the dataframe 
            # of positives
            df = data[
                i * samples_per_process: 
                (i + 1) * samples_per_process][['chrom', 'pos',
                                                'weight']].copy()
                
            num_batches.append(len(df) // self._batch_size)
            
            if df.shape[0] != 0:
                logging.debug(
                    "{} spawning process {}, df size {} "
                    "sum(num_batches) {}".format(
                        self._mode, i, df.shape, sum(num_batches)))

                # spawn and start the batch generation process 
                p = mp.Process(target=self._proc_target, args=[df, mpq, i])
                p.start()
                procs.append(p)
                mp_queues.append(mpq)
                
            else:
                if not warning_dispatched:
                    logging.warn(
                        "One or more process threads are not being assigned "
                        "data for parallel batch generation. You should "
                        "reduce the number of threads using the --threads "
                        "option for better performance. Inspect logs for "
                        "batch assignments.")
                    warning_dispatched = True
                
                logging.debug(
                    "{} skipping process {}, df size {}, num_batches "
                    "{}".format(self._mode, i, df.shape, sum(num_batches)))
                
                procs.append(None)
                mp_queues.append(None)

        logging.debug(
            "{} num_batches list {}".format(self._mode, num_batches))
                
        # the threads that will "get" from mp queues 
        # and put into the regular queue
        # this speeds up yielding of batches, because "get"
        # from mp queue is very slow
        for i in range(self._num_threads):
            # start a stealer thread only if data was assigned to
            # the i-th  process
            if num_batches[i] > 0:
                
                logging.debug(
                    "{} starting stealer thread {} [{}] ".format(
                        self._mode, i, num_batches[i]))
                
                mp_q = mp_queues[i]
                
                stealerThread = Thread(
                    target=self._stealer, args=[mp_q, q, num_batches[i], i])
                stealerThread.start()
                threads.append(stealerThread)
            else:
                threads.append(None)
                
                logging.debug(
                    "{} skipping stealer thread {} ".format(
                        self._mode, i, num_batches))

        return procs, threads, q, sum(num_batches)

    def gen(self):
        """
            Generator function to yield one epoch of data

        """
        
        if self._shuffle:
            # shuffle at the beginning of each epoch
            data = self._resized_loci[self.curr_epoch].sample(frac=1.0)
            logging.debug("{} Shuffling complete".format(self._mode))
        else:
            data = self._resized_loci[self.curr_epoch]

        # spawn multiple processes to generate batches of data in
        # parallel for each epoch
        procs, threads, q, total_batches = self._epoch_run(data)

        # yield the correct number of batches for each epoch
        for j in range(total_batches):      
            batch = q.get()
            yield batch

        # wait for batch generation processes to finish once the
        # required number of batches have been yielded
        for j in range(self._num_threads):
            if procs[j] is not None:
                logging.debug(
                    "{} waiting to join process {}".format(self._mode, j))
                procs[j].join()

            if threads[j] is not None:
                logging.debug(
                    "{} waiting to join thread {}".format(self._mode, j))
                threads[j].join()

            logging.debug(
                "{} join complete for process {}".format(self._mode, j))

        logging.debug(
            "{} Finished join for epoch".format(self._mode))

        logging.debug("{} Ready for next epoch".format(self._mode))

        # increment epoch#
        self.curr_epoch += 1


class MBPNetSequenceGenerator(MSequenceGenerator):
    """ 
        Multi task batch data generation for training BPNet
        on high-throughput sequencing data of various
        geonmics assays
    
        Args:
            tasks_json (str): path to the json file containing task
                information, required keys - 'signal', 'loci', 'bias'
                
            batch_gen_params (dictionary): python dictionary with batch
                generation parameters. Contains the following keys - 
            
                *input_seq_len (int)*
                    length of input DNA sequence
                
                *output_len (int)*
                    length of output profile
                
                *max_jitter (int)*
                    maximum value for randomized jitter to offset the 
                    peaks from the exact center of the input
                
                *rev_comp_aug (boolean)*
                    enable reverse complement augmentation
                
                *negative_sampling_rate (float)*
                    the fraction of batch_size that determines how many 
                    negative samples are added to each batch
            
                *sampling_mode (str)*
                    the mode of sampling chromosome positions - one of
                    ['peaks', 'sequential', 'random', 'manual']. In 
                    'peaks' mode the data samples are fetched from the
                    peaks bed file specified in the json file 
                    input_config['data']. In 'manual' mode, the two 
                    column pandas dataframe containing the chromosome  
                    position information is passed to the 'samples' 
                    argument of the class
                
                *shuffle (boolean)*
                    specify whether input data is shuffled at the 
                    begininning of each epoch
                
                *mode (str)*
                    'train', 'val' or 'test'
                 
            reference_genome (str): the path to the reference genome 
                fasta file
                
            chrom_sizes (str): path to the chromosome sizes file
            
            chroms (str): the list of chromosomes that will be sampled
                for batch generation
                
            loci_indices (list): list of indices to filter rows from
                 the 'loci' peaks file
                 
            background_loci_indices (list): list of indices to filter rows from
                 the 'loci' background file
                
            num_threads (int): number of parallel threads for batch
                generation, default = 10
                
            batch_size (int): size of each generated batch of data, 
                default = 64
            
            epochs (int): the number of epochs for which data has to 
                 be generated
                 
             background_only (boolean): True, if batches are to be 
                 generated with background samples alone (i.e. in the 
                 case where we are training a background model)
                 
             foreground_weight (float): sample weight for foreground
                 samples
             
             background_weight (float): sample weight for background
                 samples
                        
    """

    def __init__(self, tasks_json, batch_gen_params, reference_genome, 
                 chrom_sizes, chroms=None, loci_indices=None,
                 background_loci_indices=None, num_threads=10, batch_size=64, 
                 epochs=100, background_only=False, foreground_weight=1, 
                 background_weight=0, set_bias_as_zero=False):
        
        # name of the generator class
        self.name = "BPNet"
        
        # call base class constructor
        super().__init__(tasks_json, batch_gen_params, reference_genome, 
                         chrom_sizes, chroms, loci_indices, 
                         background_loci_indices, num_threads, 
                         batch_size, epochs, background_only, 
                         foreground_weight, background_weight, set_bias_as_zero)
        
    def get_name(self):
        """ 
            Name of the sequence generator
            
            Returns:
                str: name of the sequence generator

        """
        return self.name

    def _generate_batch(self, coords):
        """
            Generate one batch of inputs and outputs for training BPNet
            
            For all coordinates in "coords" fetch sequences &
            one hot encode the sequences. Fetch corresponding
            signal and bias values (from bigwig files). 
            Package the one hot encoded sequences and the output
            values as a tuple.
            
            Args:
                coords (pandas.DataFrame): dataframe with 'chrom', 
                    'pos', 'weight' & 'status' columns specifying the 
                    chromosome, the coordinate and whether the loci is
                    a positive(1) or negative sample(-1)
                
            Returns:
                tuple: 
                    When 'mode' is 'train' or 'val' a batch tuple 
                    with one hot encoded sequences and corresponding 
                    outputs and when 'mode' is 'test' tuple of 
                    cordinates & the inputs
        """
        
        # create a new reverse complement column in the dataframe
        # and set it to 0 for all loci (0-not reverse complemented, 
        # 1-reverse complemented)
        coords['rev_comp'] = 0
        
        # if reverse complement is enabled double the 'coords'
        if self._rev_comp_aug:
            # make a copy of coords, but set the rev_comp flag to 1
            coords_copy = coords.copy()
            coords_copy['rev_comp'] = 1
            coords = pd.concat([coords, coords_copy])
        
        # Initialize arrays to hold expected profile predictions,
        # counts predictions and profile bias and counts bias inputs
        
        # profile predictions across all tasks 
        profile_predictions = np.zeros(
            (coords.shape[0], self._output_flank * 2, 
             self._total_signal_tracks), dtype=np.float32)
        
        # counts predictions across all tasks 
        logcounts_predictions = np.zeros(
            (coords.shape[0], self._total_signal_tracks), dtype=np.float32)

        # profile and counts bias inputs for each task separately
        profile_bias_input = {}
        counts_bias_input = {}
        for i in range(self._num_tasks):
            # the number of bias tracks for the ith tasks, which
            # includes smoothed versions
            num_bias_tracks = self._get_num_bias_tracks_for_task(
                self._tasks[i])

            if num_bias_tracks > 0:
                # profile bias input for the ith task
                profile_bias_input[i] = np.zeros(
                    (coords.shape[0], self._output_flank * 2, num_bias_tracks), 
                    dtype=np.float32)

                # counts bias input for the ith task
                counts_bias_input[i] = np.zeros(
                    (coords.shape[0], num_bias_tracks), dtype=np.float32)
 
        # Initialization done.

        # list of sequences in the batch, these will be one hot
        # encoded together as a single sequence after iterating
        # over the batch
        sequences = []  
        
        # list of chromosome start/end coordinates for the batch
        coordinates = []

        # list of jitter values for the batch coordinates
        jitters = []
        
        # list of index values for the batch coordinates
        idxs = coords['idx'].values
        
        # sample weights
        weights = coords['weight'].astype(float).values

        # open all the signal and bias bigwig files and store the  
        # file objects in a dictionary
        signal_files = {}
        bias_files = {}
        for task in self._tasks:
            signal_files[task] = []
            for signal_file in self._tasks[task]['signal']['source']:
                signal_files[task].append(pyBigWig.open(signal_file))

            bias_files[task] = []
            for bias_file in self._tasks[task]['bias']['source']:
                bias_files[task].append(pyBigWig.open(bias_file))
        
        # reference file to fetch sequences
        fasta_ref = pyfaidx.Fasta(self._reference)
                                          
        # iterate over the batch
        rowCnt = 0
        for _, row in coords.iterrows():
            # randomly set a jitter value to move the peak summit 
            # slightly away from the exact center (only for samples 
            # whose weight is not 0)
            jitter = 0
            if self._mode == "train" and self._max_jitter and \
                row['weight'] != 0:
                
                jitter = random.randint(-self._max_jitter, self._max_jitter)
            
            # record the jitter for this sample
            jitters.append(jitter)
                                          
            # Step 1. get the sequence 
            chrom = row['chrom']
            # we use self._input_flank here and not self._output_flank because
            # input_seq_len is different from output_len
            start = row['pos'] - self._input_flank + jitter
            end = row['pos'] + self._input_flank + jitter
            seq = fasta_ref[chrom][start:end].seq.upper()
            
            # collect all the sequences into a list
            sequences.append(seq)
            
            start = row['pos'] - self._output_flank + jitter
            end = row['pos'] + self._output_flank + jitter
            
            # record the start/end coordinates for this sample
            coordinates.append((chrom, start, end))
                                    
            # track profile tracks across all tasks
            profile_track_idx = 0

            # iterate over each task and read the signal and bias
            # values from the bigWig files
            for i in range(self._num_tasks):
                                          
                # Step 2. get the profile signal value
                for signal_file in signal_files[i]:
                    profile_predictions[rowCnt, :, profile_track_idx] = \
                        np.nan_to_num(signal_file.values(chrom, start, end))  
                        
                    profile_track_idx += 1
                    
                if not self._set_bias_as_zero:
                #skip setting the bias values. Initialization value of zero will be used
                    # Step 3. get the bias values
                    bias_track_idx = 0
                    for j in range(len(bias_files[i])):
                        bias_file = bias_files[i][j]
                        profile_bias_input[i][rowCnt, :, bias_track_idx] = \
                            np.nan_to_num(bias_file.values(chrom, start, end))

                        bias_track_idx += 1

                        # add the smoothed track if 'smoothing' has been
                        # specified
                        if self._tasks[i]['bias']['smoothing'][j] is not None:
                            # get the smoothing params
                            sigma = self._tasks[i]['bias']['smoothing'][j][0]
                            window_size = \
                                self._tasks[i]['bias']['smoothing'][j][1]

                            # the smoothed bias track will immediately 
                            # follow the original bias track in the last
                            # dimension
                            profile_bias_input[i][rowCnt, :, bias_track_idx] = \
                                utils.gaussian1D_smoothing(
                                    profile_bias_input[i][
                                        rowCnt, :, bias_track_idx - 1],
                                    sigma, window_size)

                            bias_track_idx += 1

            rowCnt += 1

        # Step 4. one hot encode all the sequences in the batch 
        if len(sequences) == profile_predictions.shape[0]:
            X = sequtils.one_hot_encode(sequences, self._input_flank * 2)
        else:
            raise NoTracebackException(
                "Unable to generate enough sequences for the batch")

        # we can now compute the log(sum) of the profiles and bias
        # profiles for the entire batch
        logcounts_predictions = np.log(
            np.sum(profile_predictions, axis=1) + 1)
        
        if not self._set_bias_as_zero:
            #skip setting the bias values. Initialization value of zero will be used
            for key in profile_bias_input:
                counts_bias_input[key] = np.log(
                    np.sum(profile_bias_input[key], axis=1) + 1)
        
        # inputs to train, val & test
        # 'coordinates', 'jitters', 'index', 'status' & 'rev_comp'
        # are not inputs to the model, so you will see a warning 
        # about unused inputs while training. It's safe to ignore
        # the warning. We pass 'coordinates' so we can track the exact
        # coordinates of the inputs (because jitter is random)
        # 'status' refers to whether the data sample is a +ve (1)
        # or -ve (-1) example and is used by the attribution
        # prior loss function        
        inputs = {
            'sequence': X, 
            'coordinates': np.array(coordinates)}

        # add profile bias input
        for key in profile_bias_input:
            _key = 'profile_bias_input_' + str(key)
            inputs[_key] = profile_bias_input[key]

        # add counts bias input
        for key in counts_bias_input:
            _key = 'counts_bias_input_' + str(key)
            inputs[_key] = counts_bias_input[key]
                                          
        # in 'train' mode we add some extras to track all the
        # loci used in every batch of training
        if self._mode == 'train':
            inputs['jitters'] = np.array(jitters),
            inputs['index'] = np.array(idxs),
            inputs['status'] = coords['status'].values,
            inputs['rev_comp'] = coords['rev_comp'].values
        
        # in 'train' and 'val' mode we need outputs as well     
        if self._mode == 'train' or self._mode == 'val':
            outputs = {
                'profile_predictions': profile_predictions,
                'logcounts_predictions': logcounts_predictions}

            return (inputs, outputs, weights)

        # in 'test' mode we only return inputs
        elif self._mode == 'test':
            # we add the true profiles & counts so we can use those to
            # compute metrics
            inputs['true_profiles'] = profile_predictions
            inputs['true_logcounts'] = logcounts_predictions
            return inputs


def list_generator_names():
    """
       List all available sequence generators that are derived
       classes of the base class MSequenceGenerator
       
       Returns:
           list: list of sequence generator names
    """
    
    generator_names = []
    for c in MSequenceGenerator.__subclasses__():        
        result = re.search('M(.*)SequenceGenerator', c.__name__)
        generator_names.append(result.group(1))

    return generator_names


def find_generator_by_name(generator_name):
    """
        Get the sequence generator class name given its name
        
        Returns:
            str: sequence generator class name
    """
    
    for c in MSequenceGenerator.__subclasses__():
        result = re.search('M(.*)SequenceGenerator', c.__name__)
        if generator_name == result.group(1):
            return c.__name__