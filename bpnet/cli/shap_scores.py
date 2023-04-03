import h5py
import json
import numpy as np
import pandas as pd
import pyBigWig
import pysam
import shap
import tensorflow as tf

from bpnet.cli.argparsers import shap_scores_argsparser
from bpnet.utils.datetime import *
from bpnet.utils.exceptionhandler import NoTracebackException
from bpnet.utils.shaputils import *
from bpnet.utils.logger import *
from bpnet.generators.sequtils import one_hot_encode
from bpnet.utils.misc import gaussian1D_smoothing
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

from bpnet.model.custommodel \
    import CustomModel

import hdf5plugin



def save_scores(peaks_df, one_hot_sequences, hyp_shap_scores, output_fname):
    """
        Function to save shap scores to HDF5 file
        
        Args:
            peaks_df (pandas.Dataframe): a pandas dataframe that
                has 'chrom', 'start', & 'end' columns for chrom
                positions
                
            one_hot_sequences (numpy.ndarray): numpy array of shape
                N x sequence_length x 4

            hyp_shap_scores (numpy.ndarray): shap scores corresponding
                to the input sequences (hypothetical contributions);
                numpy array of shape N x sequence_length x 4
                
            output_fname (str): path to the output .h5 file
            
    """
    
    # get the chroms, starts and ends as lists
    coords_chrom = peaks_df['chrom'].values
    coords_start = peaks_df['start'].values
    coords_end = peaks_df['end'].values
    
    num_examples = peaks_df.shape[0]
    seq_len = one_hot_sequences.shape[1]
    
    # open the HDF% file for writing
    f = h5py.File(output_fname, "w")

    coords_chrom_dset = f.create_dataset(
        "coords_chrom", (num_examples,),
        dtype=h5py.string_dtype(encoding="ascii") 
    )
    coords_chrom_dset[:] = coords_chrom.astype('U8')
    
    coords_start_dset = f.create_dataset(
        "coords_start", (num_examples,), dtype="i4", 
        **hdf5plugin.Blosc()
    )
    coords_start_dset[:] = coords_start
    
    coords_end_dset = f.create_dataset(
        "coords_end", (num_examples,), dtype="i4", 
        **hdf5plugin.Blosc()
    )
    coords_end_dset[:] = coords_end
        
    hyp_scores_dset = f.create_dataset(
        "hyp_scores", (num_examples, seq_len, 4), dtype="f2",
        **hdf5plugin.Blosc()
    )
    hyp_scores_dset[:, :, :] = hyp_shap_scores.astype(np.float16)

    input_seqs_dset = f.create_dataset(
        "input_seqs", (num_examples, seq_len, 4), dtype="i1",
        **hdf5plugin.Blosc()
    )
    input_seqs_dset[:, :, :] = one_hot_sequences.astype(np.int8)
    
    f.close()
    
    

def shap_scores(args, shap_dir):
    # load the model
    model = load_model(args.model, compile=False)
    
    # read all the peaks into a pandas dataframe
    peaks_df = pd.read_csv(args.bed_file, sep='\t', header=None, 
                           names=['chrom', 'st', 'stop', 'name', 'score',
                                  'strand', 'signalValue', 'p', 'q', 'summit'])

    if args.chroms is not None:
        # keep only those rows corresponding to the required 
        # chromosomes
        peaks_df = peaks_df[peaks_df['chrom'].isin(args.chroms)]
           
    if args.sample is not None:
        # randomly sample rows
        logging.info("Sampling {} rows from {}".format(
            args.sample, args.bed_file))
        peaks_df = peaks_df.sample(n=args.sample, random_state=args.seed)
    
    if args.presort_bed_file:
        # sort the bed file in descending order of peak strength
        peaks_df = peaks_df.sort_values(['signalValue'], ascending=False)
    
    # reset index (if any of the above 3 filters have been applied, 
    # no harm if they haven't)
    peaks_df = peaks_df.reset_index(drop=True)
    
    # add new columns for start and stop based on 'summit' position
    peaks_df['start'] = peaks_df['st'] + peaks_df['summit'] - \
        (args.input_seq_len // 2)
    peaks_df['end'] = peaks_df['st'] + peaks_df['summit'] + \
        (args.input_seq_len // 2)
        
    # get final number of peaks
    num_peaks = peaks_df.shape[0]
    logging.info("#Peaks - {}".format(num_peaks))
    
    # reference file to fetch sequences
    logging.info("Opening reference file ...")
    fasta_ref = pysam.FastaFile(args.reference_genome)
        
    # if controls have been specified we to need open the control files
    # for reading
    control_bigWigs = []
    smoothing = []
    # load the control info json file
    with open(args.input_data, 'r') as inp_json:
        try:
            tasks = json.loads(inp_json.read())
            tasks = {int(k): v for k, v in tasks.items()}
        except json.decoder.JSONDecodeError:
            raise NoTracebackException(
                "Unable to load json file {}. Valid json expected. "
                "Check the file for syntax errors.".format(
                    args.input_data))

    # get the control bigWig for each task
    for task in tasks:
        if task == args.task_id:
            _control_bigWigs = tasks[task]['bias']['source']
            _smoothing = tasks[task]['bias']['smoothing'][:]
            
            if len(_control_bigWigs) > 0:
                logging.info("Opening control bigWigs ...")
    
            for control_bigWig_path in _control_bigWigs:

                # check if the file exists
                if not os.path.exists(control_bigWig_path):
                    raise NoTracebackException(
                        "File {} does not exist".format(
                            control_bigWig_path))

                logging.info(control_bigWig_path)

                # open the bigWig and add the file object to the 
                # list
                control_bigWigs.append(pyBigWig.open(control_bigWig_path))

            # add smoothing params if the input json has them
            for smoothing_val in _smoothing:
                if smoothing_val is not None:
                    smoothing.append(smoothing_val)
                    
            
    bias_counts_input = None
    bias_profile_input = None
    # if bias is part of the model then model inputs will be a list
    if isinstance(model.input, list):
        # log of sum of counts of the control track
        # if multiple control files are specified this would be
        # log(sum(position_wise_sum_from_all_files))
        bias_counts_input = np.zeros((num_peaks, 
                                      len(control_bigWigs) + len(smoothing)))

        # the control profile and the smoothed version of the control 
        bias_profile_input = np.zeros((num_peaks, args.control_len, 
                                       len(control_bigWigs) + len(smoothing)))
    
    ## IF NO CONTROL BIGWIGS ARE SPECIFIED THEN THE TWO NUMPY ARRAYS
    ## bias_counts_input AND bias_profile_input WILL REMAIN ZEROS
    
    # list to hold all the sequences for the peaks
    sequences = []
    
    # iterate through all the peaks
    for idx, row in peaks_df.iterrows():
        start = row['start']
        end = row['end']
        
        # fetch the reference sequence at the peak location
        try:
            seq = fasta_ref.fetch(row['chrom'], start, end).upper()        
        except ValueError: # start/end out of range
            logging.warn("Unable to fetch reference sequence at peak: "
                         "{} {}-{}.".format(row['chrom'], start, end))
            
            # use string of N's as a substitute
            seq = 'N'*args.input_seq_len
            
        # check if we have the required length
        if len(seq) != args.input_seq_len:
            logging.warn("Reference genome doesn't have required sequence " 
                         "length ({}) at peak: {} {}-{}. Returned length {}. "
                         "Using all N's.".format(
                             args.input_seq_len, row['chrom'], start, end, 
                             len(seq)))
            
            # use string of N's as a substitute
            seq = 'N'*args.input_seq_len     
        
        # fetch control values
        if bias_counts_input is not None and bias_profile_input is not None:
            if len(control_bigWigs) > 0:
                # a different start and end for controls since control_len
                # is usually not the same as input_seq_len
                start = row['st'] + row['summit'] - (args.control_len // 2)
                end =  row['st'] + row['summit'] + (args.control_len // 2)

                # read the values from the control bigWigs
                for i in range(len(control_bigWigs)):
                    try:
                        vals = np.nan_to_num(
                            control_bigWigs[i].values(row['chrom'], start, end))
                    except RuntimeError:
                        print("Invalid interval bounds", row['chrom'], start, end)
                        continue
                    bias_counts_input[idx, i] = np.log(np.sum(vals) + 1)
                    bias_profile_input[idx, :, i] = vals


                # compute the smoothed control profile
                start_idx = len(control_bigWigs)
                for i in range(len(smoothing)):
                    sigma = float(smoothing[i][0])
                    window_width = int(smoothing[i][1])
                    bias_profile_input[idx, :, start_idx + i] = \
                        gaussian1D_smoothing(
                            bias_profile_input[idx, :, i], sigma, window_width)

        # append to the list of sequences
        sequences.append(seq)

    # if null distribution is requested
    null_sequences = []
    if args.gen_null_dist:
        logging.info("generating null sequences ...")
        rng = np.random.RandomState(args.seed)
        
        # iterate over sequences and get the dinucleotide shuffled
        # sequence for each of them
        for seq in sequences:
            # get a list of shuffled seqs. Since we are setting
            # num_shufs to 1, the returned list will be of size 1
            shuffled_seqs = dinuc_shuffle(seq, 1, rng)
            null_sequences.append(shuffled_seqs[0])
        
        # null sequences are now our actual sequences
        sequences = null_sequences[:]

    # one hot encode all the sequences
    X = one_hot_encode(sequences, args.input_seq_len)
    print("X shape", X.shape)
        
    # inline function to handle dinucleotide shuffling
    def data_func(model_inputs):
        rng = np.random.RandomState(args.seed)
        dinucs =  [dinuc_shuffle(model_inputs[0], args.num_shuffles, rng)] + \
        [
            np.tile(
                np.zeros_like(model_inputs[i]),
                (args.num_shuffles,) + (len(model_inputs[i].shape) * (1,))
            ) for i in range(1, len(model_inputs))
        ]
        
        return dinucs
    
    print('bias_counts_input:',bias_counts_input)
    print('bias_profile_input:',bias_profile_input)
    if bias_counts_input is None and bias_profile_input is None:
        counts_explainer_inputs = [model.input]
        profile_explainer_inputs =[model.input]
        
        counts_shap_inputs = [X]
        profile_shap_inputs = [X]
    else:
        counts_explainer_inputs = [model.input[0], model.input[2]]
        profile_explainer_inputs = [model.input[0], model.input[1]]
        
        counts_shap_inputs = [X, bias_counts_input]
        profile_shap_inputs = [X, bias_profile_input]

    profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
        (counts_explainer_inputs, 
         tf.reduce_sum(model.outputs[1], axis=-1)),
        data_func, 
        combine_mult_and_diffref=combine_mult_and_diffref)

    # explainer for the profile head
    weightedsum_meannormed_logits = get_weightedsum_meannormed_logits(
        model, task_id=args.task_id, stranded=True, orig_multi_loss=args.orig_multi_loss)
    
    profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
        (profile_explainer_inputs, weightedsum_meannormed_logits),
        data_func, 
        combine_mult_and_diffref=combine_mult_and_diffref)
    


    logging.info("Generating 'counts' shap scores")
    counts_shap_scores = profile_model_counts_explainer.shap_values(
        counts_shap_inputs, progress_message=100)
    
    # save the dictionary in HDF5 formnat
    logging.info("Saving 'counts' scores")
    output_fname = '{}/counts_scores.h5'.format(shap_dir)
    
    # save the hyp shap scores, one hot sequences & chrom positions
    # to a HDF5 file
    save_scores(peaks_df, X, counts_shap_scores[0], output_fname)
    
    logging.info("Generating 'profile' shap scores")
    profile_shap_scores = profile_model_profile_explainer.shap_values(
        profile_shap_inputs, progress_message=100)
    
    # save the dictionary in HDF5 formnat
    logging.info("Saving 'profile' scores")
    output_fname = '{}/profile_scores.h5'.format(shap_dir)

    # save the profile hyp shap scores, one hot sequences & chrom 
    # positions to a HDF5 file
    save_scores(peaks_df, X, profile_shap_scores[0], output_fname)
    
    # save the dataframe as a new .bed file 
    peaks_df.to_csv('{}/peaks_valid_scores.bed'.format(shap_dir), 
                           sep='\t', header=False, index=False)
    
    # write all the command line arguments to a json file
    config_file = '{}/config.json'.format(shap_dir)
    with open(config_file, 'w') as fp:
        config = vars(args)
        json.dump(config, fp)
        
        
def shap_scores_main():
    # disable eager execution so shap deep explainer wont break
    tf.compat.v1.disable_eager_execution()
    
    # parse the command line arguments
    parser = shap_scores_argsparser()
    args = parser.parse_args()
    
    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        raise NoTracebackException(
            "Directory {} does not exist".format(args.output_directory))

    # check if the output directory is a directory path
    if not os.path.isdir(args.output_directory):
        raise NoTracebackException(
            "{} is not a directory".format(args.output_directory))
    
    # check if the reference genome file exists
    if not os.path.exists(args.reference_genome):
        raise NoTracebackException(
            "File {} does not exist".format(args.reference_genome))

    # check if the model file exists
    if not os.path.exists(args.model):
        raise NoTracebackException(
            "File {} does not exist".format(args.model))

    # check if the bed file exists
    if not os.path.exists(args.bed_file):
        raise NoTracebackException(
            "File {} does not exist".format(args.bed_file))
    
    # if controls are specified check if the control_info json exists
    if args.input_data is not None:
        if not os.path.exists(args.input_data):
            raise NoTracebackException(
                "Input data file {} does not exist".format(args.control_info))
            
    # check if both args.chroms and args.sample are specified, only
    # one of the two is allowed
    if args.chroms is not None and args.sample is not None:
        raise NoTracebackException(
            "Only one of [--chroms, --sample]  is allowed")
            
    if args.automate_filenames:
        # create a new directory using current date/time to store the
        # shapation scores
        date_time_str = local_datetime_str(args.time_zone)
        shap_scores_dir = '{}/{}'.format(args.output_directory, date_time_str)
        os.mkdir(shap_scores_dir)
    else:
        shap_scores_dir = args.output_directory    

    # filename to write debug logs
    logfname = "{}/shap_scores.log".format(shap_scores_dir)
    
    # set up the loggers
    init_logger(logfname)
    
    # shap
    logging.info("Loading {}".format(args.model))
    with CustomObjectScope({'tf': tf,
                            'CustomModel': CustomModel}):
        shap_scores(args, shap_scores_dir)

if __name__ == '__main__':
    shap_scores_main()
