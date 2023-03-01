"""
    This module contains the main script and the supporting functions
    to compute embeddings of a BPNet model given a list of chromosome
    positions
    
    Functions:
    
        get_sequences: Get one hot encoded sequences for specified  
            chromosome positions
        
        dataframe_batcher: Batch a dataframe into "batch_size" chunks
        
        compute_embeddings: Compute embeddings for specified chromosome
            positions and save as compressed numpy array
            
        find_input_layer: Find a matching input layer given a name and 
            shape

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

"""


from bpnet.cli.argparsers import embeddings_argsparser
from bpnet.utils.exceptionhandler import NoTracebackException
from bpnet.utils.logger import *
from bpnet.model.custommodel \
    import CustomModel
from bpnet.generators.sequtils import one_hot_encode

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.layers import Flatten, Cropping1D, Lambda, Reshape
from tqdm import tqdm 

import tensorflow.keras.backend as K
import h5py
import numpy as np
import os
import pandas as pd
import pysam
import tensorflow as tf

def get_sequences(chrom_positions, reference_genome_file, seq_len):
    """
        Get one hot encoded sequences for specified chromosome 
        positions
        
        Args:
            chrom_positions (pandas.Dataframe): two column dataframe 
                with 'chrom' and 'pos' columns
            reference_genome_file (str): path to the reference genome 
                fasta file
            seq_len (int): sequence length            
            
       Returns:
           numpy.ndarray: 
               3-dimension numpy array with shape 
               (len(chrom_positions), input_seq_len, 4)
    
    """
    
    # reference file to fetch sequences
    fasta_ref = pysam.FastaFile(reference_genome_file)
        
    # list to hold all dna sequences 
    sequences = []
    
    # iterate over the chromosome positions and fetch the dna sequences
    for idx, row in chrom_positions.iterrows():
        
        start = row['pos'] - seq_len // 2
        end = start + seq_len
        
        sequence = fasta_ref.fetch(row['chrom'], start, end).upper()
        
        sequences.append(sequence)
                        
    # one hot encode all the sequences in the batch 
    return one_hot_encode(sequences, seq_len)

    
def dataframe_batcher(df, batch_size):
    """
        Batch a dataframe into "batch_size" chunks
        
        Args:
            df (padas.Dataframe): pandas dataframe
            batch_size (int): the size of each chunk to divide the
                dataframe into
                
        Returns:
            generator: tuples containing dataframe chunk, and start &
                end indices of the chunk (df_chunk, start_idx, end_idx)
    """
    
    return ((df[pos:pos + batch_size], pos, pos + batch_size)
            for pos in range(0, len(df), batch_size))

    
def compute_embeddings(model, referece_genome_file, input_seq_len, 
                       embeddings_output_shape, chrom_positions, batch_size, 
                       output_filename, embeddings_layer_flattened):
    """
        Compute embeddings for specified chromosome positions and save
        as compressed numpy array
       
        Args:
            model (keras.models.Model): keras model to compute 
                embeddings
            referece_genome_file (str): path to the reference genome fasta 
                file
            input_seq_len (int): input sequence length
            embeddings_output_shape (list): shape of the embeddings 
                output of the model
            chrom_positions (pandas.Dataframe): two column dataframe 
                with 'chrom' and 'pos' columns
            batch_size (int): the size of batches to divide and process
                the chromosome positions
            output_filename (str): the name of the output file with
                '.npz' extension
            embeddings_layer_flattened (boolean): True if the final 
                layer of the embeddings models has been flattened
                 
    """
    
    # the shape of the embeddings output for all N examples
    embeddings_output_shape.insert(0, len(chrom_positions))

    # the shape of the embeddings output after aggregation along  
    # the sequence length dimension (2)
    embeddings_aggreagation_shape = (embeddings_output_shape[0], 
                                     embeddings_output_shape[1],
                                     embeddings_output_shape[3])
    
    # we'll write all the aggregated (mean, min, max & standard
    # deviation) outputs to a hdf5 file
    # open h5py file for writing
    h5_file = h5py.File(output_filename, "w")
   
    # create the coords group
    coord_group = h5_file.create_group("coords")
    num_examples = len(chrom_positions)
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", (num_examples,),
        dtype=h5py.string_dtype(encoding="ascii"), compression="gzip")
    coords_start_dset = coord_group.create_dataset(
        "coords_start", (num_examples,), dtype=int, compression="gzip")
    coords_end_dset = coord_group.create_dataset(
        "coords_end", (num_examples,), dtype=int, compression="gzip")
    
    # create the embeddings group
    emb_group = h5_file.create_group("embeddings")
    embeddings_mean = emb_group.create_dataset(
        "mean", embeddings_aggreagation_shape, compression="gzip")
    embeddings_std = emb_group.create_dataset(
        "std", embeddings_aggreagation_shape, compression="gzip")
    embeddings_max = emb_group.create_dataset(
        "max", embeddings_aggreagation_shape, compression="gzip")
    embeddings_min = emb_group.create_dataset(
        "min", embeddings_aggreagation_shape, compression="gzip")
   
    # the lists of chroms, starts & ends
    coords_chrom_dset = chrom_positions['chrom'].values
    coords_start_dset = (chrom_positions['pos'] - input_seq_len // 2).values
    coords_end_dset = (chrom_positions['pos'] + input_seq_len // 2).values

    # batch the chromosome positions dataframe and process one batch
    # at a time
    num_batches = (len(chrom_positions) // batch_size) + 1
    for batch, start, end in tqdm(
        dataframe_batcher(chrom_positions, batch_size), desc="batch", 
        total=num_batches):
        
        # get the one hot encoded sequences for the batch
        sequences = get_sequences(batch, referece_genome_file, input_seq_len)
        
        # compute the predictions
        predictions =  model.predict(sequences)

        # assign mean values for the batch
        embeddings_mean[start:end, ...] = np.mean(predictions, axis=2)
        
        # assign min values for the batch
        embeddings_min[start:end, ...] = np.min(predictions, axis=2)
        
        # assign max values for the batch
        embeddings_max[start:end, ...] = np.max(predictions, axis=2)
        
        # assign max values for the batch
        embeddings_std[start:end, ...] = np.std(predictions, axis=2)
 
    h5_file.close()
    

def find_input_layer(model, input_layer_name, input_layer_shape):
    """ 
        Find a matching input layer given a name and shape

        Args:
            model (keras.models.Model): original keras model
            input_layer_name (str): name of the input layer
            input_layer_shape (list): shape of the input layer
            
        Returns:
            int: if found, index of the matching input layer, -1 
                 otherwise
                 
    """
    
    idx = 0
    # iterate through all the inputs to the model
    for inp in model.input:
        # match a substring in the name of the input layer and 
        # match the shapes
        if inp.name.find(input_layer_name) == 0 and \
            inp.shape.as_list() == input_layer_shape:
            # match found
            return idx
        idx += 1
    
    # not match found
    return -1

    
def embeddings_main():
    """
        Main function to compute embeddings
    """
    
    # parse the command line arguments
    parser = embeddings_argsparser()
    args = parser.parse_args()

    # check if the model file exists
    if not os.path.exists(args.model):
        raise NoTracebackException(
            "Model {} does not exist".format(args.model))

    # check if the peaks file exists
    if not os.path.exists(args.peaks):
        raise NoTracebackException(
            "peaks file {} does not exist".format(args.peaks))

    # check if the output directory exists
    if not os.path.exists(args.output_directory):
        raise NoTracebackException(
            "Directory {} does not exist".format(args.output_directory))
    
    if (args.embeddings_layer_name is not None) and \
        (args.numbered_embeddings_layers_prefix is not None):
        raise NoTracebackException(
            "Only one of [--embeddings-layer-name, "
            "--numbered-embeddings-layers-prefix] can be used")
    
    if (args.embeddings_layer_name is None) and \
        (args.numbered_embeddings_layers_prefix is None):
        raise NoTracebackException(
            "One of [--embeddings-layer-name, "
            "--numbered-embeddings-layers-prefix] must be used")
        
    # filename to write debug logs
    logfname = "{}/embeddings.log".format(args.output_directory)
    
    # set up the loggers
    init_logger(logfname)

    with CustomObjectScope({'tf': tf,  
                            'CustomModel': CustomModel}):
        
        # load the model 
        model = load_model(args.model)
        logging.info("loaded model {}".format(args.model))
        
        # load the chromosome positions as a pandas dataframe
        peaks_df = pd.read_csv(args.peaks, sep='\t', header=None, 
                               names=['chrom', 'st', 'e', 'name', 'score',
                                      'strand', 'signal', 'p', 'q', 'summit'])

        # create new column for peak position
        peaks_df['pos'] = peaks_df['st'] + peaks_df['summit']

        # create a new dataframe with just 2 columns 'chrom' & 'pos'
        chrom_positions = peaks_df[['chrom', 'pos']]
        
        logging.info("Total embeddings to be computed - {}".format(
            len(chrom_positions)))
        
        # infer input seq len from input layer shape
        input_layer_shape = args.input_layer_shape
        input_seq_len = input_layer_shape[0]
        
        # add the "batch" dimension to the input layer shape so we can
        # match against the model's input layer
        input_layer_shape.insert(0, None)
        
        # find the required input layer from among all possible inputs
        # to the model and get the corresponnding index
        input_idx = find_input_layer(model, args.input_layer_name, 
                                     input_layer_shape)
        
        if input_idx == -1:
            raise NoTracebackException.NoTracebackException(
                "No match found for input layer {} with shape {}".format(
                    args.input_layer_name. args.input_layer_shape))
        else:
            # construct a new model that's a branch of the original 
            # model that computes just the embeddings given an input
            # dna sequence 
            
            # lists to hold layer names and the corresponding layers
            # from the model
            layer_names = []
            layers = []
            
            # if only a single layer is requested
            if args.embeddings_layer_name is not None:
                layer_names.append(args.embeddings_layer_name)
            
            # if many layers with same prefix are requested
            elif args.numbered_embeddings_layers_prefix is not None:
                for i in range(args.num_numbered_embeddings_layers):
                    layer_name = '{}_{:d}'.format(
                        args.numbered_embeddings_layers_prefix, i + 1)
                    layer_names.append(layer_name)
                
            
            # iterate over layer names and fetch the model layer
            for layer_name in layer_names:
                try:
                    layer = model.get_layer(layer_name).output
                        
                    # crop the layer to required size
                    if args.cropped_size is not None:
                        crop_size = (layer.shape.as_list()[1] - \
                           args.cropped_size) // 2
                        layer = Cropping1D(crop_size)(layer)
                                            
                    layers.append(layer)
                except ValueError:
                    raise NoTracebackException.NoTracebackException(
                        "No match found for {}".format(layer_name))

            if len(layers) == 1:
                # we'll reshape the output to create a dimension
                # at axis = 1, the stack operation in the else 
                # block doesn't seem to like lists of one element
                
                # get the existing shape of the layer output
                old_shape = layers[0].shape.as_list()
                
                # replace the None at dimension 0 (batch) to 1
                # the Reshape will add a new None dimension
                new_shape = old_shape[:]
                new_shape[0] = 1

                # reshape the output to mimic the stack operation
                embeddings_output = Reshape(new_shape)(layers[0])
            else:                
                
                # the final output is a vertical stacking of 
                # all embeddings layers
                embeddings_output = Lambda(
                    lambda x: K.stack(x, axis=1)
                )(list(layers))
                                        
            # create the emneddings model
            embeddings_model = Model(inputs=model.input[input_idx], 
                                     outputs=embeddings_output)
                        
            # get embeddings output shape, omit the batch dimension
            embeddings_output_shape = embeddings_output.shape.as_list()[1:]

            # output compressed numpy file containing the embeddings
            output_filename = os.path.join(args.output_directory, 
                                           args.output_filename)

            # compute the embeddings
            compute_embeddings(
                embeddings_model, args.reference_genome, input_seq_len, 
                embeddings_output_shape, chrom_positions, args.batch_size, 
                output_filename, args.flatten_embeddings_layer)

if __name__ == '__main__':
    embeddings_main()
