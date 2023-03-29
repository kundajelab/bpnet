"""
    This module containins training functions that are common to
    the CLI & the API
    
    Functions:
    
        train_and_validate: Train and validate on a single train and
            validation set
        
        train_and_validate_ksplits: Train and validate on one or 
            more train/val splits specified via a json file
        
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

# set random seed
from numpy.random import seed
seed(1234)
from tensorflow.random import set_seed 
set_seed(1234)

import copy
import json
import logging
import multiprocessing as mp
import os
import pandas as pd
import time

from bpnet.utils.datetime import *
from bpnet.utils.exceptionhandler import NoTracebackException
from bpnet.utils import logger
from bpnet.model import arch
from tensorflow.keras.optimizers import Adam
from bpnet.generators import generators 


def early_stopping_check(losses, patience=5, min_delta=1e-3):
    """
        Function to check if early stopping criteria are met
        
        Args:
            losses (list): list of all losses in order of the epochs,
                these could be training or validation losses
            patience (int): the number of epochs with no improvement
                greater than `min_delta`
            min_delta (float): The smallest amount that signals  
                sufficienct decrease in validation loss to justify
                continuation of training for a further #patience
                epochs
                
        Returns:
            bool: True, if early stopping criteria are satisfied, 
                False otherwise
    """
    
    # if sufficient number of epochs have not elapsed yet
    if len(losses) <= patience:
        return False
    
    # the loss value upon which the patience check will be performed
    anchor_loss = losses[-(patience+1)]
    
    for i in range(patience):
        if (anchor_loss - losses[i-patience]) > min_delta:
            return False
    
    return True

    
def reduce_lr_on_plateau(losses, current_lr, factor=0.5, patience=2, 
                         min_lr=1e-4):
    """
        Function to compute the new learning rate if loss is
        plateauing 
        
        Args:
            losses (list): list of all losses in order of the epochs,
                these could be training or validation losses
            current_lr (float): current learning rate
            factor (float): the factor by which the learning rate is
                to be reduced in case the plateau criteria is met
            patience (int): number of epochs with no improvement after 
                which learning rate will be reduced.
            min_lr (float): lower bound on the learning rate
                
        Returns:
            float: new learning rate
                
    """
    
    # if sufficient number of epochs have not elapsed yet 
    if len(losses) <= patience:
        return current_lr
    
    # the loss value upon which the patience check will be performed
    anchor_loss = losses[-(patience+1)]
    
    for i in range(patience):
        # improvement found
        if losses[i-patience] < anchor_loss:
            # no change in learning rate
            return current_lr
    
    # new learning rate
    new_lr = current_lr * factor
    
    # check if it's below lower bound
    if new_lr < min_lr:
        return current_lr
    
    return new_lr


def train_and_validate(
    input_data, model_arch_name, model_arch_params_json, output_params, 
    genome_params, batch_gen_params, hyper_params, parallelization_params, model_dir,
    train_chroms, val_chroms, train_indices=None, 
    val_indices=None, background_train_indices=None, 
    background_val_indices=None, bias_input_data=None, 
    mnll_loss_sample_weight=1.0, mnll_loss_background_sample_weight=0.0, 
    orig_multi_loss=False, suffix_tag=None):

    """
        Train and validate on a single train and validation set
        
        Note: the list & description for each of the required keys
            in all of the json parameter files passed to this 
            fucntion can be found here:
            http://
        
        Args:
            input_data (str): path to the tasks json file
            
            model_arch_name (str): name of the model definition 
                function in the model_archs module
                
            model_arch_params_json (str): path to json file containing
                model architecture params

            output_params (dict): dictionary containing output 
                parameters
            
            genome_params (dict): dictionary containing genome
                parameters
            
            batch_gen_params (dict): dictionary containing batch
                generation parameters
            
            hyper_params (dict): dictionary containing containing 
                training & validation hyper parameters
            
            parallelization_params (dict): dictionary containing
                parameters for parallelization options
                
            model_dir (str): the path to the output directory
            
            train_chroms (list): list of training chromosomes
            
            val_chroms (list): list of validation chromosomes
            
            train_indices (list): list of indices that index to 
                 training peaks from the signal peaks file
                 
            val_indices (list): list of indices that index to 
                 validation peaks from the singal peaks file
             
            background_train_indices (list): list of indices that index to 
                 training peaks from the background peaks file
                 
            background_val_indices (list): list of indices that index to 
                 validation peaks from the background peaks file
            
            bias_input_data (str): path to the bias tasks json file                        
                
            mnll_loss_sample_weight (float): weight for each (foreground)
                training sample for computing mnll loss

            mnll_loss_background_sample_weight (float): weight for each
                background sample for computing mnll loss
                
            orig_multi_loss (boolean): True if original multinomial loss function - one for
                each strand is to be used
                
            suffix_tag (str): optional tag to add as a suffix to files
                (model, log, history & config params files) created in
                the model directory
         
         Returns:
             keras.models.Model
             
    """
    
    # make sure the input_data json file exists
    if not os.path.isfile(input_data):
        raise NoTracebackException(
            "File not found: {} ".format(input_data))

    # load the json file
    with open(input_data, 'r') as inp_json:
        try:
            tasks = json.loads(inp_json.read())
            # since the json has keys as strings, we convert the 
            # top level keys to int so we can used them later for
            # indexing
            #: dictionary of tasks for training
            tasks = {int(k): v for k, v in tasks.items()}
        except json.decoder.JSONDecodeError:
            raise NoTracebackException(
                "Unable to load json file {}. Valid json expected. "
                "Check the file for syntax errors.".format(
                    input_data))
                
    # make sure the params json file exists
    if not os.path.isfile(model_arch_params_json):
        raise NoTracebackException(
            "File not found: {} ".format(model_arch_params_json))
            
    # load the params json file
    with open(model_arch_params_json, 'r') as inp_json:
        try:
            model_arch_params = json.loads(inp_json.read())
        except json.decoder.JSONDecodeError:
            raise NoTracebackException(
                "Unable to load json file {}. Valid json expected. "
                "Check the file for syntax errors.".format(
                    model_arch_params_json))

    if bias_input_data is not None:
        # load the bias json file
        with open(bias_input_data, 'r') as inp_json:
            try:
                bias_tasks = json.loads(inp_json.read())
                # since the json has keys as strings, we convert the 
                # top level keys to int so we can used them later for
                # indexing
                #: dictionary of tasks for training
                bias_tasks = {int(k): v for k, v in bias_tasks.items()}
            except json.decoder.JSONDecodeError:
                raise NoTracebackException(
                    "Unable to load json file {}. Valid json expected. "
                    "Check the file for syntax errors.".format(
                        bias_input_data))
                
    # filename to write debug logs
    if suffix_tag is not None:
        logfname = '{}/trainer_{}.log'.format(model_dir, suffix_tag)
    else:
        logfname = '{}/trainer.log'.format(model_dir)
        
    # we need to initialize the logger for each process
    logger.init_logger(logfname)
    
    # parameters that are specific to the training batch generation
    # process
    train_batch_gen_params = batch_gen_params
    train_batch_gen_params['mode'] = 'train'
    
    # parameters that are specific to the validation batch generation
    # process. For validation we dont use jitter, reverse complement 
    # augmentation
    val_batch_gen_params = copy.deepcopy(batch_gen_params)
    val_batch_gen_params['max_jitter'] = 0
    val_batch_gen_params['rev_comp_aug'] = False    
    val_batch_gen_params['mode'] = 'val'

    # get the corresponding batch generator class for this model
    sequence_generator_class_name = generators.find_generator_by_name(
        batch_gen_params['sequence_generator_name'])
    logging.info("SEQGEN Class Name: {}".format(sequence_generator_class_name))
    BatchGenerator = getattr(generators, sequence_generator_class_name)

    # instantiate the batch generator class for training
    train_gen = BatchGenerator(input_data, train_batch_gen_params, 
                               genome_params['reference_genome'], 
                               genome_params['chrom_sizes'],
                               train_chroms,loci_indices=train_indices,
                               background_loci_indices=background_train_indices,
                               num_threads=parallelization_params['threads'], 
                               batch_size=hyper_params['batch_size'],                                
                               foreground_weight=mnll_loss_sample_weight,
                               background_weight=mnll_loss_background_sample_weight)


    # instantiate the batch generator class for validation
    val_gen = BatchGenerator(input_data, val_batch_gen_params, 
                             genome_params['reference_genome'], 
                             genome_params['chrom_sizes'],
                             val_chroms,loci_indices=val_indices,
                             background_loci_indices=background_val_indices, 
                             num_threads=parallelization_params['threads'], 
                             batch_size=hyper_params['batch_size'],                              
                             foreground_weight=mnll_loss_sample_weight,
                             background_weight=mnll_loss_background_sample_weight)

    # we need to calculate the number of training steps and 
    # validation steps in each epoch, fit/evaluate requires this
    # to determine the end of an epoch
    train_steps = train_gen.len()
    val_steps = val_gen.len()

    # we may have to reduce the --threads sometimes
    # if the peak file has very few peaks, so we need to
    # check if these numbers will be 0
    logging.info("TRAINING STEPS - {}".format(train_steps))
    logging.info("VALIDATION STEPS - {}".format(val_steps))

    # get an instance of the model
    logging.debug("New {} model".format(model_arch_name))
    get_model = getattr(arch, model_arch_name)
    
    model = get_model(tasks, 
                        model_arch_params, 
                        orig_multi_loss=orig_multi_loss, 
                        name_prefix="main")
    
    # print out the model summary
    model.summary()
        
    logging.info(f"model.num_tasks: {model.num_tasks}")
    logging.info(f"model.num_output_tracks: {model.num_output_tracks}")
    logging.info(f"model.orig_multi_loss: {model.orig_multi_loss}")

    # compile the model
    logging.debug("Compiling model")
    logging.info("loss weights - {}".format(model_arch_params['loss_weights']))
    logging.info("counts loss - {}".format(model_arch_params['counts_loss']))
    model.compile(Adam(learning_rate=hyper_params['learning_rate']), 
					  loss = None,
                    loss_weights=model_arch_params['loss_weights'])
    
    # begin time for training
    t1 = time.time()

    # track training losses, validation losses and start & end
    # times
    custom_history = {
        'learning_rate': {},
        'loss': {},
        'batch_loss': {},
        'profile_predictions_loss': {},
        'logcounts_predictions_loss': {},
        'val_loss': {},
        'val_batch_loss': {},
        'val_profile_predictions_loss': {},
        'val_logcounts_predictions_loss': {},
        'start_time': {},
        'end_time': {},
        'elapsed': {}        
    }
    
    # we maintain a separate list to track validation losses to make it 
    # easier for early stopping 
    val_losses = []
    
    # track validation losses for learning rate update 
    val_losses_lr = []
    
    # track best loss so we can restore weights 
    best_loss = 1e32
    
    # keep a copy of the best weights
    best_weights = None
    
    # the epoch with the best validation loss
    best_epoch = 1
    
    # start training
    logging.debug("Training started ...")
    for epoch in range(hyper_params['epochs']):
        # First, let's train for one epoch
        logging.info("Training Epoch {}".format(epoch + 1))
        train_start_time = time.time()
        custom_history['learning_rate'][str(epoch + 1)] = \
            model.optimizer.learning_rate.numpy()
        custom_history['start_time'][str(epoch + 1)] = \
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(train_start_time))
        # training generator function that will be passed to fit
        train_generator = train_gen.gen()
        history = model.fit(
            train_generator, epochs=1, steps_per_epoch=train_steps)
        train_end_time = time.time()
        
        # record the training losses
        for key in history.history:
            custom_history[key][str(epoch + 1)] = history.history[key][0]
        
        # Then, we evaluate on the validation set
        logging.info("Validation Epoch {}".format(epoch + 1))
        val_start_time = time.time()
        # validation generator function that will be passed to evaluate 
        val_generator = val_gen.gen()
        val_loss = model.evaluate(
            val_generator, steps=val_steps, return_dict=True)
        val_losses.append(val_loss['loss'])
        val_losses_lr.append(val_loss['loss'])
        val_end_time = time.time()
        custom_history['end_time'][str(epoch + 1)] = \
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(val_end_time))
        custom_history['elapsed'][str(epoch + 1)] = \
            val_end_time - train_start_time
        
        # record the validation losses
        for key in val_loss:
            custom_history['val_' + key][str(epoch + 1)] = \
                val_loss[key]

        # update best weights and loss 
        if val_loss['loss'] < best_loss:
            best_weights = model.get_weights()
            best_loss = val_loss['loss']
            best_epoch = epoch + 1
        
        # check if early stopping criteria are satisfied
        if early_stopping_check(
            val_losses,
            patience=hyper_params['early_stopping_patience'],
            min_delta=hyper_params['early_stopping_min_delta']):
            
            # restore best weights
            logging.info("Restoring best weights from epoch {}".format(
                best_epoch))
            model.set_weights(best_weights)
            break

        # lower learning rate if criteria are satisfied
        current_lr = model.optimizer.learning_rate.numpy()
        new_lr = reduce_lr_on_plateau(
            val_losses_lr,
            current_lr,
            factor=hyper_params['lr_reduction_factor'], 
            patience=hyper_params['reduce_lr_on_plateau_patience'],
            min_lr=hyper_params['min_learning_rate'])
        
        # reset the validation losses tracker for learning rate update
        if new_lr != current_lr:
            val_losses_lr = [val_losses_lr[-1]]
        
        # set the new learning rate
        model.optimizer.learning_rate.assign(new_lr)

        # display current learning rate and training status
        logging.info("Current learning rate - {:5f}, Stop Training - {}".format(
            model.optimizer.learning_rate.numpy(),
            model.stop_training))

    # end time for training
    t2 = time.time() 
    logging.info("Total Elapsed Time: {}".format(t2-t1))

    # base model filename
    if output_params['automate_filenames']:
        # get random alphanumeric tag for model
        model_tag = getAlphaNumericTag(output_params['tag_length'])
        model_fname = "{}/{}".format(model_dir, model_tag)
    elif output_params['model_output_filename'] is not None:
        model_fname = "{}/{}".format(model_dir, 
                                     output_params['model_output_filename'])
    else:
        model_fname = "{}/model".format(model_dir)
    
    # add suffix tag to model name
    if suffix_tag is not None:
        model_fname += "_{}".format(suffix_tag)
    
    # extension
    # model_fname += ".h5"
    
    # save HDF5 model file
    model.save(model_fname)
    logging.info("Finished saving model: {}".format(model_fname))

    # save history to json:  
    # Step 1. convert the custom history dict to a pandas DataFrame:  
    hist_df = pd.DataFrame(custom_history) 

    # file name for json file
    hist_json = model_fname+'.history.json'

    # Step 2. write the dataframe to json
    with open(hist_json, mode='w') as f:
        hist_df.to_json(f)
    
    logging.info("Finished saving training and validation history: {}".format(
        hist_json))

    # write all the command line arguments to a json file
    # & include the number of epochs the training lasted for, and the
    # validation and testchroms
    config_file = '{}/config'.format(model_dir)
    # add suffix tag to model name
    if suffix_tag is not None:
        config_file += "_{}".format(suffix_tag)
    # extension
    config_file += ".json"
    
    with open(config_file, 'w') as fp:
        config = {}        
        config['input_data'] = input_data
        config['output_params'] = output_params
        config['genome_params'] = genome_params
        config['batch_gen_params'] = batch_gen_params
        config['hyper_params'] = hyper_params
        config['parallelization_params'] = parallelization_params
        
        # the number of epochs the training lasted
        config['training_epochs'] = epoch + 1

        # the epoch with best validation loss
        config['best_epoch'] = best_epoch 
        
        config['train_chroms'] = train_chroms
        config['val_chroms'] = val_chroms
        config['model_filename'] = model_fname

        json.dump(config, fp)

    return model


def train_and_validate_ksplits(
    input_data, model_arch_name, model_arch_params_json, output_params, 
    genome_params, batch_gen_params, hyper_params, parallelization_params, 
    splits, bias_input_data=None, mnll_loss_sample_weight=1.0, 
    mnll_loss_background_sample_weight=0.0,orig_multi_loss=False):

    """
        Train and validate on one or more train/val splits
        
        Args:
            input_data (str): path to the tasks json file
            
            model_arch_name (str): name of the model definition 
                function in the model_archs module
                
            model_arch_params_json (str): path to json file containing
                model architecture params
            
            output_params (dict): dictionary containing output 
                parameters
            
            genome_params (dict): dictionary containing genome
                parameters
            
            batch_gen_params (dict): dictionary containing batch
                generation parameters
            
            hyper_params (dict): dictionary containing containing 
                training & validation hyper parameters
            
            parallelization_params (dict): dictionary containing
                parameters for parallelization options
            
            splits (str): path to the json file containing train & 
                validation splits
            
            bias_input_data (str): path to the bias tasks json file
        
            mnll_loss_sample_weight (float): weight for each (foreground)
                training sample for computing mnll loss

            mnll_loss_background_sample_weight (float): weight for each
                background sample for computing mnll loss
                
            orig_multi_loss (boolean): True if original multinomial loss function - one for
                each strand is to be used
    """
    
    chroms = genome_params['chroms']
        
    # list of models from all of the splits
    models = []
    
    # run training for each validation/test split
    num_splits = len(list(splits.keys()))
    for i in range(num_splits):
        
        if output_params['automate_filenames']:
            # create a new directory using current date/time to store the
            # model, the loss history and logs 
            date_time_str = local_datetime_str(output_params['time_zone'])
            model_dir = '{}/{}_split{:03d}'.format(
                output_params['output_dir'], date_time_str, i)
            os.mkdir(model_dir)
            split_tag = None
        elif os.path.isdir(output_params['output_dir']):
            model_dir = output_params['output_dir']     
            split_tag = "split{:03d}".format(i)
        else:
            logging.error("Directory does not exist {}.".format(
                output_params['output_dir']))
            return
            
        # filename to write debug logs
        logfname = '{}/trainer.log'.format(model_dir)
        # set up logger for main procecss
        logger.init_logger(logfname)
    
        # train & validation chromosome split
        
        # we'll make val or val_indices_file the starting point
        train_chroms = None
        val_chroms = None
        train_indices = None
        val_indices = None
        background_train_indices=None
        background_val_indices=None
        if 'val' in splits[str(i)]:
            val_chroms = splits[str(i)]['val']
            if 'train' in splits[str(i)]:
                train_chroms = splits[str(i)]['train']
            # if 'test' key is present but train is not
            elif 'test' in splits[str(i)]:
                test_chroms = splits[str(i)]['test']
                # take the set difference of the whole list of
                # chroms with the union of val and test
                train_chroms = list(chroms.difference(
                    set(val_chroms + test_chroms)))
            else:
                # take the set difference of the whole list of
                # chroms with val
                train_chroms = list(chroms.difference(val_chroms))
                
            logging.info("Train chroms: {}".format(train_chroms))
            logging.info("Val chroms: {}".format(val_chroms))
        elif 'loci_val_indices_file' in splits[str(i)]:
            loci_val_indices_file = splits[str(i)]['loci_val_indices_file']
            loci_train_indices_file = splits[str(i)]['loci_train_indices_file']
            
            # make sure the val_indices_file file exists
            if not os.path.isfile(loci_val_indices_file):
                raise NoTracebackException(
                    "File not found: {} ".format(loci_val_indices_file))
            
            # make sure the train_indices_file file exists
            if not os.path.isfile(loci_train_indices_file):
                raise NoTracebackException(
                    "File not found: {} ".format(loci_train_indices_file))

            # load val_indices
            f = open(loci_val_indices_file)
            lines = f.readlines()
            val_indices = [int(line.rstrip('\r').rstrip('\n')) for line in lines]
            f.close()
            
            # load val_indices
            f = open(loci_train_indices_file)
            lines = f.readlines()
            train_indices = [int(line.rstrip('\r').rstrip('\n')) for line in lines]
            f.close()
        
            logging.info("Train indices length: {}".format(len(train_indices)))
            logging.info("Val indices length: {}".format(len(val_indices)))

            if 'background_val_indices_file' in splits[str(i)]:
                background_val_indices_file = splits[str(i)]['background_val_indices_file']
                background_train_indices_file = splits[str(i)]['background_train_indices_file']

                # make sure the background_val_indices_file file exists
                if not os.path.isfile(background_val_indices_file):
                    raise NoTracebackException(
                        "File not found: {} ".format(background_val_indices_file))

                # make sure the background_train_indices_file file exists
                if not os.path.isfile(background_train_indices_file):
                    raise NoTracebackException(
                        "File not found: {} ".format(background_train_indices_file))    
            
                # load background_val_indices
                f = open(background_val_indices_file)
                lines = f.readlines()
                background_val_indices = [int(line.rstrip('\r').rstrip('\n'))
                                          for line in lines]
                f.close()

                # load background_train_indices
                f = open(background_train_indices_file)
                lines = f.readlines()
                background_train_indices = [int(line.rstrip('\r').rstrip('\n'))
                                            for line in lines]
                f.close()

                logging.info("Background Train indices length: {}".format(
                    len(background_train_indices)))
                logging.info("Background Val indices length: {}".format(
                    len(background_val_indices)))

        
        logging.info("Split #{}".format(i))
        logging.info("Training chromosomes, if chromosome wise training regime: {}".format(train_chroms))
        logging.info("Validation chromosomes, if chromosome wise training regime: {}".format(val_chroms))
        logging.info("orig_multi_loss: {}".format(orig_multi_loss))
        # Start training for the split in a separate process
        # This ensures that all resources are freed, when the 
        # process terminates, & available for training the next split
        # Mitigates the problem where training subsequent splits
        # is considerably slow
        logging.debug("Split {}: Creating training process".format(i))
        p = mp.Process(
            target=train_and_validate, 
            args=[input_data, model_arch_name, model_arch_params_json,
                  output_params, genome_params, batch_gen_params, hyper_params,
                  parallelization_params, model_dir, train_chroms, val_chroms, train_indices,
                  val_indices, background_train_indices, background_val_indices,
                  bias_input_data, mnll_loss_sample_weight, 
                  mnll_loss_background_sample_weight,orig_multi_loss,
                  split_tag])
        p.start()
        
        # wait for the process to finish
        p.join()
