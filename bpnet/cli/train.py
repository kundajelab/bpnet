"""Train models using gin configuration
"""
import gin
import json
import sys
import os
import yaml
from argh.decorators import aliases, named
from uuid import uuid4
from fs.osfs import OSFS
import numpy as np
from tqdm import tqdm
from bpnet.cli.schemas import DataSpec
from bpnet.utils import (create_tf_session, write_json,
                         Logger, NumpyAwareJSONEncoder,
                         dict_prefix_key, kv_string2dict)

# try to import differnet experiment tracking frameworks
try:
    from comet_ml import Experiment
except ImportError:
    Experiment = None
try:
    import wandb
except ImportError:
    wandb = None

# import all modules registering any gin configurables
from bpnet.trainers import SeqModelTrainer
from bpnet import metrics
from bpnet import trainers
from bpnet import samplers

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def add_file_logging(output_dir, logger, name='stdout'):
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)
    log = Logger(os.path.join(output_dir, 'log', name + '.log'), 'a+')  # log to the file
    fh = logging.FileHandler(os.path.join(output_dir, 'log', name + '.log'), 'a+')
    fh.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return log


def start_experiment(output_dir,
                     cometml_project="",
                     wandb_project="",
                     run_id=None,
                     note_params="",
                     force_overwrite=False):
    """Start a model training experiment. This will create a new output directory
    and setup the experiment management handles
    """
    sys.path.append(os.getcwd())
    if cometml_project:
        logger.info("Using comet.ml")
        if Experiment is None:
            from comet_ml import Experiment
            raise ImportError("Comet.ml could not be imported")
        workspace, project_name = cometml_project.split("/")
        cometml_experiment = Experiment(project_name=project_name, workspace=workspace)
        # TODO - get the experiment id
        # specify output_dir to that directory
    else:
        cometml_experiment = None

    if wandb_project:
        assert "/" in wandb_project
        entity, project = wandb_project.split("/")
        if wandb is None:
            logger.warn("wandb not installed. Not using it")
            wandb_run = None
        else:
            wandb._set_stage_dir("./")  # Don't prepend wandb to output file
            if run_id is not None:
                wandb.init(project=project,
                           dir=output_dir,
                           entity=entity,
                           resume=run_id)
            else:
                # automatically set the output
                wandb.init(project=project,
                           entity=entity,
                           dir=output_dir)
            wandb_run = wandb.run
            logger.info("Using wandb")
            print(wandb_run)
    else:
        wandb_run = None

    # update the output directory
    if run_id is None:
        if wandb_run is not None:
            run_id = os.path.basename(wandb_run.dir)
        elif cometml_experiment is not None:
            run_id = cometml_experiment.id
        else:
            # random run_id
            run_id = str(uuid4())
    output_dir = os.path.join(output_dir, run_id)

    if wandb_run is not None:
        # make sure the output directory is the same
        # wandb_run._dir = os.path.normpath(output_dir)  # This doesn't work
        # assert os.path.normpath(wandb_run.dir) == os.path.normpath(output_dir)
        # TODO - fix this assertion-> the output directories should be the same
        # in order for snakemake to work correctly
        pass
    # -----------------------------

    if os.path.exists(os.path.join(output_dir, 'config.gin')):
        if force_overwrite:
            logger.info(f"config.gin already exists in the output "
                        "directory {output_dir}. Removing the whole directory.")
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} shouldn't exist!")
    os.makedirs(output_dir, exist_ok=True)  # make the output directory. It shouldn't exist

    # add logging to the file
    add_file_logging(output_dir, logger)

    # write note_params.json
    if note_params:
        logger.info(f"note_params: {note_params}")
        note_params_dict = kv_string2dict(note_params)
    else:
        note_params_dict = dict()
    write_json(note_params_dict,
               os.path.join(output_dir, "note_params.json"),
               sort_keys=True,
               indent=2)

    if cometml_experiment is not None:
        cometml_experiment.log_multiple_params(note_params_dict)
        cometml_experiment.log_multiple_params(dict(output_dir=output_dir), prefix='cli/')

        exp_url = f"https://www.comet.ml/{cometml_experiment.workspace}/{cometml_experiment.project_name}/{cometml_experiment.id}"
        logger.info("Comet.ml url: " + exp_url)
        # write the information about comet.ml experiment
        write_json({"url": exp_url,
                    "key": cometml_experiment.id,
                    "project": cometml_experiment.project_name,
                    "workspace": cometml_experiment.workspace},
                   os.path.join(output_dir, "cometml.json"),
                   sort_keys=True,
                   indent=2)

    if wandb_run is not None:
        wandb_run.config.update(note_params_dict)
        write_json({"url": wandb_run.get_url(),
                    "key": wandb_run.id,
                    "project": wandb_run.project,
                    "path": wandb_run.path,
                    "group": wandb_run.group
                    },
                   os.path.join(output_dir, "wandb.json"),
                   sort_keys=True,
                   indent=2)
        wandb_run.config.update(dict_prefix_key(dict(output_dir=output_dir), prefix='cli/'))

    return cometml_experiment, wandb_run


def log_gin_config(output_dir, cometml_experiment=None, wandb_run=None):
    """Save the config.gin file containing the whole config, convert it
    to a dictionary and upload it to cometml and wandb.
    """
    gin_config_str = gin.operative_config_str()

    print("Used config: " + "-" * 40)
    print(gin_config_str)
    print("-" * 52)
    with open(os.path.join(output_dir, "config.gin"), "w") as f:
        f.write(gin_config_str)
    # parse the gin config string to dictionary
    gin_config_str = "\n".join([x for x in gin_config_str.split("\n")
                                if not x.startswith("import")])
    gin_config_dict = yaml.load(gin_config_str
                                .replace("@", "")
                                .replace(" = %", ": ")
                                .replace(" = ", ": "))
    write_json(gin_config_dict,
               os.path.join(output_dir, "config.gin.json"),
               sort_keys=True,
               indent=2)

    if cometml_experiment is not None:
        # Skip any rows starting with import
        cometml_experiment.log_multiple_params(gin_config_dict)

    if wandb_run is not None:
        # This allows to display the metric on the dashboard
        wandb_run.config.update({k.replace(".", "/"): v for k, v in gin_config_dict.items()})


@gin.configurable
def train(output_dir,
          model=gin.REQUIRED,
          data=gin.REQUIRED,
          eval_metric=None,
          eval_train=False,
          eval_skip=[],
          trainer_cls=SeqModelTrainer,
          # shared
          batch_size=256,
          num_workers=8,
          # train-specific
          epochs=100,
          early_stop_patience=4,
          train_epoch_frac=1.0,
          valid_epoch_frac=1.0,
          train_samples_per_epoch=None,
          validation_samples=None,
          train_batch_sampler=None,
          stratified_sampler_p=None,
          tensorboard=True,
          seed=None,
          cometml_experiment=None,
          wandb_run=None
          ):
    """Main entry point to configure in the gin config

    Args:
      model: compiled keras model
      data: tuple of (train, valid) Datasets
      eval_train: if True, also compute the evaluation metrics for the final model
        on the training set
      eval_skip List[str]: datasets to skip during evaluation
      seed: random seed to use (in numpy and tensorflow)
    """
    # from this point on, no configurable should be added. Save the gin config
    log_gin_config(output_dir, cometml_experiment, wandb_run)

    train_dataset, valid_dataset = data[0], data[1]

    if seed is not None:
        # Set the random seed
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.set_random_seed(seed)
        except Exception:
            logger.info("Unable to set random seed for tensorflow")

    # make sure the validation dataset names are unique
    if isinstance(valid_dataset, list):
        dataset_names = []
        for d in valid_dataset:
            dataset_name = d[0]
            if dataset_name in dataset_names:
                raise ValueError("The dataset names are not unique")
            dataset_names.append(dataset_name)

    if stratified_sampler_p is not None and train_batch_sampler is not None:
        raise ValueError("stratified_sampler_p and train_batch_sampler are mutually exclusive."
                         " Please specify only one of them.")

    if stratified_sampler_p is not None and train_batch_sampler is None:
        # HACK - there is no guarantee that train_dataset.get_targets() will exist
        # Maybe we have to introduce a ClassificationDataset instead which will
        # always implement get_targets()
        logger.info(f"Using stratified samplers with p: {stratified_sampler_p}")
        train_batch_sampler = samplers.StratifiedRandomBatchSampler(train_dataset.get_targets().max(axis=1),
                                                                    batch_size=batch_size,
                                                                    p_vec=stratified_sampler_p,
                                                                    verbose=True)

    tr = trainer_cls(model, train_dataset, valid_dataset, output_dir, cometml_experiment, wandb_run)
    tr.train(batch_size=batch_size,
             epochs=epochs,
             early_stop_patience=early_stop_patience,
             num_workers=num_workers,
             train_epoch_frac=train_epoch_frac,
             valid_epoch_frac=valid_epoch_frac,
             train_samples_per_epoch=train_samples_per_epoch,
             validation_samples=validation_samples,
             train_batch_sampler=train_batch_sampler,
             tensorboard=tensorboard)
    final_metrics = tr.evaluate(eval_metric, batch_size=batch_size, num_workers=num_workers,
                                eval_train=eval_train, eval_skip=eval_skip, save=True)
    # pass
    logger.info("Done!")
    print("-" * 40)
    print("Final metrics: ")
    print(json.dumps(final_metrics, cls=NumpyAwareJSONEncoder, indent=2))

    # upload files to comet.ml
    if cometml_experiment is not None:
        logger.info("Uploading files to comet.ml")
        for f in tqdm(list(OSFS(output_dir).walk.files())):
            # [1:] removes trailing slash
            cometml_experiment.log_asset(file_path=os.path.join(output_dir, f[1:]),
                                         file_name=f[1:])
    return final_metrics


def get_premade_path(premade, raise_error=False):
    """Get the pre-made file from ../premade/ directory
    """
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_dir = os.path.dirname(os.path.abspath(filename))
    premade_dir = os.path.join(this_dir, "../premade/")
    # TODO - generalize this to multiple files?
    premade_file = os.path.join(premade_dir, premade + '.gin')
    if not os.path.exists(premade_file):
        if raise_error:
            # list all file
            available_premade_files = [f.replace(".gin", "")
                                       for f in os.listdir(premade_dir)
                                       if f.endswith(".gin")]
            premade_str = "\n".join(available_premade_files)
            raise FileNotFoundError(
                f"Premade model {premade} doesn't exist. Available models:\n{premade_str}"
            )
        else:
            return None
    return premade_file


@named('train')
def bpnet_train(dataspec,
                output_dir,
                premade='bpnet9',  # TODO - make the premade model optional?
                config=None,
                override='',
                eval_report='../templates/basepair-template.ipynb',  # TODO specify the correct path
                gpu=0,
                memfrac_gpu=0.45,
                wandb_project="",
                cometml_project="",
                run_id=None,
                note_params="",
                force_overwrite=False,
                skip_evaluation=False):
    """Train a model using gin-config

    Args:
      dataspec: dataspec.yaml file
      output_dir: where to store the results. Note: a subdirectory `run_id`
        will be created in `output_dir`.
      premade: pre-made config file to use (e.g. use the default architecture). See TODO - X for available premade models.
      config(optional): gin config file path(s) specifying the model architecture and the loss etc. They override the premade model
        single model example: config.gin
        mutltiple file example: data.gin,model.gin
      override: semi-colon separated list of additional gin-bindings to use
      eval_report: path to the ipynb report file. Use the default one. If set to None or none, then the report will not be generated.
      gpu: which gpu to use. Example: gpu=1
      memfrac_gpu: what fraction of the GPU's memory to use
      wandb_project: path to the wandb (https://www.wandb.com/) project name `<entity>/<project>`. Example: Avsecz/test.
        This will track and upload your metrics. Make sure you have specified the following environemnt variable: TODO.
        If not specified, wandb will not be used
      cometml_project: path to the comet.ml (https://www.comet.ml/) project specified as <username>/<project>.
        This will track and upload your metrics. Make sure you have specified the following environemnt variable: TODO.
        If not specified, cometml will not get used
      run_id: manual run id. If not specified, it will be either randomly
        generated or re-used from wandb or comet.ml.
      note_params: take note of additional key=value pairs.
        Example: --note-params note='my custom note',feature_set=this
      force_overwrite: if True, the output directory will be overwritten
      skip_evaluation: if True, model evaluation will be skipped


    Output:
      train.log - log file
      model.h5 - Keras model HDF5 file
      seqmodel.pkl - Serialized SeqModel. This is the main trained model.
      eval-report.ipynb/.html - evaluation report containing training loss curves and some example model predictions.
        You can specify your own ipynb using --eval-report=my-notebook.ipynb.
      model.gin -> copied from the input
      dataspec.yaml -> copied from the input
    """
    cometml_experiment, wandb_run = start_experiment(output_dir=output_dir,
                                                     cometml_project=cometml_project,
                                                     wandb_project=wandb_project,
                                                     run_id=run_id,
                                                     note_params=note_params,
                                                     force_overwrite=force_overwrite)
    # parse and validate the dataspec
    ds = DataSpec.load(dataspec)

    # --------------------------------------------
    # Parse the config file
    import gin.tf
    if gpu is not None:
        logger.info(f"Using gpu: {gpu}, memory fraction: {memfrac_gpu}")
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac_gpu)

    gin_files = []
    if premade in ['none', 'None', '']:
        logger.info(f"premade model not specified")
    else:
        gin_files.append(get_premade_path(premade, raise_error=True))
        logger.info(f"Using the following premade model: {premade}")

    if config is not None:
        logger.info(f"Using the following config.gin files: {config}")
        gin_files += config.split(",")

    if len(gin_files) == 0:
        raise ValueError("Please specify at least one of the two: --premade or --config")

    # infer differnet hyper-parameters from the dataspec file
    if len(ds.bias_specs) > 0:
        use_bias = True
        if len(ds.bias_specs) > 1:
            # TODO - allow multiple bias track
            # - split the heads separately
            raise ValueError("Only a single bias track is currently supported")

        bias = [v for k, v in ds.bias_specs.items()][0]
        n_bias_tracks = len(bias.tracks)
    else:
        use_bias = False
        n_bias_tracks = 0
    tasks = list(ds.task_specs)
    # figure out the right hyper-parameters
    dataspec_bindings = [f'dataspec="{dataspec}"',
                         f'use_bias={use_bias}',
                         f'n_bias_tracks={n_bias_tracks}',
                         f'tasks={tasks}'
                         ]

    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=dataspec_bindings + override.split(";"),
                                        # NOTE: custom files were inserted right after
                                        # ther user's config file and before the `override`
                                        # parameters specified at the command-line
                                        # This allows the user to disable the bias correction
                                        # despite being specified in the config file
                                        skip_unknown=False)

    # --------------------------------------------
    # Remember the parsed configs

    # comet - log environment
    if cometml_experiment is not None:
        # log other parameters
        cometml_experiment.log_multiple_params(dict(premade=premade,
                                                    config=config,
                                                    override=override,
                                                    gin_files=gin_files,
                                                    gpu=gpu), prefix='cli/')

    # wandb - log environment
    if wandb_run is not None:

        # store general configs
        wandb_run.config.update(dict_prefix_key(dict(premade=premade,
                                                     config=config,
                                                     override=override,
                                                     gin_files=gin_files,
                                                     gpu=gpu), prefix='cli/'))

    return train(output_dir=output_dir, cometml_experiment=cometml_experiment, wandb_run=wandb_run)
