"""Train models using gin configuration
"""
import gin
import gc
import json
import sys
import os
import yaml
import shutil
from argh.decorators import named, arg
from uuid import uuid4
import numpy as np
from tqdm import tqdm
import keras.backend as K
from bpnet.dataspecs import DataSpec
from bpnet.data import NumpyDataset
from bpnet.utils import (create_tf_session, write_json,
                         render_ipynb,
                         related_dump_yaml,
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
                     extra_kwargs=None,
                     overwrite=False):
    """Start a model training experiment. This will create a new output directory
    and setup the experiment management handles
    """
    sys.path.append(os.getcwd())
    if cometml_project:
        logger.info("Using comet.ml")
        if Experiment is None:
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
            logger.info("Using wandb. Running wandb.init()")
            wandb._set_stage_dir("./")  # Don't prepend wandb to output file
            if run_id is not None:
                wandb.init(project=project,
                           dir=output_dir,
                           entity=entity,
                           reinit=True,
                           resume=run_id)
            else:
                # automatically set the output
                wandb.init(project=project,
                           entity=entity,
                           reinit=True,
                           dir=output_dir)
            wandb_run = wandb.run
            if wandb_run is None:
                logger.warn("Wandb run is None")
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
        if overwrite:
            logger.info(f"config.gin already exists in the output "
                        "directory {output_dir}. Removing the whole directory.")
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
        cometml_experiment.log_parameters(note_params_dict)
        cometml_experiment.log_parameters(dict(output_dir=output_dir), prefix='cli/')

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

    return cometml_experiment, wandb_run, output_dir


def gin2dict(gin_config_str):
    # parse the gin config string to dictionary
    gin_config_lines = [x for x in gin_config_str.split("\n")
                        if not x.startswith("import")]
    gin_config_str = "\n".join(gin_config_lines)

    # insert macros
    macros = []
    track = False
    for line in gin_config_lines:
        if line.startswith("# Macros"):
            track = True
        if " = %" in line:
            track = False
            break
        if track:
            macros.append(line)
    gin_macro_dict = yaml.load("\n".join(macros).replace("@", "").replace(" = %", ": ").replace(" = ", ": "))
    lines = []

    for linei in gin_config_lines:
        if " = %" in linei:
            k, v = linei.split(" = %")
            lines.append(f"{k} = {gin_macro_dict[v]}")
        else:
            lines.append(linei)

    gin_config_dict = yaml.load("\n".join(lines)
                                .replace("@", "")
                                .replace(" = %", ": ")
                                .replace(" = ", ": "))
    return gin_config_dict


def log_gin_config(output_dir, cometml_experiment=None, wandb_run=None, prefix=''):
    """Save the config.gin file containing the whole config, convert it
    to a dictionary and upload it to cometml and wandb.
    """
    gin_config_str = gin.operative_config_str()

    print("Used config: " + "-" * 40)
    print(gin_config_str)
    print("-" * 52)
    with open(os.path.join(output_dir, f"{prefix}config.gin"), "w") as f:
        f.write(gin_config_str)

    gin_config_dict = gin2dict(gin_config_str)
    write_json(gin_config_dict,
               os.path.join(output_dir, f"{prefix}config.gin.json"),
               sort_keys=True,
               indent=2)

    if cometml_experiment is not None:
        # Skip any rows starting with import
        cometml_experiment.log_parameters(gin_config_dict)

    if wandb_run is not None:
        # This allows to display the metric on the dashboard
        wandb_run.config.update({k.replace(".", "/"): v for k, v in gin_config_dict.items()})


def _track_stats(tracks):
    """Compute the statistics of different tracks
    """
    stats = {}
    for k, x in tracks.items():
        total = x.sum(axis=(-1, -2))
        stats[k] = {"per-base max": x.max(),
                    "per-base fraction of 0": np.mean(x == 0),
                    "per-base mean": np.mean(x),
                    "per-base median": np.median(x),
                    "total mean": np.mean(total),
                    "total median": np.median(total),
                    "total max": total.max(),
                    "total fraction of 0": np.mean(total == 0)
                    }
    return stats


@arg('dataspec',
     help='dataspec.yaml file')
@arg('--regions',
     help='Path to the interval BED file. If not specified, files specified in dataspec.yml will be used')
@arg('--sample', type=int,
     help='Specifies the number of randomly selected regions on which the stats are computed.')
@arg('--peak-width',
     help='Resize all regions to that specific width using interval center as an anchorpoint.')
def dataspec_stats(dataspec,
                   regions=None,
                   sample=None,
                   peak_width=1000):
    """Compute the stats about the tracks
    """
    import random
    from pybedtools import BedTool
    from bpnet.preproc import resize_interval
    from genomelake.extractors import FastaExtractor

    ds = DataSpec.load(dataspec)

    if regions is not None:
        regions = list(BedTool(regions))
    else:
        regions = ds.get_all_regions()

    if sample is not None and sample < len(regions):
        logger.info(f"Using {sample} randomly sampled regions instead of {len(regions)}")
        regions = random.sample(regions, k=sample)

    # resize the regions
    regions = [resize_interval(interval, peak_width, ignore_strand=True)
               for interval in regions]

    base_freq = FastaExtractor(ds.fasta_file)(regions).mean(axis=(0, 1))

    count_stats = _track_stats(ds.load_counts(regions, progbar=True))
    bias_count_stats = _track_stats(ds.load_bias_counts(regions, progbar=True))

    print("")
    print("Base frequency")
    for i, base in enumerate(['A', 'C', 'G', 'T']):
        print(f"- {base}: {base_freq[i]}")
    print("")
    print("Count stats")
    for task, stats in count_stats.items():
        print(f"- {task}")
        for stat_key, stat_value in stats.items():
            print(f"  {stat_key}: {stat_value}")
    print("")
    print("Bias stats")
    for task, stats in bias_count_stats.items():
        print(f"- {task}")
        for stat_key, stat_value in stats.items():
            print(f"  {stat_key}: {stat_value}")


@gin.configurable
def train(output_dir,
          model=gin.REQUIRED,
          data=gin.REQUIRED,
          eval_metric=None,
          eval_train=False,
          eval_skip=[],
          trainer_cls=SeqModelTrainer,
          eval_report=None,
          # shared
          batch_size=256,
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
          # specified by bpnet_train
          in_memory=False,
          num_workers=8,
          gpu=None,
          memfrac_gpu=None,
          cometml_experiment=None,
          wandb_run=None,
          ):
    """Main entry point to configure in the gin config

    Args:
      model: compiled keras model
      data: tuple of (train, valid) Datasets
      eval_train: if True, also compute the evaluation metrics for the final model
        on the training set
      eval_report: path to the ipynb report file. Use the default one. If set to empty string, the report will not be generated.
      eval_skip List[str]: datasets to skip during evaluation
      seed: random seed to use (in numpy and tensorflow)
    """
    # from this point on, no configurable should be added. Save the gin config
    log_gin_config(output_dir, cometml_experiment, wandb_run)

    train_dataset, valid_dataset = data[0], data[1]

    if eval_report is not None:
        eval_report = os.path.abspath(os.path.expanduser(eval_report))
        if not os.path.exists(eval_report):
            raise ValueError(f"Evaluation report {eval_report} doesn't exist")

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

    num_workers_orig = num_workers  # remember the old number of workers before overwriting it
    if in_memory:
        # load the training datasets to memory
        logger.info("Loading the training data into memory")
        train_dataset = NumpyDataset(train_dataset.load_all(batch_size=batch_size,
                                                            num_workers=num_workers))
        logger.info("Loading the validation data into memory")
        if isinstance(valid_dataset, list):
            # appropriately handle the scenario where multiple
            # validation data may be provided as a list of (name, Dataset) tuples
            valid_dataset = [(k, NumpyDataset(data.load_all(batch_size=batch_size,
                                                            num_workers=num_workers)))
                             for k, data in valid_dataset]
        else:
            # only a single Dataset was provided
            valid_dataset = NumpyDataset(valid_dataset.load_all(batch_size=batch_size,
                                                                num_workers=num_workers))

        num_workers = 1  # don't use multi-processing any more

    tr = trainer_cls(model,
                     train_dataset,
                     valid_dataset,
                     output_dir,
                     cometml_experiment,
                     wandb_run)

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

    if eval_report is not None:
        logger.info("Running the evaluation report")
        # Release the GPU
        K.clear_session()

        # remove memory
        del tr, train_dataset, valid_dataset, data
        gc.collect()

        if num_workers_orig != num_workers:
            # recover the original number of workers
            num_workers = num_workers_orig

        # Run the jupyter notebook
        render_ipynb(eval_report,
                     os.path.join(output_dir, os.path.basename(eval_report)),
                     params=dict(model_dir=os.path.abspath(output_dir),
                                 gpu=gpu,
                                 memfrac_gpu=memfrac_gpu,
                                 in_memory=in_memory,
                                 num_workers=num_workers))

    # upload all files in output_dir to comet.ml
    # Note: wandb does this automatically
    if cometml_experiment is not None:
        logger.info("Uploading files to comet.ml")
        cometml_experiment.log_asset_folder(folder=output_dir)

    logger.info(f"Done training and evaluating the model. Model and metrics can be found in: {output_dir}")

    return final_metrics


def _get_premade_path(premade, raise_error=False):
    """Get the pre-made file from ../premade/ directory
    """
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_dir = os.path.dirname(os.path.abspath(filename))
    premade_dir = os.path.join(this_dir, "../premade/")
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


def _get_gin_files(premade, config):
    """Get the gin files given premade and config
    """
    gin_files = []
    if premade in ['none', 'None', '']:
        logger.info(f"premade config not specified")
    else:
        gin_files.append(_get_premade_path(premade, raise_error=True))
        logger.info(f"Using the following premade configuration: {premade}")

    if config is not None:
        logger.info(f"Using the following config.gin files: {config}")
        gin_files += config.split(",")
    if len(gin_files) == 0:
        raise ValueError("Please specify at least one of the two: --premade or --config")
    return gin_files


@named('train')
@arg('dataspec',
     help='dataspec.yaml file')
@arg('output_dir',
     help='where to store the results. Note: a subdirectory `run_id` will be created in `output_dir`.')
@arg('--premade',
     help='pre-made config file to use (e.g. use the default architecture). See TODO - X for available premade models.')
@arg('--config',
     help='gin config file path(s) specifying the model architecture and the loss etc.'
     ' They override the premade model. Single model example: config.gin. Multiple file example: data.gin,model.gin')
@arg('--override',
     help='semi-colon separated list of additional gin bindings to use')
@arg('--gpu',
     help='which gpu to use. Example: gpu=1')
@arg('--memfrac-gpu',
     help='what fraction of the GPU memory to use')
@arg('--num-workers',
     help='number of workers to use in parallel for loading the data the model')
@arg('--vmtouch',
     help='if True, use vmtouch to load the files in dataspec into Linux page cache')
@arg('--in-memory',
     help='if True, load the entire dataset into the memory first')
@arg('--wandb-project',
     help='path to the wandb (https://www.wandb.com/) project name `<entity>/<project>`. '
     'Example: Avsecz/test. This will track and upload your metrics. '
     'Make sure you have specified the following environemnt variable: TODO. If not specified, wandb will not be used')
@arg('--cometml-project',
     help='path to the comet.ml (https://www.comet.ml/) project specified as <username>/<project>.'
     ' This will track and upload your metrics. Make sure you have specified the following environemnt '
     'variable: TODO. If not specified, cometml will not get used')
@arg('--run-id',
     help='manual run id. If not specified, it will be either randomly generated or re-used from wandb or comet.ml.')
@arg('--note-params',
     help='take note of additional key=value pairs. Example: --note-params note="my custom note",feature_set=this')
@arg('--overwrite',
     help='if True, the output directory will be overwritten')
def bpnet_train(dataspec,
                output_dir,
                premade='bpnet9',
                config=None,
                override='',
                gpu=0,
                memfrac_gpu=0.45,
                num_workers=8,
                vmtouch=False,
                in_memory=False,
                wandb_project="",
                cometml_project="",
                run_id=None,
                note_params="",
                overwrite=False):
    """Train a model using gin-config

    Output files:
      train.log - log file
      model.h5 - Keras model HDF5 file
      seqmodel.pkl - Serialized SeqModel. This is the main trained model.
      eval-report.ipynb/.html - evaluation report containing training loss curves and some example model predictions.
        You can specify your own ipynb using `--override='report_template.name="my-template.ipynb"'`.
      model.gin -> copied from the input
      dataspec.yaml -> copied from the input
    """
    cometml_experiment, wandb_run, output_dir = start_experiment(output_dir=output_dir,
                                                                 cometml_project=cometml_project,
                                                                 wandb_project=wandb_project,
                                                                 run_id=run_id,
                                                                 note_params=note_params,
                                                                 overwrite=overwrite)
    # remember the executed command
    write_json({
        "dataspec": dataspec,
        "output_dir": output_dir,
        "premade": premade,
        "config": config,
        "override": override,
        "gpu": gpu,
        "memfrac_gpu": memfrac_gpu,
        "num_workers": num_workers,
        "vmtouch": vmtouch,
        "in_memory": in_memory,
        "wandb_project": wandb_project,
        "cometml_project": cometml_project,
        "run_id": run_id,
        "note_params": note_params,
        "overwrite": overwrite},
        os.path.join(output_dir, 'bpnet-train.kwargs.json'),
        indent=2)

    # copy dataspec.yml and input config file over
    if config is not None:
        shutil.copyfile(config, os.path.join(output_dir, 'input-config.gin'))

    # parse and validate the dataspec
    ds = DataSpec.load(dataspec)
    related_dump_yaml(ds.abspath(), os.path.join(output_dir, 'dataspec.yml'))
    if vmtouch:
        if shutil.which('vmtouch') is None:
            logger.warn("vmtouch is currently not installed. "
                        "--vmtouch disabled. Please install vmtouch to enable it")
        else:
            # use vmtouch to load all file to memory
            ds.touch_all_files()

    # --------------------------------------------
    # Parse the config file
    # import gin.tf
    if gpu is not None:
        logger.info(f"Using gpu: {gpu}, memory fraction: {memfrac_gpu}")
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac_gpu)

    gin_files = _get_gin_files(premade, config)

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
    # TODO - handle multiple track widths?
    tracks_per_task = [len(v.tracks) for k, v in ds.task_specs.items()][0]
    # figure out the right hyper-parameters
    dataspec_bindings = [f'dataspec="{dataspec}"',
                         f'use_bias={use_bias}',
                         f'n_bias_tracks={n_bias_tracks}',
                         f'tracks_per_task={tracks_per_task}',
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
        cometml_experiment.log_parameters(dict(premade=premade,
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

    return train(output_dir=output_dir,
                 cometml_experiment=cometml_experiment,
                 wandb_run=wandb_run,
                 num_workers=num_workers,
                 in_memory=in_memory,
                 # to execute the sub-notebook
                 memfrac_gpu=memfrac_gpu,
                 gpu=gpu)
