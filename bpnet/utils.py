import sys
import os
import pickle
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import papermill as pm  # Render the ipython notebook
from kipoi_utils.external.flatten_json import flatten, unflatten
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def jupyter_nbconvert(input_ipynb):
    # NOTE: cwd is used since the input_ipynb could contain some strange output
    # characters like '[' which would mess up the paths
    subprocess.call(["jupyter",
                     "nbconvert",
                     os.path.basename(input_ipynb),
                     "--to", "html"],
                    cwd=os.path.dirname(input_ipynb))


def render_ipynb(template_ipynb, rendered_ipynb, params=dict()):
    """Render the ipython notebook

    Args:
      template_ipynb: template ipython notebook where one cell defines the following metadata:
        {"tags": ["parameters"]}
      render_ipynb: output ipython notebook path
      params: parameters used to execute the ipython notebook
    """
    import jupyter_client

    os.makedirs(os.path.dirname(rendered_ipynb), exist_ok=True)
    kernel_name = os.environ.get("CONDA_DEFAULT_ENV", 'python3')
    if kernel_name not in jupyter_client.kernelspec.find_kernel_specs():
        logger.info(f"Installing the ipython kernel for the current conda environment: {kernel_name}")
        from ipykernel.kernelspec import install
        install(user=True, kernel_name=kernel_name)

    pm.execute_notebook(
        template_ipynb,  # input template
        rendered_ipynb,
        kernel_name=kernel_name,  # default kernel
        parameters=params
    )
    jupyter_nbconvert(rendered_ipynb)


def tqdm_restart():
    """Restart tqdm to not print to every line
    """
    from tqdm import tqdm as tqdm_cls
    inst = tqdm_cls._instances
    for i in range(len(inst)):
        inst.pop().close()


def touch_file(file, verbose=True):
    import subprocess
    if verbose:
        add = "v"
    else:
        add = ""
    subprocess.run(["vmtouch", f'-{add}tf', file])


def remove_exists(output_path, overwrite=False):
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            raise ValueError(f"File exists {str(output_path)}. Use overwrite=True to overwrite it")


def write_pkl(obj, fname, create_dirs=True, protocol=2):
    import cloudpickle
    if create_dirs:
        if os.path.dirname(fname):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
    cloudpickle.dump(obj, open(fname, 'wb'), protocol=protocol)


def read_pkl(fname):
    import cloudpickle
    return cloudpickle.load(open(fname, 'rb'))


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


class NumpyAwareJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def write_json(obj, fname, **kwargs):
    with open(fname, "w") as f:
        return json.dump(obj, f, cls=NumpyAwareJSONEncoder, **kwargs)


dump = write_pkl
load = read_pkl


def _listify(arg):
    if hasattr(type(arg), '__len__'):
        return arg
    return [arg, ]


def reverse_complement(seq):
    alt_map = {'ins': '0'}
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    for k, v in alt_map.items():
        seq = seq.replace(k, v)
    bases = list(seq)
    bases = reversed([complement.get(base, base) for base in bases])
    bases = ''.join(bases)
    for k, v in alt_map.items():
        bases = bases.replace(v, k)
    return bases


def related_dump_yaml(obj, path, verbose=False):
    import related
    generated_yaml = related.to_yaml(obj,
                                     suppress_empty_values=False,
                                     suppress_map_key_values=True)  # .strip()
    if verbose:
        print(generated_yaml)

    with open(path, "w") as f:
        f.write(generated_yaml)


def shuffle_list(l):
    return pd.Series(l).sample(frac=1).tolist()


def flatten_list(l):
    """Flattens a nested list
    """
    return [x for nl in l for x in nl]


class Logger(object):
    """tee functionality in python. If this object exists,
    then all of stdout gets logged to the file

    Adoped from:
    https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/3423392#3423392
    """

    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        # flush right away
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def add_file_logging(output_dir, logger, name='stdout'):
    os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)
    log = Logger(os.path.join(output_dir, 'log', name + '.log'), 'a+')  # log to the file
    fh = logging.FileHandler(os.path.join(output_dir, 'log', name + '.log'), 'a+')
    fh.setFormatter(logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return log


def halve(n):
    """Halve an integer"""
    return n // 2 + n % 2, n // 2


def expand_str_list(l, prefix="", suffix=""):
    """add strings to the beginning or to the end of the string
    """
    return [prefix + x + suffix for x in l]


def kv_string2dict(s):
    """Convert a key-value string: k=v,k2=v2,... into a dictionary
    """
    import yaml
    return yaml.load(s.replace(",", "\n").replace("=", ": "))


def dict_suffix_key(d, suffix):
    return {k + suffix: v for k, v in d.items()}


def dict_prefix_key(d, prefix):
    return {prefix + k: v for k, v in d.items()}


def kwargs_str2kwargs(hparams):
    """Converts a string to a dictionary of kwargs

    In[22]: params_str2kwargs("a=1;b=[1,2]")
    Out[22]: {'a': 1, 'b': [1, 2]}

    In[30]: hparams_str2kwargs("a=null")
    Out[30]: {'a': None}

    """
    import yaml
    return yaml.load(hparams.replace(";", "\n").replace("=", ": "))


def apply_parallel(df_grouped, func, n_jobs=-1, verbose=True):
    from joblib import Parallel, delayed
    import pandas as pd
    from tqdm import tqdm
    retLst = Parallel(n_jobs=n_jobs)(delayed(func)(group)
                                     for name, group in tqdm(df_grouped, disable=not verbose))
    return pd.concat(retLst)


def get_timestamp():
    """Get current time-stamp as a string: 2018-12-10_14:20:04
    """
    import datetime
    import time
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


class ConditionalRun:
    """Simple class keeping track whether the command has already
    been ran or not
    """

    def __init__(self, main_cmd, cmd, output_dir, force=False):
        self.main_cmd = main_cmd
        self.cmd = cmd
        self.output_dir = output_dir
        self.force = force

    def set_cmd(self, cmd):
        self.cmd = cmd
        return self

    def get_done_file(self):
        return os.path.join(self.output_dir, f".{self.main_cmd}/{self.cmd}.done")

    def done(self):
        ret = os.path.exists(self.get_done_file())
        if self.force:
            # always run the command
            ret = False
        if ret:
            logger.info(f"Skipping {self.cmd}")
        else:
            logger.info(f"Running {self.cmd}")
        return ret

    def write(self):
        fname = self.get_done_file()
        os.makedirs(os.path.dirname(fname),
                    exist_ok=True)
        with open(fname, "w") as f:
            f.write(get_timestamp())


class Logger(object):
    """tee functionality in python. If this object exists,
    then all of stdout gets logged to the file
    Adoped from:
    https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python/3423392#3423392
    """

    def __init__(self, name, mode='a'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        # flush right away
        self.file.flush()
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def fnmatch_any(s, patterns):
    from fnmatch import fnmatch  # unix-style pattern matching
    return any([fnmatch(s, p) for p in patterns])


def to_list(l):
    if isinstance(l, list):
        return l
    else:
        return [l]


def pd_first_cols(df: pd.DataFrame, cols):
    """Set `cols` to be the first columns in pd.DataFrame df
    """
    for c in cols:
        assert c in df
    other_cols = [c for c in df.columns if c not in cols]
    return df[cols + other_cols]


def pd_col_prepend(df: pd.DataFrame, column, prefix='', suffix=''):
    """Add a prefix or suffix to all the columns names in pd.DataFrame
    """
    if isinstance(column, list):
        for c in column:
            df[c] = prefix + df[c] + suffix
    else:
        df[column] = prefix + df[column] + suffix
    return df


def create_tf_session(visiblegpus, per_process_gpu_memory_fraction=0.45):
    import os
    import tensorflow as tf
    import keras.backend as K
    os.environ['CUDA_VISIBLE_DEVICES'] = str(visiblegpus)
    session_config = tf.ConfigProto()
    # session_config.gpu_options.deferred_deletion_bytes = DEFER_DELETE_SIZE
    session_config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
    session = tf.Session(config=session_config)
    K.set_session(session)
    return session
