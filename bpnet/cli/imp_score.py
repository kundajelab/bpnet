"""Module containing code for importance scoring
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import warnings
from kipoi.readers import HDF5Reader
from kipoi.writers import HDF5BatchWriter
from bpnet.BPNet import BPNet, BiasModel
from bpnet.seqmodel import SeqModel
from bpnet.cli.schemas import DataSpec, HParams, ModiscoHParams
from bpnet.functions import mean
from bpnet.preproc import onehot_dinucl_shuffle
from bpnet.utils import add_file_logging, fnmatch_any, create_tf_session
import h5py
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def imp_score(model_dir,
              output_file,
              method="grad",
              split='all',
              batch_size=512,
              num_workers=10,
              h5_chunk_size=512,
              max_batches=-1,
              shuffle_seq=False,
              memfrac=0.45,
              exclude_chr='',
              overwrite=False,
              gpu=None):
    """Run importance scores for a BPNet model

    Args:
      model_dir: path to the model directory
      output_file: output file path (HDF5 format)
      method: which importance scoring method to use ('grad', 'deeplift' or 'ism')
      split: for which dataset split to compute the importance scores
      h5_chunk_size: hdf5 chunk size.
      exclude_chr: comma-separated list of chromosomes to exclude
      overwrite: if True, overwrite the output directory
      gpu (int): which GPU to use locally. If None, GPU is not used
    """
    add_file_logging(os.path.dirname(output_file), logger, 'modisco-score')
    if gpu is not None:
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac)
    else:
        # Don't use any GPU's
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise ValueError(f"File exists {output_file}. Use overwrite=True to overwrite it")

    if exclude_chr:
        exclude_chr = exclude_chr.split(",")
    else:
        exclude_chr = []
    # load the config files
    logger.info("Loading the config files")
    model_dir = Path(model_dir)
    hp = HParams.load(model_dir / "hparams.yaml")
    ds = DataSpec.load(model_dir / "dataspec.yaml")
    tasks = list(ds.task_specs)
    # validate that the correct dataset was used
    if hp.data.name != 'get_StrandedProfile_datasets':
        logger.warn("hp.data.name != 'get_StrandedProfile_datasets'")

    if split == 'valid':
        assert len(exclude_chr) == 0
        incl_chromosomes = hp.data.kwargs['valid_chr']
        excl_chromosomes = None
    elif split == 'test':
        assert len(exclude_chr) == 0
        incl_chromosomes = hp.data.kwargs['test_chr']
        excl_chromosomes = None
    elif split == 'train':
        assert len(exclude_chr) == 0
        incl_chromosomes = None
        excl_chromosomes = hp.data.kwargs['valid_chr'] + hp.data.kwargs['test_chr'] + hp.data.kwargs.get('exclude_chr', [])
    elif split == 'all':
        incl_chromosomes = None
        excl_chromosomes = hp.data.kwargs.get('exclude_chr', []) + exclude_chr
        logger.info("Excluding chromosomes: {excl_chromosomes}")
    else:
        raise ValueError("split needs to be from {train,valid,test,all}")

    logger.info("Creating the dataset")
    from bpnet.datasets import StrandedProfile
    seq_len = hp.data.kwargs['peak_width']
    dl_valid = StrandedProfile(ds,
                               incl_chromosomes=incl_chromosomes,
                               excl_chromosomes=excl_chromosomes,
                               peak_width=seq_len,
                               shuffle=False,
                               target_transformer=None)

    bpnet = BPNet.from_mdir(model_dir)

    # # setup the bias model
    # if [task for task, task_spec in ds.task_specs.items() if task_spec.bias_model]:
    #     bm = BiasModel(ds)
    # else:
    # bm = lambda x: x

    writer = HDF5BatchWriter(output_file, chunk_size=h5_chunk_size)
    for i, batch in enumerate(tqdm(dl_valid.batch_iter(batch_size=batch_size, num_workers=num_workers))):
        if max_batches > 0:
            logging.info(f"max_batches: {max_batches} exceeded. Stopping the computation")
            if i > max_batches:
                break
        # append the bias model predictions
        # (batch['inputs'], batch['targets']) = bm((batch['inputs'], batch['targets']))

        # store the original batch containing 'inputs' and 'targets'
        wdict = batch

        if shuffle_seq:
            # Di-nucleotide shuffle the sequences
            if 'seq' in batch['inputs']:
                batch['inputs']['seq'] = onehot_dinucl_shuffle(batch['inputs']['seq'])
            else:
                batch['inputs'] = onehot_dinucl_shuffle(batch['inputs'])

        # loop through all tasks, pred_summary and strands
        for task_i, task in enumerate(tasks):
            for pred_summary in ['count', 'weighted']:
                # figure out the number of channels
                nstrands = batch['targets'][f'profile/{task}'].shape[-1]
                strand_hash = ["pos", "neg"]

                for strand_i in range(nstrands):
                    hyp_imp = bpnet.imp_score(batch['inputs'],
                                              task=task,
                                              strand=strand_hash[strand_i],
                                              method=method,
                                              pred_summary=pred_summary,
                                              batch_size=None)  # don't second-batch
                    # put importance scores to the dictionary
                    wdict[f"/hyp_imp/{task}/{pred_summary}/{strand_i}"] = hyp_imp
        writer.batch_write(wdict)
    writer.close()


def avail_imp_scores(model_dir):
    """List the available interpretation targets
    """
    # don't use any gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    seqmodel = SeqModel.from_mdir(model_dir)
    print("Available Interpretation targets>")
    for name, _ in seqmodel.get_intp_tensors(preact_only=False):
        print(name)


def imp_score_seqmodel(model_dir,
                       output_file,
                       dataspec=None,
                       peak_width=1000,
                       seq_width=None,
                       intp_pattern='*',  # specifies which imp. scores to compute
                       # skip_trim=False,  # skip trimming the output
                       method="deeplift",
                       batch_size=512,
                       max_batches=-1,
                       shuffle_seq=False,
                       memfrac=0.45,
                       num_workers=10,
                       h5_chunk_size=512,
                       exclude_chr='',
                       include_chr='',
                       overwrite=False,
                       skip_bias=False,
                       gpu=None):
    """Run importance scores for a BPNet model

    Args:
      model_dir: path to the model directory
      output_file: output file path (HDF5 format)
      method: which importance scoring method to use ('grad', 'deeplift' or 'ism')
      split: for which dataset split to compute the importance scores
      h5_chunk_size: hdf5 chunk size.
      exclude_chr: comma-separated list of chromosomes to exclude
      overwrite: if True, overwrite the output directory
      skip_bias: if True, don't store the bias tracks in teh output
      gpu (int): which GPU to use locally. If None, GPU is not used
    """
    add_file_logging(os.path.dirname(output_file), logger, 'modisco-score-seqmodel')
    if gpu is not None:
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac)
    else:
        # Don't use any GPU's
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise ValueError(f"File exists {output_file}. Use overwrite=True to overwrite it")

    if seq_width is None:
        logger.info("Using seq_width = peak_width")
        seq_width = peak_width

    # make sure these are int's
    seq_width = int(seq_width)
    peak_width = int(peak_width)

    # Split
    intp_patterns = intp_pattern.split(",")

    # Allow chr inclusion / exclusion
    if exclude_chr:
        exclude_chr = exclude_chr.split(",")
    else:
        exclude_chr = None
    if include_chr:
        include_chr = include_chr.split(",")
    else:
        include_chr = None

    logger.info("Loading the config files")
    model_dir = Path(model_dir)

    if dataspec is None:
        # Specify dataspec
        dataspec = model_dir / "dataspec.yaml"
    ds = DataSpec.load(dataspec)

    logger.info("Creating the dataset")
    from bpnet.datasets import StrandedProfile
    dl_valid = StrandedProfile(ds,
                               incl_chromosomes=include_chr,
                               excl_chromosomes=exclude_chr,
                               peak_width=peak_width,
                               seq_width=seq_width,
                               shuffle=False,
                               taskname_first=True,  # Required to work nicely with imp-score
                               target_transformer=None)

    # Setup importance score trimming
    if seq_width > peak_width:
        # Trim
        # make sure we can nicely trim the peak
        logger.info("Trimming the output")
        assert (seq_width - peak_width) % 2 == 0
        trim_start = (seq_width - peak_width) // 2
        trim_end = seq_width - trim_start
        assert trim_end - trim_start == peak_width
    elif seq_width == peak_width:
        trim_start = 0
        trim_end = peak_width
    else:
        raise ValueError("seq_width < peak_width")

    seqmodel = SeqModel.from_mdir(model_dir)

    # get all possible interpretation names
    # make sure they match the specified glob
    intp_names = [name for name, _ in seqmodel.get_intp_tensors(preact_only=False)
                  if fnmatch_any(name, intp_patterns)]
    logger.info(f"Using the following interpretation targets:")
    for n in intp_names:
        print(n)

    writer = HDF5BatchWriter(output_file, chunk_size=h5_chunk_size)
    for i, batch in enumerate(tqdm(dl_valid.batch_iter(batch_size=batch_size, num_workers=num_workers))):
        # store the original batch containing 'inputs' and 'targets'
        wdict = batch
        if skip_bias:
            wdict['inputs'] = {'seq': wdict['inputs']['seq']}  # ignore all other inputs

        if max_batches > 0:
            logging.info(f"max_batches: {max_batches} exceeded. Stopping the computation")
            if i > max_batches:
                break

        if shuffle_seq:
            # Di-nucleotide shuffle the sequences
            batch['inputs']['seq'] = onehot_dinucl_shuffle(batch['inputs']['seq'])

        for name in intp_names:
            hyp_imp = seqmodel.imp_score(batch['inputs']['seq'],
                                         name=name,
                                         method=method,
                                         batch_size=None)  # don't second-batch

            # put importance scores to the dictionary
            # also trim the importance scores appropriately so that
            # the output will always be w.r.t. the peak center
            wdict[f"/hyp_imp/{name}"] = hyp_imp[:, trim_start:trim_end]

        # trim the sequence as well
        if isinstance(wdict['inputs'], dict):
            # Trim the sequence
            wdict['inputs']['seq'] = wdict['inputs']['seq'][:, trim_start:trim_end]
        else:
            wdict['inputs'] = wdict['inputs'][:, trim_start:trim_end]

        writer.batch_write(wdict)
    writer.close()


class ImpScoreFile:
    """Importance score container file

    Note: Use this class typically with `ImpScoreFile.from_modisco_dir()`

    Args:
      file_path: path to the hdf5 file
      include_samples: boolean vector of the same length
        as the arrays in the hdf5 file denoting which samples
        to keep and which ones to omit
      default_imp_score: which importance score should be the
        default one
    """

    def __init__(self, file_path,
                 include_samples=None,
                 default_imp_score='weighted'):
        self.file_path = file_path
        self.f = HDF5Reader(self.file_path)
        self.f.open()

        # use the hdf5 file handle
        self.data = self.f.f

        self.include_samples = include_samples

        self._hyp_contrib_cache = dict()
        self.default_imp_score = default_imp_score

    @classmethod
    def from_modisco_dir(cls, modisco_dir, ignore_include_samples=False):
        from bpnet.cli.modisco import load_included_samples
        from bpnet.utils import read_json
        if ignore_include_samples:
            include_samples = None
        else:
            include_samples = load_included_samples(modisco_dir)
            if include_samples.all():
                # All are true, we can ignore that
                include_samples = None
        kwargs = read_json(os.path.join(modisco_dir, "kwargs.json"))
        return cls(kwargs["imp_scores"],
                   include_samples,
                   default_imp_score=kwargs['grad_type'])

    def close(self):
        self.f.close()

    def __del__(self):
        self.close()

    def get_tasks(self):
        key = '/hyp_imp/'
        if isinstance(self.data, dict):
            return [x.replace(key, "").split("/")[0]
                    for x in self.data.keys()
                    if x.startswith(key)]
        else:
            return list(self.data[key].keys())

    def __len__(self):
        """Get the length of the dataset
        """
        if self.include_samples is not None:
            return self.include_samples.sum()
        else:
            return len(self.f)

    def _subset(self, x, idx=None):
        if self.include_samples is not None:
            if idx is None:
                return x[:][self.include_samples]
            else:
                new_idx = np.arange(len(self.include_samples))[self.include_samples][idx]
                return x[new_idx]
        else:
            if idx is None:
                return x[:]
            else:
                return x[idx]

    def _data_keys(self):
        if isinstance(self.data, dict):
            return list(self.data.keys())
        else:
            return list(list(zip(*self.f.ls()))[0])

    def _data_subkeys(self, key):
        return [k for k in self._data_keys()
                if k.startswith(key)]

    def get_example_idx(self, idx=None):
        if idx is None:
            idx = slice(None, None)
        return np.arange(len(self))[idx]

    def get_ranges(self, idx=None):
        return pd.DataFrame({"chrom": self._subset(self.data["/metadata/range/chr"], idx),
                             "start": self._subset(self.data["/metadata/range/start"], idx),
                             "end": self._subset(self.data["/metadata/range/end"], idx),
                             "strand": self._subset(self.data["/metadata/range/strand"], idx),
                             "interval_from_task": self._subset(self.data["/metadata/interval_from_task"], idx),
                             "idx": self.get_example_idx(idx),
                             })

    def _get_profile_key(self):
        template = '/targets/profile/{t}'
        if isinstance(self.data, dict):
            if template.format(t=self.get_tasks()[0]) in self.data:
                return template
            else:
                return '/targets/{t}/profile'
        else:
            if 'profile' in self.data['targets']:
                return template
            else:
                return '/targets/{t}/profile'

    def get_profiles(self, idx=None):
        tmpl = self._get_profile_key()
        return {t: self._subset(self.data[tmpl.format(t=t)], idx)
                for t in self.get_tasks()}

    def _get_seq_key(self):
        """Figure out the right sequence key
        """
        if isinstance(self.data, dict):
            if '/inputs' in self.data:
                return '/inputs'
            elif '/inputs/seq' in self.data:
                return '/inputs/seq'
            else:
                input_keys = [x.replace(key, "")
                              for x in self.data.keys()
                              if x.startswith('/inputs')]
                if len(input_keys) == 0:
                    raise ValueError("No entry with '/inputs'")
                else:
                    return input_keys[0]
        else:
            if isinstance(self.data['/inputs'], h5py.Group):
                return '/inputs/seq'
            else:
                return '/inputs'

    def get_seq(self, idx=None):
        return self._subset(self.data[self._get_seq_key()], idx)

    def get_seqlen(self):
        return self.data[self._get_seq_key()].shape[1]

    def contains_imp_score(self, imp_score):
        """Test if it contains `imp_score` importance score
        """
        task = self.get_tasks()[0]
        return len(self._data_subkeys(f'/hyp_imp/{task}/{imp_score}')) > 0

    def get_hyp_contrib(self, imp_score=None, idx=None, pred_summary=None):
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `imp_score`")
            imp_score = pred_summary

        imp_score = (imp_score if imp_score is not None
                     else self.default_imp_score)
        if imp_score in self._hyp_contrib_cache and idx is None:
            return self._hyp_contrib_cache[imp_score]
        else:
            # NOTE: this line averages any additional axes after {imp_score} like
            # strands denoted with:
            # /hyp_imp/{task}/{imp_score}/{strand}, where strand = 0 or 1
            out = {task: mean([self._subset(self.data[k], idx)
                               for k in self._data_subkeys(f'/hyp_imp/{task}/{imp_score}')])
                   for task in self.get_tasks()
                   }
            if idx is None:
                self._hyp_contrib_cache[imp_score] = out
            return out

    def get_contrib(self, imp_score=None, idx=None, pred_summary=None):
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `imp_score`")
            imp_score = pred_summary

        imp_score = (imp_score if imp_score is not None
                     else self.default_imp_score)
        seq = self.get_seq(idx=idx)
        return {task: hyp_contrib * seq
                for task, hyp_contrib in self.get_hyp_contrib(imp_score, idx=idx).items()}

    def cache(self):
        """Cache the data in memory
        """
        self.data = self.f.load_all(unflatten=False)
        return self

    def _extract(self, seqlet, seq, hyp_contrib, profiles, profile_width=None):
        """Extract all the values using the seqlet
        """
        from bpnet.modisco.core import Pattern
        ex_hyp_contrib = {task: seqlet.extract(arr)
                          for task, arr in hyp_contrib.items()}
        ex_seq = seqlet.extract(seq)
        # TODO - implement this to work also for the out-of-core case efficiently
        wide_seqlet = seqlet.resize(profile_width)
        return Pattern(name=seqlet.name,
                       seq=ex_seq,
                       contrib={task: arr * ex_seq for task, arr in ex_hyp_contrib.items()},
                       hyp_contrib=ex_hyp_contrib,
                       profile={task: wide_seqlet.extract(arr)
                                for task, arr in profiles.items()},
                       attrs=dict()
                       )

    def get_all(self, imp_score=None):
        return (self.get_seq(),
                self.get_contrib(imp_score=imp_score),
                self.get_hyp_contrib(imp_score=imp_score),
                self.get_profiles(),
                self.get_ranges())

    def extract(self, seqlets, profile_width=None, imp_score=None, verbose=False, pred_summary=None):
        """Extract multiple seqlets
        """
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `imp_score`")
            imp_score = pred_summary
        from bpnet.modisco.core import StackedSeqletImp

        imp_score = (imp_score if imp_score is not None
                     else self.default_imp_score)
        seq = self.get_seq()
        hyp_contrib = self.get_hyp_contrib(imp_score=imp_score)
        profile = self.get_profiles()
        return StackedSeqletImp.from_seqlet_imps([
            self._extract(s, seq, hyp_contrib, profile, profile_width=profile_width)
            for s in tqdm(seqlets, disable=not verbose)
        ])

    # StackedSeqletImp
    def extract_dfi(self, dfi, profile_width=None, imp_score=None, verbose=False, pred_summary=None):
        """Extract multiple seqlets
        """
        from bpnet.modisco.core import StackedSeqletImp
        from bpnet.modisco.pattern_instances import dfi_row2seqlet, dfi_filter_valid
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `imp_score`")
            imp_score = pred_summary

        if profile_width is not None:
            df_valid = dfi_filter_valid(dfi, profile_width=profile_width, seqlen=self.get_seqlen())
            if len(df_valid) != len(dfi):
                print(f"Removed {len(dfi) - len(df_valid)}/{len(dfi)} instances at the boundaries")
            dfi = df_valid

        imp_score = (imp_score if imp_score is not None
                     else self.default_imp_score)
        seq = self.get_seq()
        hyp_contrib = self.get_hyp_contrib(imp_score=imp_score)
        profile = self.get_profiles()
        out = StackedSeqletImp.from_seqlet_imps([
            self._extract(dfi_row2seqlet(dfi.iloc[i]),
                          seq,
                          hyp_contrib,
                          profile,
                          profile_width=profile_width)
            for i in tqdm(range(len(dfi)), disable=not verbose)
        ])
        out.dfi = dfi
        return out
