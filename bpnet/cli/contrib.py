"""Module containing code for contribution scoring
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from argh.decorators import aliases, named, arg
import os
import warnings
from kipoi.readers import HDF5Reader
from kipoi.writers import HDF5BatchWriter
from bpnet.seqmodel import SeqModel
from bpnet.cli.schemas import DataSpec, ModiscoHParams
from bpnet.functions import mean
from bpnet.preproc import onehot_dinucl_shuffle
from bpnet.utils import add_file_logging, fnmatch_any, create_tf_session, read_json
import h5py
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def avail_contrib_scores(model_dir):
    """List the available interpretation targets
    """
    # don't use any gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    seqmodel = SeqModel.from_mdir(model_dir)
    print("Available Interpretation targets>")
    for name, _ in seqmodel.get_intp_tensors(preact_only=False):
        print(name)


@named('contrib')
@arg('--max-regions', type=int,
     help='Maximum number of regions to score.')
@arg("model_dir",
     help='path to the model directory')
@arg("output_file",
     help='output file path (example: deeplift.imp-score.h5)')
@arg("--method",
     help="which contribution scoring method to use ('grad', 'deeplift' or 'ism')")
@arg("--dataspec",
     help="Dataspec yaml file path. If not specified, dataspec used to train the model will be used")
@arg("--regions",
     help="Regions BED3 file. If not specified, regions specified in the dataspec file will be used")
@arg("--fasta-file",
     help="Reference genome fasta file. If specified, the dataspec argument will be ignored and the "
     "experimental track values will not be stored to the output file. Requires --regions to be specified. ")
@arg("--fasta-file",
     help="Reference genome fasta file. If specified, the dataspec argument will be ignored and the "
     "experimental track values will not be stored to the output file. Requires --regions to be specified. ")
@arg("--shuffle-seq",
     help="Compute the contribution scores on the shufled DNA sequences. Used to generate the background or 'null' distribution "
     "of the contribution scores used by TF-MoDISco.")
@arg("--shuffle-regions",
     help="Shuffle the order in which the regions are scores. Useful when using --max-regions.")
@arg("--max-regions",
     help="Compute the contribution scores only for the top `max-regions` instead of all the regions specified "
     "in the dataspec or the regions document.")
def bpnet_contrib(model_dir,
                  output_file,
                  method="deeplift",
                  dataspec=None,
                  regions=None,
                  fasta_file=None,  # alternative to dataspec
                  shuffle_seq=False,
                  shuffle_regions=False,
                  max_regions=None,
                  # reference='zeroes', # Currently the only option
                  # peak_width=1000,  # automatically inferred from 'config.gin.json'
                  # seq_width=None,

                  intp_pattern='*/profile/wn,*/counts/pre-act',  # specifies which contrib. scores to compute
                  batch_size=512,
                  memfrac_gpu=0.45,
                  num_workers=10,
                  h5_chunk_size=512,
                  exclude_chr='',
                  include_chr='',
                  overwrite=False,
                  skip_bias=False,
                  gpu=None):
    """Run contribution scores for a BPNet model

      method: 
      exclude_chr: comma-separated list of chromosomes to exclude
      overwrite: if True, overwrite the output directory
      skip_bias: if True, don't store the bias tracks in the output
      gpu (int): which GPU to use locally. If None, GPU is not used
      h5_chunk_size: hdf5 chunk size.
    """
    add_file_logging(os.path.dirname(output_file), logger, 'bpnet-contrib')
    if gpu is not None:
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac_gpu)
    else:
        # Don't use any GPU's
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            raise ValueError(f"File exists {output_file}. Use overwrite=True to overwrite it")

    config = read_json(os.path.join(model_dir, 'config.gin.json'))
    seq_width = config['seq_width']
    peak_width = config['seq_width']

    # NOTE - seq_width has to be the same for the input and the target
    #
    # infer from the command line
    # if seq_width is None:
    #     logger.info("Using seq_width = peak_width")
    #     seq_width = peak_width

    # # make sure these are int's
    # seq_width = int(seq_width)
    # peak_width = int(peak_width)

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

    logger.info("Creating the dataset")
    from bpnet.datasets import StrandedProfile, SeqClassification
    if fasta_file is not None:
        if regions is None:
            raise ValueError("fasta_file specified. Expecting regions to be specified as well")
        dl_valid = SeqClassification(fasta_file=fasta_file,
                                     interval_file=regions,
                                     incl_chromosomes=include_chr,
                                     excl_chromosomes=exclude_chr,
                                     auto_resize_len=seq_width,
                                     )
    else:
        if dataspec is None:
            logger.info("Using dataspec used to train the model")
            # Specify dataspec
            dataspec = model_dir / "dataspec.yaml"

        ds = DataSpec.load(dataspec)
        dl_valid = StrandedProfile(ds,
                                   incl_chromosomes=include_chr,
                                   excl_chromosomes=exclude_chr,
                                   intervals_file=regions,
                                   peak_width=peak_width,
                                   seq_width=seq_width,
                                   taskname_first=True,  # Required to work nicely with contrib-score
                                   target_transformer=None)

    # Setup contribution score trimming (not required currently)
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

    if max_regions is not None:
        if len(dl_valid) > max_regions:
            logging.info(f"Using {max_regions} regions instead of the original {len(dl_valid)}")
        else:
            logging.info(f"--max-regions={max_regions} is larger than the dataset size: {len(dl_valid)}. "
                         "Using the dataset size for max-regions")
            max_regions = len(dl_valid)
    else:
        max_regions = len(dl_valid)

    max_batches = np.ceil(max_regions / batch_size)

    writer = HDF5BatchWriter(output_file, chunk_size=h5_chunk_size)
    for i, batch in enumerate(tqdm(dl_valid.batch_iter(batch_size=batch_size,
                                                       shuffle=shuffle_regions,
                                                       num_workers=num_workers),
                                   total=max_batches)):
        # store the original batch containing 'inputs' and 'targets'
        wdict = batch
        if skip_bias:
            wdict['inputs'] = {'seq': wdict['inputs']['seq']}  # ignore all other inputs

        if max_batches > 0:
            if i > max_batches:
                break

        if shuffle_seq:
            # Di-nucleotide shuffle the sequences
            batch['inputs']['seq'] = onehot_dinucl_shuffle(batch['inputs']['seq'])

        for name in intp_names:
            hyp_contrib = seqmodel.contrib_score(batch['inputs']['seq'],
                                                 name=name,
                                                 method=method,
                                                 batch_size=None)  # don't second-batch

            # put contribution scores to the dictionary
            # also trim the contribution scores appropriately so that
            # the output will always be w.r.t. the peak center
            wdict[f"/hyp_contrib/{name}"] = hyp_contrib[:, trim_start:trim_end]

        # trim the sequence as well
        if isinstance(wdict['inputs'], dict):
            # Trim the sequence
            wdict['inputs']['seq'] = wdict['inputs']['seq'][:, trim_start:trim_end]
        else:
            wdict['inputs'] = wdict['inputs'][:, trim_start:trim_end]

        writer.batch_write(wdict)
    writer.close()


class ContribScoreFile:
    """Contribution score container file

    Note: Use this class typically with `ContribScoreFile.from_modisco_dir()`

    Args:
      file_path: path to the hdf5 file
      include_samples: boolean vector of the same length
        as the arrays in the hdf5 file denoting which samples
        to keep and which ones to omit
      default_contrib_score: which contribution score should be the
        default one
    """

    def __init__(self, file_path,
                 include_samples=None,
                 default_contrib_score='weighted'):
        self.file_path = file_path
        self.f = HDF5Reader(self.file_path)
        self.f.open()

        # use the hdf5 file handle
        self.data = self.f.f

        self.include_samples = include_samples

        self._hyp_contrib_cache = dict()
        self.default_contrib_score = default_contrib_score

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
        return cls(kwargs["contrib_scores"],
                   include_samples,
                   default_contrib_score=kwargs['contrib_type'])

    def close(self):
        self.f.close()

    def __del__(self):
        self.close()

    def get_tasks(self):
        key = '/hyp_contrib/'
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
                input_keys = [x.replace(x.split("/")[1], "")
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

    def contains_contrib_score(self, contrib_score):
        """Test if it contains `contrib_score` contribution score
        """
        task = self.get_tasks()[0]
        return len(self._data_subkeys(f'/hyp_contrib/{task}/{contrib_score}')) > 0

    def get_hyp_contrib(self, contrib_score=None, idx=None, pred_summary=None):
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `contrib_score`")
            contrib_score = pred_summary

        contrib_score = (contrib_score if contrib_score is not None
                         else self.default_contrib_score)
        if contrib_score in self._hyp_contrib_cache and idx is None:
            return self._hyp_contrib_cache[contrib_score]
        else:
            # NOTE: this line averages any additional axes after {contrib_score} like
            # strands denoted with:
            # /hyp_contrib/{task}/{contrib_score}/{strand}, where strand = 0 or 1
            out = {task: mean([self._subset(self.data[k], idx)
                               for k in self._data_subkeys(f'/hyp_contrib/{task}/{contrib_score}')])
                   for task in self.get_tasks()
                   }
            if idx is None:
                self._hyp_contrib_cache[contrib_score] = out
            return out

    def get_contrib(self, contrib_score=None, idx=None, pred_summary=None):
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `contrib_score`")
            contrib_score = pred_summary

        contrib_score = (contrib_score if contrib_score is not None
                         else self.default_contrib_score)
        seq = self.get_seq(idx=idx)
        return {task: hyp_contrib * seq
                for task, hyp_contrib in self.get_hyp_contrib(contrib_score, idx=idx).items()}

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

    def get_all(self, contrib_score=None):
        return (self.get_seq(),
                self.get_contrib(contrib_score=contrib_score),
                self.get_hyp_contrib(contrib_score=contrib_score),
                self.get_profiles(),
                self.get_ranges())

    def extract(self, seqlets, profile_width=None, contrib_score=None, verbose=False, pred_summary=None):
        """Extract multiple seqlets
        """
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `contrib_score`")
            contrib_score = pred_summary
        from bpnet.modisco.core import StackedSeqletContrib

        contrib_score = (contrib_score if contrib_score is not None
                         else self.default_contrib_score)
        seq = self.get_seq()
        hyp_contrib = self.get_hyp_contrib(contrib_score=contrib_score)
        profile = self.get_profiles()
        return StackedSeqletContrib.from_seqlet_contribs([
            self._extract(s, seq, hyp_contrib, profile, profile_width=profile_width)
            for s in tqdm(seqlets, disable=not verbose)
        ])

    # StackedSeqletContrib
    def extract_dfi(self, dfi, profile_width=None, contrib_score=None, verbose=False, pred_summary=None):
        """Extract multiple seqlets
        """
        from bpnet.modisco.core import StackedSeqletContrib
        from bpnet.modisco.pattern_instances import dfi_row2seqlet, dfi_filter_valid
        if pred_summary is not None:
            warnings.warn("pred_summary is deprecated. Use `contrib_score`")
            contrib_score = pred_summary

        if profile_width is not None:
            df_valid = dfi_filter_valid(dfi, profile_width=profile_width, seqlen=self.get_seqlen())
            if len(df_valid) != len(dfi):
                print(f"Removed {len(dfi) - len(df_valid)}/{len(dfi)} instances at the boundaries")
            dfi = df_valid

        contrib_score = (contrib_score if contrib_score is not None
                         else self.default_contrib_score)
        seq = self.get_seq()
        hyp_contrib = self.get_hyp_contrib(contrib_score=contrib_score)
        profile = self.get_profiles()
        out = StackedSeqletContrib.from_seqlet_contribs([
            self._extract(dfi_row2seqlet(dfi.iloc[i]),
                          seq,
                          hyp_contrib,
                          profile,
                          profile_width=profile_width)
            for i in tqdm(range(len(dfi)), disable=not verbose)
        ])
        out.dfi = dfi
        return out
