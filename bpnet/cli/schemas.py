"""
Related schemas
"""
from __future__ import absolute_import
from __future__ import print_function

from copy import deepcopy
import os
from collections import OrderedDict
import related
from kipoi.external.related.mixins import RelatedConfigMixin, RelatedLoadSaveMixin
from kipoi.external.related.fields import AnyField
from kipoi.external.related.fields import StrSequenceField
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------
# data_spec


@related.mutable(strict=False)
class TaskSpec(RelatedConfigMixin):
    # Bigwig file paths to counts from
    # the positive and negative strands
    task = related.StringField()
    pos_counts = AnyField()
    neg_counts = AnyField()
    peaks = related.StringField(None, required=False)

    # if True the profile array will be single-stranded
    ignore_strand = related.BooleanField(False, required=False)

    # bias_model = related.StringField(None, required=False)  # if available, provide the bias model
    # implements .predict_on_batch(onehot_seq)

    # bias_bigwig = related.StringField(None, required=False)

    def load_counts(self, intervals, use_strand=True, progbar=False):
        import numpy as np
        # from genomelake.extractors import BigwigExtractor
        from bpnet.extractors import StrandedBigWigExtractor
        if isinstance(self.pos_counts, str):
            pos_counts = StrandedBigWigExtractor(self.pos_counts,
                                                 use_strand=use_strand,
                                                 nan_as_zero=True).extract(intervals)
            neg_counts = StrandedBigWigExtractor(self.neg_counts,
                                                 use_strand=use_strand,
                                                 nan_as_zero=True).extract(intervals)
        elif isinstance(self.pos_counts, list):
            pos_counts = sum([StrandedBigWigExtractor(counts,
                                                      use_strand=use_strand,
                                                      nan_as_zero=True).extract(intervals, progbar=progbar)
                              for counts in self.pos_counts])
            neg_counts = sum([StrandedBigWigExtractor(counts,
                                                      use_strand=use_strand,
                                                      nan_as_zero=True).extract(intervals, progbar=progbar)
                              for counts in self.neg_counts])
        else:
            raise ValueError('pos_counts is not a str or a list')
        if self.ignore_strand:
            return (pos_counts + neg_counts)[..., np.newaxis]  # keep the same dimension
        else:
            if use_strand:
                neg_strand = np.array([s.strand == '-' for s in intervals]).reshape((-1, 1))
                return np.stack([np.where(neg_strand, neg_counts, pos_counts),
                                 np.where(neg_strand, pos_counts, neg_counts)], axis=-1)
            else:
                return np.stack([pos_counts, neg_counts], axis=-1)

    def get_bw_dict(self):
        return {"pos": self.pos_counts, "neg": self.neg_counts}

    def list_all_files(self, include_peaks=False):
        """List all file paths specified
        """
        files = []
        if isinstance(self.pos_counts, str):
            files += [self.pos_counts,
                      self.neg_counts]
        elif isinstance(self.pos_counts, list):
            for counts in self.pos_counts:
                files.append(counts)
            for counts in self.neg_counts:
                files.append(counts)
        else:
            raise ValueError('pos_counts is not a str or a list')
        if include_peaks and self.peaks is not None:
            files.append(self.peaks)
        return files

    def touch_all_files(self, verbose=True):
        from bpnet.utils import touch_file
        for f in self.list_all_files(include_peaks=False):
            touch_file(f, verbose)

    def abspath(self):
        """Use absolute filepaths
        """
        if self.peaks is None:
            peaks_abspath = None
        else:
            peaks_abspath = os.path.abspath(self.peaks)
        obj = deepcopy(self)
        if isinstance(self.pos_counts, str):
            obj.pos_counts = os.path.abspath(self.pos_counts)
            obj.neg_counts = os.path.abspath(self.neg_counts)
        elif isinstance(self.pos_counts, list):
            obj.pos_counts = [os.path.abspath(counts) for counts in self.pos_counts]
            obj.neg_counts = [os.path.abspath(counts) for counts in self.neg_counts]
        else:
            raise ValueError('pos_counts is not a str or a list')
        obj.peaks = peaks_abspath
        return obj

    def __attrs_post_init__(self):
        if not isinstance(self.pos_counts, str) and not isinstance(self.pos_counts, list):
            raise ValueError('pos_counts is not a str or a list')
        if type(self.neg_counts) != type(self.pos_counts):
            raise ValueError('neg_counts has to be same type as pos_counts')
        if isinstance(self.pos_counts, list) and len(self.pos_counts) != len(self.neg_counts):
            raise ValueError('neg_counts has to be same length as pos_counts')


@related.immutable(strict=True)
class BiasSpec(TaskSpec):
    # specifies for which tasks does this bias track apply
    tasks = related.SequenceField(str, required=False)


@related.immutable(strict=False)
class DataSpec(RelatedLoadSaveMixin):
    """Dataset specification
    """
    # Dictionary of different bigwig files
    task_specs = related.MappingField(TaskSpec, "task",
                                      required=True,
                                      repr=True)

    # Path to the reference genome fasta file
    fasta_file = related.StringField(required=True)

    # Bias track specification
    bias_specs = related.MappingField(BiasSpec, "task",
                                      required=False,
                                      repr=True)

    # # Set of peaks to consider
    # peaks = related.StringField(required=True)

    # Original path to the file
    path = related.StringField(required=False)

    def abspath(self):
        return DataSpec(
            task_specs={k: v.abspath() for k, v in self.task_specs.items()},
            fasta_file=os.path.abspath(self.fasta_file),
            # peaks=os.path.abspath(self.peaks),
            path=self.path)

    def task2idx(self, task, dtype='counts'):
        """Get the index output

        Args:
          task: task name
          dtype: 'counts' or 'profile'

        Returns:
          index for the list of predicted arrays
        """
        # TODO - this is not the right location for the code
        # it should be next to the model
        n2idx = {k: i for i, k in enumerate(self.task_specs)}
        if dtype == "counts":
            return len(self.task_specs) + n2idx[task]
        elif dtype == "profile":
            return n2idx[task]
        else:
            raise ValueError("type is not from 'counts' or 'profile'")

    def get_bws(self):
        return OrderedDict([(task, task_spec.get_bw_dict())
                            for task, task_spec in self.task_specs.items()])

    def list_all_files(self, include_peaks=False):
        """List all file paths specified
        """
        files = []
        files.append(self.fasta_file)
        for ts in self.task_specs.values():
            files += ts.list_all_files(include_peaks=include_peaks)

        for ts in self.bias_specs.values():
            files += ts.list_all_files(include_peaks=include_peaks)
        return files

    def touch_all_files(self, verbose=True):
        from bpnet.utils import touch_file
        touch_file(self.fasta_file, verbose)

        for ts in self.task_specs.values():
            ts.touch_all_files(verbose=verbose)

        for ts in self.bias_specs.values():
            ts.touch_all_files(verbose=verbose)

    def load_counts(self, intervals, use_strand=True, progbar=False):
        return {task: ts.load_counts(intervals, use_strand=use_strand, progbar=progbar)
                for task, ts in self.task_specs.items()}

    def load_bias_counts(self, intervals, use_strand=True, progbar=False):
        return {task: ts.load_counts(intervals, use_strand=use_strand, progbar=progbar)
                for task, ts in self.bias_specs.items()}

# --------------------------------------------
# hparams


@related.mutable(strict=True)
class ModelHParams(RelatedConfigMixin):
    """Model hyper-parameters / configuration
    """
    # Name of the model function from bpnet.models
    name = related.StringField("seq_multitask", required=False)
    # Kwargs for the model function
    kwargs = related.ChildField(dict, default={}, required=False)

    # Other function needs to fill in [tasks and seq_len]


@related.immutable(strict=True)
class DataHParams(RelatedConfigMixin):
    """Dataset hyper-parameters / configuration
    """
    name = related.StringField("chip_exo_nexus", required=False)  # function name
    type = related.StringField(default="function", required=False)  # function name

    # Preproc for smoothing
    preproc = related.ChildField(dict, default={}, required=False)
    # preproc
    # # BinAndSmooth params
    # mode = related.StringField('gauss', required=False)
    # sigma = related.FloatField(1.2, required=False)
    # binsize = related.IntegerField(10, required=False)

    # Kwargs for the dataset function
    kwargs = related.ChildField(dict, default={}, required=False)
    # kwargs
    # # Sequence width
    # peak_width = related.IntegerField(200, required=False)

    # # whether to shuffle the samples
    # shuffle = related.BooleanField(True, required=False)

    # # valid, test chromosomes
    # valid_chr = related.SequenceField(str, default=['chr2', 'chr3', 'chr4'], required=False)
    # test_chr = related.SequenceField(str, default=['chr1', 'chr8', 'chr9'], required=False)


# @related.immutable(strict=True)
# class DataHParams(RelatedConfigMixin):
#     """Dataset hyper-parameters / configuration
#     """
#     # Sequence width
#     peak_width = related.IntegerField(200, required=False)

#     # whether to shuffle the samples
#     shuffle = related.BooleanField(True, required=False)

#     # valid, test chromosomes
#     valid_chr = related.SequenceField(str, default=['chr2', 'chr3', 'chr4'], required=False)
#     test_chr = related.SequenceField(str, default=['chr1', 'chr8', 'chr9'], required=False)
#     # all other chromosomes are used to train the model

#     # TODO - assert valid and test don't overlap


@related.mutable(strict=True)
class TrainHParams(RelatedConfigMixin):
    """Training hyper-parameters
    """
    # patience in keras.callbacks.EarlyStopping
    early_stop_patience = related.IntegerField(5, required=False)

    # maximual number of epochs to train the model for
    epochs = related.IntegerField(200, required=False)

    batch_size = related.IntegerField(256, required=False)

    balance_peak_classes = related.BooleanField(False, required=False)

    # how frequently to evaluate (in fraction of the epoch)
    train_epoch_frac = related.FloatField(1.0, required=False)
    valid_epoch_frac = related.FloatField(1.0, required=False)


@related.mutable(strict=True)
class EvalHParams(RelatedConfigMixin):
    """Training hyper-parameters
    """
    # Hyper-parameters to compute the auPR
    pos_min_threshold = related.FloatField(0.05, required=False)
    neg_max_threshold = related.FloatField(0.01, required=False)

    # each position in the positive class should be
    # supported by at least `required_min_pos_counts` of reads
    required_min_pos_counts = related.FloatField(2.5, required=False)

    # For which pools to compute the evaluation metrics
    binsizes = related.SequenceField(int, [1, 10], required=False)


# shall we make a separate hparams.yaml file for modisco?
@related.immutable(strict=True)
class ModiscoHParams(RelatedLoadSaveMixin):
    """Modisco hyper-parameters
    """
    # Modisco kwargs
    sliding_window_size = related.IntegerField(21, required=False)
    flank_size = related.IntegerField(10, required=False)  # old=5
    target_seqlet_fdr = related.FloatField(0.01, required=False)

    min_seqlets_per_task = related.IntegerField(1000, required=False)  # NOTE - this is not used as of modisco 0.5

    # Deprecated in modisco 0.5
    min_passing_windows_frac = related.FloatField(0.03, required=False)
    max_passing_windows_frac = related.FloatField(0.2, required=False)

    # metaclustering
    min_metacluster_size = related.IntegerField(2000, required=False)
    min_metacluster_size_frac = related.FloatField(0.02, required=False)

    # seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory
    trim_to_window_size = related.IntegerField(30, required=False)  # default: 30, old=15
    initial_flank_to_add = related.IntegerField(10, required=False)  # default 10, old=5
    kmer_len = related.IntegerField(8, required=False)  # default 8, old=5
    num_gaps = related.IntegerField(3, required=False)  # default 3, old=1
    num_mismatches = related.IntegerField(2, required=False)  # default 2, old=0
    final_min_cluster_size = related.IntegerField(60, required=False)
    max_seqlets_per_metacluster = related.IntegerField(20000, required=False)

    # Other modisc-related kwargs

    # What's the maximum cosine distance between strands to
    # still consider the examples as representative?

    # TODO - update back
    # max_strand_distance = related.FloatField(0.2, required=False)

    # TODO - specify for which ones to run?

    # Original path to the file
    path = related.StringField(required=False)

    def get_modisco_kwargs(self):
        d = self.get_config()
        d.pop('counts', None)
        return d


@related.mutable(strict=True)
class HParams(RelatedLoadSaveMixin):
    """Hyper-nparameters yaml file
    """
    model = related.ChildField(ModelHParams, ModelHParams(), required=False)
    data = related.ChildField(DataHParams, DataHParams(), required=False)
    train = related.ChildField(TrainHParams, TrainHParams(), required=False)
    evaluate = related.ChildField(EvalHParams, EvalHParams(), required=False)
    modisco = related.ChildField(dict, default={}, required=False)
    # modisco = related.ChildField(ModiscoHParams, ModiscoHParams(), required=False)
    path = related.StringField(required=False)
