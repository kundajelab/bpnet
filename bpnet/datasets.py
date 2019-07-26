import pandas as pd
import numpy as np
from copy import deepcopy
import json
from pathlib import Path
from pysam import FastaFile
from pybedtools import BedTool, Interval
import pybedtools
from kipoi.data import Dataset
# try:
# import torch
# from bpnet.data import Dataset
# torch.multiprocessing.set_sharing_strategy('file_system')
# except:
#     print("PyTorch not installed. Using Dataset from kipoi.data")
#    from kipoi.data import Dataset

from kipoi.metadata import GenomicRanges
from bpnet.config import get_data_dir, valid_chr, test_chr, all_chr
from bpnet.utils import to_list
from bpnet.cli.schemas import DataSpec
from bpnet.preproc import bin_counts, keep_interval, moving_average, IntervalAugmentor
from bpnet.losses import MultichannelMultinomialNLL, mc_multinomial_nll_2, mc_multinomial_nll_1, twochannel_multinomial_nll
from concise.utils.helper import get_from_module
from tqdm import tqdm
from concise.preprocessing import encodeDNA
from random import Random
import joblib
from bpnet.preproc import resize_interval, AppendTotalCounts, AppendCounts
from genomelake.extractors import FastaExtractor, BigwigExtractor, ArrayExtractor
from kipoi.data_utils import get_dataset_item
from kipoiseq.dataloaders.sequence import BedDataset
import gin
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
mc_multinomial_nll = mc_multinomial_nll_1


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        import bisect
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        import warnings
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


@gin.configurable
def chip_exo_nexus(dataspec,
                   peak_width=200,
                   shuffle=True,
                   preprocessor=AppendTotalCounts(),
                   interval_augm=lambda x: x,
                   valid_chr=valid_chr,
                   test_chr=test_chr):
    """
    General dataloading function for ChIP-exo or ChIP-nexus data

    Args:
      dataspec: bpnet.schemas.DataSpec object containing information about
        the bigwigs, fasta_file and
      peak_width: final width of the interval to extract
      shuffle: if true, the order of the peaks will get shuffled
      preprocessor: preprocessor object - needs to implement .fit() and .predict() methods
      interval_augm: interval augmentor.
      valid_chr: list of chromosomes in the validation split
      test_chr: list of chromosomes in the test split

    Returns:
      (train, valid, test) tuple where train consists of:
        - x: one-hot encoded sequence, sample shape: (peak_width, 4)
        - y: dictionary containing fields:
          {task_id}/profile: sample shape - (peak_width, 2), count profile
          {task_id}/counts: sample shape - (2, ), total number of counts per strand
        - metadata: pandas dataframe storing the original intervals

    """
    for v in valid_chr:
        assert v not in test_chr

    def set_attrs_name(interval, name):
        """Add a name to the interval
        """
        interval.attrs['name'] = name
        return interval

    # Load intervals for all tasks.
    #   remember the task name in interval.name
    def get_bt(peaks):
        if peaks is None:
            return []
        else:
            return BedTool(peaks)

    # Resize and skip infervals outside of the genome
    from pysam import FastaFile
    fa = FastaFile(dataspec.fasta_file)
#     intervals = len(get_bt(peaks))
#     n_int = len(intervals)

    intervals = [set_attrs_name(resize_interval(interval_augm(interval), peak_width), task)
                 for task, ds in dataspec.task_specs.items()
                 for i, interval in enumerate(get_bt(ds.peaks))
                 if keep_interval(interval, peak_width, fa)]
#     if len(intervals) != n_int:
#         logger.warn(f"Skipped {n_int - len(intervals)} intervals"
#                     " outside of the genome size")

    if shuffle:
        Random(42).shuffle(intervals)

    # Setup metadata
    dfm = pd.DataFrame(dict(id=np.arange(len(intervals)),
                            chr=[x.chrom for x in intervals],
                            start=[x.start for x in intervals],
                            end=[x.stop for x in intervals],
                            task=[x.attrs['name'] for x in intervals]
                            ))

    logger.info("extract sequence")
    seq = FastaExtractor(dataspec.fasta_file)(intervals)

    logger.info("extract counts")
    cuts = {f"profile/{task}": spec.load_counts(intervals)
            for task, spec in tqdm(dataspec.task_specs.items())}
    # # sum across the sequence
    # for task in dataspec.task_specs:
    #     cuts[f"counts/{task}"] = cuts[f"profile/{task}"].sum(axis=1)
    assert len(seq) == len(dfm)
    assert len(seq) == len(cuts[list(cuts.keys())[0]])

    # Split by chromosomes
    is_test = dfm.chr.isin(test_chr)
    is_valid = dfm.chr.isin(valid_chr)
    is_train = (~is_test) & (~is_valid)

    train = [seq[is_train], get_dataset_item(cuts, is_train), dfm[is_train]]
    valid = [seq[is_valid], get_dataset_item(cuts, is_valid), dfm[is_valid]]
    test = [seq[is_test], get_dataset_item(cuts, is_test), dfm[is_test]]

    if preprocessor is not None:
        preprocessor.fit(train[1])
        train[1] = preprocessor.transform(train[1])
        valid[1] = preprocessor.transform(valid[1])
        test[1] = preprocessor.transform(test[1])

    train.append(preprocessor)
    return (train, valid, test)


# helper functions for StrandedProfile


def load_beds(bed_files, excl_chromosomes=None,
              incl_chromosomes=None, chromosome_lens=None, resize_width=None):
    """Load the bed files as a pandas.DataFrame

    Args:
      bed_files (list of str): a dictionary of bed files
      incl_chromosomes (list of str): list of chromosomes to keep.
          Intervals from other chromosomes are dropped.
      excl_chromosomes (list of str): list of chromosomes to exclude.
          Intervals from other chromosomes are dropped.
      chromosome_lens (dict of int): dictionary with chromosome lengths
      resize_width (int): desired interval width. The resize fixes the center
          of the interval.

    Returns:
      pandas.DataFrame with columns: chrom, start, end, task, (maybe others)
    """
    # load all the intervals
    def set_first_colnames(df, colnames):
        cols = list(df.columns)
        cols[:len(colnames)] = colnames
        df.columns = cols
        return df

    dfm = pd.concat([
        set_first_colnames(pd.read_csv(bed_file, sep='\t',
                                       header=None, usecols=[0, 1, 2]),
                           ['chrom', 'start', 'end']
                           ).assign(task=task)
        for task, bed_file in bed_files.items()])
    dfm.start = dfm.start.astype(int)
    dfm.end = dfm.end.astype(int)
    # filter the data frame columns
    dfm = dfm[['chrom', 'start', 'end', 'task']]

    # omit data outside chromosomes
    if incl_chromosomes is not None:
        dfm = dfm[dfm.chrom.isin(incl_chromosomes)]
    if excl_chromosomes is not None:
        dfm = dfm[~dfm.chrom.isin(excl_chromosomes)]

    # resize the interval
    if resize_width is not None:
        dfm = resize_interval(dfm, resize_width, ignore_strand=True)

    # Skip intervals outside of the genome
    if chromosome_lens is not None:
        n_int = len(dfm)
        dfm = dfm[(0 <= dfm.start) &
                  (dfm.start < dfm.end) &
                  (dfm.end < dfm.chrom.map(chromosome_lens))]

        if len(dfm) != n_int:
            print(f"Skipped {n_int - len(dfm)} intervals"
                  " outside of the genome size")
    # make chrom a pd.Categorical so that reference on copy doesn't occur
    dfm['chrom'] = pd.Categorical(dfm['chrom'])

    return dfm


def run_extractors(extractors, intervals, ignore_strand=False):
    if ignore_strand:
        out = sum([np.abs(ex(intervals, nan_as_zero=True))
                   for ex in extractors])[..., np.newaxis]  # keep the same dimension
    else:
        out = np.stack([np.abs(ex(intervals, nan_as_zero=True))
                        for ex in extractors], axis=-1)

    # Take the strand into account
    for i, interval in enumerate(intervals):
        if interval.strand == '-':
            out[i, :, :] = out[i, ::-1, ::-1]
    return out


@gin.configurable
def get_StrandedProfile_datasets(dataspec,
                                 peak_width=200,
                                 seq_width=None,
                                 shuffle=True,
                                 target_transformer=AppendCounts(),
                                 valid_chr=['chr2', 'chr3', 'chr4'],
                                 test_chr=['chr1', 'chr8', 'chr9'],
                                 all_chr=all_chr,
                                 exclude_chr=[],
                                 vmtouch=True):
    # test and valid shouldn't be in the valid or test sets
    for vc in valid_chr:
        assert vc not in exclude_chr
    for vc in test_chr:
        assert vc not in exclude_chr

    if isinstance(dataspec, str):
        dataspec = DataSpec.load(dataspec)

    if vmtouch:
        # use vmtouch to load all file to memory
        dataspec.touch_all_files()

    return (StrandedProfile(dataspec, peak_width,
                            seq_width=seq_width,
                            # Only include chromosomes from `all_chr`
                            incl_chromosomes=[c for c in all_chr
                                              if c not in valid_chr + test_chr + exclude_chr],
                            excl_chromosomes=valid_chr + test_chr + exclude_chr,
                            shuffle=shuffle, target_transformer=target_transformer),
            StrandedProfile(dataspec, peak_width,
                            seq_width=seq_width,
                            incl_chromosomes=valid_chr,
                            shuffle=shuffle, target_transformer=target_transformer),
            StrandedProfile(dataspec, peak_width,
                            seq_width=seq_width,
                            incl_chromosomes=test_chr,
                            shuffle=shuffle, target_transformer=target_transformer))


@gin.configurable
def get_StrandedProfile_datasets2(dataspec,
                                  peak_width=200,
                                  intervals_file=None,
                                  seq_width=None,
                                  shuffle=True,
                                  target_transformer=AppendCounts(),
                                  include_metadata=False,
                                  valid_chr=['chr2', 'chr3', 'chr4'],
                                  test_chr=['chr1', 'chr8', 'chr9'],
                                  tasks=None,
                                  taskname_first=False,
                                  exclude_chr=[],
                                  augment_interval=False,
                                  interval_augmentation_shift=200,
                                  vmtouch=True,
                                  profile_bias_pool_size=None):
    from bpnet.metrics import BPNetMetric, PeakPredictionProfileMetric, pearson_spearman
    # test and valid shouldn't be in the valid or test sets
    for vc in valid_chr:
        assert vc not in exclude_chr
    for vc in test_chr:
        assert vc not in exclude_chr

    dataspec = DataSpec.load(dataspec)
    if vmtouch:
        # use vmtouch to load all file to memory
        dataspec.touch_all_files()

    if tasks is None:
        tasks = list(dataspec.task_specs)

    if augment_interval:
        interval_transformer = IntervalAugmentor(max_shift=interval_augmentation_shift,
                                                 flip_strand=True)
    else:
        interval_transformer = None

    return (StrandedProfile(dataspec, peak_width,
                            intervals_file=intervals_file,
                            seq_width=seq_width,
                            include_metadata=include_metadata,
                            incl_chromosomes=[c for c in all_chr
                                              if c not in valid_chr + test_chr + exclude_chr],
                            excl_chromosomes=valid_chr + test_chr + exclude_chr,
                            tasks=tasks,
                            taskname_first=taskname_first,
                            shuffle=shuffle,
                            target_transformer=target_transformer,
                            interval_transformer=interval_transformer,
                            profile_bias_pool_size=profile_bias_pool_size),
            [('valid-peaks', StrandedProfile(dataspec,
                                             peak_width,
                                             intervals_file=intervals_file,
                                             seq_width=seq_width,
                                             include_metadata=include_metadata,
                                             incl_chromosomes=valid_chr,
                                             tasks=tasks,
                                             taskname_first=taskname_first,
                                             interval_transformer=interval_transformer,
                                             shuffle=shuffle, target_transformer=target_transformer,
                                             profile_bias_pool_size=profile_bias_pool_size)),
             ('train-peaks', StrandedProfile(dataspec, peak_width,
                                             intervals_file=intervals_file,
                                             seq_width=seq_width,
                                             include_metadata=include_metadata,
                                             incl_chromosomes=[c for c in all_chr
                                                               if c not in valid_chr + test_chr + exclude_chr],
                                             excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                             tasks=tasks,
                                             taskname_first=taskname_first,
                                             interval_transformer=interval_transformer,
                                             shuffle=shuffle, target_transformer=target_transformer,
                                             profile_bias_pool_size=profile_bias_pool_size)),
             ])


@gin.configurable
def get_gw_StrandedProfile_datasets(dataspec,
                                    intervals_file=None,
                                    peak_width=200,
                                    seq_width=None,
                                    shuffle=True,
                                    target_transformer=AppendCounts(),
                                    include_metadata=False,
                                    taskname_first=False,
                                    include_classes=False,
                                    only_classes=False,
                                    tasks=None,
                                    valid_chr=['chr2', 'chr3', 'chr4'],
                                    test_chr=['chr1', 'chr8', 'chr9'],
                                    exclude_chr=[],
                                    vmtouch=True,
                                    profile_bias_pool_size=None):
    # NOTE = only chromosomes from chr1-22 and chrX and chrY are considered here
    # (e.g. all other chromosomes like ChrUn... are omitted)
    from bpnet.metrics import BPNetMetric, PeakPredictionProfileMetric, pearson_spearman
    # test and valid shouldn't be in the valid or test sets
    for vc in valid_chr:
        assert vc not in exclude_chr
    for vc in test_chr:
        assert vc not in exclude_chr

    dataspec = DataSpec.load(dataspec)
    if vmtouch:
        # use vmtouch to load all file to memory
        dataspec.touch_all_files()

    if tasks is None:
        tasks = list(dataspec.task_specs)

    train = StrandedProfile(dataspec, peak_width,
                            seq_width=seq_width,
                            intervals_file=intervals_file,
                            include_metadata=include_metadata,
                            taskname_first=taskname_first,
                            include_classes=include_classes,
                            only_classes=only_classes,
                            tasks=tasks,
                            incl_chromosomes=[c for c in all_chr
                                              if c not in valid_chr + test_chr + exclude_chr],
                            excl_chromosomes=valid_chr + test_chr + exclude_chr,
                            shuffle=shuffle, target_transformer=target_transformer,
                            profile_bias_pool_size=profile_bias_pool_size)

    valid = [('train-valid-genome-wide',
              StrandedProfile(dataspec, peak_width,
                              seq_width=seq_width,
                              intervals_file=intervals_file,
                              include_metadata=include_metadata,
                              include_classes=include_classes,
                              only_classes=only_classes,
                              taskname_first=taskname_first,
                              tasks=tasks,
                              incl_chromosomes=valid_chr,
                              shuffle=shuffle, target_transformer=target_transformer,
                              profile_bias_pool_size=profile_bias_pool_size))]
    if include_classes:
        # Only use binary classification for genome-wide evaluation
        valid = valid + [('valid-genome-wide',
                          StrandedProfile(dataspec, peak_width,
                                          seq_width=seq_width,
                                          intervals_file=intervals_file,
                                          include_metadata=include_metadata,
                                          include_classes=True,
                                          only_classes=True,
                                          taskname_first=taskname_first,
                                          tasks=tasks,
                                          incl_chromosomes=valid_chr,
                                          shuffle=shuffle,
                                          target_transformer=target_transformer,
                                          profile_bias_pool_size=profile_bias_pool_size))]

    if not only_classes:
        # Add also the peak regions
        valid = valid + [
            ('valid-peaks', StrandedProfile(dataspec, peak_width,
                                            seq_width=seq_width,
                                            intervals_file=None,
                                            include_metadata=include_metadata,
                                            taskname_first=taskname_first,
                                            tasks=tasks,
                                            include_classes=False,  # dataspec doesn't contain labels
                                            only_classes=only_classes,
                                            incl_chromosomes=valid_chr,
                                            shuffle=shuffle, target_transformer=target_transformer,
                                            profile_bias_pool_size=profile_bias_pool_size)),
            ('train-peaks', StrandedProfile(dataspec, peak_width,
                                            seq_width=seq_width,
                                            intervals_file=None,
                                            include_metadata=include_metadata,
                                            taskname_first=taskname_first,
                                            tasks=tasks,
                                            include_classes=False,  # dataspec doesn't contain labels
                                            only_classes=only_classes,
                                            incl_chromosomes=[c for c in all_chr
                                                              if c not in valid_chr + test_chr + exclude_chr],
                                            excl_chromosomes=valid_chr + test_chr + exclude_chr,
                                            shuffle=shuffle, target_transformer=target_transformer,
                                            profile_bias_pool_size=profile_bias_pool_size)),
            # use the default metric for the peak sets
        ]
    return train, valid


@gin.configurable
class StrandedProfile(Dataset):

    def __init__(self, ds,
                 peak_width=200,
                 seq_width=None,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 intervals_file=None,
                 bcolz=False,
                 in_memory=False,
                 include_metadata=True,
                 taskname_first=False,
                 tasks=None,
                 include_classes=False,
                 only_classes=False,
                 shuffle=True,
                 interval_transformer=None,
                 target_transformer=None,
                 profile_bias_pool_size=None):
        """Dataset for loading the bigwigs and fastas

        Args:
          ds (bpnet.cli.schemas.DataSpec): data specification containing the
            fasta file, bed files and bigWig file paths
          chromosomes (list of str): a list of chor
          peak_width: resize the bed file to a certain width
          intervals_file: if specified, use these regions to train the model.
            If not specified, the regions are inferred from the dataspec.
          only_classes: if True, load only classes
          bcolz: If True, the bigwig/fasta files are in the genomelake bcolz format
          in_memory: If True, load the whole bcolz into memory. Only applicable when bcolz=True
          shuffle: True
          preprocessor: trained preprocessor object containing the .transform methods
        """
        if isinstance(ds, str):
            self.ds = DataSpec.load(ds)
        else:
            self.ds = ds
        self.peak_width = peak_width
        if seq_width is None:
            self.seq_width = peak_width
        else:
            self.seq_width = seq_width
        self.shuffle = shuffle
        self.intervals_file = intervals_file
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.target_transformer = target_transformer
        self.include_classes = include_classes
        self.only_classes = only_classes
        self.taskname_first = taskname_first
        if self.only_classes:
            assert self.include_classes
        self.profile_bias_pool_size = profile_bias_pool_size
        # not specified yet
        self.fasta_extractor = None
        self.bw_extractors = None
        self.bias_bw_extractors = None
        self.include_metadata = include_metadata
        self.interval_transformer = interval_transformer
        self.bcolz = bcolz
        self.in_memory = in_memory
        if not self.bcolz and self.in_memory:
            raise ValueError("in_memory option only applicable when bcolz=True")

        # Load chromosome lengths
        if self.bcolz:
            p = json.loads((Path(self.ds.fasta_file) / "metadata.json").read_text())
            self.chrom_lens = {c: v[0] for c, v in p['file_shapes'].items()}
        else:
            fa = FastaFile(self.ds.fasta_file)
            self.chrom_lens = {name: l for name, l in zip(fa.references, fa.lengths)}
            if len(self.chrom_lens) == 0:
                raise ValueError(f"no chromosomes found in fasta file: {self.ds.fasta_file}. "
                                 "Make sure the file path is correct and that the fasta index file {self.ds.fasta_file}.fai is up to date")
            del fa

        if self.intervals_file is None:
            self.dfm = load_beds(bed_files={task: task_spec.peaks
                                            for task, task_spec in self.ds.task_specs.items()
                                            if task_spec.peaks is not None},
                                 chromosome_lens=self.chrom_lens,
                                 excl_chromosomes=self.excl_chromosomes,
                                 incl_chromosomes=self.incl_chromosomes,
                                 resize_width=max(self.peak_width, self.seq_width))
            assert list(self.dfm.columns)[:4] == ["chrom", "start", "end", "task"]
            if self.shuffle:
                self.dfm = self.dfm.sample(frac=1)
            self.tsv = None
            self.dfm_tasks = None
        else:
            self.tsv = TsvReader(self.intervals_file,
                                 num_chr=False,
                                 label_dtype=int,
                                 mask_ambigous=-1,
                                 incl_chromosomes=incl_chromosomes,
                                 excl_chromosomes=excl_chromosomes,
                                 chromosome_lens=self.chrom_lens,
                                 resize_width=max(self.peak_width, self.seq_width)
                                 )
            if self.shuffle:
                self.tsv.shuffle_inplace()
            self.dfm = self.tsv.df  # use the data-frame from tsv
            self.dfm_tasks = self.tsv.get_target_names()

        # remember the tasks
        if tasks is None:
            self.tasks = list(self.ds.task_specs)
        else:
            self.tasks = tasks

        if self.bcolz and self.in_memory:
            self.fasta_extractor = ArrayExtractor(self.ds.fasta_file, in_memory=True)
            self.bw_extractors = {task: [ArrayExtractor(task_spec.pos_counts, in_memory=True),
                                         ArrayExtractor(task_spec.neg_counts, in_memory=True)]
                                  for task, task_spec in self.ds.task_specs.items() if task in self.tasks}
            self.bias_bw_extractors = {task: [ArrayExtractor(task_spec.pos_counts, in_memory=True),
                                              ArrayExtractor(task_spec.neg_counts, in_memory=True)]
                                       for task, task_spec in self.ds.bias_specs.items() if task in self.tasks}

        if self.include_classes:
            assert self.dfm_tasks is not None

        if self.dfm_tasks is not None:
            assert set(self.tasks).issubset(self.dfm_tasks)

        # setup bias maps per task
        self.task_bias_tracks = {task: [bias for bias, spec in self.ds.bias_specs.items()
                                        if task in spec.tasks]
                                 for task in self.tasks}

    def __len__(self):
        return len(self.dfm)

    def get_targets(self):
        """
        'targets'
        """
        assert self.intervals_file is not None
        return self.tsv.get_targets()

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            # Use array extractors
            if self.bcolz:
                self.fasta_extractor = ArrayExtractor(self.ds.fasta_file, in_memory=False)
                self.bw_extractors = {task: [ArrayExtractor(task_spec.pos_counts, in_memory=False),
                                             ArrayExtractor(task_spec.neg_counts, in_memory=False)]
                                      for task, task_spec in self.ds.task_specs.items() if task in self.tasks}
                self.bias_bw_extractors = {task: [ArrayExtractor(task_spec.pos_counts, in_memory=False),
                                                  ArrayExtractor(task_spec.neg_counts, in_memory=False)]
                                           for task, task_spec in self.ds.bias_specs.items() if task in self.tasks}
            else:
                # Use normal fasta/bigwig extractors
                assert not self.bcolz
                # first call
                self.fasta_extractor = FastaExtractor(self.ds.fasta_file, use_strand=True)
                self.bw_extractors = {task: [BigwigExtractor(task_spec.pos_counts),
                                             BigwigExtractor(task_spec.neg_counts)]
                                      for task, task_spec in self.ds.task_specs.items() if task in self.tasks}
                self.bias_bw_extractors = {task: [BigwigExtractor(task_spec.pos_counts),
                                                  BigwigExtractor(task_spec.neg_counts)]
                                           for task, task_spec in self.ds.bias_specs.items()}

        # Setup the intervals
        interval = Interval(self.dfm.iat[idx, 0],  # chrom
                            self.dfm.iat[idx, 1],  # start
                            self.dfm.iat[idx, 2])  # end

        # Transform the input interval (for say augmentation...)
        if self.interval_transformer is not None:
            interval = self.interval_transformer(interval)

        target_interval = resize_interval(deepcopy(interval), self.peak_width)
        seq_interval = resize_interval(deepcopy(interval), self.seq_width)

        # This only kicks in when we specify the taskname from dataspec
        # to the 3rd column. E.g. it doesn't apply when using intervals_file
        interval_from_task = self.dfm.iat[idx, 3] if self.intervals_file is None else ''

        # extract seq + tracks
        sequence = self.fasta_extractor([seq_interval])[0]

        if not self.only_classes:
            if self.taskname_first:
                cuts = {f"{task}/profile": run_extractors(self.bw_extractors[task],
                                                          [target_interval],
                                                          ignore_strand=spec.ignore_strand)[0]
                        for task, spec in self.ds.task_specs.items() if task in self.tasks}
            else:
                cuts = {f"profile/{task}": run_extractors(self.bw_extractors[task],
                                                          [target_interval],
                                                          ignore_strand=spec.ignore_strand)[0]
                        for task, spec in self.ds.task_specs.items() if task in self.tasks}

            # Add counts
            if self.target_transformer is not None:
                cuts = self.target_transformer.transform(cuts)

            # Add bias tracks
            if len(self.ds.bias_specs) > 0:

                biases = {bias_task: run_extractors(self.bias_bw_extractors[bias_task],
                                                    [target_interval],
                                                    ignore_strand=spec.ignore_strand)[0]
                          for bias_task, spec in self.ds.bias_specs.items()}

                task_biases = {f"bias/{task}/profile": np.concatenate([biases[bt]
                                                                       for bt in self.task_bias_tracks[task]],
                                                                      axis=-1)
                               for task in self.tasks}

                if self.target_transformer is not None:
                    for task in self.tasks:
                        task_biases[f'bias/{task}/counts'] = np.log(1 + task_biases[f'bias/{task}/profile'].sum(0))
                    # total_count_bias = np.concatenate([np.log(1 + x[k].sum(0))
                    #                                    for k, x in biases.items()], axis=-1)
                    # task_biases['bias/total_counts'] = total_count_bias

                if self.profile_bias_pool_size is not None:
                    for task in self.tasks:
                        task_biases[f'bias/{task}/profile'] = np.concatenate(
                            [moving_average(task_biases[f'bias/{task}/profile'], n=pool_size)
                             for pool_size in to_list(self.profile_bias_pool_size)], axis=-1)

                sequence = {"seq": sequence, **task_biases}
        else:
            cuts = dict()

        if self.include_classes:
            if self.taskname_first:
                # Get the classes from the tsv file
                classes = {f"{task}/class": self.dfm.iat[idx, i + 3]
                           for i, task in enumerate(self.dfm_tasks) if task in self.tasks}
            else:
                classes = {f"class/{task}": self.dfm.iat[idx, i + 3]
                           for i, task in enumerate(self.dfm_tasks) if task in self.tasks}
            cuts = {**cuts, **classes}

        out = {"inputs": sequence,
               "targets": cuts}

        if self.include_metadata:
            out['metadata'] = {"range": GenomicRanges(chr=target_interval.chrom,
                                                      start=target_interval.start,
                                                      end=target_interval.stop,
                                                      id=idx,
                                                      strand=(target_interval.strand
                                                              if target_interval.strand is not None
                                                              else "*"),
                                                      ),
                               "interval_from_task": interval_from_task}
        return out

# ----------------------------------------------------------------
# Binary classification datasets


class TsvReader:
    """Reads a tsv file in the following format:
    chr  start  stop  task1  task2 ...
    Args:
      tsv_file: tsv file type
      label_dtype: data type of the labels
    """

    def __init__(self, tsv_file, num_chr=False,
                 label_dtype=None,
                 mask_ambigous=None,
                 # task_prefix='task/',
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 chromosome_lens=None,
                 resize_width=None
                 ):
        """

          chromosome_lens: chromosome lengths (dictionary chrom -> len)
          resize_width: desired resize with
        """
        self.tsv_file = tsv_file
        self.num_chr = num_chr
        self.label_dtype = label_dtype
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.chromosome_lens = chromosome_lens
        self.resize_width = resize_width

        columns = list(pd.read_csv(self.tsv_file, nrows=0, sep='\t').columns)

        if not columns[0].startswith("CHR") and not columns[0].startswith("#CHR"):
            # No classes were provided
            self.columns = None
            self.tasknames = None
            skiprows = None
        else:
            # There exists a header
            self.columns = columns
            self.tasknames = list(columns)[3:]
            skiprows = [0]
            # self.tasknames = [c.replace(task_prefix, "") for c in columns if task_prefix in c]

        df_peek = pd.read_csv(self.tsv_file,
                              header=None,
                              nrows=1,
                              skiprows=skiprows,
                              comment='#',
                              sep='\t')
        self.n_tasks = df_peek.shape[1] - 3
        assert self.n_tasks >= 0

        if self.tasknames is not None:
            # make sure the task-names match
            assert self.n_tasks == len(self.tasknames)

        self.df = pd.read_csv(self.tsv_file,
                              header=None,
                              comment='#',
                              skiprows=skiprows,
                              dtype={i: d
                                     for i, d in enumerate([str, int, int] +
                                                           [self.label_dtype] * self.n_tasks)},
                              sep='\t')
        if self.num_chr and self.df.iloc[0][0].startswith("chr"):
            self.df[0] = self.df[0].str.replace("^chr", "")
        if not self.num_chr and not self.df.iloc[0][0].startswith("chr"):
            self.df[0] = "chr" + self.df[0]

        if mask_ambigous is not None and self.n_tasks > 0:
            # exclude regions where only ambigous labels are present
            self.df = self.df[~np.all(self.df.iloc[:, 3:] == mask_ambigous, axis=1)]

        # omit data outside chromosomes
        if incl_chromosomes is not None:
            self.df = self.df[self.df[0].isin(incl_chromosomes)]
        if excl_chromosomes is not None:
            self.df = self.df[~self.df[0].isin(excl_chromosomes)]

        # make the chromosome name a categorical variable
        self.df[0] = pd.Categorical(self.df[0])

        # Skip intervals outside of the genome
        if self.chromosome_lens is not None:
            n_int = len(self.df)
            center = (self.df[1] + self.df[2]) // 2
            valid_seqs = ((center > self.resize_width // 2 + 1) &
                          (center < self.df[0].map(chromosome_lens) - self.resize_width // 2 - 1))
            self.df = self.df[valid_seqs]

            if len(self.df) != n_int:
                print(f"Skipped {n_int - len(self.df)} intervals"
                      " outside of the genome size")

    def __getitem__(self, idx):
        """Returns (pybedtools.Interval, labels)
        """
        # TODO - speedup using: iat[idx, .]
        # interval = Interval(self.dfm.iat[idx, 0],  # chrom
        #                     self.dfm.iat[idx, 1],  # start
        #                     self.dfm.iat[idx, 2])  # end
        # intervals = [interval]
        # task = self.dfm.iat[idx, 3]  # task
        row = self.df.iloc[idx]
        interval = Interval(row[0], row[1], row[2])

        if self.n_tasks == 0:
            labels = {}
        else:
            labels = row.iloc[3:].values.astype(self.label_dtype)
        return interval, labels

    def __len__(self):
        return len(self.df)

    def get_target_names(self):
        return self.tasknames

    def get_targets(self):
        return self.df.iloc[:, 3:].values.astype(self.label_dtype)

    def shuffle_inplace(self):
        """Shuffle the interval
        """
        self.df = self.df.sample(frac=1)


@gin.configurable
class SeqClassification(Dataset):
    """
    Args:
        intervals_file: bed3+<columns> file containing intervals+labels
        fasta_file: file path; Genome sequence
        label_dtype: label data type
        num_chr_fasta: if True, the tsv-loader will make sure that the chromosomes
          don't start with chr
    """

    def __init__(self, intervals_file,
                 fasta_file,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 num_chr_fasta=False):
        self.num_chr_fasta = num_chr_fasta
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes

        self.tsv = TsvReader(self.intervals_file,
                             num_chr=self.num_chr_fasta,
                             label_dtype=int,
                             mask_ambigous=-1,
                             incl_chromosomes=incl_chromosomes,
                             excl_chromosomes=excl_chromosomes,
                             )
        self.fasta_extractor = None

    def __len__(self):
        return len(self.tsv)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)

        interval, labels = self.tsv[idx]

        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]))

        return {
            "inputs": {"seq": seq},
            "targets": labels,
            "metadata": {
                "ranges": GenomicRanges(interval.chrom, interval.start, interval.stop, str(idx))
            }
        }

    def get_targets(self):
        return self.tsv.get_targets()


@gin.configurable
def chrom_dataset(dataset_cls, valid_chr=valid_chr, holdout_chr=test_chr):
    return (dataset_cls(excl_chromosomes=valid_chr + holdout_chr),
            dataset_cls(incl_chromosomes=valid_chr))


class ActivityDataset(Dataset):
    """
    Args:
        intervals_file: bed4 file containing chrom  start  end  name
        fasta_file: file path; Genome sequence
        label_dtype: label data type
        num_chr_fasta: if True, the tsv-loader will make sure that the chromosomes
          don't start with chr
    """

    def __init__(self, intervals_file,
                 fasta_file,
                 bigwigs,
                 track_width=2000,
                 incl_chromosomes=None,
                 excl_chromosomes=None,
                 num_chr_fasta=False):
        self.num_chr_fasta = num_chr_fasta
        self.intervals_file = intervals_file
        self.fasta_file = fasta_file
        self.bigwigs = bigwigs
        self.incl_chromosomes = incl_chromosomes
        self.excl_chromosomes = excl_chromosomes
        self.track_width = track_width

        self.tsv = BedDataset(self.intervals_file,
                              num_chr=self.num_chr_fasta,
                              bed_columns=4,
                              ignore_targets=True,
                              incl_chromosomes=incl_chromosomes,
                              excl_chromosomes=excl_chromosomes,
                              )
        self.fasta_extractor = None
        self.bigwig_extractors = None

    def __len__(self):
        return len(self.tsv)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaExtractor(self.fasta_file)
            self.bigwig_extractors = {a: [BigwigExtractor(f) for f in self.bigwigs[a]]
                                      for a in self.bigwigs}

        interval, labels = self.tsv[idx]
        interval = resize_interval(interval, 1000)
        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]))

        interval_wide = resize_interval(deepcopy(interval), self.track_width)

        return {
            "inputs": {"seq": seq},
            "targets": {a: sum([e([interval_wide])[0] for e in self.bigwig_extractors[a]]).sum()
                        for a in self.bigwig_extractors},
            "metadata": {
                "ranges": GenomicRanges(interval.chrom, interval.start, interval.stop, str(idx)),
                "ranges_wide": GenomicRanges.from_interval(interval_wide),
                "name": interval.name
            }
        }

    def get_targets(self):
        return self.tsv.get_targets()


def get(name):
    return get_from_module(name, globals())
