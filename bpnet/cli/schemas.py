"""
Schemas describing the following configuration YAML files:
- dataspec.yml
- modisco.yml
"""
from __future__ import absolute_import
from __future__ import print_function
from copy import deepcopy
import os
from collections import OrderedDict
import related
from kipoi_utils.external.related.mixins import RelatedConfigMixin, RelatedLoadSaveMixin
from kipoi_utils.external.related.fields import AnyField
from kipoi_utils.external.related.fields import StrSequenceField
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@related.mutable(strict=False)
class TaskSpec(RelatedConfigMixin):
    task = related.StringField()
    # Bigwig file paths to tracks (e.g. ChIP-nexus read counts for positive and negative strand)
    tracks = StrSequenceField(str)
    peaks = related.StringField(None, required=False)

    # if True the tracks will be simply added together
    # instead of predicting them separately
    sum_tracks = related.BooleanField(False, required=False)

    # One could in the future add the assay type
    # assay = related.StringField(None, required=False)

    def load_counts(self, intervals, use_strand=True, progbar=False):
        import numpy as np
        # from genomelake.extractors import BigwigExtractor
        from bpnet.extractors import StrandedBigWigExtractor
        tracks = [StrandedBigWigExtractor(track,
                                          use_strand=use_strand,
                                          nan_as_zero=True).extract(intervals, progbar=progbar)
                  for track in self.tracks]
        if self.sum_tracks:
            return sum(tracks)[..., np.newaxis]  # keep the same dimension
        else:
            if use_strand:
                neg_strand = np.array([s.strand == '-' for s in intervals]).reshape((-1, 1))
                # NOTE: this assumes that there are exactly 2 strands
                if len(tracks) != 2:
                    raise ValueError(f"use_strand is True. However, there are {len(tracks)} "
                                     "and not 2 as expected")
                pos_counts = tracks[0]
                neg_counts = tracks[1]
                return np.stack([np.where(neg_strand, neg_counts, pos_counts),
                                 np.where(neg_strand, pos_counts, neg_counts)], axis=-1)
            else:
                return np.stack(tracks, axis=-1)

    def list_all_files(self, include_peaks=False):
        """List all file paths specified
        """
        files = self.tracks
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
        obj.tracks = [os.path.abspath(track) for track in self.tracks]
        obj.peaks = peaks_abspath
        return obj


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

    def get_bws(self):
        return OrderedDict([(task, task_spec.tracks)
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

    def get_all_regions(self):
        """Get all the regions
        """
        from pybedtools import BedTool
        regions = []
        for task, task_spec in self.task_specs.items():
            if task_spec.peaks is not None:
                regions += list(BedTool(task_spec.peaks))
        return regions


# --------------------------------------------
# hparams
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
