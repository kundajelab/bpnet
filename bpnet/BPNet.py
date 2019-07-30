import os
from kipoi_utils.data_utils import get_dataset_item, numpy_collate_concat
from kipoi_utils.utils import unique_list
import keras.backend as K
import matplotlib.ticker as ticker
from bpnet.functions import softmax
from genomelake.extractors import FastaExtractor
from keras.models import load_model
from collections import OrderedDict
from bpnet.plot.tracks import plot_tracks, filter_tracks
from bpnet.extractors import extract_seq
from bpnet.data import numpy_minibatch, nested_numpy_minibatch
from tqdm import tqdm
from bpnet.utils import flatten_list
from concise.utils.plot import seqlogo
from bpnet.functions import mean
from concise.preprocessing import encodeDNA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from genomelake.extractors import BigwigExtractor
import pyBigWig
from pysam import FastaFile
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO - remove the fasta file


class BPNetSeqModel:
    """BPNet based on SeqModel
    """

    def __init__(self, seqmodel, fasta_file=None):
        self.seqmodel = seqmodel
        self.tasks = self.seqmodel.tasks
        self.fasta_file = fasta_file
        # TODO - add some sanity checks (profile head available etc)

    @classmethod
    def from_mdir(cls, model_dir):
        from bpnet.seqmodel import SeqModel
        # TODO - figure out also the fasta_file if present (from dataspec)
        # from bpnet.cli.schemas import DataSpec
        # ds_path = os.path.join(model_dir, "dataspec.yaml")
        # if os.path.exists(ds_path):
        #     ds = DataSpec.load(ds_path)
        #     fasta_file = ds.fasta_file
        return cls(SeqModel.from_mdir(model_dir))

    def input_seqlen(self):
        return self.seqmodel.seqlen

    def predict(self, seq, batch_size=512):
        """Make model prediction

        Args:
          seq: numpy array of one-hot-encoded array of sequences
          batch_size: batch size

        Returns:
          dictionary key=task and value=prediction for the task
        """

        preds = self.seqmodel.predict(seq, batch_size=batch_size)
        return {task: preds[f'{task}/profile'] * np.exp(preds[f'{task}/counts'][:, np.newaxis])
                for task in self.seqmodel.tasks}

    def imp_score_all(self, seq, method='deeplift', aggregate_strand=True, batch_size=512,
                      pred_summaries=['weighted', 'count']):
        """Compute all importance scores

        Args:
          seq: one-hot encoded DNA sequences
          method: 'grad', 'deeplift' or 'ism'
          aggregate_strands: if True, the average importance scores across strands will be returned
          batch_size: batch size when computing the importance scores

        Returns:
          dictionary with keys: {task}/{pred_summary}/{strand_i} or {task}/{pred_summary}
          and values with the same shape as `seq` corresponding to importance scores
        """
        assert aggregate_strand

        imp_scores = self.seqmodel.imp_score_all(seq, method=method)

        return {f"{task}/" + self._get_old_imp_score_name(pred_summary): imp_scores[f"{task}/{pred_summary}"]
                for task in self.seqmodel.tasks
                for pred_summary in ['profile/wn', 'counts/pre-act']}

    def _get_old_imp_score_name(self, s):
        s2s = {"profile/wn": 'weighted', 'counts/pre-act': 'count'}
        return s2s[s]

    def sim_pred(self, central_motif, side_motif=None, side_distances=[], repeat=128, importance=[]):
        """
        Args:
          importance: list of importance scores
        """
        from bpnet.simulate import generate_seq, average_profiles, flatten
        batch_size = repeat
        seqlen = self.seqmodel.seqlen
        tasks = self.seqmodel.tasks

        # simulate sequence
        seqs = encodeDNA([generate_seq(central_motif, side_motif=side_motif,
                                       side_distances=side_distances, seqlen=seqlen)
                          for i in range(repeat)])

        # get predictions
        scaled_preds = self.predict(seqs, batch_size=batch_size)

        if importance:
            # get the importance scores (compute only the profile and counts importance)
            imp_scores_all = self.seqmodel.imp_score_all(seqs, intp_pattern=['*/profile/wn', '*/counts/pre-act'])
            imp_scores = {t: {self._get_old_imp_score_name(imp_score_name): seqs * imp_scores_all[f'{t}/{imp_score_name}']
                              for imp_score_name in importance}
                          for t in tasks}

            # merge and aggregate the profiles
            out = {"imp": imp_scores, "profile": scaled_preds}
        else:
            out = {"profile": scaled_preds}
        return average_profiles(flatten(out, "/"))

    def get_seq(self, intervals, variants=None, use_strand=False):
        """Get the one-hot-encoded sequence used to make model predictions and
        optionally augment it with the variants
        """
        if variants is not None:
            if use_strand:
                raise NotImplementedError("use_strand=True not implemented for variants")
            # Augment the intervals using a variant
            if not isinstance(variants, list):
                variants = [variants] * len(intervals)
            else:
                assert len(variants) == len(intervals)
            seq = np.stack([extract_seq(interval, variant, self.fasta_file, one_hot=True)
                            for variant, interval in zip(variants, intervals)])
        else:
            variants = [None] * len(intervals)
            seq = FastaExtractor(self.fasta_file, use_strand=use_strand)(intervals)
        return seq

    def predict_all(self, seq, imp_method='grad', batch_size=512, pred_summaries=['weighted', 'count']):
        """Make model prediction based
        """
        preds = self.predict(seq, batch_size=batch_size)

        if imp_method is not None:
            imp_scores = self.imp_score_all(seq, method=imp_method, aggregate_strand=True,
                                            batch_size=batch_size, pred_summaries=pred_summaries)
        else:
            imp_scores = dict()

        out = [dict(
            seq=get_dataset_item(seq, i),
            # interval=intervals[i],
            pred=get_dataset_item(preds, i),
            # TODO - shall we call it hyp_imp score or imp_score?
            imp_score=get_dataset_item(imp_scores, i),
        ) for i in range(len(seq))]
        return out

    def predict_intervals(self, intervals,
                          variants=None,
                          imp_method='grad',
                          use_strand=False,
                          batch_size=512):
        """
        Args:
          intervals: list of pybedtools.Interval
          variant: a single instance or a list bpnet.extractors.Variant
          pred_summary: 'mean' or 'max', summary function name for the profile gradients
          compute_grads: if False, skip computing gradients
        """
        # TODO - support also other importance scores
        seq = self.get_seq(intervals, variants, use_strand=use_strand)

        preds = self.predict_all(seq, imp_method, batch_size)

        # append intervals
        for i in range(len(seq)):
            preds[i]['interval'] = intervals[i]
            if variants is not None:
                preds[i]['variant'] = variants[i]
        return preds

    def plot_intervals(self, intervals, ds=None, variants=None,
                       seqlets=[],
                       pred_summary='weighted',
                       imp_method='grad',
                       batch_size=128,
                       # ylim=None,
                       xlim=None,
                       # seq_height=1,
                       rotate_y=0,
                       add_title=True,
                       fig_height_per_track=2,
                       same_ylim=False,
                       fig_width=20):
        """Plot predictions

        Args:
          intervals: list of pybedtools.Interval
          variant: a single instance or a list of bpnet.extractors.Variant
          ds: DataSpec. If provided, the ground truth will be added to the plot
          pred_summary: 'mean' or 'max', summary function name for the profile gradients
        """
        out = self.predict_intervals(intervals,
                                     variants=variants,
                                     imp_method=imp_method,
                                     # pred_summary=pred_summary,
                                     batch_size=batch_size)
        figs = []
        if xlim is None:
            xmin = 0
        else:
            xmin = xlim[0]
        shifted_seqlets = [s.shift(-xmin) for s in seqlets]

        for i in range(len(out)):
            pred = out[i]
            interval = out[i]['interval']

            if ds is not None:
                obs = {task: ds.task_specs[task].load_counts([interval])[0] for task in self.tasks}
            else:
                obs = None

            title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', ''))

            # handle the DNase case
            if isinstance(pred['seq'], dict):
                seq = pred['seq']['seq']
            else:
                seq = pred['seq']

            if obs is None:
                # TODO - simplify?
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Imp profile", pred['imp_score'][f"{task}/{pred_summary}"] * seq),
                    # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                ] for task_idx, task in enumerate(self.tasks)]))
            else:
                viz_dict = OrderedDict(flatten_list([[
                    (f"{task} Pred", pred['pred'][task]),
                    (f"{task} Obs", obs[task]),
                    (f"{task} Imp profile", pred['imp_score'][f"{task}/{pred_summary}"] * seq),
                    # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
                ] for task_idx, task in enumerate(self.tasks)]))

            if add_title:
                title = "{i.chrom}:{i.start}-{i.end}, {i.name} {v}".format(i=interval, v=pred.get('variant', '')),
            else:
                title = None

            if same_ylim:
                fmax = {feature: max([np.abs(viz_dict[f"{task} {feature}"]).max() for task in self.tasks])
                        for feature in ['Pred', 'Imp profile', 'Obs']}

                ylim = []
                for k in viz_dict:
                    f = k.split(" ", 1)[1]
                    if "Imp" in f:
                        ylim.append((-fmax[f], fmax[f]))
                    else:
                        ylim.append((0, fmax[f]))
            else:
                ylim = None
            fig = plot_tracks(filter_tracks(viz_dict, xlim),
                              seqlets=shifted_seqlets,
                              title=title,
                              fig_height_per_track=fig_height_per_track,
                              rotate_y=rotate_y,
                              fig_width=fig_width,
                              ylim=ylim,
                              legend=True)
            figs.append(fig)
        return figs

    # TODO also allow imp_scores
    def export_bw(self,
                  intervals,
                  output_dir,
                  # pred_summary='weighted',
                  imp_method='grad',
                  batch_size=512,
                  scale_importance=False,
                  chromosomes=None):
        """Export predictions and model importances to big-wig files

        Args:
          intervals: list of genomic intervals
          output_dir: output directory

          batch_size:
          scale_importance: if True, multiple the importance scores by the predicted count value
          chromosomes: a list of chromosome names consisting a genome
        """
        #          pred_summary: which operation to use for the profile gradients
        logger.info("Get model predictions and importance scores")
        out = self.predict_intervals(intervals,
                                     imp_method=imp_method,
                                     batch_size=batch_size)

        logger.info("Setup bigWigs for writing")
        # Get the genome lengths
        fa = FastaFile(self.fasta_file)
        if chromosomes is None:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths)])
        else:
            genome = OrderedDict([(c, l) for c, l in zip(fa.references, fa.lengths) if c in chromosomes])
        fa.close()

        output_feats = ['preds.pos', 'preds.neg', 'importance.profile', 'importance.counts']

        # make sure the intervals are in the right order
        first_chr = list(np.unique(np.array([interval.chrom for interval in intervals])))
        last_chr = [c for c, l in genome.items() if c not in first_chr]
        genome = [(c, genome[c]) for c in first_chr + last_chr]

        # open bigWigs for writing
        bws = {}
        for task in self.tasks:
            bws[task] = {}
            for feat in output_feats:
                bw_preds_pos = pyBigWig.open(f"{output_dir}/{task}.{feat}.bw", "w")
                bw_preds_pos.addHeader(genome)
                bws[task][feat] = bw_preds_pos

        def add_entry(bw, arr, interval, start_idx=0):
            """Macro for adding an entry to the bigwig file

            Args:
              bw: pyBigWig file handle
              arr: 1-dimensional numpy array
              interval: genomic interval pybedtools.Interval
              start_idx: how many starting values in the array to skip
            """
            assert arr.ndim == 1
            assert start_idx < len(arr)

            if interval.stop - interval.start != len(arr):
                logger.error(f"interval.stop - interval.start ({interval.stop - interval.start})!= len(arr) ({len(arr)})")
                logger.error(f"Skipping the entry: {interval}")
                return
            bw.addEntries(interval.chrom, interval.start + start_idx,
                          values=arr[start_idx:],
                          span=1, step=1)

        # interval logic to handle overlapping intervals
        #   assumption: all intervals are sorted w.r.t the start coordinate
        #   strategy: don't write values at the same position twice (skip those)
        #
        # graphical representation:
        # ...     ]    - prev_stop
        #      [     ]   - new interval 1
        #         [  ]   - added chunk from interval 1
        #   [  ]         - new interval 2 - skip
        #          [   ] - new interval 3, fully add

        logger.info("Writing to bigWigs")
        prev_stop = None   # Keep track of what the previous interval already covered
        prev_chrom = None
        for i in tqdm(range(len(out))):
            interval = out[i]['interval']

            if prev_chrom != interval.chrom:
                # Encountered a new chromosome
                prev_stop = 0  # Restart the end-counter
                prev_chrom = interval.chrom

            if prev_stop >= interval.stop:
                # Nothing new to add to that range
                continue
            start_idx = max(prev_stop - interval.start, 0)

            for tid, task in enumerate(self.tasks):
                # Write predictions
                preds = out[i]['pred'][task]
                add_entry(bws[task]['preds.pos'], preds[:, 0],
                          interval, start_idx)
                add_entry(bws[task]['preds.neg'], preds[:, 1],
                          interval, start_idx)

                # Get the importance scores
                seq = out[i]['seq']
                hyp_imp = out[i]['imp_score']

                if scale_importance:
                    si_profile = preds.sum()  # Total number of counts in the region
                    si_counts = preds.sum()
                else:
                    si_profile = 1
                    si_counts = 1

                # profile - multipl
                add_entry(bws[task]['importance.profile'],
                          hyp_imp[f'{task}/weighted'][seq.astype(bool)] * si_profile,
                          interval, start_idx)
                add_entry(bws[task]['importance.counts'],
                          hyp_imp[f'{task}/count'][seq.astype(bool)] * si_counts,
                          interval, start_idx)

            prev_stop = max(interval.stop, prev_stop)

        logger.info("Done writing. Closing bigWigs")
        # Close all the big-wig files
        for task in self.tasks:
            for feat in output_feats:
                bws[task][feat].close()
        logger.info(f"Done! Files located at: {output_dir}")
