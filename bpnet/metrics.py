from scipy.stats import pearsonr, spearmanr
import logging
import matplotlib.pyplot as plt
from bpnet.utils import read_pkl
from keras.models import load_model
from bpnet.utils import _listify, create_tf_session
from bpnet.datasets import chip_exo_nexus
from bpnet.stats import permute_array
from bpnet.cli.schemas import DataSpec, HParams
from bpnet.functions import softmax, mean
from concise.utils.helper import write_json
from concise.eval_metrics import auprc, auc, accuracy
import os
import json
from tqdm import tqdm
import matplotlib
import pandas as pd
import numpy as np
from collections import OrderedDict
import concise
import gin

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Metric helpers
def average_profile(pe):
    tasks = list(pe)
    binsizes = list(pe[tasks[0]])
    return {binsize: {"auprc": mean([pe[task][binsize]['auprc'] for task in tasks])}
            for binsize in binsizes}


def average_counts(pe):
    tasks = list(pe)
    metrics = list(pe[tasks[0]])
    return {metric: mean([pe[task][metric] for task in tasks])
            for metric in metrics}


def bin_counts_max(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout


def bin_counts_summary(x, binsize=2, fn=np.max):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = np.apply_along_axis(fn, 1, x[:, (binsize * i):(binsize * (i + 1)), :])
    return xout


def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 2, 4, 10]):
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold

    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)

    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = []
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        # TODO - I used to have bin_counts_max over here instead of bin_counts_sum
        try:
            res = auprc(y_true,
                        np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true,
                               np.ravel(bin_counts_max(yp_random, binsize)))
        except ValueError:
            res = np.nan
            res_random = np.nan

        out.append({"binsize": binsize,
                    "auprc": res,
                    "random_auprc": res_random,
                    "n_positives": n_positives,
                    "frac_ambigous": frac_ambigous,
                    "imbalance": imbalance
                    })

    return pd.DataFrame.from_dict(out)

# --------------------------------------------


@gin.configurable
class BPNetSeparatePostproc:

    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, y_true, preds):
        profile_preds = {task: softmax(preds[task_i])
                         for task_i, task in enumerate(self.tasks)}
        count_preds = {task: preds[len(self.tasks) + task_i].sum(axis=-1)
                       for task_i, task in enumerate(self.tasks)}
        profile_true = {task: y_true[f'profile/{task}']
                        for task in self.tasks}
        counts_true = {task: y_true[f'counts/{task}'].sum(axis=-1)
                       for task in self.tasks}
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetSinglePostproc:
    """Example where we predict a single track
    """

    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, y_true, preds):
        profile_preds = {task: preds[task_i] / preds[task_i].sum(axis=-2, keepdims=True)
                         for task_i, task in enumerate(self.tasks)}
        count_preds = {task: np.log(1 + preds[task_i].sum(axis=(-2, -1)))
                       for task_i, task in enumerate(self.tasks)}

        profile_true = {task: y_true[f'profile/{task}']
                        for task in self.tasks}
        counts_true = {task: np.log(1 + y_true[f'profile/{task}'].sum(axis=(-2, -1)))
                       for task in self.tasks}
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetMetric:
    """BPNet metrics when the net is predicting counts and profile separately
    """

    def __init__(self, tasks, count_metric,
                 profile_metric=None,
                 postproc_fn=None):
        """

        Args:
          tasks: tasks
          count_metric: count evaluation metric
          profile_metric: profile evaluation metric
        """
        self.tasks = tasks
        self.count_metric = count_metric
        self.profile_metric = profile_metric

        if postproc_fn is None:
            self.postproc_fn = BPNetSeparatePostproc(tasks=self.tasks)
        else:
            self.postproc_fn = postproc_fn

    def __call__(self, y_true, preds):
        # extract the profile and count predictions

        y_true, preds = self.postproc_fn(y_true, preds)

        out = {}
        out["counts"] = {task: self.count_metric(y_true['counts'][task],
                                                 preds['counts'][task])
                         for task in self.tasks}
        out["counts"]['avg'] = average_counts(out["counts"])

        out["avg"] = {"counts": out["counts"]['avg']}  # new system compatibility
        if self.profile_metric is not None:
            out["profile"] = {task: self.profile_metric(y_true['profile'][task],
                                                        preds['profile'][task])
                              for task in self.tasks}
            out["profile"]['avg'] = average_profile(out["profile"])
            out["avg"]['profile'] = out["profile"]['avg']
        return out


@gin.configurable
class BPNetMetricSingleProfile:
    """BPNet metrics when the net is predicting the total counts + profile at the same time
    """

    def __init__(self, count_metric,
                 profile_metric=None):
        """

        Args:
          tasks: tasks
          count_metric: count evaluation metric
          profile_metric: profile evaluation metric
        """
        # self.tasks = tasks
        self.count_metric = count_metric
        self.profile_metric = profile_metric

    def __call__(self, y_true, preds):
        # extract the profile and count predictions
        out = {}

        # sum across positions + strands
        out["counts"] = self.count_metric(np.log(1 + y_true.sum(axis=(-2, -1))),
                                          np.log(1 + preds.sum(axis=(-2, -1))))

        if self.profile_metric is not None:
            out["profile"] = self.profile_metric(y_true, preds)
        return out


@gin.configurable
class PeakPredictionProfileMetric:

    def __init__(self, pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 10]):

        self.pos_min_threshold = pos_min_threshold
        self.neg_max_threshold = neg_max_threshold
        self.required_min_pos_counts = required_min_pos_counts
        self.binsizes = binsizes

    def __call__(self, y_true, y_pred):
        out = eval_profile(y_true, y_pred,
                           pos_min_threshold=self.pos_min_threshold,
                           neg_max_threshold=self.neg_max_threshold,
                           required_min_pos_counts=self.required_min_pos_counts,
                           binsizes=self.binsizes)

        return {f"binsize={k}": v for k, v in out.set_index("binsize").to_dict("index").items()}


default_peak_pred_metric = PeakPredictionProfileMetric(pos_min_threshold=0.015,
                                                       neg_max_threshold=0.005,
                                                       required_min_pos_counts=2.5,
                                                       binsizes=[1, 10])


@gin.configurable
def pearson_spearman(yt, yp):
    return {"pearsonr": pearsonr(yt, yp)[0],
            "spearmanr": spearmanr(yt, yp)[0]}
