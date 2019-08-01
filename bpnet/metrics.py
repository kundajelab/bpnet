import sklearn.metrics as skm
import logging
import matplotlib.pyplot as plt
from bpnet.utils import read_pkl
from keras.models import load_model
from bpnet.utils import _listify, create_tf_session
from bpnet.stats import permute_array
from bpnet.functions import softmax, mean
import os
import json
from tqdm import tqdm
import matplotlib
import pandas as pd
import numpy as np
from collections import OrderedDict
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
        except Exception:
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


# --------------------------------------------
# Combined metrics


@gin.configurable
class BootstrapMetric:
    def __init__(self, metric, n):
        """
        Args:
          metric: a function accepting (y_true and y_pred) and
             returning the evaluation result
          n: number of bootstrap samples to draw
        """
        self.metric = metric
        self.n = n

    def __call__(self, y_true, y_pred):
        outl = []
        for i in range(self.n):
            bsamples = (
                pd.Series(np.arange(len(y_true))).sample(frac=1, replace=True).values
            )
            outl.append(self.metric(y_true[bsamples], y_pred[bsamples]))
        return outl


@gin.configurable
class MetricsList:
    """Wraps a list of metrics into a single metric returning a list"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [metric(y_true, y_pred) for metric in self.metrics]


@gin.configurable
class MetricsDict:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return {k: metric(y_true, y_pred) for k, metric in self.metrics.items()}


@gin.configurable
class MetricsTupleList:
    """Wraps a dictionary of metrics into a single metric returning a dictionary"""

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return [(k, metric(y_true, y_pred)) for k, metric in self.metrics]


@gin.configurable
class MetricsOrderedDict:
    """Wraps a OrderedDict/tuple list of metrics into a single metric
    returning an OrderedDict
    """

    def __init__(self, metrics):
        self.metrics = metrics

    def __call__(self, y_true, y_pred):
        return OrderedDict([(k, metric(y_true, y_pred)) for k, metric in self.metrics])


@gin.configurable
class MetricsMultiTask:
    """Run the same metric across multiple tasks
    """

    def __init__(self, metrics, task_names=None):
        self.metrics = metrics
        self.task_names = task_names

    def __call__(self, y_true, y_pred):
        n_tasks = y_true.shape[1]
        if self.task_names is None:
            self.task_names = [i for i in range(n_tasks)]
        else:
            assert len(self.task_names) == n_tasks
        return OrderedDict([(task, self.metrics(y_true[:, i], y_pred[:, i]))
                            for i, task in enumerate(self.task_names)])


@gin.configurable
class MetricsAggregated:

    def __init__(self,
                 metrics,
                 agg_fn={"mean": np.mean, "std": np.std},
                 prefix=""):
        self.metrics
        self.agg_fn = agg_fn
        self.prefix = prefix

    def __call__(self, y_true, y_pred):
        out = self.metrics(y_true, y_pred)
        # TODO - generalize using numpy_collate?
        m = np.array(list(out.values()))
        return {self.prefix + k: fn(m) for k, fn in self.agg_fn}


@gin.configurable
class MetricsConcise:

    def __init__(self, metrics):
        import concise
        self.metrics_dict = OrderedDict([(m, concise.eval_metrics.get(m))
                                         for m in metrics])

    def __call__(self, y_true, y_pred):
        return OrderedDict([(m, fn(y_true, y_pred))
                            for m, fn in self.metrics_dict.items()])


# -----------------------------
# Binary classification
# Metric helpers
MASK_VALUE = -1
# Binary classification


def _mask_nan(y_true, y_pred):
    mask_array = ~np.isnan(y_true)
    if np.any(np.isnan(y_pred)):
        print("WARNING: y_pred contains {0}/{1} np.nan values. removing them...".
              format(np.sum(np.isnan(y_pred)), y_pred.size))
        mask_array = np.logical_and(mask_array, ~np.isnan(y_pred))
    return y_true[mask_array], y_pred[mask_array]


def _mask_value(y_true, y_pred, mask=MASK_VALUE):
    mask_array = y_true != mask
    return y_true[mask_array], y_pred[mask_array]


def _mask_value_nan(y_true, y_pred, mask=MASK_VALUE):
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return _mask_value(y_true, y_pred, mask)


@gin.configurable
def n_positive(y_true, y_pred):
    return y_true.sum()


@gin.configurable
def n_negative(y_true, y_pred):
    return (1 - y_true).sum()


@gin.configurable
def frac_positive(y_true, y_pred):
    return y_true.mean()


@gin.configurable
def accuracy(y_true, y_pred, round=True):
    """Classification accuracy
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.accuracy_score(y_true, y_pred)


@gin.configurable
def auc(y_true, y_pred, round=True):
    """Area under the ROC curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)

    if round:
        y_true = y_true.round()
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan
    return skm.roc_auc_score(y_true, y_pred)


@gin.configurable
def auprc(y_true, y_pred):
    """Area under the precision-recall curve
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    return skm.average_precision_score(y_true, y_pred)


@gin.configurable
def mcc(y_true, y_pred, round=True):
    """Matthews correlation coefficient
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.matthews_corrcoef(y_true, y_pred)


@gin.configurable
def f1(y_true, y_pred, round=True):
    """F1 score: `2 * (p * r) / (p + r)`, where p=precision and r=recall.
    """
    y_true, y_pred = _mask_value_nan(y_true, y_pred)
    if round:
        y_true = np.round(y_true)
        y_pred = np.round(y_pred)
    return skm.f1_score(y_true, y_pred)


@gin.configurable
def cat_acc(y_true, y_pred):
    """Categorical accuracy
    """
    return np.mean(y_true.argmax(axis=1) == y_pred.argmax(axis=1))


classification_metrics = [
    ("auPR", auprc),
    ("auROC", auc),
    ("accuracy", accuracy),
    ("n_positive", n_positive),
    ("n_negative", n_negative),
    ("frac_positive", frac_positive),
]


@gin.configurable
class ClassificationMetrics:
    """All classification metrics
    """
    cls_metrics = classification_metrics

    def __init__(self):
        self.classification_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        return self.classification_metric(y_true, y_pred)
# TODO - add gin macro for a standard set of classification and regession metrics


# --------------------------------------------
# Regression

@gin.configurable
def cor(y_true, y_pred):
    """Compute Pearson correlation coefficient.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]


@gin.configurable
def kendall(y_true, y_pred, nb_sample=100000):
    """Kendall's tau coefficient, Kendall rank correlation coefficient
    """
    from scipy.stats import kendalltau
    y_true, y_pred = _mask_nan(y_true, y_pred)
    if len(y_true) > nb_sample:
        idx = np.arange(len(y_true))
        np.random.shuffle(idx)
        idx = idx[:nb_sample]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    return kendalltau(y_true, y_pred)[0]


@gin.configurable
def mad(y_true, y_pred):
    """Median absolute deviation
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


@gin.configurable
def rmse(y_true, y_pred):
    """Root mean-squared error
    """
    return np.sqrt(mse(y_true, y_pred))


@gin.configurable
def rrmse(y_true, y_pred):
    """1 - rmse
    """
    return 1 - rmse(y_true, y_pred)


@gin.configurable
def mse(y_true, y_pred):
    """Mean squared error
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return ((y_true - y_pred) ** 2).mean(axis=None)


@gin.configurable
def ermse(y_true, y_pred):
    """Exponentiated root-mean-squared error
    """
    return 10**np.sqrt(mse(y_true, y_pred))


@gin.configurable
def var_explained(y_true, y_pred):
    """Fraction of variance explained.
    """
    y_true, y_pred = _mask_nan(y_true, y_pred)
    var_resid = np.var(y_true - y_pred)
    var_y_true = np.var(y_true)
    return 1 - var_resid / var_y_true


@gin.configurable
def pearsonr(y_true, y_pred):
    from scipy.stats import pearsonr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return pearsonr(y_true, y_pred)[0]


@gin.configurable
def spearmanr(y_true, y_pred):
    from scipy.stats import spearmanr
    y_true, y_pred = _mask_nan(y_true, y_pred)
    return spearmanr(y_true, y_pred)[0]


@gin.configurable
def pearson_spearman(yt, yp):
    return {"pearsonr": pearsonr(yt, yp),
            "spearmanr": spearmanr(yt, yp)}


regression_metrics = [
    ("mse", mse),
    ("var_explained", var_explained),
    ("pearsonr", pearsonr),  # pearson and spearman correlation
    ("spearmanr", spearmanr),
    ("mad", mad),  # median absolute deviation
]


@gin.configurable
class RegressionMetrics:
    """All classification metrics
    """
    cls_metrics = regression_metrics

    def __init__(self):
        self.regression_metric = MetricsOrderedDict(self.cls_metrics)

    def __call__(self, y_true, y_pred):
        # squeeze the last dimension
        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = np.ravel(y_true)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.ravel(y_pred)

        return self.regression_metric(y_true, y_pred)


# available eval metrics --------------------------------------------


BINARY_CLASS = ["auc", "auprc", "accuracy", "tpr", "tnr", "f1", "mcc"]
CATEGORY_CLASS = ["cat_acc"]
REGRESSION = ["mse", "mad", "cor", "ermse", "var_explained"]

AVAILABLE = BINARY_CLASS + CATEGORY_CLASS + REGRESSION
