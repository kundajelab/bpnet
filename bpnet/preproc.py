import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from copy import deepcopy
import pybedtools
from bpnet.external.deeplift.dinuc_shuffle import dinuc_shuffle
from concise.preprocessing.sequence import one_hot2string, encodeDNA, DNA
from pybedtools import Interval, BedTool
from scipy.ndimage.filters import gaussian_filter1d
import gin
import random


def moving_average(x, n=1):
    """Compute the moving average along the first axis
    """
    from bpnet.modisco.sliding_similarities import pad_same
    if n == 1:
        return x
    x = pad_same(x[np.newaxis], motif_len=n)[0]
    ret = np.cumsum(x, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def scale_min_max(x):
    return (x - x.min()) / (x.max() - x.min())


def bin_counts(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].sum(1)
    return xout


def transform_data(data, use_profile, use_counts):

    out = []
    for i, d in enumerate(data):
        ys = np.log(1 + d[1]['sox2'].sum(1))
        yo = np.log(1 + d[1]['oct4'].sum(1))
        if i == 0:
            scaler_s = StandardScaler()
            scaler_s.fit(ys)
            scaler_o = StandardScaler()
            scaler_o.fit(yo)
        ys = scaler_s.transform(ys)
        yo = scaler_s.transform(yo)
        if use_profile:
            out.append((d[0], [d[1]['sox2'], d[1]['oct4'], ys, yo], d[2]))
        else:
            out.append((d[0], [ys, yo], d[2]))
    return tuple(out)


def transform_data_single_task(data, use_profile, use_counts):

    out = []
    for i, d in enumerate(data):
        y = np.log(1 + d[1].sum(1))
        if i == 0:
            scaler = StandardScaler()
            scaler.fit(y)

        y = scaler.transform(y)
        if use_profile:
            out.append((d[0], [d[1], y], d[2]))
        else:
            out.append((d[0], [y], d[2]))
    return tuple(out)


class AppendCounts:
    def fit(self, x, *_):
        return self

    def transform(self, x, *_):
        counts = {k.replace("profile", "counts"): np.log(1 + x[k].sum(0))
                  for k in x}
        return {**x, **counts}


class TotalCount:
    def fit(self, x, *_):
        return self

    def transform(self, x, *_):
        return x.sum(1)

    def fit_transform(self, x, *_):
        self.fit(x)
        return self.transform(x)


class CountLog:

    def fit(self, x, *_):
        return self

    def transform(self, x, *_):
        """
        """
        return np.log(1 + x.sum(1))

    def fit_transform(self, x, *_):
        self.fit(x)
        return self.transform(x)


class AppendTotalCounts:
    """Takes a dictionary of arrays:
    profile/<task1>: (batch, seqlen, 2)
    profile/<task2>: (batch, seqlen, 2)
    ...

    and appends a dictionary:
    counts/<task1>: (batch, 2)
    counts/<task2>: (batch, 2)
    ...
    """

    def __init__(self, preproc_unit=make_pipeline(CountLog(),
                                                  StandardScaler())):
        self.preproc_unit = preproc_unit
        self.objects = None

    def fit(self, x, *_):
        def fit_single(x, preproc):
            preproc_cp = deepcopy(preproc)
            preproc_cp.fit(x)
            return preproc_cp
        self.objects = {k: fit_single(v, self.preproc_unit)
                        for k, v in x.items()}
        return self

    def transform(self, x, *_):
        # Append counts to profiles
        assert self.objects is not None
        counts = {k.replace("profile/", "counts/"): pp.transform(x[k])
                  for k, pp in self.objects.items()}
        return {**x, **counts}

    def fit_transform(self, x, *_):
        self.fit(x)
        return self.transform(x)


class BinAndSmooth:
    """Takes a dictionary of arrays with keys:
    profile/<task1>: (batch, seqlen, 2)
    ...

    and adds a smoothing filter to its binned version, editing in place.
    This can be mode='gauss' (set the sigma for locality)
                mode='bin' (only bin it)
       (# TODO) mode='fft' (FFTs the data, takes the top n_freq frequencies,
                            and performs IFFT on it)
    """

    def __init__(self, binsize=10, mode='gauss', sigma=1.2):
        self.mode = mode
        self.binsize = binsize
        self.sigma = sigma

    def fit(self, x, *_):
        return self

    def transform(self, x, *_):
        for k in x.keys():
            if 'profile' not in k:
                continue
            profile = np.expand_dims(x[k], 0)

            profile = bin_counts(profile, self.binsize)
            if self.mode == 'gauss':
                profile = gaussian_filter1d(profile, axis=1, sigma=self.sigma)

            x[k] = profile.squeeze()

        return x

    def fit_transform(self, x, *_):
        self.fit(x)
        return self.transform(x)


# Inteval operations - TODO -> put these to Kipoi.GenomicRanges
def parse_interval(s):
    import pybedtools
    chrom, ranges = s.replace(",", "").split(":")
    start, end = ranges.split("-")
    return pybedtools.create_interval_from_list([chrom, int(start), int(end)])


def interval_center(interval, ignore_rc=False):
    """Get the center of the interval

    Note: it takes care of the strand
       >>>>
         |

       <<<<
        |
    """
    if ignore_rc:
        add_offset = 0
    else:
        if isinstance(interval, pd.DataFrame):
            if 'strand' in interval.columns:
                add_offset = interval.strand.map({"+": 1, "-": 0})
            else:
                add_offset = 1  # ignore strand
        else:
            add_offset = 0 if interval.strand == "-" else 1
    delta = (interval.end + interval.start) % 2
    uncorrected_center = (interval.end + interval.start) // 2
    return uncorrected_center + add_offset * delta


def resize_interval_ij(interval, width, ignore_strand=False):
    """Resize the bedtools interval

    Note: it takes care of the strand
    """
    center = interval_center(interval, ignore_rc=ignore_strand)

    if not ignore_strand:
        pos_strand = interval.strand != '-'
    else:
        pos_strand = True

    start = center - width // 2 - (width % 2) * (~pos_strand)
    end = center + width // 2 + (width % 2) * pos_strand
    return start, end


def resize_interval(interval, width, ignore_strand=False):
    """Resize the bedtools interval

    Note: it takes care of the strand
    """
    start, end = resize_interval_ij(interval, width,
                                    ignore_strand=ignore_strand)
    return update_interval(interval, start, end)


def update_interval(interval, start, end):
    if isinstance(interval, Interval):
        name = interval.name if interval.name is not None else ''
        return pybedtools.create_interval_from_list([interval.chrom,
                                                     start,
                                                     end,
                                                     name,
                                                     interval.score,
                                                     interval.strand])
    else:
        # interval = deepcopy(interval)
        interval.start = start
        interval.end = end
        return interval


def update_interval_strand(interval, strand):
    if isinstance(interval, Interval):
        name = interval.name if interval.name is not None else ''
        return pybedtools.create_interval_from_list([interval.chrom,
                                                     interval.start,
                                                     interval.end,
                                                     name,
                                                     interval.score,
                                                     strand])
    else:
        # interval = deepcopy(interval)
        interval.strand = strand
        return interval


def keep_interval(interval, width, fa):
    """Returns True if the interval can be validly resized
    """
    start, stop = resize_interval_ij(interval, width)
    return start >= 0 and stop > start and stop < fa.get_reference_length(interval.chrom)


def shift_interval(interval, shift):
    return update_interval(interval,
                           start=interval.start + shift,
                           end=interval.end + shift)


def random_strand(interval):
    strand = ['+', '-'][random.randint(0, 1)]
    return update_interval_strand(interval, strand)


@gin.configurable
class IntervalAugmentor:
    """Randomly shift and swap strands

    Args:
      max_shift: Inteval shift is sampled uniformly from [-max_shift, max_shift]
      flip_strand: if True, strand is randomly sampled
    """

    def __init__(self, max_shift, flip_strand=True):
        self.max_shift = max_shift
        self.flip_strand = flip_strand

    def __call__(self, interval):
        # Generate the random shift
        shift = random.randint(-self.max_shift, self.max_shift)
        if self.flip_strand:
            interval = random_strand(interval)
        # Return the interval
        return shift_interval(interval, shift)


def label_bed(a, b_dict):
    """Compute if any feature overlaps with the bed file

    Args:
      a: bed file of interest
      b: list of bed files
    """
    # load the pandas dataframe
    dfg = BedTool(a).to_dataframe()
    dfg['name'] = dfg.index
    btg = BedTool.from_dataframe(dfg)

    for task, b in b_dict.items():
        feature = f'task/{task}'
        intersected = btg.intersect(BedTool(b), wa=True, u=True).to_dataframe()['name']
        dfg[feature] = 0
        dfg.loc[intersected, feature] = 1
    del dfg['name']
    dfg = dfg.rename(columns={"chrom": "#chrom"})
    return dfg


def dfint_intersects(dfa, dfb):
    return ~dfint_no_intersection(dfa, dfb)


def dfint_no_intersection(dfa, dfb):
    """Search if two data-frames have no intersection

    Args:
      dfa,dfb: each data frame has to contain three columns
        with the following entries: chr, start and end

    Returns:
      dfa with 
    """
    from pybedtools import BedTool
    assert len(dfa.columns) == 3
    assert len(dfb.columns) == 3
    dfa = dfa.copy()
    dfa['id'] = np.arange(len(dfa))
    bta = BedTool.from_dataframe(dfa)
    btb = BedTool.from_dataframe(dfb)
    not_intersected_id = bta.intersect(btb, v=True).to_dataframe().name
    return dfa['id'].isin(not_intersected_id)


def dfint_overlap_idx(dfa, dfb):
    """Overlap dfa with dfb

    Returns:
      np.array with length `len(dfa)` of the matching row indices in dfab

    Note:
      if multiple rows in dfab overlap a row in dfa,
      then the first mathing row in dfb is returned
    """
    from pybedtools import BedTool
    assert len(dfa.columns) == 3
    assert len(dfb.columns) == 3
    dfa = dfa.copy()
    dfa['id'] = np.arange(len(dfa))
    dfb = dfb.copy()
    dfb['id'] = np.arange(len(dfb))
    bta = BedTool.from_dataframe(dfa)
    btb = BedTool.from_dataframe(dfb)
    dfi = bta.intersect(btb, wa=True, loj=True).to_dataframe()
    keep = ~dfi[['chrom', 'start', 'end', 'name']].duplicated()
    out = dfi[keep].iloc[:, -1]  # final column
    out[out == '.'] = '-1'
    return out.astype(int).values


def balance_class_weight(labels):
    """Compute the class balances
    """
    counts = pd.Series(labels).value_counts()
    norm_value = counts.min()
    hash_map = norm_value / counts
    return labels.map(hash_map).values


def rc_seq(seq):
    """
    Reverse complement the sequence
    >>> assert rc_seq("TATCG") == "CGATA"
    """
    rc_hash = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
    }
    return "".join([rc_hash[s] for s in reversed(seq)])


def onehot_dinucl_shuffle(seqs):
    """Di-nucleotide shuffle the sequences
    """
    return encodeDNA([dinuc_shuffle(s) for s in one_hot2string(seqs, vocab=DNA)])
