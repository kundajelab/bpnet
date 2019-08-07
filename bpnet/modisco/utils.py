"""Small helper-functions for used by modisco classes
"""
import pandas as pd
import numpy as np
from kipoi.readers import HDF5Reader
from bpnet.cli.contrib import ContribFile
from bpnet.functions import mean
import warnings


def bootstrap_mean(x, n=100):
    """Bootstrap the mean computation"""
    out = []

    for i in range(n):
        idx = pd.Series(np.arange(len(x))).sample(frac=1.0, replace=True).values
        out.append(x[idx].mean(0))
    outm = np.stack(out)
    return outm.mean(0), outm.std(0)


def nan_like(a, dtype=float):
    a = np.empty(a.shape, dtype)
    a.fill(np.nan)
    return a


def ic_scale(x):
    from modisco.visualization import viz_sequence
    background = np.array([0.27, 0.23, 0.23, 0.27])
    return viz_sequence.ic_scale(x, background=background)


def shorten_pattern(pattern):
    """metacluster_0/pattern_1 -> m1_p1
    """
    if "/" not in pattern:
        # input is already a short pattern
        return pattern
    else:
        # two slashes -> prepended name
        return pattern.replace("metacluster_", "m").replace("/pattern_", "_p")


def longer_pattern(shortpattern):
    """m1_p1 -> metacluster_0/pattern_1
    """
    if "/" in shortpattern:
        # input is already a long pattern
        return shortpattern
    else:
        return shortpattern.replace("_p", "/pattern_").replace("m", "metacluster_")


def extract_name_short(ps):
    m, p = ps.split("_")
    return {"metacluster": int(m.replace("m", "")), "pattern": int(p.replace("p", ""))}


def extract_name_long(ps):
    m, p = ps.split("/")
    return {"metacluster": int(m.replace("metacluster_", "")), "pattern": int(p.replace("pattern_", ""))}


def trim_pssm_idx(pssm, frac=0.05):
    if frac == 0:
        return 0, len(pssm)
    pssm = np.abs(pssm)
    threshold = pssm.sum(axis=-1).max() * frac
    for i in range(len(pssm)):
        if pssm[i].sum(axis=-1) > threshold:
            break

    for j in reversed(range(len(pssm))):
        if pssm[j].sum(axis=-1) > threshold:
            break
    return i, j + 1  # + 1 is for using 0-based indexing


def trim_pssm(pssm, frac=0.05):
    i, j = trim_pssm_idx(pssm, frac=frac)
    return pssm[i:j]
