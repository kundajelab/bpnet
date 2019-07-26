"""Implements various similarity metrics in a sliding-window fashion

Main function: sliding_similarity

General remarks:
- qa: query array (pattern) of shape (query_seqlen, channels) used for scanning
- ta: target array which gets scanned by qa of shape (..., target_seqlen, channels)

by Ziga Avsec
"""
from __future__ import division, print_function, absolute_import
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import correlate


def halve(n):
    return n // 2 + n % 2, n // 2


def pad_same(track, motif_len, mode='median', **kwargs):
    """Pad along the last axis
    """
    if mode is None:
        return track
    i, j = halve(motif_len - 1)
    return np.pad(track, [(0, 0), (i, j)] + [(0, 0)] * (len(track.shape) - 2), mode=mode, **kwargs)


def rolling_window(a, window_width):
    """Create a new array suitable for rolling window operation

    Adopted from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html,
    also discussed in this PR: https://github.com/numpy/numpy/issues/7753

    Args:
      a: input array of shape (..., positions)
      window_width: width of the window to scan

    Returns:
      array of shape (..., positions - window_width + 1, window_width)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_width + 1, window_width)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sliding_continousjaccard(qa, ta):
    """Score the region with contionous jaccard
    Args:
      qa: query array (pattern) of shape (query_seqlen, channels) used for scanning
      ta: target array which gets scanned by qa of shape (..., target_seqlen, channels)

    Returns:
      a tuple: (jaccard score of the normalized array, L1 magnitude of the scanned window)
        both are of shape (..., target_seqlen - query_seqlen + 1)

      if pad_mode is used, the output shape is: (..., target_seqlen)
    """
    qa = qa.swapaxes(-2, -1)  #
    ta = ta.swapaxes(-2, -1)

    assert ta.shape[-1] >= qa.shape[-1]  # target needs to be longer than the query

    window_len = qa.shape[-1]
    # out_len = qa.shape[-1] - window_len + 1

    ta_strided = rolling_window(ta, window_len).swapaxes(-2, -1)

    # compute the normalization factor
    qa_L1_norm = np.sum(np.abs(qa))
    ta_L1_norm = np.sum(np.abs(ta_strided), axis=(-3, -2))
    per_pos_scale_factor = qa_L1_norm / (ta_L1_norm + (0.0000001 * (ta_L1_norm == 0)))

    ta_strided_normalized = ta_strided * per_pos_scale_factor[..., np.newaxis, np.newaxis, :]

    qa_strided = qa[..., np.newaxis]

    ta_strided_normalized_abs = np.abs(ta_strided_normalized)
    qa_strided_abs = np.abs(qa_strided)
    union = np.sum(np.maximum(ta_strided_normalized_abs, qa_strided_abs), axis=(-3, -2))
    # union = np.sum(np.maximum(np.abs(ta_strided_normalized), np.abs(qa_strided)), axis=(-3, -2))
    intersection = (np.sum(np.minimum(ta_strided_normalized_abs, qa_strided_abs) *
                           np.sign(ta_strided_normalized) * np.sign(qa_strided), axis=(-3, -2)))
    return intersection / union, ta_L1_norm


def parallel_sliding_continousjaccard(qa, ta, pad_mode=None, n_jobs=10, verbose=True):
    """Parallel version of sliding_continousjaccard

    pad: if not None, pad to achieve same padding. pad can be a constant value
        mean, median, symmetric

    if pad_mode is used, the output shape is: (..., target_seqlen)
    """
    r = np.stack(Parallel(n_jobs)(delayed(sliding_continousjaccard)(qa, ta[i])
                                  for i in tqdm(range(len(ta)), disable=not verbose)))
    return pad_same(r[:, 0], len(qa), pad_mode), pad_same(r[:, 1], len(qa), pad_mode)

# ------------------------------------------------
# PWM scanning


def sliding_dotproduct(qa, ta, pad_mode=None):
    """'convolution' implemented in numpy with valid padding
    """
    if qa.ndim < ta.ndim:
        qa = qa[np.newaxis]
    return correlate(ta, qa, mode='valid')[..., 0]


def parallel_sliding_dotproduct(qa, ta, pad_mode=None, n_jobs=10, verbose=True):
    """Parallel version of sliding_dotproduct
    """
    return pad_same(np.stack(Parallel(n_jobs)(delayed(sliding_dotproduct)(qa, ta[i][np.newaxis])
                                              for i in tqdm(range(len(ta)), disable=not verbose)))[:, 0],
                    len(qa), pad_mode)


def sliding_kl_divergence(qa, ta, kind='simmetric'):
    """Score the region with contionous jaccard
    Args:
      qa: query array (pattern) of shape (query_seqlen, channels) used for scanning
      ta: target array which gets scanned by qa of shape (..., target_seqlen, channels)
      kind: either qt, tq or simmetric: qt -> KL(qa, ta),  tq -> KL(tq, qa)
        simmetric -> (KL(qa, ta) + KL(ta, qa)) / 2
      pad: if not None, pad to achieve same padding. pad can be a constant value
        mean, median, symmetric

    Returns:
      a tuple: (jaccard score of the normalized array, L1 magnitude of the scanned window)
        both are of shape (..., target_seqlen - query_seqlen + 1)

      if pad_mode is used, the output shape is: (..., target_seqlen)
    """
    qa = qa.swapaxes(-2, -1)  #
    ta = ta.swapaxes(-2, -1)

    assert ta.shape[-1] >= qa.shape[-1]  # target needs to be longer than the query

    window_len = qa.shape[-1]
    # out_len = qa.shape[-1] - window_len + 1

    ta_strided = rolling_window(ta, window_len).swapaxes(-2, -1)
    # (..., channels, query_seqlen, target_seqlen)

    qa_exp = qa[..., np.newaxis]
    # (..., channels, query_seqlen, 1)

    # first sum computes the KL diver
    if kind == 'qt':
        return (qa_exp * np.log(qa_exp / ta_strided)).sum(axis=1).mean(axis=1)
    elif kind == 'tq':
        return (ta_strided * np.log(ta_strided / qa_exp)).sum(axis=1).mean(axis=1)
    elif kind == 'simmetric':
        qt = (qa_exp * np.log(qa_exp / ta_strided)).sum(axis=1).mean(axis=1)
        tq = (ta_strided * np.log(ta_strided / qa_exp)).sum(axis=1).mean(axis=1)
        return (qt + tq) / 2


def parallel_sliding_kl_divergence(qa, ta, kind='simmetric', pseudo_p=1e-6, pad_mode=None, n_jobs=10, verbose=True):
    """Parallel version of sliding_continousjaccard

    pad: if not None, pad to achieve same padding. pad can be a constant value
        mean, median, symmetric

    if pad_mode is used, the output shape is: (..., target_seqlen)
    """
    # add pseudo-counts
    qa = qa + pseudo_p
    qa = qa / qa.sum(axis=-1, keepdims=True)
    ta = ta + pseudo_p
    ta = ta / ta.sum(axis=-1, keepdims=True)

    return pad_same(np.concatenate(Parallel(n_jobs)(delayed(sliding_kl_divergence)(qa, ta[i][np.newaxis], kind=kind)
                                                    for i in tqdm(range(len(ta)), disable=not verbose))),
                    len(qa), pad_mode)


def sliding_similarity(qa, ta, metric='continousjaccard', pad_mode=None, n_jobs=10, verbose=True):
    """
    Args:
      qa (np.array): query array (pattern) of shape (query_seqlen, channels) used for scanning
      ta (np.array): target array which gets scanned by qa of shape (..., target_seqlen, channels)
      metric (str): similarity metric to use. Can be either from continousjaccard, dotproduct.
        dotproduct implements 'convolution' in numpy

    Returns:
      single array for dotproduct or a tuple of two arrays for continousjaccard (match and magnitude)

      if pad_mode is used, the output shape is: (..., target_seqlen)
    """
    if metric == 'continousjaccard':
        return parallel_sliding_continousjaccard(qa, ta, n_jobs=n_jobs, verbose=verbose, pad_mode=pad_mode)
    elif metric == 'dotproduct':
        return parallel_sliding_dotproduct(qa, ta, n_jobs=n_jobs, verbose=verbose, pad_mode=pad_mode)
    elif metric == 'simmetric_kl':
        return - parallel_sliding_kl_divergence(qa, ta, n_jobs=n_jobs, kind='simmetric', verbose=verbose, pad_mode=pad_mode)
    else:
        raise ValueError("metric needs to be from: 'continousjaccard', 'dotproduct', 'simmetric_kl'")


# --------------------------------------------
# Example on how to implement pwm scanning using
#
def pssm_scan(pwm, seqs, background_probs=[0.27, 0.23, 0.23, 0.27], pad_mode='median', n_jobs=10, verbose=True):
    """
    """
    def pwm2pssm(arr, background_probs):
        """Convert pwm array to pssm array
        pwm means that rows sum to one
        """
        arr = arr / arr.sum(1, keepdims=True)
        arr = arr + 0.01  # add pseudo-counts
        arr = arr / arr.sum(1, keepdims=True)
        b = np.array(background_probs)[np.newaxis]
        return np.log(arr / b).astype(arr.dtype)
    pssm = pwm2pssm(pwm, background_probs)
    return sliding_similarity(pssm, seqs, 'dotproduct', pad_mode, n_jobs, verbose)
