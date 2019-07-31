import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib import colors
from bpnet.plot.utils import MidpointNormalize


class QuantileTruncateNormalizer:
    def __init__(self, pmin=50, pmax=99):
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, signal):
        norm_signal = np.minimum(signal, np.percentile(signal, self.pmax))
        norm_signal = np.maximum(norm_signal, np.percentile(signal, self.pmin))
        return norm_signal


class RowQuantileNormalizer:
    def __init__(self, pmin=50, pmax=99):
        """Row-normalize the profile matrix

        Args:
          pmin: minimum percentile
          pmax: maximum percentile
        """
        self.pmin = pmin
        self.pmax = pmax

    def __call__(self, signal):
        s = signal.copy()
        p50 = np.percentile(s, self.pmin, axis=1)
        p99 = np.percentile(s, self.pmax, axis=1)

        # mask all values < p50
        s[s < p50[:, np.newaxis]] = np.nan
        snorms = np.minimum(s / p99[:, np.newaxis], 1)
        return snorms


def normalize(p, pmin=50, pmax=99):
    """Back-compatibility
    """
    return RowQuantileNormalizer(pmin, pmax)(p)


def heatmap_stranded_profile(signal, ax=None, figsize=(5, 20),
                             aspect=0.2, normalizer=RowQuantileNormalizer(),
                             interpolation='nearest', tick_step=25):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    norm_signal = normalizer(signal)
    ax.imshow(norm_signal[:, :, 0], cmap=plt.cm.Reds, interpolation=interpolation, aspect=aspect)
    ax.imshow(norm_signal[:, :, 1], alpha=0.5, cmap=plt.cm.Blues, interpolation=interpolation, aspect=aspect)
    seq_len = signal.shape[1]
    ticks = np.arange(0, seq_len + 1 - tick_step, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks - seq_len // 2)
    ax.set_ylabel("Seqlet index")
    ax.set_xlabel("Position")
    return fig


def multiple_heatmap_stranded_profile(signal_dict, figsize=(20, 20), sort_idx=None, **kwargs):
    """Plot a dictionary of profiles
    """
    tasks = list(signal_dict.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize)

    # pre-sort
    if sort_idx is None:
        total_counts = sum([x.sum(axis=-1).sum(axis=-1) for x in signal_dict.values()])
        sort_idx = np.argsort(-total_counts)

    for i, (task, ax) in enumerate(zip(tasks, axes)):
        heatmap_stranded_profile(signal_dict[task][sort_idx], ax=ax, **kwargs)
        ax.set_title(task)
    fig.subplots_adjust(wspace=0)  # no space between plots
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)  # no numbers
    plt.setp([a.get_yaxis() for a in fig.axes[1:]], visible=False)  # no numbers
    return fig


def heatmap_contribution_profile(signal, ax=None, figsize=(5, 20), aspect=0.2, sort_idx=None, tick_step=25):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if sort_idx is None:
        sort_idx = np.arange(signal.shape[0])

    interpolation = 'nearest'
    ax.imshow(signal[sort_idx],
              cmap=plt.cm.RdBu, norm=MidpointNormalize(midpoint=0),
              interpolation=interpolation, aspect=aspect)

    seq_len = signal.shape[1]
    ticks = np.arange(0, seq_len + 1 - tick_step, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks - seq_len // 2)
    ax.set_ylabel("Seqlet index")
    ax.set_xlabel("Position")


def multiple_heatmap_contribution_profile(signal_dict, sort_idx=None,
                                          figsize=(20, 20), **kwargs):
    """Plot a dictionary of profiles
    """
    tasks = list(signal_dict.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize)

    # --------------------
    # special. TODO - re-factor
    if sort_idx is None:
        sort_idx = np.arange([x for x in signal_dict.values()][0].shape[0])

    for i, (task, ax) in enumerate(zip(tasks, axes)):
        heatmap_contribution_profile(signal_dict[task][sort_idx],
                                     ax=ax, **kwargs)
        # --------------------
        ax.set_title(task)
    fig.subplots_adjust(wspace=0)  # no space between plots
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)  # no numbers
    plt.setp([a.get_yaxis() for a in fig.axes[1:]], visible=False)  # no numbers
    return fig


def multiple_heatmaps(signal_dict, plot_fn, sort_idx=None, figsize=(20, 20), **kwargs):
    tasks = list(signal_dict.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=figsize)
    if sort_idx is None:
        sort_idx = np.arange([x for x in signal_dict.values()][0].shape[0])

    for i, (task, ax) in enumerate(zip(tasks, axes)):
        plot_fn(signal_dict[task][sort_idx],
                ax=ax, **kwargs)
        ax.set_title(task)
    fig.subplots_adjust(wspace=0)  # no space between plots
    plt.setp([a.get_yticklabels() for a in fig.axes[1:]], visible=False)  # no numbers
    plt.setp([a.get_yaxis() for a in fig.axes[1:]], visible=False)  # no numbers
    return fig


def heatmap_sequence(one_hot, ax=None, sort_idx=None, aspect='auto', figsize_tmpl=(8, 4), cbar=True):
    """Plot a heatmap of sequences
    """
    if ax is None:
        figsize = (figsize_tmpl[0] * one_hot.shape[1] / 200,
                   figsize_tmpl[1] * one_hot.shape[0] / 2000)
        fig, ax = plt.subplots(figsize=figsize)

    if sort_idx is None:
        sort_idx = np.arange(one_hot.shape[0])

    cmap = colors.ListedColormap(["red", "orange", "blue", "green"][::-1])
    qrates = np.array(list("TGCA"))
    bounds = np.linspace(-.5, 3.5, 5)
    norm = colors.BoundaryNorm(bounds, 4)
    fmt = mpl.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

    img = ax.imshow(one_hot.argmax(axis=-1)[sort_idx], aspect=aspect, cmap=cmap, norm=norm, alpha=0.8)
    if cbar:
        ax2_divider = make_axes_locatable(ax)
        cax2 = ax2_divider.append_axes("top", size="5%", pad=0.05)
        # cb2 = colorbar(im2, cax=cax2, orientation="horizontal")
        cb2 = colorbar(img, cax=cax2, cmap=cmap, norm=norm, boundaries=bounds,
                       orientation="horizontal",
                       ticks=[0, 1, 2, 3], format=fmt)
        cax2.xaxis.set_ticks_position("top")
    seq_len = one_hot.shape[1]
    ticks = np.arange(0, seq_len + 1, 25)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks - seq_len // 2)
    ax.set_ylabel("Seqlet index")
    ax.set_xlabel("Position")
    return fig
