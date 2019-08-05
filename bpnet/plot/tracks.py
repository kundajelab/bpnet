"""Plots genomic tracks
"""
import attr
import collections
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from collections import OrderedDict
from concise.utils.pwm import seqlogo
from bpnet.plot.utils import simple_yaxis_format, strip_axis, spaced_xticks, draw_box, spine_subset, draw_hline


@attr.s
class TrackInterval:
    start = attr.ib()
    end = attr.ib()
    name = attr.ib(None)

    @classmethod
    def from_pybedtools_interval(cls, interval):
        return cls(interval.start, interval.end, interval.name)


def plot_seqlet_box(seqlet, ax, add_label=False):
    """
    Args:
      seqlet: object with start, end, name, strand attribues
      if Seqname is available, then we can plot it to the right position
    """
    xlim = ax.get_xlim()

    xmin = seqlet.start + 0.5
    xmax = seqlet.end + 0.5
    if xmax < 0 or xmin > xlim[1] - xlim[0]:
        return

    draw_box(xmin, xmax, ax, 'g')  # trimmed pattern location

    if add_label:
        y = ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15

        ax.text(xmin + 0.5, y,
                s=seqlet.strand + str(seqlet.name),
                fontsize=8)


def plot_seqlet_underscore(seqlet, ax, add_label=False):
    """
    Args:
      seqlet: object with start, end, name, strand attribues
      if Seqname is available, then we can plot it to the right position
    """
    xlim = ax.get_xlim()

    xmin = seqlet.start + 0.5
    xmax = seqlet.end + 0.5
    if xmax < 0 or xmin > xlim[1] - xlim[0]:
        return

    # TODO - it would be nice to also have some differnet colors here
    draw_hline(xmin, xmax, ax.get_ylim()[0], col='r', linewidth=5, alpha=0.3)

    if add_label:
        y = ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15

        ax.text(xmin + 0.5, y,
                s=seqlet.strand + str(seqlet.name),
                fontsize=5)


def get_items(tracks):
    if isinstance(tracks, dict):
        return list(tracks.items())
    else:
        return tracks


def seqlen_tracks(tracks):
    # check that all the arrays have the same number of dimensions
    seqlens = {arr.shape[0] for k, arr in get_items(tracks)}
    assert len(seqlens) == 1
    return seqlens.pop()


def filter_tracks(tracks, xlim=None):
    """Filter tracks

    Args:
      tracks: dictionary or OrderedDict of arrays with ndim >=2 and same second dimension
      xlim: tuple/dict containing xmin and xmax

    Returns:
      OrderedDict of tracks
    """
    if tracks is None:
        return None

    if isinstance(tracks, dict) and len(tracks) == 0:
        return tracks

    seqlen = seqlen_tracks(tracks)
    if xlim is None:
        return tracks

    xmin, xmax = xlim
    assert xmax > xmin
    assert xmin >= 0
    assert xmax <= seqlen

    if isinstance(tracks, OrderedDict):
        return OrderedDict([(track, arr[xmin:xmax])
                            for track, arr in tracks.items()])
    elif isinstance(tracks, dict):
        return {track: arr[xmin:xmax]
                for track, arr in tracks.items()}
    elif isinstance(tracks, list):
        return [(track, arr[xmin:xmax])
                for track, arr in tracks]
    else:
        raise ValueError("Tracks need to be either dict or OrderedDict")


def rc_tracks(tracks):
    """Filter tracks

    Args:
      tracks: dictionary or OrderedDict of arrays with ndim >=2 and same second dimension

    Returns:
      OrderedDict of tracks
    """
    if tracks is None:
        return tracks

    if isinstance(tracks, OrderedDict):
        return OrderedDict([(track, arr[::-1, ::-1])
                            for track, arr in tracks.items()])
    elif isinstance(tracks, dict):
        return {track: arr[::-1, ::-1]
                for track, arr in tracks.items()}
    elif isinstance(tracks, list):
        return [(track, arr[::-1, ::-1])
                for track, arr in tracks]
    else:
        raise ValueError("Tracks need to be either dict or OrderedDict")


# pad a single track
def pad_track(track, new_len, value=0):
    """if delta%2==1, then one more is added to the end
    """
    assert track.ndim == 2
    assert new_len >= len(track)

    out = np.empty((new_len, track.shape[1]))
    out[:] = value
    delta = new_len - len(track)
    i = delta // 2
    j = i + len(track)
    out[i:j] = track
    return out


def pad_tracks(tracks, new_len, value=0):
    """Filter tracks

    Args:
      tracks: dictionary or OrderedDict of arrays with ndim >=2 and same second dimension

    Returns:
      OrderedDict of tracks
    """
    from bpnet.modisco.sliding_similarities import pad_same
    if tracks is None:
        return tracks

    if isinstance(tracks, OrderedDict):
        return OrderedDict([(track, pad_track(arr, new_len, value=value))
                            for track, arr in tracks.items()])
    elif isinstance(tracks, dict):
        return {track: pad_track(arr, new_len, value=value)
                for track, arr in tracks.items()}
    elif isinstance(tracks, list):
        return [(track, pad_track(arr, new_len, value=value))
                for track, arr in tracks]
    else:
        raise ValueError("Tracks need to be either dict or OrderedDict")


def skip_nan_tracks(tracks):
    """Filter tracks

    Args:
      tracks: dictionary or OrderedDict of arrays with ndim >=2 and same second dimension

    Returns:
      OrderedDict of tracks
    """
    if tracks is None:
        return tracks

    if isinstance(tracks, OrderedDict):
        return OrderedDict([(track, arr)
                            for track, arr in tracks.items() if arr is not None])
    elif isinstance(tracks, dict):
        return {track: arr
                for track, arr in tracks.items() if arr is not None}
    elif isinstance(tracks, list):
        return [(track, arr)
                for track, arr in tracks if arr is not None]
    else:
        raise ValueError("Tracks need to be either dict or OrderedDict")


def plot_track(arr, ax, legend=False, ylim=None, color=None, track=None):
    """Plot a track
    """
    seqlen = len(arr)
    if arr.ndim == 1 or arr.shape[1] == 1:
        # single track
        if color is not None:
            if isinstance(color, collections.Sequence):
                color = color[0]
        ax.plot(np.arange(1, seqlen + 1), np.ravel(arr), color=color)
    elif arr.shape[1] == 4:
        # plot seqlogo
        seqlogo(arr, ax=ax)
    elif arr.shape[1] == 2:
        # plot both strands
        if color is not None:
            assert isinstance(color, collections.Sequence)
            c1 = color[0]
            c2 = color[1]
        else:
            c1, c2 = None, None
        ax.plot(np.arange(1, seqlen + 1), arr[:, 0], label='pos', color=c1)
        ax.plot(np.arange(1, seqlen + 1), arr[:, 1], label='neg', color=c2)
        if legend:
            ax.legend()
    else:
        raise ValueError(f"Don't know how to plot array with shape[1] != {arr.shape[1]}. Valid values are: 1,2 or 4.")
    if ylim is not None:
        ax.set_ylim(ylim)


def get_list_value(lv, i):
    if lv is None:
        return lv
    elif isinstance(lv, list):
        return lv[i]
    else:
        return lv


def to_neg(track):
    """Use the negative sign for reads on the reverse strand
    """
    track = track.copy()
    track[:, 1] = - track[:, 1]
    return track


def plot_tracks(tracks, seqlets=[],
                title=None,
                rotate_y=90,
                legend=False,
                fig_width=20,
                fig_height_per_track=2,
                ylim=None,
                same_ylim=False,
                use_spine_subset=False,
                seqlet_plot_fn=plot_seqlet_box,
                ylab=True,
                color=None,
                height_ratios=None,
                plot_track_fn=plot_track):
    """Plot a multiple tracks.

    One-hot-encoded sequence as a logo,
    and 1 or 2 dim tracks as normal line-plots.

    Args:
      tracks: dictionary of numpy arrays with the same axis0 length
      fig_width: figure width
      fig_height_per_track: figure height per track.
      ylim: if not None, a single tuple or a list of tuples representing the ylim to use

    Returns:
      matplotlib.figure.Figure
    """

    if height_ratios is not None:
        gridspec_kw = {"height_ratios": height_ratios}
    else:
        gridspec_kw = dict()

    tracks = skip_nan_tracks(tracks)  # ignore None values
    fig, axes = plt.subplots(len(tracks), 1,
                             figsize=(fig_width, fig_height_per_track * len(tracks)),
                             gridspec_kw=gridspec_kw,
                             sharex=True)

    if len(tracks) == 1:
        axes = [axes]

    if same_ylim:
        ylim = (0, max([v.max() for k, v in get_items(tracks)]))

    for i, (ax, (track, arr)) in enumerate(zip(axes, get_items(tracks))):
        yl = get_list_value(ylim, i)
        plot_track_fn(arr, ax, legend, yl,
                      color=get_list_value(color, i),
                      track=track)
        if use_spine_subset:
            spine_subset(ax, max(yl[0], 0), yl[1])

        # TODO allow to specify separate seqlets for different regions (e.g. via dicts)
        for seqlet in seqlets:
            if seqlet.seqname == track:
                seqlet_plot_fn(seqlet, ax, add_label=True)
        # ax.set_ylabel(track)
        if ylab:
            if rotate_y == 90:
                ax.set_ylabel(track)
            else:
                ax.set_ylabel(track, rotation=rotate_y,
                              multialignment='center',
                              ha='right', labelpad=5)
        simple_yaxis_format(ax)
        if i != len(tracks) - 1:
            ax.xaxis.set_ticks_position('none')
        if i == 0 and title is not None:
            ax.set_title(title)

        # if seqlets:
        #    pass

    # add ticks to the final axis
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    # spaced_xticks(ax, spacing=5)
    fig.subplots_adjust(hspace=0)
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # cleanup the plot
    return fig


def tidy_motif_plot(ax=None):
    if ax is None:
        ax = plt.gca()
    strip_axis(ax)
    ax.set_xlabel(None)
    ax.get_xaxis().set_visible(False)
    ax.set_ylim([0, 2.0])
    ax.set_ylabel("IC")
