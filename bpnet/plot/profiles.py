from tqdm import tqdm
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from bpnet.plot.utils import simple_yaxis_format, strip_axis, spaced_xticks
from bpnet.modisco.utils import bootstrap_mean, nan_like, ic_scale
from bpnet.plot.utils import show_figure


# TODO - make it as a bar-plot with two standard colors:
# #B23F49 (pos), #045CA8 (neg)
def plot_stranded_profile(profile, ax=None, ymax=None, profile_std=None, flip_neg=True, set_ylim=True):
    """Plot the stranded profile
    """
    if ax is None:
        ax = plt.gca()

    if profile.ndim == 1:
        # also compatible with single dim
        profile = profile[:, np.newaxis]
    assert profile.ndim == 2
    assert profile.shape[1] <= 2
    labels = ['pos', 'neg']

    # determine ymax if not specified
    if ymax is None:
        if profile_std is not None:
            ymax = (profile.max() - 2 * profile_std).max()
        else:
            ymax = profile.max()

    if set_ylim:
        if flip_neg:
            ax.set_ylim([-ymax, ymax])
        else:
            ax.set_ylim([0, ymax])

    ax.axhline(y=0, linewidth=1, linestyle='--', color='black')
    # strip_axis(ax)

    xvec = np.arange(1, len(profile) + 1)

    for i in range(profile.shape[1]):
        sign = 1 if not flip_neg or i == 0 else -1
        ax.plot(xvec, sign * profile[:, i], label=labels[i])

        # plot also the ribbons
        if profile_std is not None:
            ax.fill_between(xvec,
                            sign * profile[:, i] - 2 * profile_std[:, i],
                            sign * profile[:, i] + 2 * profile_std[:, i],
                            alpha=0.1)
    # return ax


def multiple_plot_stranded_profile(d_profile, figsize_tmpl=(4, 3), normalize=False):
    fig, axes = plt.subplots(1, len(d_profile),
                             figsize=(figsize_tmpl[0] * len(d_profile), figsize_tmpl[1]),
                             sharey=True)
    for i, (task, ax) in enumerate(zip(d_profile, axes)):
        arr = d_profile[task].mean(axis=0)
        if normalize:
            arr = arr / arr.max()
        plot_stranded_profile(arr, ax=ax, set_ylim=False)
        ax.set_title(task)
        if i == 0:
            ax.set_ylabel("Avg. counts")
            ax.set_xlabel("Position")
    fig.subplots_adjust(wspace=0)  # no space between plots
    return fig


def aggregate_profiles(profile_arr, n_bootstrap=None, only_idx=None):
    if only_idx is not None:
        return profile_arr[only_idx], None

    if n_bootstrap is not None:
        return bootstrap_mean(profile_arr, n=n_bootstrap)
    else:
        return profile_arr.mean(axis=0), None


def extract_signal(x, seqlets, rc_fn=lambda x: x[::-1, ::-1]):
    def optional_rc(x, is_rc):
        if is_rc:
            return rc_fn(x)
        else:
            return x
    return np.stack([optional_rc(x[s['example'], s['start']:s['end']], s['rc'])
                     for s in seqlets])


def plot_profiles(seqlets_by_pattern,
                  x,
                  tracks,
                  contribution_scores={},
                  figsize=(20, 2),
                  start_vec=None,
                  width=20,
                  legend=True,
                  rotate_y=90,
                  seq_height=1,
                  ymax=None,  # determine y-max
                  n_limit=35,
                  n_bootstrap=None,
                  flip_neg=False,
                  patterns=None,
                  fpath_template=None,
                  only_idx=None,
                  mkdir=False,
                  rc_fn=lambda x: x[::-1, ::-1]):
    """
    Plot the sequence profiles
    Args:
      x: one-hot-encoded sequence
      tracks: dictionary of profile tracks
      contribution_scores: optional dictionary of contribution scores

    """
    import matplotlib.pyplot as plt
    from concise.utils.plot import seqlogo_fig, seqlogo

    # Setup start-vec
    if start_vec is not None:
        if not isinstance(start_vec, list):
            start_vec = [start_vec] * len(patterns)
    else:
        start_vec = [0] * len(patterns)
        width = len(x)

    if patterns is None:
        patterns = list(seqlets_by_pattern)
    # aggregated profiles
    d_signal_patterns = {pattern:
                         {k: aggregate_profiles(
                             extract_signal(y, seqlets_by_pattern[pattern])[:, start_vec[ip]:(start_vec[ip] + width)],
                             n_bootstrap=n_bootstrap, only_idx=only_idx)
                          for k, y in tracks.items()}
                         for ip, pattern in enumerate(patterns)}
    if ymax is None:
        # infer ymax
        def take_max(x, dx):
            if dx is None:
                return x.max()
            else:
                # HACK - hard-coded 2
                return (x + 2 * dx).max()

        ymax = [max([take_max(*d_signal_patterns[pattern][k])
                     for pattern in patterns])
                for k in tracks]  # loop through all the tracks
    if not isinstance(ymax, list):
        ymax = [ymax] * len(tracks)

    figs = []
    for i, pattern in enumerate(tqdm(patterns)):
        j = i
        # --------------
        # extract signal
        seqs = extract_signal(x, seqlets_by_pattern[pattern])[:, start_vec[i]:(start_vec[i] + width)]
        ext_contribution_scores = {s: extract_signal(contrib, seqlets_by_pattern[pattern])[:, start_vec[i]:(start_vec[i] + width)]
                                   for s, contrib in contribution_scores.items()}
        d_signal = d_signal_patterns[pattern]
        # --------------
        if only_idx is None:
            sequence = ic_scale(seqs.mean(axis=0))
        else:
            sequence = seqs[only_idx]

        n = len(seqs)
        if n < n_limit:
            continue
        fig, ax = plt.subplots(1 + len(contribution_scores) + len(tracks),
                               1, sharex=True,
                               figsize=figsize,
                               gridspec_kw={'height_ratios': [1] * len(tracks) + [seq_height] * (1 + len(contribution_scores))})

        # signal
        ax[0].set_title(f"{pattern} ({n})")
        for i, (k, signal) in enumerate(d_signal.items()):
            signal_mean, signal_std = d_signal_patterns[pattern][k]
            plot_stranded_profile(signal_mean, ax=ax[i], ymax=ymax[i],
                                  profile_std=signal_std, flip_neg=flip_neg)
            simple_yaxis_format(ax[i])
            strip_axis(ax[i])
            ax[i].set_ylabel(f"{k}", rotation=rotate_y, ha='right', labelpad=5)

            if legend:
                ax[i].legend()

        # -----------
        # contribution scores (seqlogo)
        # -----------
        # average the contribution scores
        if only_idx is None:
            norm_contribution_scores = {k: v.mean(axis=0)
                                        for k, v in ext_contribution_scores.items()}
        else:
            norm_contribution_scores = {k: v[only_idx]
                                        for k, v in ext_contribution_scores.items()}

        max_scale = max([np.maximum(v, 0).sum(axis=-1).max() for v in norm_contribution_scores.values()])
        min_scale = min([np.minimum(v, 0).sum(axis=-1).min() for v in norm_contribution_scores.values()])
        for k, (contrib_score_name, logo) in enumerate(norm_contribution_scores.items()):
            ax_id = len(tracks) + k

            # Trim the pattern if necessary
            # plot
            ax[ax_id].set_ylim([min_scale, max_scale])
            ax[ax_id].axhline(y=0, linewidth=1, linestyle='--', color='grey')
            seqlogo(logo, ax=ax[ax_id])

            # style
            simple_yaxis_format(ax[ax_id])
            strip_axis(ax[ax_id])
            # ax[ax_id].set_ylabel(contrib_score_name)
            ax[ax_id].set_ylabel(contrib_score_name, rotation=rotate_y, ha='right', labelpad=5)  # va='bottom',

        # -----------
        # information content (seqlogo)
        # -----------
        # plot
        seqlogo(sequence, ax=ax[-1])

        # style
        simple_yaxis_format(ax[-1])
        strip_axis(ax[-1])
        ax[-1].set_ylabel("Inf. content", rotation=rotate_y, ha='right', labelpad=5)
        ax[-1].set_xticks(list(range(0, len(sequence) + 1, 5)))

        figs.append(fig)
        # save to file
        if fpath_template is not None:
            pname = pattern.replace("/", ".")
            basepath = fpath_template.format(pname=pname, pattern=pattern)
            if mkdir:
                os.makedirs(os.path.dirname(basepath), exist_ok=True)
            plt.savefig(basepath + '.png', dpi=600)
            plt.savefig(basepath + '.pdf', dpi=600)
            plt.close(fig)    # close the figure
            show_figure(fig)
            plt.show()
    return figs


def plot_profiles_single(seqlet,
                         x,
                         tracks,
                         contribution_scores={},
                         figsize=(20, 2),
                         legend=True,
                         rotate_y=90,
                         seq_height=1,
                         flip_neg=False,
                         rc_fn=lambda x: x[::-1, ::-1]):
    """
    Plot the sequence profiles
    Args:
      x: one-hot-encoded sequence
      tracks: dictionary of profile tracks
      contribution_scores: optional dictionary of contribution scores

    """
    import matplotlib.pyplot as plt
    from concise.utils.plot import seqlogo_fig, seqlogo

    # --------------
    # extract signal
    seq = seqlet.extract(x)
    ext_contribution_scores = {s: seqlet.extract(contrib) for s, contrib in contribution_scores.items()}

    fig, ax = plt.subplots(1 + len(contribution_scores) + len(tracks),
                           1, sharex=True,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': [1] * len(tracks) + [seq_height] * (1 + len(contribution_scores))})

    # signal
    for i, (k, signal) in enumerate(tracks.items()):
        plot_stranded_profile(seqlet.extract(signal), ax=ax[i],
                              flip_neg=flip_neg)
        simple_yaxis_format(ax[i])
        strip_axis(ax[i])
        ax[i].set_ylabel(f"{k}", rotation=rotate_y, ha='right', labelpad=5)

        if legend:
            ax[i].legend()

    # -----------
    # contribution scores (seqlogo)
    # -----------
    max_scale = max([np.maximum(v, 0).sum(axis=-1).max() for v in ext_contribution_scores.values()])
    min_scale = min([np.minimum(v, 0).sum(axis=-1).min() for v in ext_contribution_scores.values()])
    for k, (contrib_score_name, logo) in enumerate(ext_contribution_scores.items()):
        ax_id = len(tracks) + k
        # plot
        ax[ax_id].set_ylim([min_scale, max_scale])
        ax[ax_id].axhline(y=0, linewidth=1, linestyle='--', color='grey')
        seqlogo(logo, ax=ax[ax_id])

        # style
        simple_yaxis_format(ax[ax_id])
        strip_axis(ax[ax_id])
        # ax[ax_id].set_ylabel(contrib_score_name)
        ax[ax_id].set_ylabel(contrib_score_name, rotation=rotate_y, ha='right', labelpad=5)  # va='bottom',

    # -----------
    # information content (seqlogo)
    # -----------
    # plot
    seqlogo(seq, ax=ax[-1])

    # style
    simple_yaxis_format(ax[-1])
    strip_axis(ax[-1])
    ax[-1].set_ylabel("Inf. content", rotation=rotate_y, ha='right', labelpad=5)
    ax[-1].set_xticks(list(range(0, len(seq) + 1, 5)))
    return fig


def hist_position(dfp, tasks):
    """Make the positional histogram

    Args:
      dfp: pd.DataFrame with columns: peak_id, and center
      tasks: list of tasks for which to plot the different peak_id columns
    """
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 2),
                             sharey=True, sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    for i, (task, ax) in enumerate(zip(tasks, axes)):
        ax.hist(dfp[dfp.peak_id == task].center, bins=100)
        ax.set_title(task)
        ax.set_xlabel("Position")
        ax.set_xlim([0, 1000])
        if i == 0:
            ax.set_ylabel("Frequency")
    plt.subplots_adjust(wspace=0)
    return fig


def bar_seqlets_per_example(dfp, tasks):
    """Make the positional histogram

    Args:
      dfp: pd.DataFrame with columns: peak_id, and center
      tasks: list of tasks for which to plot the different peak_id columns
    """
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 2),
                             sharey=True, sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    for i, (task, ax) in enumerate(zip(tasks, axes)):
        dfpp = dfp[dfp.peak_id == task]
        ax.set_title(task)
        ax.set_xlabel("Frequency")
        if i == 0:
            ax.set_ylabel("# per example")
        if not len(dfpp):
            continue
        dfpp.groupby("example_idx").\
            size().value_counts().plot(kind="barh", ax=ax)
    plt.subplots_adjust(wspace=0)
    return fig


def box_counts(total_counts, pattern_idx):
    """Make a box-plot with total counts in the region

    Args:
      total_counts: dict per task
      pattern_idx: array with example_idx of the pattern 
    """
    dfs = pd.concat([total_counts.melt().assign(subset="all peaks"),
                     total_counts.iloc[pattern_idx].melt().assign(subset="contains pattern")])
    dfs.value = np.log10(1 + dfs.value)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot("variable", "value", hue="subset", data=dfs, ax=ax)
    ax.set_xlabel("Task")
    ax.set_ylabel("log10(1+counts)")
    ax.set_title("Total number of counts in the region")
    return fig
