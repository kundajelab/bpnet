"""Evaluation plots
"""
import numpy as np
import matplotlib.pyplot as plt


def regression_eval(y_true, y_pred, alpha=0.5, markersize=2, task="",
                    ax=None, same_lim=False, loglog=False, use_density=False, rasterized=True):

    if ax is None:
        fig, ax = plt.subplots(1)
    from scipy.stats import pearsonr, spearmanr
    xmax = max([y_true.max(), y_pred.max()])
    xmin = min([y_true.min(), y_pred.min()])

    if loglog:
        pearson, pearson_pval = pearsonr(np.log10(y_true), np.log10(y_pred))
        spearman, spearman_pval = spearmanr(np.log10(y_true), np.log(y_pred))
    else:
        pearson, pearson_pval = pearsonr(y_true, y_pred)
        spearman, spearman_pval = spearmanr(y_true, y_pred)
    if loglog:
        plt_fn = ax.loglog
    else:
        plt_fn = ax.plot

    plt_fn(y_pred, y_true, ".",
           markersize=markersize,
           rasterized=rasterized,
           alpha=alpha)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")

    if same_lim:
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((xmin, xmax))
    rp = r"$R_{p}$"
    rs = r"$R_{s}$"
    ax.set_title(task)
    ax.text(.95, .2, f"{rp}={pearson:.2f}",
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes)
    ax.text(.95, .05, f"{rs}={spearman:.2f}",
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes)


def plot_loss(dfh, metrics, figsize=(8, 3)):
    """
    Plot the loss-curves.

    Args:
      dfh: pd.DataFrame obtained from history.csv keras file
      metrics: list of metrics to plot. could be also a single string
      figsize: output figure size
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    # plt.figure(figsize=figsize)
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        axes[i].plot(dfh[metric], label='train')
        axes[i].plot(dfh["val_" + metric], label='val')
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("Epoch")
        axes[i].set_title(metric)
        axes[i].legend()
    plt.tight_layout()
