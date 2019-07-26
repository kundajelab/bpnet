"""Shared utils between plots
"""
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np


def show_figure(fig):
    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)


def simple_yaxis_format(ax, n=1):
    """Use a single-digit format for the y-axis
    """
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter(f'%.{n}f'))


def strip_axis(ax):
    """Omit top,right,bottom spines and x-ticks
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_ticks_position('none')


def spine_subset(ax, minimum, maximum):
    """Show the spines (e.g. axis lines) only for a subset of points
    """
    ax.spines['left'].set_bounds(minimum, maximum)
    ax.set_yticks([minimum, maximum])


def stylize_axes(ax):
    # Drop top-right axis, also
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ticks should point outwards
    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)


# TODO - replace with MaxNLocator
# https://matplotlib.org/api/ticker_api.html#matplotlib.ticker.MaxNLocator
def spaced_xticks(ax, spacing=5):
    ax.set_xticks(list(range(int(ax.get_xlim()[0]) + 1,
                             int(ax.get_xlim()[1]) + 1,
                             int(spacing))))


def draw_box(start, end, ax, col='r', alpha=0.1):
    """
    Plot genomic intervals

    Args:
      start: start x coordinate
      end: end x coordinate
      ax: matplotlib axis
      col: plot color
      alpha: plot alpha
    """

    # get axis height
    ymin = ax.get_ylim()[0]
    ywidth = ax.get_ylim()[1] - ax.get_ylim()[0]

    rects = [Rectangle((start, ymin), end - start, ywidth, fill=False)]
    ax.add_collection(PatchCollection(rects, facecolor=col, alpha=alpha, edgecolor=col))


def draw_hline(start, end, y, ax, col='r', linewidth=5, alpha=0.1):
    from matplotlib.collections import LineCollection
    lc = LineCollection([[(start, y), (end, y)]],
                        colors=col,
                        linewidths=linewidth,
                        alpha=alpha)
    ax.add_collection(lc)


# set the colormap and centre the colorbar
class MidpointNormalize(plt.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        plt.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class PlotninePalette:
    """Use matplotlib colors with plotnine.scale_fill_gradient2

    ```
    scale_fill_gradient2(low=cb_red, mid='white', high=cb_blue, midpoint=.5, limits=[0, 1], 
                            palette=PlotninePalette("RdBu", [.1, .9]))
    ```
    """

    def __init__(self, mpl_cmap, limits=[0, 1]):
        """
        Args:
          mpl_cmap (str): Matplotlib colormap
          limits: what range of the colormap to explore. For example: [.1, .9] only
            explores the .1-.9 range of the colormap
          reverse: if True, the colormap is reversed
        """
        self.mpl_cmap = mpl_cmap
        self.limits = limits
        # self.reverse = reverse

    def __call__(self, values):
        # if self.reverse:
        #     values = 1 - values
        values = self.limits[0] + values * (self.limits[1] - self.limits[0])
        return [plt.cm.colors.to_hex(x) for x in plt.get_cmap(self.mpl_cmap)(values)]


def seqlogo_clean(seq, letter_width=0.2, height=0.8, title=None):
    import matplotlib.pyplot as plt
    from concise.utils.plot import seqlogo
    fig, ax = plt.subplots(figsize=(letter_width * len(seq), height))
    ax.axison = False
    seqlogo(seq, ax=ax)
    if title is not None:
        ax.set_title(title)
    else:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot_colormap(cmap_dict, label='', ax=None, figwidth=.5):
    """Plot a color map

    Args:
      cmap_dict: color map dictionary: key: color (hex or rgb)
      label: name of the colormap (e.g. x-axix
    """
    import matplotlib as mpl

    D = 1 / len(cmap_dict) / 2

    colors = list(cmap_dict.values())
    keys = list(cmap_dict.keys())

    @mpl.ticker.FuncFormatter
    def major_formatter(x, pos):
        return {D + i / len(keys): k
                for i, k in enumerate(keys)}[x]

    if ax is None:
        fig, ax = plt.subplots(figsize=(figwidth, figwidth * len(cmap_dict)))

    cb3 = mpl.colorbar.ColorbarBase(ax,
                                    cmap=mpl.colors.ListedColormap(colors),
                                    ticks=[D + i / len(keys) for i in range(len(keys))],
                                    format=major_formatter,
                                    spacing='uniform',
                                    orientation='vertical')
    cb3.set_label(label)


# TODO - remove?
def plt9_tilt_xlab(angle=45):
    from plotnine import theme, element_text
    return theme(axis_text_x=element_text(rotation=angle, hjust=1))
