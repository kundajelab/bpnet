"""Vdom visualization for modisco
"""
import pandas as pd
from bpnet.plot.heatmaps import multiple_heatmap_stranded_profile, multiple_heatmap_contribution_profile, heatmap_sequence
from bpnet.cli.contrib import ContribFile
from collections import OrderedDict
from bpnet.plot.profiles import extract_signal, multiple_plot_stranded_profile, hist_position, bar_seqlets_per_example, box_counts
from bpnet.functions import mean
import numpy as np
import pandas as pd
from vdom.helpers import (h1, p, li, img, div, b, br, ul, img,
                          details, summary,
                          table, thead, th, tr, tbody, td, ol)
import io
import base64
import urllib
import matplotlib.pyplot as plt
import os


def fig2vdom(fig, **kwargs):
    """Convert a matplotlib figure to an online image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    plt.close()
    return img(src='data:image/png;base64,' + urllib.parse.quote(string), **kwargs)


def n_seqlets(self, metacluster, pattern):
    pattern_grp = self.get_pattern_grp(metacluster, pattern)
    return pattern_grp['seqlets_and_alnmts/seqlets'].shape[0]


def get_pattern_grp(mr, metacluster, pattern):
    return mr.f.f[f'/metacluster_idx_to_submetacluster_results/{metacluster}/seqlets_to_patterns_result/patterns/{pattern}']


def vdom_pssm(pssm, letter_width=0.2, letter_height=0.8, **kwargs):
    """Nicely plot the pssm
    """
    import matplotlib.pyplot as plt
    from concise.utils.plot import seqlogo_fig, seqlogo
    fig, ax = plt.subplots(figsize=(letter_width * len(pssm), letter_height))
    ax.axison = False
    seqlogo(pssm, ax=ax)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig2vdom(fig, **kwargs)


def vdom_footprint(arr, r_height=None, text=None,
                   fontsize=32, figsize=(3, 1), **kwargs):
    """Plot the sparkline for the footprint

    Args:
      arr: np.array of shape (seq_len, 2)
      r_height: if not None, add a rectangle with heigth = r_height
      text: add additional text to top right corner
      fontsize: size of the additional font
      figsize: figure size
      **kwargs: additional kwargs passed to `fig2vdom`

    Returns:
      VDOM object containing the image
    """
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=figsize)
    # print(arr.max())
    if r_height is not None:
        rect = patches.Rectangle((0, 0), len(arr),
                                 r_height,  # / arr.max(),
                                 linewidth=1,
                                 edgecolor=None,
                                 alpha=0.3,
                                 facecolor='lightgrey')
        ax.add_patch(rect)
        ax.set_ylim([0, max(r_height, arr.max())])
        ax.axhline(r_height, alpha=0.3, color='black', linestyle='dashed')
    ax.plot(arr[:, 0])
    ax.plot(arr[:, 1])

    if text is not None:
        # Annotate text top-left
        pass
        ax.text(1, 1, text,
                fontsize=fontsize,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right')

    ax.axison = False
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig2vdom(fig, **kwargs)


def template_vdom_pattern(name, n_seqlets, trimmed_motif,
                          full_motif, figures_url, add_plots={}, metacluster=""):

    return details(summary(name, f": # seqlets: {n_seqlets}",
                           # br(),
                           trimmed_motif),  # ", rc: ",  motif_rc),
                   details(summary("Aggregated profiles and contribution scores)"),
                           img(src=figures_url + "/agg_profile_contribcores.png", width=840),
                           ),
                   details(summary("Aggregated hypothetical contribution scores)"),
                           img(src=figures_url + "/agg_profile_hypcontribscores.png", width=840),
                           ),
                   details(summary("Sequence"),
                           full_motif,
                           br(),
                           img(src=figures_url + "/heatmap_seq.png", width=840 // 2),
                           ),
                   details(summary("ChIP-nexus counts"),
                           img(src=figures_url + "/profile_aggregated.png", width=840),
                           img(src=figures_url + "/profile_heatmap.png", width=840),
                           ),
                   details(summary("Contribution scores (profile)"),
                           img(src=figures_url + "/contrib_profile.png", width=840),
                           ),
                   details(summary("Contribution scores (counts)"),
                           img(src=figures_url + "/contrib_counts.png", width=840),
                           ),
                   *[details(summary(k), *v) for k, v in add_plots.items()],
                   id=metacluster + "/" + name
                   )


def vdom_pattern(mr, metacluster, pattern,
                 figdir,
                 total_counts,
                 dfp,
                 trim_frac=0.05,
                 letter_width=0.2, height=0.8):

    # get the trimmed motifs
    trimmed_motif = vdom_pssm(mr.get_pssm(metacluster, pattern,
                                          rc=False, trim_frac=trim_frac),
                              letter_width=letter_width,
                              height=height)
    full_motif = vdom_pssm(mr.get_pssm(metacluster, pattern,
                                       rc=False, trim_frac=0),
                           letter_width=letter_width,
                           height=height)

    # ----------------
    # add new plots here
    dfpp = dfp[dfp.pattern == (metacluster + "/" + pattern)]
    tasks = dfp.peak_id.unique()
    pattern_idx = dfpp.example_idx.unique()
    add_plots = OrderedDict([
        ("Positional distribution",
         [fig2vdom(hist_position(dfpp, tasks=tasks)),
          fig2vdom(bar_seqlets_per_example(dfpp, tasks=tasks))
          ]),
        ("Total count distribution",
         [p(f"Pattern occurs in {len(pattern_idx)} / {len(total_counts)} regions"
            f" ({100*len(pattern_idx)/len(total_counts):.1f}%)"),
          fig2vdom(box_counts(total_counts, pattern_idx))]
         )
    ])
    # ----------------

    return template_vdom_pattern(name=pattern,
                                 n_seqlets=mr.n_seqlets(metacluster, pattern),
                                 trimmed_motif=trimmed_motif,
                                 full_motif=full_motif,
                                 figures_url=os.path.join(figdir, f"{metacluster}/{pattern}"),
                                 add_plots=add_plots,
                                 metacluster=metacluster,
                                 )


def template_vdom_metacluster(name, n_patterns, n_seqlets, important_for, patterns, is_open=False):
    return details(summary(b(name), f", # patterns: {n_patterns},"
                           f" # seqlets: {n_seqlets}, "
                           "important for: ", b(important_for)),
                   ul([li(pattern) for pattern in patterns], start=0),
                   id=name,
                   open=is_open)


def vdom_metacluster(mr, metacluster, figdir, total_counts, dfp=None, is_open=True,
                     **kwargs):
    patterns = mr.patterns(metacluster)
    n_seqlets = sum([mr.n_seqlets(metacluster, pattern)
                     for pattern in patterns])
    n_patterns = len(patterns)

    def render_act(task, act):
        """Render the activity vector
        """
        task = task.replace("/weighted", "").replace("/profile", "")  # omit weighted or profile
        if act == 0:
            return ""
        elif act < 0:
            return f"-{task}"
        else:
            return task
    activities = mr.metacluster_activity(metacluster)

    # tasks = mr.tasks()
    # tasks = unique_list([task.split("/")[0] for task in tasks])  # HACK. For some
    # TODO - one could pretify this here by using Task, and cTask

    important_for = ",".join([render_act(task, act)
                              for task, act in zip(mr.tasks(), activities)
                              if act != 0])
    pattern_vdoms = [vdom_pattern(mr, metacluster, pattern, figdir, total_counts,
                                  dfp, **kwargs)
                     for pattern in patterns]
    return template_vdom_metacluster(metacluster,
                                     n_patterns,
                                     n_seqlets,
                                     important_for,
                                     pattern_vdoms,
                                     is_open=is_open
                                     )


def vdom_modisco(mr, figdir, total_counts, dfp=None, is_open=True, **kwargs):
    return div([vdom_metacluster(mr, metacluster, figdir, total_counts, dfp=dfp,
                                 is_open=is_open, **kwargs)
                for metacluster in mr.metaclusters()
                if len(mr.patterns(metacluster)) > 0])


def get_signal(seqlets, d: ContribFile, tasks, resize_width=200):
    thr_one_hot = d.get_seq()

    if resize_width is None:
        # width = first seqlets
        resize_width = seqlets[0].end - seqlets[0].start

    # get valid seqlets
    start_pad = np.ceil(resize_width / 2)
    end_pad = thr_one_hot.shape[1] - start_pad
    valid_seqlets = [s.resize(resize_width)
                     for s in seqlets
                     if (s.center() > start_pad) and (s.center() < end_pad)]

    # prepare data
    ex_signal = {task: extract_signal(d.get_profiles()[task], valid_seqlets)
                 for task in tasks}

    ex_contrib_profile = {task: extract_signal(d.get_contrib()[task],
                                               valid_seqlets).sum(axis=-1)
                          for task in tasks}

    if d.contains_contrib_score('count'):
        ex_contrib_counts = {task: extract_signal(d.get_contrib("count")[task],
                                                  valid_seqlets).sum(axis=-1) for task in tasks}
    elif d.contains_contrib_score('counts/pre-act'):
        ex_contrib_counts = {task: extract_signal(d.get_contrib("counts/pre-act")[task],
                                                  valid_seqlets).sum(axis=-1) for task in tasks}
    else:
        ex_contrib_counts = None

    ex_seq = extract_signal(thr_one_hot, valid_seqlets)

    seq, contrib, hyp_contrib, profile, ranges = d.get_all()

    total_counts = sum([x.sum(axis=-1).sum(axis=-1) for x in ex_signal.values()])
    sort_idx = np.argsort(-total_counts)
    return ex_signal, ex_contrib_profile, ex_contrib_counts, ex_seq, sort_idx


def vdm_heatmaps(seqlets, d, included_samples, tasks, pattern, top_n=None, pssm_fig=None, opened=False):
    ex_signal, ex_contrib_profile, ex_contrib_counts, ex_seq, sort_idx = get_signal(seqlets, d, included_samples, tasks)

    if top_n is not None:
        sort_idx = sort_idx[:top_n]
    return div(details(summary("Sequence:"),
                       pssm_fig,
                       br(),
                       fig2vdom(heatmap_sequence(ex_seq, sort_idx=sort_idx, figsize_tmpl=(10, 15), aspect='auto')),
                       open=opened
                       ),

               details(summary("ChIP-nexus counts:"),
                       fig2vdom(multiple_plot_stranded_profile(ex_signal, figsize_tmpl=(20 / len(ex_signal), 3))),
                       # TODO - change
                       fig2vdom(multiple_heatmap_stranded_profile(ex_signal, sort_idx=sort_idx, figsize=(20, 20))),
                       open=opened
                       ),
               details(summary("Contribution scores (profile)"),
                       fig2vdom(multiple_heatmap_contribution_profile(ex_contrib_profile, sort_idx=sort_idx, figsize=(20, 20))),
                       open=opened
                       ),
               details(summary("Contribution scores (counts)"),
                       fig2vdom(multiple_heatmap_contribution_profile(ex_contrib_counts, sort_idx=sort_idx, figsize=(20, 20))),
                       open=opened
                       )
               )


def write_heatmap_pngs(seqlets, d, tasks, pattern, output_dir):
    """Write out histogram png's
    """
    # get the data
    ex_signal, ex_contrib_profile, ex_contrib_counts, ex_seq, sort_idx = get_signal(seqlets, d, tasks)
    # get the plots
    figs = dict(
        heatmap_seq=heatmap_sequence(ex_seq, sort_idx=sort_idx, figsize_tmpl=(10, 15), aspect='auto'),
        profile_aggregated=multiple_plot_stranded_profile(ex_signal, figsize_tmpl=(20 / len(ex_signal), 3)),
        profile_heatmap=multiple_heatmap_stranded_profile(ex_signal, sort_idx=sort_idx, figsize=(20, 20)),
        contrib_profile=multiple_heatmap_contribution_profile(ex_contrib_profile, sort_idx=sort_idx, figsize=(20, 20)),
    )

    if ex_contrib_counts is not None:
        figs['contrib_counts'] = multiple_heatmap_contribution_profile(ex_contrib_counts, sort_idx=sort_idx, figsize=(20, 20))
    # write the figures
    for k, fig in figs.items():
        fig.savefig(os.path.join(output_dir, k + ".png"), bbox_inches='tight')


def df2html(df, uuid='table', style='width:100%'):
    import seaborn as sns
    cm = sns.light_palette("green", as_cmap=True)
    # leverage pandas style to color cells according to values
    # https://pandas.pydata.org/pandas-docs/stable/style.html
    s = df.style.background_gradient(cmap=cm).set_precision(3).hide_index()
    return s.render(uuid=uuid).replace(f'<table id="T_{uuid}"',
                                       f'<table id="T_{uuid}" class="compact hover nowrap" style="{style}"')


def df2html_old(df, style='width:100%'):
    add_tags = f'id="table_id" style="{style}"'
    with pd.option_context('display.max_colwidth', -1):
        table = df.to_html(escape=False,
                           classes='display nowrap',
                           float_format='%.2g',
                           index=False).replace(' class="dataframe', f' {add_tags} class="dataframe')
    return table

# <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.13/css/jquery.dataTables.css">

def get_datatable_header():
    return '''
      <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
      <script type="text/javascript"  src="https://cdn.datatables.net/1.10.19/js/jquery.dataTables.min.js"></script>
      <script type="text/javascript"  src="https://cdn.datatables.net/colreorder/1.5.1/js/dataTables.colReorder.min.js"></script>
      <script type="text/javascript"  src="https://cdn.datatables.net/fixedcolumns/3.2.6/js/dataTables.fixedColumns.min.js"></script>
      
      <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.19/css/jquery.dataTables.min.css">     
      <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/colreorder/1.5.1/css/colReorder.dataTables.min.css">
      <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/fixedcolumns/3.2.6/css/fixedColumns.dataTables.min.css">
      <link rel="stylesheet" href="https://cdn.jupyter.org/notebook/5.1.0/style/style.min.css">
    '''


def style_html_table_datatable(html_str):
    from IPython.display import HTML, Javascript

    header = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    {get_datatable_header()}
    <head>
    </body>
        '''
    script = '''
    <script>
    $(document).ready( function () {
    var table = $('#T_table').DataTable({
         scrollX: true,
         scrollY: '80vh',
         scrollCollapse: true,
         paging: false,
         colReorder: true,
         columnDefs: [
            { orderable: false, targets: 0 },
            { orderable: false, targets: 1 }
        ],
        ordering: [[ 1, 'asc' ]],
        colReorder: {
            fixedColumnsLeft: 1,
            fixedColumnsRight: 0
        }
    });

    new $.fn.dataTable.FixedColumns( table, {
        leftColumns: 3,
        rightColumns: 0
    } );

    // Select rows
    $('#T_table tbody').on( 'click', 'tr', function () {
        $(this).toggleClass('selected');
    } );

    } );
    </script>
    </body>
    </html>
    '''

    return header + html_str + script

def write_datatable_html(df, output_file, other=""):
    html = style_html_table_datatable(df2html(df) + other)
    with open(output_file, "w") as f:
        f.write(html)


def render_datatable(df):
    from IPython.display import HTML, Javascript, display
    display(HTML(get_datatable_header() + df2html(df)))
    # display(Javascript(""" $(document).ready( function () {
    # $('#T_table').DataTable();
    # } );"""))


def footprint_df(footprints, dfl=None, width=120, **kwargs):
    """Draw footprints sparklines into a pandas.DataFrame

    Args:
      footprints: footprint dict with `<pattern>/<task>` nested structure
        each node contains an array of shape (seq_len, 2)
      dfl: optional pandas.DataFrame of labels. Contains columns:
        pattern <task>/l
      width: width of the final plot
      **kwargs: additional kwargs to pass to vdom_footprint
    """
    from tqdm import tqdm
    from bpnet.modisco.utils import shorten_pattern

    def map_label(l):
        """Label -> short-name
        """
        # TODO - get rid of this function
        if l is None:
            return "/"
        else:
            return l[0].upper()
    tasks = list(footprints[list(footprints)[0]].keys())
    profile_max_median = {task: np.median([np.max(v[task]) for v in footprints.values()]) for task in tasks}
    out = []

    for p, arr_d in tqdm(footprints.items()):
        try:
            labels = dfl[dfl.pattern == shorten_pattern(p)].iloc[0].to_dict()
        except Exception:
            labels = {t + "/l": None for t in tasks}
        d = {task: vdom_footprint(arr_d[task],
                                  r_height=profile_max_median[task],
                                  text=map_label(labels[task + "/l"]),
                                  **kwargs).to_html().replace("<img",
                                                              f"<img width={width}")
             for task in tasks}
        d['pattern'] = shorten_pattern(p)
        out.append(d)
    return pd.DataFrame(out)
