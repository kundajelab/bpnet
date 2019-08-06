"""Implementes ModiscoResult class
"""
from collections import OrderedDict, defaultdict
import os
from copy import deepcopy
import yaml
from bpnet.functions import mean
import h5py
from concise.utils.helper import write_json, read_json
from kipoi.readers import HDF5Reader
import numpy as np
from pybedtools import BedTool, Interval
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from bpnet.plot.utils import show_figure
from bpnet.modisco.utils import bootstrap_mean, nan_like, ic_scale
import matplotlib.pyplot as plt
import seaborn as sns
import attr
from bpnet.plot.vdom import vdom_modisco
from bpnet.plot.utils import seqlogo_clean, strip_axis
from bpnet.extractors import Interval
from tqdm import tqdm
from bpnet.modisco.utils import shorten_pattern, longer_pattern


def parse_seqlet(s):
    return {x.split(":")[0]: yaml.load(x.split(":")[1])
            for x in s.decode("utf8").split(",")}


def group_seqlets(seqlets, by='seqname'):
    """Group seqlets by a certain feature
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for seqlet in seqlets:
        groups[getattr(seqlet, by)].append(seqlet)
    return groups


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


def match_seqlets(unlabelled_seqlets, labelled_seqlets):
    """Map unlabelled seqlets to labelled ones

    Args:
      unlabelled_seqlets List[Seqlet]: seqlets without a label
      labelled_seqlets List[Seqlet]: seqlets with name corresponding
          to a specific class

    Returns:
        seqlets of the same length as unlabelled_seqlets but now with
        name being the class.
    """
    g_unlabelled_seqlets = group_seqlets(unlabelled_seqlets, by="seqname")
    g_labelled_seqlets = group_seqlets(labelled_seqlets, by="seqname")

    out_seqlets = []
    for seqname, example_unlabelled_seqlets in g_unlabelled_seqlets.items():
        # Get the labelled seqlets for that seqname / seqlet
        example_labelled_seqlets = g_labelled_seqlets.get(seqname, [])

        # for each unlabelled seqlet...
        for unlabelled_seqlet in example_unlabelled_seqlets:
            # Try to find the right labelled seqlet
            seqlet_matched = False
            for labelled_seqlet in example_labelled_seqlets:
                if labelled_seqlet.contains(unlabelled_seqlet):
                    # write the labelled_seqlet
                    out_seqlets.append(labelled_seqlet.copy())
                    seqlet_matched = True
                    break
            if not seqlet_matched:
                # No matches found
                # write the unlabelled seqlet
                out = unlabelled_seqlet.copy()
                out.name = None  # No label
                out_seqlets.append(out)
    return out_seqlets


# TODO - rename this to ModiscoFile


class ModiscoResult:

    def __init__(self, fpath):
        self.fpath = fpath
        self.f = HDF5Reader(self.fpath)
        self.f.open()

        # example ranges. loaded when needed
        self.ranges = None

    def open(self):
        self.f.open()
        return self

    def close(self):
        self.f.close()

    def tasks(self):
        return list(self.f.f['task_names'][:].astype(str))

    def patterns(self, metacluster=None):
        """List all detected patterns:

        output format: {metacluster}/{pattern}
        """
        if metacluster is None:
            # Display all
            return [f"{metacluster}/{pattern}"
                    for metacluster in self.metaclusters()
                    for pattern in self.patterns(metacluster)]
        else:
            try:
                patterns = (list(self.f.f
                                 ["metacluster_idx_to_submetacluster_results"]
                                 [f"{metacluster}"]
                                 ["seqlets_to_patterns_result"]
                                 ["patterns"]["all_pattern_names"]))
            except KeyError:
                patterns = []
            return [x.decode("utf8") for x in patterns]

    def get_all_patterns(self):
        return [self.get_pattern(pn)
                for pn in self.patterns()]

    def get_pattern(self, pattern):
        """Get the pattern name
        """
        from bpnet.modisco.core import Pattern
        return Pattern.from_hdf5_grp(self.get_pattern_grp(*pattern.split("/")), pattern)

    def metaclusters(self):
        return list(self.f.f['/metaclustering_results/all_metacluster_names'][:].astype(str))

    def metacluster_activity(self, metacluster=None):
        if metacluster is not None:
            return self.f.f[f'/metacluster_idx_to_submetacluster_results/{metacluster}/activity_pattern'][:]
        else:
            dfa = pd.DataFrame([self.metacluster_activity(mc)
                                for mc in self.metaclusters()],
                               columns=self.tasks())
            dfa.index.name = "metacluster"
            return dfa

    def metacluster_stats(self):
        """Get metacluster stats by metacsluter
        """
        mc_stat = pd.DataFrame([p.split("/") + [len(s)]
                                for p, s in self.seqlets().items()],
                               columns=["metacluster", "pattern", "n"])
        mc_stat.pattern = mc_stat.pattern.str.replace("pattern_", "").astype(int)
        mc_stat.metacluster = mc_stat.metacluster.str.replace("metacluster_", "").astype(int)
        return mc_stat

    def plot_metacluster_activity(self, ax=None, figsize=None, **kwargs):
        dfa = self.metacluster_activity()
        if ax is None:
            if figsize is None:
                figsize = (0.5 * len(dfa), 0.5 * len(dfa.T))
            fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(dfa.T, cmap=sns.color_palette("RdBu_r", 3),
                    linewidths=1,
                    ax=ax, **kwargs)
        ax.set_xlabel("Metacluster")
        ax.set_ylabel("Tasks")

    def parse_seqlet(cls, s):
        return Seqlet.from_dict({x.split(":")[0]: yaml.load(x.split(":")[1])
                                 for x in s.decode("utf8").split(",")})

    def all_seqlets(self, label=False):
        """Returns all discovered seqlets
        """
        if not label:
            return [self.parse_seqlet(x)
                    for x in self.f.f['/multitask_seqlet_creation_results/final_seqlets'][:]]
        else:
            return match_seqlets(self.all_seqlets(False),
                                 self.seqlets(None))

    def _get_seqlets(self, pattern, trim_frac=0):
        metacluster, pattern = pattern.split("/")
        i, j = trim_pssm_idx(self.get_pssm(metacluster, pattern), trim_frac)
        return [self.parse_seqlet(x).trim(i, j)
                for x in self.f.f[f'/metacluster_idx_to_submetacluster_results/{metacluster}/seqlets_to_patterns_result/patterns/{pattern}/seqlets_and_alnmts/seqlets'][:]]

    def seqlets(self, by='pattern', trim_frac=0):
        if by == 'pattern':
            return {p: self._get_seqlets(p, trim_frac=trim_frac)
                    for p in self.patterns()}
        elif by == 'example':
            examples = {}
            for p, seqlets in self.seqlets(by='pattern', trim_frac=trim_frac).items():
                for seqlet in seqlets:
                    seqlet.name = p
                    if seqlet.seqname in examples:
                        examples[seqlet.seqname].append(seqlet)
                    else:
                        examples[seqlet.seqname] = [seqlet]
            return examples
        elif by is None:
            def add_name(seqlet, name):
                seqlet.name = name
                return seqlet
            return [add_name(seqlet, p)
                    for p in self.patterns()
                    for seqlet in self._get_seqlets(p, trim_frac=trim_frac)]
        else:
            raise ValueError("by has to be from: {'pattern', 'example',None}")

    def extract_signal(self, x, pattern=None, rc_fn=lambda x: x[::-1, ::-1]):
        if pattern is None:
            # Back-compatibility, TODO - remove
            return {p: self.extract_signal(x, p, rc_fn) for p in self.patterns()}

        def optional_rc(x, is_rc):
            if is_rc:
                return rc_fn(x)
            else:
                return x
        return np.stack([optional_rc(x[s['example'], s['start']:s['end']], s['rc'])
                         for s in self._get_seqlets(pattern)])

    def plot_profiles(self, x, tracks,
                      contribution_scores={},
                      figsize=(20, 2),
                      rc_vec=None,
                      start_vec=None,
                      width=20,
                      legend=True,
                      seq_height=2,
                      ylim=[0, 3],
                      n_limit=35,
                      n_bootstrap=None,
                      patterns=None,
                      fpath_template=None,
                      rc_fn=lambda x: x[::-1, ::-1]):
        """
        Plot the sequence profiles
        Args:
          x: one-hot-encoded sequence
          tracks: dictionary of profile tracks
          contribution_scores: optional dictionary of contribution scores

        TODO - add the reverse complementation option to it
        """
        import matplotlib.pyplot as plt
        from concise.utils.plot import seqlogo_fig, seqlogo

        seqs_all = self.extract_signal(x)
        ext_contribution_scores = {s: self.extract_signal(contrib)
                                   for s, contrib in contribution_scores.items()}
        # TODO assert correct shape in contrib

        if patterns is None:
            patterns = self.patterns()
        for i, pattern in enumerate(patterns):
            j = i
            seqs = seqs_all[pattern]
            sequence = ic_scale(seqs.mean(axis=0))
            if rc_vec is not None and rc_vec[i]:
                rc_seq = True
                sequence = rc_fn(sequence)
            else:
                rc_seq = False
            if start_vec is not None:
                start = start_vec[i]
                sequence = sequence[start:(start + width)]
            n = len(seqs)
            if n < n_limit:
                continue
            fig, ax = plt.subplots(1 + len(contribution_scores) + len(tracks),
                                   1, sharex=True,
                                   figsize=figsize,
                                   gridspec_kw={'height_ratios': [1] * len(tracks) + [seq_height] * (1 + len(contribution_scores))})
            ax[0].set_title(f"{pattern} ({n})")
            for i, (k, y) in enumerate(tracks.items()):
                signal = self.extract_signal(y, rc_fn)[pattern]

                if start_vec is not None:
                    start = start_vec[i]
                    signal = signal[:, start:(start + width)]

                if n_bootstrap is None:
                    signal_mean = signal.mean(axis=0)
                    signal_std = signal.std(axis=0)
                else:
                    signal_mean, signal_std = bootstrap_mean(signal, n=n_bootstrap)
                if rc_vec is not None and rc_vec[i]:
                    signal_mean = rc_fn(signal_mean)
                    signal_std = rc_fn(signal_std)

                ax[i].plot(np.arange(1, len(signal_mean) + 1), signal_mean[:, 0], label='pos')
                if n_bootstrap is not None:
                    ax[i].fill_between(np.arange(1, len(signal_mean) + 1),
                                       signal_mean[:, 0] - 2 * signal_std[:, 0],
                                       signal_mean[:, 0] + 2 * signal_std[:, 0],
                                       alpha=0.1)
                #                   label='pos')
                # plot also the other strand
                if signal_mean.shape[1] == 2:
                    ax[i].plot(np.arange(1, len(signal_mean) + 1),
                               signal_mean[:, 1], label='neg')
                    if n_bootstrap is not None:
                        ax[i].fill_between(np.arange(1, len(signal_mean) + 1),
                                           signal_mean[:, 1] - 2 * signal_std[:, 1],
                                           signal_mean[:, 1] + 2 * signal_std[:, 1],
                                           alpha=0.1)
                #                   label='pos')
                ax[i].set_ylabel(f"{k}")
                ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['bottom'].set_visible(False)
                ax[i].xaxis.set_ticks_position('none')
                if isinstance(ylim[0], list):
                    ax[i].set_ylim(ylim[i])
                if legend:
                    ax[i].legend()

            for k, (contrib_score_name, values) in enumerate(ext_contribution_scores.items()):
                ax_id = len(tracks) + k
                logo = values[pattern].mean(axis=0)
                # Trim the pattern if necessary
                if rc_seq:
                    logo = rc_fn(logo)
                if start_vec is not None:
                    start = start_vec[j]
                    logo = logo[start:(start + width)]
                seqlogo(logo, ax=ax[ax_id])
                ax[ax_id].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax[ax_id].set_ylabel(contrib_score_name)
                ax[ax_id].spines['top'].set_visible(False)
                ax[ax_id].spines['right'].set_visible(False)
                ax[ax_id].spines['bottom'].set_visible(False)
                ax[ax_id].xaxis.set_ticks_position('none')

            seqlogo(sequence, ax=ax[-1])
            ax[-1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax[-1].set_ylabel("Inf. content")
            ax[-1].spines['top'].set_visible(False)
            ax[-1].spines['right'].set_visible(False)
            ax[-1].spines['bottom'].set_visible(False)
            ax[-1].set_xticks(list(range(0, len(sequence) + 1, 5)))

            if fpath_template is not None:
                pname = pattern.replace("/", ".")
                plt.savefig(fpath_template.format(pname) + '.png', dpi=600)
                plt.savefig(fpath_template.format(pname) + '.pdf', dpi=600)
                plt.close(fig)    # close the figure
                show_figure(fig)
                plt.show()

    def load_ranges(self):
        mdir = os.path.dirname(self.fpath)
        if self.ranges is None:
            from bpnet.cli.modisco import load_ranges
            self.ranges = load_ranges(mdir)
            self.ranges['example_idx'] = np.arange(len(self.ranges))
        return self.ranges

    def get_seqlet_intervals(self, pattern, trim_frac=0, as_df=False):
        import pybedtools
        ranges = self.load_ranges()
        dfs = self.seqlet_df_instances(patterns=[pattern], trim_frac=trim_frac)
        dfs = pd.merge(dfs, ranges, how='left',
                       left_on='seqname', right_on='example_idx',
                       suffixes=("_seqlet", "_example"))
        dfs['start'] = dfs['start_example'] + dfs['start_seqlet']
        dfs['end'] = dfs['start_example'] + dfs['end_seqlet']
        dfs['strand'] = dfs['strand_seqlet']

        dfs_intervals = dfs[['chrom', 'start', 'end', 'interval_from_task', 'seqname', 'strand', 'pattern']]
        if as_df:
            return dfs_intervals
        else:
            intervals = [Interval.from_pybedtools(pybedtools.create_interval_from_list(list(row.values)))
                         for i, row in dfs_intervals.iterrows()]
            return intervals

    def export_seqlets_bed(self, output_dir,
                           example_intervals=None,
                           trim_frac=0.08,
                           position='absolute',
                           gzip=True):
        """Export the seqlet positions to a bed file

        Args:
          output_file: output file path
          example_intervals: list of genomic intervals of the examples (regions where seqlets were extracted)
          position: 'relative' or 'absolute'
        """
        assert position in ['relative', 'absolute']
        if position == 'absolute':
            assert example_intervals is not None
            os.makedirs(output_dir, exist_ok=True)
            BedTool(example_intervals).saveas(os.path.join(output_dir, "scored_regions.bed"))

        exi = example_intervals
        for p in tqdm(self.patterns()):
            if position == 'relative':
                nintervals = [Interval(str(seqlet.seqname),
                                       seqlet.start,
                                       seqlet.end,
                                       p,  # name
                                       0,  # score
                                       seqlet.strand).to_pybedtools()  # strand
                              for seqlet in self._get_seqlets(p, trim_frac=trim_frac)]
            else:
                # absolute position
                nintervals = [Interval(exi[seqlet.seqname].chrom,
                                       exi[seqlet.seqname].start + seqlet.start,
                                       exi[seqlet.seqname].start + seqlet.end,
                                       p,  # name
                                       0,  # score
                                       seqlet.strand).to_pybedtools()  # strand
                              for seqlet in self._get_seqlets(p, trim_frac=trim_frac)]

            # write the bed-file
            fpath = os.path.join(output_dir, f"{p}.bed")
            if gzip:
                fpath = fpath + ".gz"
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            BedTool(nintervals).saveas(fpath)

    def seqlet_df_instances(self, patterns=None, trim_frac=0):
        if patterns is None:
            patterns = self.patterns()
        if isinstance(patterns, str):
            patterns = [patterns]
        dfp = pd.concat([pd.DataFrame([s.to_dict()
                                       for s in self._get_seqlets(pattern, trim_frac=trim_frac)]).
                         assign(pattern=pattern)
                         for pattern in patterns
                         ])
        dfp['center'] = (dfp.end + dfp.start) / 2
        return dfp

    def stats(self, verbose=True):
        mc_stat = self.metacluster_stats()
        stats = {"patterns": len(self.patterns()),
                 "clustered_seqlets": mc_stat.n.sum(),
                 "metaclusters": len(self.metaclusters()),
                 "all_seqlets": self.f.f['/multitask_seqlet_creation_results/final_seqlets'].shape[0]
                 }
        stats['clustered_seqlets_frac'] = stats['clustered_seqlets'] / stats['all_seqlets']
        if verbose:
            print(f"# seqlets assigned to patterns: {stats['clustered_seqlets']} "
                  f"/ {stats['all_seqlets']} ({round(stats['clustered_seqlets_frac']*100)}%)"
                  )
        return stats

    def n_seqlets(self, metacluster, pattern):
        pattern_grp = self.get_pattern_grp(metacluster, pattern)
        return pattern_grp['seqlets_and_alnmts/seqlets'].shape[0]

    def get_pattern_grp(self, metacluster, pattern):
        return self.f.f[f'/metacluster_idx_to_submetacluster_results/{metacluster}/seqlets_to_patterns_result/patterns/{pattern}']

    def get_pssm(self, metacluster, pattern,
                 rc=False, trim_frac=None):
        pattern_grp = self.get_pattern_grp(metacluster, pattern)
        pssm = ic_scale(pattern_grp["sequence"]['fwd'][:])
        if trim_frac is not None:
            pssm = trim_pssm(pssm, trim_frac)
        if rc:
            pssm = pssm[::-1, ::-1]
        return pssm

    def plot_position_hist(self, pattern, ax=None):
        dfp = self.seqlet_df_instances([pattern])
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))
        else:
            fig = plt.gcf()
        dfp.center.plot.hist(100, ax=ax)
        ax.set_xticks(np.arange(0, 1001, 100))   # TODO un-hardcode
        ax.set_xlabel("Position")
        return fig

    def plot_pssm(self, metacluster, pattern, rc=False,
                  letter_width=0.2, height=0.8, trim_frac=None, title=None):
        pssm = self.get_pssm(metacluster, pattern, rc, trim_frac=trim_frac)
        return seqlogo_clean(pssm, letter_width, height, title)

    def plot_pattern(self,
                     pattern_name,
                     kind='all',
                     rc=False,
                     trim_frac=None,
                     letter_width=0.2,
                     height=0.8,
                     rotate_y=0,
                     ylab=True):
        pattern = self.get_pattern(pattern_name)
        pattern = pattern.trim_seq_ic(trim_frac)
        ns = self.n_seqlets(*pattern_name.split("/"))
        pattern.name = shorten_pattern(pattern_name) + f" ({ns})"
        if rc:
            pattern = pattern.rc()
        return pattern.plot(kind, letter_width=letter_width,
                            height=height, rotate_y=rotate_y, ylab=ylab)

    def vdom(self, figdir, is_open=True, **kwargs):
        return vdom_modisco(self, figdir,
                            is_open=True, **kwargs)

    def plot_all_patterns(self,
                          kind='seq',
                          trim_frac=0,
                          n_min_seqlets=None,
                          ylim=None,
                          no_axis=False,
                          **kwargs):
        """
        Args:
          kind:
        """
        self.stats()  # print stats
        for pattern in self.patterns():
            if n_min_seqlets is not None and self.n_seqlets(*pattern.split("/")) < n_min_seqlets:
                continue
            self.plot_pattern(pattern,
                              kind=kind,
                              trim_frac=trim_frac,
                              **kwargs)
            if ylim is not None:
                plt.ylim(ylim)
            if no_axis:
                strip_axis(plt.gca())
                plt.gca().axison = False


class MultipleModiscoResult:
    def __init__(self, modisco_paths):
        self.modisco_paths = modisco_paths
        # load the modisco directory
        self.mr_dict = {k: ModiscoResult(v).open()
                        for k, v in modisco_paths.items()}

    def close(self):
        for mr in self.mr_dict.values():
            mr.close()

    def parse_pattern_name(self, pattern_name):
        task, name = pattern_name.split("/", 1)
        if "/" not in name:
            name = longer_pattern(name)
        return task, name

    def get_pattern(self, pattern_name):
        task, name = self.parse_pattern_name(pattern_name)
        p = self.mr_dict[task].get_pattern(name)
        p.name = pattern_name
        return p

    def n_seqlets(self, pattern_name):
        task, name = self.parse_pattern_name(pattern_name)
        return self.mr_dict[task].n_seqlets(*name.split("/"))

    def tasks(self):
        return list({x for mr in self.mr_dict.values()
                     for x in mr.tasks()})

    def get_seqlet_intervals(self, pattern_name,
                             trim_frac=0,
                             as_df=False):
        task, name = self.parse_pattern_name(pattern_name)
        mr = self.mr_dict[task]
        return mr.get_seqlet_intervals(self,
                                       name,
                                       trim_frac=trim_frac,
                                       as_df=as_df)

    def patterns(self):
        """List all patterns
        """
        return [task + "/" + pn
                for task, mr in self.mr_dict.items()
                for pn in mr.patterns()]

    def get_all_patterns(self):
        return [self.get_pattern(pn)
                for pn in self.patterns()]

    def _get_seqlets(self, pattern_name, trim_frac=0):
        task, name = self.parse_pattern_name(pattern_name)
        return self.mr_dict[task]._get_seqlets(name, trim_frac=trim_frac)


@attr.s
class Seqlet:
    # Have a proper seqlet class (interiting from the interval?)

    seqname = attr.ib()
    start = attr.ib()
    end = attr.ib()
    name = attr.ib()
    strand = attr.ib(".")

    @classmethod
    def from_dict(cls, d):
        return cls(seqname=d['example'],
                   start=d['start'],
                   end=d['end'],
                   name="",
                   strand="-" if d['rc'] else "+")

    def center(self, ignore_rc=False):
        if ignore_rc:
            add_offset = 0
        else:
            add_offset = 0 if self.strand == "-" else 1
        delta = (self.end + self.start) % 2
        center = (self.end + self.start) // 2
        return center + add_offset * delta

    def set_seqname(self, seqname):
        obj = self.copy()
        obj.seqname = seqname
        return obj

    def shift(self, x):
        obj = self.copy()
        obj.start = self.start + x
        obj.end = self.end + x
        return obj

    def swap_strand(self):
        obj = self.copy()
        if obj.strand == "+":
            obj.strand = "-"
        elif obj.strand == "-":
            obj.strand = "+"
        return obj

    def to_dict(self):
        return OrderedDict([("seqname", self.seqname),
                            ("start", self.start),
                            ("end", self.end),
                            ("name", self.name),
                            ("strand", self.strand)])

    def __getitem__(self, item):
        if item == "example":
            return self.seqname
        elif item == "start":
            return self.start
        elif item == "end":
            return self.end
        elif item == "pattern":
            return self.name
        elif item == "rc":
            return self.rc
        else:
            raise ValueError("item needs to be from:"
                             "example, start, end, pattern, rc")

    @property
    def rc(self):
        return self.strand == "-"

    def copy(self):
        return deepcopy(self)

    def contains(self, seqlet, ignore_strand=True):
        """Check if one seqlet contains the other seqlet
        """
        if self.seqname != seqlet.seqname:
            return False
        if self.start > seqlet.start:
            return False
        if self.end < seqlet.end:
            return False
        if not ignore_strand and self.strand != seqlet.strand:
            return False
        return True

    def extract(self, x, rc_fn=lambda x: x[::-1, ::-1]):

        if isinstance(x, OrderedDict):
            return OrderedDict([(track, self.extract(arr, rc_fn))
                                for track, arr in x.items()])
        elif isinstance(x, dict):
            return {track: self.extract(arr, rc_fn)
                    for track, arr in x.items()}
        elif isinstance(x, list):
            return [(track, self.extract(arr, rc_fn))
                    for track, arr in x]
        else:
            # Normal array
            def optional_rc(x, is_rc):
                if is_rc:
                    return rc_fn(x)
                else:
                    return x
            return optional_rc(
                x[self['example']][self['start']:self['end']],
                self['rc']
            )

    def resize(self, width):
        obj = deepcopy(self)

        if width is None or self.width() == width:
            # no need to resize
            return obj

        if not self['rc']:
            obj.start = self.center() - width // 2 - width % 2
            obj.end = self.center() + width // 2
        else:
            obj.start = self.center() - width // 2
            obj.end = self.center() + width // 2 + width % 2
        return obj

    def width(self):
        return self.end - self.start

    def trim(self, i, j):
        if i == 0 and j == self.width():
            return self
        obj = self.copy()
        assert j > i
        if self.strand == "-":
            w = self.width()
            obj.start = self.start + w - j
            obj.end = self.start + w - i
        else:
            obj.start = self.start + i
            obj.end = self.start + j
        return obj

    def valid_resize(self, width, max_width):
        if width is None:
            width = self.width()
        return self.center() > width // 2 and self.center() < max_width - width // 2

    def pattern_align(self, offset=0, use_rc=False):
        """Align the seqlet accoring to the pattern alignmet

        Example:
        `seqlet.pattern_align(**pattern.attrs['align'])`
        """
        seqlet = self.copy()
        if use_rc:
            seqlet = seqlet.swap_strand()
        return seqlet.shift((seqlet.rc * 2 - 1) * offset)


def resize_seqlets(seqlets, resize_width, seqlen):
    return [s.resize(resize_width) for s in seqlets
            if s.valid_resize(resize_width, seqlen)]


def labelled_seqlets2df(seqlets):
    """Convert a list of sequences to a dataframe

    Args:
      seqlets: list of seqlets returned by find_instances

    Returns:
      pandas.DataFrame with one row per seqlet
    """
    def seqlet2row(seqlet):
        """Convert a single seqlete to a pandas array
        """
        return OrderedDict([
            ("example_idx", seqlet.coor.example_idx),
            ("seqlet_start", seqlet.coor.start),
            ("seqlet_end", seqlet.coor.end),
            ("seqlet_is_revcomp", seqlet.coor.is_revcomp),
            ("seqlet_score", seqlet.coor.score),
            ("metacluster", seqlet.metacluster),
            ("pattern", seqlet.pattern),
            ("percnormed_score", seqlet.score_result.percnormed_score),
            ("score", seqlet.score_result.score),
            ("offset", seqlet.score_result.offset),
            ("revcomp", seqlet.score_result.revcomp),
        ])

    return pd.DataFrame([seqlet2row(seqlet) for seqlet in seqlets])
