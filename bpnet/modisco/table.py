"""Create an overview table for modisco
"""
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
import os
from bpnet.modisco.results import ModiscoResult, trim_pssm_idx
from bpnet.plot.vdom import write_datatable_html
from bpnet.plot.profiles import extract_signal
from kipoi.readers import HDF5Reader
from collections import OrderedDict
from bpnet.functions import mean
from bpnet.utils import read_json
from bpnet.modisco.utils import shorten_pattern, longer_pattern
from tqdm import tqdm
from joblib import Parallel, delayed
import attr
from bpnet.plot.vdom import fig2vdom, vdom_pssm
from vdom.helpers import a
from bpnet.modisco.periodicity import periodicity_10bp_frac
from bpnet.cli.modisco import load_included_samples


class ModiscoData:
    """Container for all the required data from a modisco run
    """
    # global variables
    trim_frac = 0.08

    def __init__(self, mr, d, included_samples, tasks):
        self.mr = mr
        self.d = d
        # if 'hyp_contrib' not in self.d:
        # backcompatibility
        #    self.d['hyp_contrib'] = self.d['grads']
        if included_samples is not None:
            warnings.warn("included_samples deprecated. use included_samples=None")
        self.included_samples = included_samples  # TODO - remove this
        self.tasks = tasks

        self.seqlets_per_task = self.mr.seqlets()
        self.profile = {}
        self.profile_wide = {}
        self.grad_profile = {}
        self.grad_counts = {}
        self.seq = {}

        self.footprint_width = 200

        # Setup all the required matrices
        for pattern, seqlets in tqdm(self.seqlets_per_task.items()):
            # get wide seqlets

            wide_seqlets = [s.resize(self.footprint_width)
                            for s in seqlets
                            if s.center() > self.footprint_width // 2 and
                            s.center() < self.get_seqlen(pattern) - self.footprint_width // 2
                            ]

            # profile data
            self.profile[pattern] = {task: extract_signal(self.get_region_profile(task), seqlets)
                                     for task in tasks}

            self.profile_wide[pattern] = {task: extract_signal(self.get_region_profile(task), wide_seqlets)
                                          for task in tasks}
            # contribution scores
            self.grad_profile[pattern] = {task: extract_signal(self.get_region_grad(task, 'profile'),
                                                               seqlets)
                                          for task in tasks}
            self.grad_counts[pattern] = {task: extract_signal(self.get_region_grad(task, 'counts'), seqlets)
                                         for task in tasks}
            # seq
            self.seq[pattern] = extract_signal(self.get_region_seq(), seqlets)

    @classmethod
    def load(cls, modisco_dir, contrib_scores_h5, contribsf=None):
        """Instantiate ModiscoData from tf-modisco run folder
        """
        from bpnet.cli.contrib import ContribFile
        modisco_dir = Path(modisco_dir)

        # Load the contribution scores and the data
        # d = HDF5Reader.load(contrib_scores_h5)

        # load modisco
        mr = ModiscoResult(modisco_dir / "modisco.h5")
        mr.open()

        if contribsf is not None:
            # Cache the results
            d = contribsf
        else:
            d = ContribFile.from_modisco_dir(modisco_dir)
            d.cache()
        # load included samples
        # included_samples = load_included_samples(modisco_dir)
        included_samples = None

        tasks = d.get_tasks()  # list(d['targets']['profile'].keys())
        return cls(mr, d, included_samples, tasks)

    # Explicitly defined getters to abstract the underlying storage

    def get_seqlen(self, pattern):
        return self.d.get_seqlen()

    def get_seqlets(self, pattern):
        return self.seqlets_per_task[pattern]

    def get_profile(self, pattern, task):
        """Get profile counts associated with the pattern
        """
        return self.profile[pattern][task]

    def get_profile_wide(self, pattern, task):
        """Get profile counts associated with the pattern
        """
        return self.profile_wide[pattern][task]

    def get_region_profile(self, task):
        """Return all profiles of `task` found in peaks of `peak_task`
        """
        return self.d.get_profiles()[task]

    def get_region_seq(self):
        """Return all profiles of `task` found in peaks of `peak_task`
        """
        return self.d.get_seq()

    def get_region_grad(self, task, which='profile'):
        if which == 'profile':
            return self.d.get_hyp_contrib()[task]  # expect the default contribution score
        elif which == 'counts':
            if self.d.contains_contrib_score('count'):
                return self.d.get_hyp_contrib(contrib_score='count')[task]
            elif self.d.contains_contrib_score('counts/pre-act'):
                return self.d.get_hyp_contrib(contrib_score='counts/pre-act')[task]
            else:
                warnings.warn("region counts not present. Returning the default contribution scores")
                return self.d.get_hyp_contrib()[task]  # expect the default contribution score
        else:
            raise ValueError("which needs to be from {'profile', 'counts'}")

    def get_peak_task_idx(self, peak_task):
        task_regions = self.d.get_ranges()['interval_from_task'].values == peak_task
        return np.arange(len(task_regions))[task_regions]

    def get_contrib(self, pattern, task, which='profile'):
        """Get contribution scores associated with the pattern
        """
        if which == 'profile':
            return self.grad_profile[pattern][task] * self.get_seq(pattern)
        elif which == 'counts':
            return self.grad_counts[pattern][task] * self.get_seq(pattern)
        else:
            raise ValueError("which needs to be from {'profile', 'counts'}")

    def get_seq(self, pattern):
        """Get sequences associated with the pattern
        """
        return self.seq[pattern]

    def get_tasks(self):
        return self.tasks

    def get_trim_idx(self, pattern):
        """Return the trimming indices
        """
        return trim_pssm_idx(self.mr.get_pssm(*pattern.split("/")), frac=self.trim_frac)

    def get_centroid_seqlet_matches(self, trim_frac=0.08, n_jobs=1):
        """Get match statistics for all the seqlets for all patterns

        Args:
          trim_frac: how much to trim the original pattern
          n_jobs: Number of jobs to run in parallel
        """
        return pd.concat([self._get_centroid_seqlet_matches(pattern, trim_frac, n_jobs)
                          for pattern in tqdm(self.mr.patterns())])

    def _get_centroid_seqlet_matches(self, pattern_name, trim_frac=0.08, n_jobs=1, verbose=False):
        """get_centroid_seqlet_matches for a single pattern
        """
        pattern = self.mr.get_pattern(pattern_name).trim_seq_ic(trim_frac)
        tasks = pattern.tasks()  # get tasks from the pattern
        i, j = self.get_trim_idx(pattern_name)

        seq = self.get_seq(pattern_name)[:, i:j]
        # profile = {task: self.get_profile_wide(pattern_name, task) for task in tasks}
        contrib = {task: self.get_contrib(pattern_name, task, 'profile')[:, i:j]
                   for task in tasks}

        match, contribution = pattern.scan_contribution(contrib, hyp_contrib=None, tasks=tasks,
                                                        n_jobs=n_jobs, verbose=False, pad_mode=None)
        seq_match = pattern.scan_seq(seq, n_jobs=n_jobs, verbose=False, pad_mode=None)

        dfm = pattern.get_instances(tasks, match, contribution, seq_match, fdr=1, verbose=verbose, plot=verbose)
        dfm = dfm[dfm.seq_match > 0]
        return dfm


DOC = OrderedDict([
    ("logo pwm", "sequence information content"),
    ("logo contrib", "average contribution score logo"),
    ("n seqlets", "total number of seqlets"),
    ("ic pwm mean", "average information content per base"),
    ("<task> contrib profile / counts", "Average per-base profile/total count prediction contribution scores for <task>. Seqlets are trimmed to the core motif displayed in logo pwm (typically 10bp)"),
    ("<task> footprint entropydiff", "average entropy difference compared to the uniform distribution (computed at non-trimmed seqlets regions, typically 40bp). More positive numbers represent stronger deviation from the uniform distribution"),
    ("<task> footprint max",
     "(max(pos) + max(neg))/2 where `pos` are the "
     "maximum counts of the average profilefor the positive strand. Higher number means better agreement between the strands."),
    ("<task> footprint standcor", "Maximum auto-correlation between positive and reversed negative strand"),
    ("<task> footprint counts", "average number of counts in the seqlet region"),
    ("<task> region counts", "average number of counts in the whole region where seqlets are located (typically 1kb)"),
    ("<task> pos absmean", "Absolute value of the mean seqlet position with 0 being the region center"),
    ("<task> pos std", "Standard deviation of the position"),
    ("<task> pos unimodal", "If True, the distribution of positions is estimated to be uni-modal"),
    ("<task> periodicity 10bp", "Strength of the 10bp periodicity in the profile contribution scores. "
     "Measured as the fraction of the fourier power spectrum at 10bp."),
    ("consensus", "consensus sequence (tallest letters from logo pwm)"),
])


def modisco_table(data):
    """Main function

    Args:
      data: instance of ModiscoData
      report_url: if provided, the pattern will be a link

    Returns:
      pd.DataFrame containing all the information about the patterns
    """
    df = pd.DataFrame([pattern_features(pattern, data)
                       for pattern in tqdm(data.mr.patterns())])
    df.insert(0, 'idx', df.index)  # add also idx to the front

    df.doc = DOC
    return df


def pattern_features(pattern, data):
    """

    Returns:
      OrderedDict of columns for pandas dataFrame
    """
    return OrderedDict([
        ("pattern", shorten_pattern(pattern)),
        ("logo pwm", logo_pwm(pattern, data)),
        ("logo contrib", logo_contrib(pattern, data)),
        ("n seqlets", len(data.get_seqlets(pattern))),
        ("ic pwm mean", pwm_mean_ic(pattern, data)),
    ] +
        [res for task in data.get_tasks()
         for res in pattern_task_features(pattern, task, data)] +
        [("consensus", consensus(pattern, data))]
    )


# --------------------------------------------
# individual features


def pattern_url(shortpattern, report_url):
    return a(shortpattern, href=report_url + "#" + longer_pattern(shortpattern)).to_html()


def logo_pwm(pattern, data, width=80):
    fig = data.mr.plot_pssm(*pattern.split("/"), trim_frac=data.trim_frac)
    return fig2vdom(fig).to_html().replace("<img", f"<img width={width}")  # hack


def logo_contrib(pattern, data, width=80):
    arr = mean([data.get_contrib(pattern, task, 'profile').mean(axis=0)
                for task in data.get_tasks()])

    # trim array to match the pwm
    i, j = data.get_trim_idx(pattern)
    arr = arr[i:j]
    return vdom_pssm(arr).to_html().replace("<img", f"<img width={width}")  # hack


def consensus(pattern, data):
    pssm = data.mr.get_pssm(*pattern.split("/"), trim_frac=data.trim_frac)
    return "".join(["ACGT"[i] for i in pssm.argmax(axis=1)])


def pwm_mean_ic(pattern, data):
    """Average per-base information content
    of the PWM matrix
    """
    pssm = data.mr.get_pssm(*pattern.split("/"), trim_frac=data.trim_frac)
    return pssm.sum(axis=1).mean(axis=0)

# --------------------------------------------
# task/TF - related features


def pattern_task_features(pattern, task, data):
    # TODO - have the order TF1_f1, TF2, f2

    def format_feature_groups(feature_groups, task):
        """Nicely format feature groups
        """
        return [
            (f"{task} {feature_group} {subfeature}", value)
            for feature_group, feature_group_values in feature_groups
            for subfeature, value in feature_group_values.items()
        ]

    return format_feature_groups([
        ("contrib", task_contrib(pattern, task, data)),
        ("footprint", task_footprint(pattern, task, data)),
        ("region", task_region(pattern, task, data)),
        ("pos", task_pos(pattern, task, data)),
        ("periodicity", task_periodicity(pattern, task, data)),
    ], task)


def task_contrib(pattern, task, data):
    """Average contribution of pattern for task
    """
    i, j = data.get_trim_idx(pattern)
    # TODO - adopt to the scenario where "counts" contribution scores are not present
    return OrderedDict([
        # 1. aggregate across channels (sum)
        # 2. aggregate accross examples
        # 3. trim to the pattern
        ("profile", data.get_contrib(pattern, task, 'profile').sum(axis=-1).mean(axis=0)[i:j].mean()),
        ("counts", data.get_contrib(pattern, task, 'counts').sum(axis=-1).mean(axis=0)[i:j].mean()),
    ])


def task_footprint(pattern, task, data):
    """Average contribution of pattern for task
    """
    from scipy.stats import entropy
    from scipy.signal import correlate

    # question: in case of many seqlets, take only the top N?
    # profile = data.get_profile(patern, task)
    # p = profile / profile.sum(axis=1, keepdims=True)
    # # drop NA's
    # notnan = ~np.any(np.any(np.isnan(p), axis=-1), axis=-1)
    # total_counts = profile[notnan].sum(axis=1).mean(axis=-1)
    # p = p[notnan]
    # entropies = entropy(p.swapaxes(0, 1)).sum(axis=-1)

    agg_profile = data.get_profile_wide(pattern, task).mean(axis=0)
    agg_profile_norm = agg_profile / agg_profile.sum(axis=0, keepdims=True)

    return OrderedDict([
        # entropy = diff to the uniform entropy
        ("entropydiff", entropy(np.ones_like(agg_profile_norm)).sum(axis=-1) - entropy(agg_profile_norm).sum(axis=-1)),
        ("max", agg_profile.max(axis=0).mean()),
        ("strandcor", correlate(agg_profile_norm[:, 0], agg_profile_norm[::-1, 1]).max()),
        ("counts", agg_profile.sum()),
    ])


def task_region(pattern, task, data):
    """Average amount of counts in the whole region when the footprint was present
    """
    profiles = data.get_region_profile(task)
    seqlet_idx = np.array([s.seqname for s in data.get_seqlets(pattern)])

    return OrderedDict([
        ("counts", profiles[seqlet_idx].mean(axis=0).sum()),
    ])


def dist_n_modes(values):
    """Given an empirical distribution, return the number of modes
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelextrema

    try:
        density = gaussian_kde(values)
    except:
        import pdb
        pdb.set_trace()

    xs = np.linspace(values.min(), values.max(), 100)
    smooth_hist = density(xs)
    maxima = argrelextrema(smooth_hist, np.greater)
    # for multiple maxima, the next found maxima should be at least at
    # 1/2 of the highest
    return np.sum(smooth_hist[maxima[0]] > 0.5 * smooth_hist.max())


def task_pos(pattern, task, data):
    """Average contribution of pattern for task
    """
    task_idx = data.get_peak_task_idx(task)
    if len(task_idx) == 0:
        # none of the peaks were found. Skip this analysis
        return OrderedDict([])

    positions = np.array([s.center() for s in data.get_seqlets(pattern)
                          if s.seqname in task_idx])
    if len(positions) <= 1:
        # patern was not found in any of the peaks. Skip this analysis
        return OrderedDict([])

    return OrderedDict([
        ("meandiff", np.abs(data.get_seqlen(pattern) / positions.mean())),
        ("std", positions.std()),
        ("unimodal", dist_n_modes(positions) == 1),
    ])


def task_periodicity(pattern, task, data):
    """Task periodicity
    """
    return OrderedDict([
        ("10bp", periodicity_10bp_frac(pattern, task, data)),
    ])

# ----------------------
def write_modisco_table(df, output_dir, report_url=None, prefix='pattern_table',
                        exclude_when_writing=["logo pwm", "logo contrib"], doc=DOC, write_csv=True):
    """Write the pattern table to as .html and .csv
    """
    from vdom.helpers import h2, h3, p, ul, ol, li, div, b
    output_dir = Path(output_dir)
    df = df.copy()

    if write_csv:
        cols_for_csv = [c for c in df.columns if c not in exclude_when_writing]
        df[cols_for_csv].to_csv(output_dir / f'{prefix}.csv', index=False)

    if report_url is not None:
        df.pattern = [pattern_url(p, report_url) for p in df.pattern]
    df.columns = [c.replace(" ", "<br>") for c in df.columns]
    # Add instructions
    instructions = div(
        h2("Column description"),
        ul([li(b(k), ": ", v) for k, v in doc.items()]),
        h2("Table interaction options"),
        ul([
            li(b("Re-order columns: "),
               "Drag column headers using the mouse pointer"),
            li(b("Sort w.r.t single column: "),
               "Click on the column header to sort the table with respect two that "
               "column in an ascending order and click again to sort in a descending order."),
            li(b("Sort w.r.t multiple columns: "),
               "Hold shift and click multiple column headers to sort w.r.t. multiple columns"),
            li(b("View logo images: "),
               "Right-click on the images and choose 'Open image in new tab'"),
            li(b("Select muliple rows: "),
               "Click on the row to select/de-select it."),
            li(b("Get more information about the pattern: "),
               "Follow the link in the pattern column."),
        ])
    )
    write_datatable_html(df, output_dir / f"{prefix}.html", instructions.to_html())
