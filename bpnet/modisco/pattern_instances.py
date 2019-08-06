"""Code for working with the pattern instances table
produced by `bpnet.cli.modisco.modisco_score2`
which calls `pattern.get_instances`
"""
from bpnet.stats import quantile_norm
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
from bpnet.modisco.utils import longer_pattern, shorten_pattern
from bpnet.cli.modisco import get_nonredundant_example_idx
import numpy as np
from bpnet.plot.profiles import extract_signal
from bpnet.modisco.results import resize_seqlets
from bpnet.modisco.results import trim_pssm_idx, Seqlet


# TODO - allow these to be of also other type?
def load_instances(parq_file, motifs=None, dedup=True, verbose=True):
    """Load pattern instances from the parquet file

    Args:
      parq_file: parquet file of motif instances
      motifs: dictionary of motifs of interest.
        key=custom motif name, value=short pattern name (e.g. {'Nanog': 'm0_p3'})

    """
    if motifs is not None:
        incl_motifs = {longer_pattern(m) for m in motifs.values()}
    else:
        incl_motifs = None

    if isinstance(parq_file, pd.DataFrame):
        dfi = parq_file
    else:
        if motifs is not None:
            from fastparquet import ParquetFile

            # Selectively load only the relevant patterns
            pf = ParquetFile(str(parq_file))
            patterns = [shorten_pattern(pn) for pn in incl_motifs]
            dfi = pf.to_pandas(filters=[("pattern_short", "in", patterns)])
        else:
            dfi = pd.read_parquet(str(parq_file), engine='fastparquet')
            if 'pattern' not in dfi:
                # assumes a hive-stored file
                dfi['pattern'] = dfi['dir0'].str.replace("pattern=", "").astype(str) + "/" + dfi['dir1'].astype(str)

    # filter
    if motifs is not None:
        dfi = dfi[dfi.pattern.isin(incl_motifs)]  # NOTE this should already be removed
        if 'pattern_short' not in dfi:
            dfi['pattern_short'] = dfi['pattern'].map({k: shorten_pattern(k) for k in incl_motifs})
        dfi['pattern_name'] = dfi['pattern_short'].map({v: k for k, v in motifs.items()})
    else:
        dfi['pattern_short'] = dfi['pattern'].map({k: shorten_pattern(k)
                                                   for k in dfi.pattern.unique()})

    # add some columns if they don't yet exist
    if 'pattern_start_abs' not in dfi:
        dfi['pattern_start_abs'] = dfi['example_start'] + dfi['pattern_start']
    if 'pattern_end_abs' not in dfi:
        dfi['pattern_end_abs'] = dfi['example_start'] + dfi['pattern_end']

    if dedup:
        # deduplicate
        dfi_dedup = dfi.drop_duplicates(['pattern',
                                         'example_chrom',
                                         'pattern_start_abs',
                                         'pattern_end_abs',
                                         'strand'])

        # number of removed duplicates
        d = len(dfi) - len(dfi_dedup)
        if verbose:
            print("number of de-duplicated instances:", d, f"({d / len(dfi) * 100}%)")

        # use de-duplicated instances from now on
        dfi = dfi_dedup
    return dfi


def multiple_load_instances(paths, motifs):
    """
    Args:
      paths: dictionary <tf> -> instances.parq
      motifs: dictinoary with <motif_name> -> pattern name of
        the form `<TF>/m0_p1`
    """
    from bpnet.utils import pd_col_prepend
    # load all the patterns

    dfi = pd.concat([load_instances(path,
                                    motifs=OrderedDict([(motif, pn.split("/", 1)[1])
                                                        for motif, pn in motifs.items()
                                                        if pn.split("/", 1)[0] == tf]),
                                    dedup=False).assign(tf=tf).pipe(pd_col_prepend, ['pattern', 'pattern_short'], prefix=tf + "/")
                     for tf, path in tqdm(paths.items())
                     ])
    return dfi


def dfi_add_ranges(dfi, ranges, dedup=False):
    # Add absolute locations
    dfi = dfi.merge(ranges, on="example_idx", how='left')
    dfi['pattern_start_abs'] = dfi['example_start'] + dfi['pattern_start']
    dfi['pattern_end_abs'] = dfi['example_start'] + dfi['pattern_end']

    if dedup:
        # deduplicate
        dfi_dedup = dfi.drop_duplicates(['pattern',
                                         'example_chrom',
                                         'pattern_start_abs',
                                         'pattern_end_abs',
                                         'strand'])

        # number of removed duplicates
        d = len(dfi) - len(dfi_dedup)
        print("number of de-duplicated instances:", d, f"({d / len(dfi) * 100}%)")

        # use de-duplicated instances from now on
        dfi = dfi_dedup
    return dfi


def dfi2pyranges(dfi):
    """Convert dfi to pyranges

    Args:
      dfi: pd.DataFrame returned by `load_instances`
    """
    import pyranges as pr
    dfi = dfi.copy()
    dfi['Chromosome'] = dfi['example_chrom']
    dfi['Start'] = dfi['pattern_start_abs']
    dfi['End'] = dfi['pattern_end_abs']
    dfi['Name'] = dfi['pattern']
    dfi['Score'] = dfi['contrib_weighted_p']
    dfi['Strand'] = dfi['strand']
    return pr.PyRanges(dfi)


def align_instance_center(dfi, original_patterns, aligned_patterns, trim_frac=0.08):
    """Align the center of the seqlets using aligned patterns

    Args:
      dfi: pd.DataFrame returned by `load_instances`
      original_patterns: un-trimmed patterns that were trimmed using
        trim_frac before scanning
      aligned_patterns: patterns that are all lined-up and that contain
        'align': {"use_rc", "offset" } information in the attrs
      trim_frac: original trim_frac used to trim the motifs

    Returns:
      dfi with 2 new columns: `pattern_center_aln` and `pattern_strand_aln`
    """
    # NOTE - it would be nice to be able to give trimmed patterns instead of
    # `original_patterns` + `trim_frac` and just extract the trim stats from the pattern
    # TODO - shall we split this function into two -> one for dealling with
    #        pattern trimming and one for dealing with aligning patterns?
    trim_shift_pos = {p.name: p._trim_center_shift(trim_frac=trim_frac)[0]
                      for p in original_patterns}
    trim_shift_neg = {p.name: p._trim_center_shift(trim_frac=trim_frac)[1]
                      for p in original_patterns}
    shift = {p.name: (p.attrs['align']['use_rc'] * 2 - 1) * p.attrs['align']['offset']
             for p in aligned_patterns}
    strand_shift = {p.name: p.attrs['align']['use_rc'] for p in aligned_patterns}

    strand_vec = dfi.strand.map({"+": 1, "-": -1})
    dfi['pattern_center_aln'] = (dfi.pattern_center -
                                 # - trim_shift since we are going from trimmed to non-trimmed
                                 np.where(dfi.strand == '-',
                                          dfi.pattern.map(trim_shift_neg),
                                          dfi.pattern.map(trim_shift_pos)) +
                                 # NOTE: `strand` should better be called `pattern_strand`
                                 dfi.pattern.map(shift) * strand_vec)

    def flip_strand(x):
        return x.map({"+": "-", "-": "+"})

    # flip the strand
    dfi['pattern_strand_aln'] = np.where(dfi.pattern.map(strand_shift),
                                         # if True, then we are on the other strand
                                         flip_strand(dfi.strand),
                                         dfi.strand)
    return dfi


def extract_ranges(dfi):
    """Extract example ranges
    """
    ranges = dfi[['example_chrom', 'example_start',
                  'example_end', 'example_idx']].drop_duplicates()
    ranges.columns = ['chrom', 'start', 'end', 'example_idx']
    return ranges


def filter_nonoverlapping_intervals(dfi):
    ranges = extract_ranges(dfi)
    keep_idx = get_nonredundant_example_idx(ranges, 200)
    return dfi[dfi.example_idx.isin(keep_idx)]


def plot_coocurence_matrix(dfi, total_examples, signif_threshold=1e-5, ax=None):
    """Test for motif co-occurence in example regions

    Args:
      dfi: pattern instance DataFrame observer by load_instances
      total_examples: total number of examples
    """
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    from sklearn.metrics import matthews_corrcoef
    from scipy.stats import fisher_exact
    import statsmodels as sm
    import seaborn as sns
    import matplotlib.pyplot as plt

    counts = pd.pivot_table(dfi, 'pattern_len', "example_idx",
                            "pattern_name", aggfunc=len, fill_value=0)
    ndxs = list(counts)
    c = counts > 0

    o = np.zeros((len(ndxs), len(ndxs)))
    op = np.zeros((len(ndxs), len(ndxs)))
    # fo = np.zeros((len(ndxs), len(ndxs)))
    # fp = np.zeros((len(ndxs), len(ndxs)))

    for i, xn in enumerate(ndxs):
        for j, yn in enumerate(ndxs):
            if xn == yn:
                continue
            ct = pd.crosstab(c[xn], c[yn])
            # add not-counted 0 values:
            ct.iloc[0, 0] += total_examples - len(c)
            t22 = sm.stats.contingency_tables.Table2x2(ct)
            o[i, j] = np.log2(t22.oddsratio)
            op[i, j] = t22.oddsratio_pvalue()
    signif = op < signif_threshold
    a = np.zeros_like(signif).astype(str)
    a[signif] = "*"
    a[~signif] = ""
    np.fill_diagonal(a, '')

    sns.heatmap(pd.DataFrame(o, columns=ndxs, index=ndxs),
                annot=a, fmt="", vmin=-4, vmax=4,
                cmap='RdBu_r', ax=ax)
    ax.set_title(f"Log2 odds-ratio. (*: p<{signif_threshold})")


def construct_motif_pairs(dfi, motif_pair,
                          features=['match_weighted_p',
                                    'contrib_weighted_p',
                                    'contrib_weighted']):
    """Construct motifs pair table
    """
    dfi_filtered = dfi.set_index('example_idx', drop=False)
    counts = pd.pivot_table(dfi_filtered,
                            'pattern_center', "example_idx", "pattern_name",
                            aggfunc=len, fill_value=0)

    if motif_pair[0] != motif_pair[1]:
        relevant_examples_idx = counts.index[np.all(counts[motif_pair] == 1, 1)]
    else:
        relevant_examples_idx = counts.index[np.all(counts[motif_pair] == 2, 1)]

    dft = dfi_filtered.loc[relevant_examples_idx]
    dft = dft[dft.pattern_name.isin(motif_pair)]

    dft = dft.sort_values(['example_idx', 'pattern_center'])
    dft['pattern_order'] = dft.index.duplicated().astype(int)
    if motif_pair[0] == motif_pair[1]:
        dft['pattern_name'] = dft['pattern_name'] + dft['pattern_order'].astype(str)
        motif_pair = [motif_pair[0] + '0', motif_pair[1] + '1']

    dftw = dft.set_index(['pattern_name'], append=True)[['pattern_center',
                                                         'strand'] + features].unstack()

    dftw['center_diff'] = dftw['pattern_center'][motif_pair].diff(axis=1).iloc[:, 1]

    dftw_filt = dftw[np.abs(dftw.center_diff) > 10]

    dftw_filt['distance'] = np.abs(dftw_filt['center_diff'])
    dftw_filt['strand_combination'] = dftw_filt['strand'][motif_pair].sum(1)
    return dftw_filt


def dfi_row2seqlet(row, short_name=False):
    return Seqlet(row.example_idx,
                  row.pattern_start,
                  row.pattern_end,
                  name=shorten_pattern(row.pattern) if short_name else row.pattern,
                  strand=row.strand)


def dfi2seqlets(dfi, short_name=False):
    """Convert the data-frame produced by pattern.get_instances()
    to a list of Seqlets

    Args:
      dfi: pd.DataFrame returned by pattern.get_instances()
      short_name: if True, short pattern name will be used for the seqlet name

    Returns:
      Seqlet list
    """
    return [dfi_row2seqlet(row, short_name=short_name)
            for i, row in dfi.iterrows()]


def profile_features(seqlets, ref_seqlets, profile, profile_width=70):
    from bpnet.simulate import profile_sim_metrics
    # resize
    seqlets = resize_seqlets(seqlets, profile_width, seqlen=profile.shape[1])
    seqlets_ref = resize_seqlets(ref_seqlets, profile_width, seqlen=profile.shape[1])
#     import pdb
#     pdb.set_trace()

    # extract the profile
    seqlet_profile = extract_signal(profile, seqlets)
    seqlet_profile_ref = extract_signal(profile, seqlets_ref)

    # compute the average profile
    avg_profile = seqlet_profile_ref.mean(axis=0)

    metrics = pd.DataFrame([profile_sim_metrics(avg_profile + 1e-6, cp + 1e-6)
                            for cp in seqlet_profile])
    metrics_ref = pd.DataFrame([profile_sim_metrics(avg_profile + 1e-6, cp + 1e-6)
                                for cp in seqlet_profile_ref])

    assert len(metrics) == len(seqlets)  # needs to be the same length

    if metrics.simmetric_kl.min() == np.inf or \
            metrics_ref.simmetric_kl.min() == np.inf:
        profile_match_p = None
    else:
        profile_match_p = quantile_norm(metrics.simmetric_kl, metrics_ref.simmetric_kl)
    return pd.DataFrame(OrderedDict([
        ("profile_match", metrics.simmetric_kl),
        ("profile_match_p", profile_match_p),
        ("profile_counts", metrics['counts']),
        ("profile_counts_p", quantile_norm(metrics['counts'], metrics_ref['counts'])),
        ("profile_max", metrics['max']),
        ("profile_max_p", quantile_norm(metrics['max'], metrics_ref['max'])),
        ("profile_counts_max_ref", metrics['counts_max_ref']),
        ("profile_counts_max_ref_p", quantile_norm(metrics['counts_max_ref'],
                                                   metrics_ref['counts_max_ref'])),
    ]))


def dfi_filter_valid(df, profile_width, seqlen):
    return df[(df.pattern_center.round() - profile_width > 0)
              & ((df.pattern_center + profile_width < seqlen))]


def annotate_profile_single(dfi, pattern_name, mr, profiles, profile_width=70, trim_frac=0.08):
    seqlen = profiles[list(profiles)[0]].shape[1]

    dfi = dfi_filter_valid(dfi.copy(), profile_width, seqlen)
    dfi['id'] = np.arange(len(dfi))
    assert np.all(dfi.pattern == pattern_name)

    dfp_pattern_list = []
    dfi_subset = dfi
    ref_seqlets = mr._get_seqlets(pattern_name, trim_frac=trim_frac)
    dfi_seqlets = dfi2seqlets(dfi_subset)
    for task in profiles:
        dfp = profile_features(dfi_seqlets,
                               ref_seqlets=ref_seqlets,
                               profile=profiles[task],
                               profile_width=profile_width)
        assert len(dfi_subset) == len(dfp)
        dfp.columns = [f'{task}/{c}' for c in dfp.columns]  # prepend task
        dfp_pattern_list.append(dfp)

    dfp_pattern = pd.concat(dfp_pattern_list, axis=1)
    dfp_pattern['id'] = dfi_subset['id'].values
    assert len(dfp_pattern) == len(dfi)
    return pd.merge(dfi, dfp_pattern, on='id')


def annotate_profile(dfi, mr, profiles, profile_width=70, trim_frac=0.08, pattern_map=None):
    """Append profile match columns to dfi

    Args:
      dfi[pd.DataFrame]: motif instances
      mr[ModiscoResult]
      profiles: dictionary of profiles with shape: (n_examples, seqlen, strand)
      profile_width: width of the profile to extract
      trim_frac: what trim fraction to use then computing the values for modisco
        seqlets.
      pattern_map[dict]: mapping from the pattern name in `dfi` to the corresponding
        pattern in `mr`. Used when dfi was for example not derived from modisco.
    """
    seqlen = profiles[list(profiles)[0]].shape[1]

    dfi = dfi_filter_valid(dfi.copy(), profile_width, seqlen)
    dfi['id'] = np.arange(len(dfi))
    # TODO - remove in-valid variables
    dfp_list = []
    for pattern in tqdm(dfi.pattern.unique()):
        dfp_pattern_list = []
        dfi_subset = dfi[dfi.pattern == pattern]
        for task in profiles:
            if pattern_map is not None:
                modisco_pattern = pattern_map[pattern]
            else:
                modisco_pattern = pattern
            dfp = profile_features(dfi2seqlets(dfi_subset),
                                   ref_seqlets=mr._get_seqlets(modisco_pattern,
                                                               trim_frac=trim_frac),
                                   profile=profiles[task],
                                   profile_width=profile_width)
            assert len(dfi_subset) == len(dfp)
            dfp.columns = [f'{task}/{c}' for c in dfp.columns]  # prepend task
            dfp_pattern_list.append(dfp)

        dfp_pattern = pd.concat(dfp_pattern_list, axis=1)
        dfp_pattern['id'] = dfi_subset['id'].values
        dfp_list.append(dfp_pattern)
    out = pd.concat(dfp_list, axis=0)
    assert len(out) == len(dfi)
    return pd.merge(dfi, out, on='id')
