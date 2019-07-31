import logging
import matplotlib.pyplot as plt
"""
Run modisco
"""
import pandas as pd
import os
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
from bpnet.utils import write_pkl, related_dump_yaml, render_ipynb, remove_exists, add_file_logging, create_tf_session
from bpnet.cli.schemas import DataSpec, ModiscoHParams
from bpnet.cli.contrib import ContribScoreFile
# ContribScoreFile
from bpnet.modisco.results import ModiscoResult
from bpnet.functions import mean
from kipoi_utils.utils import unique_list
from scipy.spatial.distance import correlation
from concise.utils.helper import write_json, read_json
from bpnet.data import numpy_minibatch
from kipoi.writers import HDF5BatchWriter
from kipoi.readers import HDF5Reader
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------
# load functions for the modisco directory


def load_included_samples(modisco_dir):
    modisco_dir = Path(modisco_dir)

    kwargs = read_json(modisco_dir / "kwargs.json")

    d = ContribScoreFile(kwargs["contrib_scores"])
    interval_from_task = d.get_ranges().interval_from_task
    n = len(d)
    d.close()

    included_samples = np.ones((n,), dtype=bool)
    if not kwargs.get("skip_dist_filter", False) and (modisco_dir / "strand_distances.h5").exists():
        included_samples = HDF5Reader.load(modisco_dir / "strand_distances.h5")['included_samples'] & included_samples

    if kwargs.get("filter_npy", None) is not None:
        included_samples = np.load(kwargs["filter_npy"]) & included_samples

    if kwargs.get("subset_tasks", None) is not None and kwargs.get("filter_subset_tasks", False):
        included_samples = interval_from_task.isin(kwargs['subset_tasks']).values & included_samples

    return included_samples


def load_ranges(modisco_dir):
    modisco_dir = Path(modisco_dir)
    included_samples = load_included_samples(modisco_dir)

    kwargs = read_json(modisco_dir / "kwargs.json")
    d = ContribScoreFile(kwargs["contrib_scores"], included_samples)
    df = d.get_ranges()
    d.close()
    return df


def get_nonredundant_example_idx(ranges, width=200):
    """Get non - overlapping intervals(in the central region)

    Args:
      ranges: pandas.DataFrame returned by bpnet.cli.modisco.load_ranges
      width: central region considered that should not overlap between
         any interval
    """
    from pybedtools import BedTool
    from bpnet.preproc import resize_interval
    # 1. resize ranges
    ranges['example_idx'] = np.arange(len(ranges))  # make sure
    r = ranges[['chrom', 'start', 'end', 'example_idx']]  # add also the strand information
    r = resize_interval(r, width, ignore_strand=True)

    bt = BedTool.from_dataframe(r)
    btm = bt.sort().merge()
    df = btm.to_dataframe()
    df = df[(df.end - df.start) < width * 2]

    r_overlaps = bt.intersect(BedTool.from_dataframe(df), wb=True).to_dataframe()
    keep_idx = r_overlaps.drop_duplicates(['score', 'strand', 'thickStart'])['name'].astype(int)

    return keep_idx


def load_profiles(modisco_dir, contrib_scores):
    """Load profiles from a modisco dir
    """
    modisco_dir = Path(modisco_dir)
    include_samples = load_included_samples(modisco_dir)
    f = ContribScoreFile(contrib_scores, include_samples)
    profiles = f.get_profiles()
    f.close()
    return profiles

# --------------------------------------------


def modisco_run(contrib_scores,
                output_dir,
                null_contrib_scores=None,
                hparams=None,
                override_hparams="",
                contrib_type="weighted",
                subset_tasks=None,
                filter_subset_tasks=False,
                filter_npy=None,
                exclude_chr="",
                seqmodel=False,  # interpretation glob
                # hparams=None,
                num_workers=10,
                max_strand_distance=0.1,
                overwrite=False,
                skip_dist_filter=False,
                use_all_seqlets=False,
                merge_tasks=False,
                gpu=None,
                ):
    """
    Run modisco

    Args:
      contrib_scores: path to the hdf5 file of contribution scores
      null_contrib_scores: Path to the null contribution scores
      contrib_type: for which output to compute the contribution scores
      hparams: None, modisco hyper - parameeters: either a path to modisco.yaml or
        a ModiscoHParams object
      override_hparams: hyper - parameters overriding the settings in the hparams file
      output_dir: output file directory
      filter_npy: path to a npy file containing a boolean vector used for subsetting
      exclude_chr: comma-separated list of chromosomes to exclude
      seqmodel: If enabled, then the contribution scores came from `contrib-score-seqmodel`
      subset_tasks: comma-separated list of task names to use as a subset
      filter_subset_tasks: if True, run modisco only in the regions for that TF
      hparams: hyper - parameter file
      summary: which summary statistic to use for the profile gradients
      skip_dist_filter: if True, distances are not used to filter
      use_all_seqlets: if True, don't restrict the number of seqlets
      split: On which data split to compute the results
      merge_task: if True, contribution scores for the tasks will be merged
      gpu: which gpu to use. If None, don't use any GPU's

    Note: when using subset_tasks, modisco will run on all the contribution scores. If you wish
      to run it only for the contribution scores for a particular task you should subset it to
      the peak regions of interest using `filter_npy`
    """
    plt.switch_backend('agg')
    add_file_logging(output_dir, logger, 'modisco-run')
    import os
    if gpu is not None:
        create_tf_session(gpu)
    else:
        # Don't use any GPU's
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # import theano
    import modisco
    import modisco.tfmodisco_workflow.workflow

    if seqmodel:
        assert '/' in contrib_type

    if subset_tasks == '':
        logger.warn("subset_tasks == ''. Not using subset_tasks")
        subset_tasks = None

    if subset_tasks == 'all':
        # Use all subset tasks e.g. don't subset
        subset_tasks = None

    if subset_tasks is not None:
        subset_tasks = subset_tasks.split(",")
        if len(subset_tasks) == 0:
            raise ValueError("Provide one or more subset_tasks. Found None")

    if filter_subset_tasks and subset_tasks is None:
        print("Using filter_subset_tasks=False since `subset_tasks` is None")
        filter_subset_tasks = False

    if exclude_chr:
        exclude_chr = exclude_chr.split(",")
    else:
        exclude_chr = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "modisco.h5"
    remove_exists(output_path, overwrite)

    output_distances = output_dir / "strand_distances.h5"
    remove_exists(output_distances, overwrite)

    if filter_npy is not None:
        filter_npy = os.path.abspath(filter_npy)

    # save the hyper-parameters
    write_json(dict(contrib_scores=os.path.abspath(contrib_scores),
                    contrib_type=contrib_type,
                    output_dir=str(output_dir),
                    subset_tasks=subset_tasks,
                    filter_subset_tasks=filter_subset_tasks,
                    hparams=hparams,
                    null_contrib_scores=null_contrib_scores,
                    # TODO - pack into hyper-parameters as well?
                    filter_npy=filter_npy,
                    exclude_chr=",".join(exclude_chr),
                    skip_dist_filter=skip_dist_filter,
                    use_all_seqlets=use_all_seqlets,
                    max_strand_distance=max_strand_distance,
                    gpu=gpu),
               os.path.join(output_dir, "kwargs.json"))

    print("-" * 40)
    # parse the hyper-parameters
    if hparams is None:
        print(f"Using default hyper-parameters")
        hp = ModiscoHParams()
    else:
        if isinstance(hparams, str):
            print(f"Loading hyper-parameters from file: {hparams}")
            hp = ModiscoHParams.load(hparams)
        else:
            assert isinstance(hparams, ModiscoHParams)
            hp = hparams
    if override_hparams:
        print(f"Overriding the following hyper-parameters: {override_hparams}")
    hp = tf.contrib.training.HParams(**hp.get_modisco_kwargs()).parse(override_hparams)

    if use_all_seqlets:
        hp.max_seqlets_per_metacluster = None

    # save the hyper-parameters
    print("Using the following hyper-parameters for modisco:")
    print("-" * 40)
    related_dump_yaml(ModiscoHParams(**hp.values()),
                      os.path.join(output_dir, "hparams.yaml"), verbose=True)
    print("-" * 40)

    # TODO - replace with contrib_scores
    d = HDF5Reader.load(contrib_scores)
    if 'hyp_contrib' not in d:
        # backcompatibility
        d['hyp_contrib'] = d['grads']

    if seqmodel:
        tasks = list(d['targets'])
    else:
        tasks = list(d['targets']['profile'])

    if subset_tasks is not None:
        # validate that all the `subset_tasks`
        # are present in `tasks`
        for st in subset_tasks:
            if st not in tasks:
                raise ValueError(f"subset task {st} not found in tasks: {tasks}")
        logger.info(f"Using the following tasks: {subset_tasks} instead of the original tasks: {tasks}")
        tasks = subset_tasks

    if isinstance(d['inputs'], dict):
        one_hot = d['inputs']['seq']
    else:
        one_hot = d['inputs']

    n = len(one_hot)

    # --------------------
    # apply filters
    if not skip_dist_filter:
        print("Using profile prediction for the strand filtering")
        contrib_type_filtered = 'weighted'
        distances = np.array([np.array([correlation(np.ravel(d['hyp_contrib'][task][contrib_type_filtered][0][i]),
                                                    np.ravel(d['hyp_contrib'][task][contrib_type_filtered][1][i]))
                                        for i in range(n)])
                              for task in tasks
                              if len(d['hyp_contrib'][task][contrib_type_filtered]) == 2]).T.mean(axis=-1)  # average the distances across tasks

        dist_filter = distances < max_strand_distance
        print(f"Fraction of sequences kept: {dist_filter.mean()}")

        HDF5BatchWriter.dump(output_distances,
                             {"distances": distances,
                              "included_samples": dist_filter})
    else:
        dist_filter = np.ones((n, ), dtype=bool)

    # add also the filter numpy
    if filter_npy is not None:
        print(f"Loading a filter file from {filter_npy}")
        filter_vec = np.load(filter_npy)
        dist_filter = dist_filter & filter_vec

    if filter_subset_tasks:
        assert subset_tasks is not None
        interval_from_task = pd.Series(d['metadata']['interval_from_task'])
        print(f"Subsetting the intervals accoring to subset_tasks: {subset_tasks}")
        print(f"Number of original regions: {dist_filter.sum()}")
        dist_filter = dist_filter & interval_from_task.isin(subset_tasks).values
        print(f"Number of filtered regions after filter_subset_tasks: {dist_filter.sum()}")

    # filter by chromosome
    if exclude_chr:
        logger.info(f"Excluding chromosomes: {exclude_chr}")
        chromosomes = d['metadata']['range']['chr']
        dist_filter = dist_filter & (~pd.Series(chromosomes).isin(exclude_chr)).values
    # -------------------------------------------------------------
    # setup contribution scores

    if seqmodel:
        thr_one_hot = one_hot[dist_filter]
        thr_hypothetical_contribs = {f"{task}/{gt}": d['hyp_contrib'][task][gt.split("/")[0]][gt.split("/")[1]][dist_filter]
                                     for task in tasks
                                     for gt in contrib_type.split(",")}
        thr_contrib_scores = {f"{task}/{gt}": thr_hypothetical_contribs[f"{task}/{gt}"] * thr_one_hot
                              for task in tasks
                              for gt in contrib_type.split(",")}
        task_names = [f"{task}/{gt}" for task in tasks for gt in contrib_type.split(",")]

    else:
        if merge_tasks:
            thr_one_hot = np.concatenate([one_hot[dist_filter]
                                          for task in tasks
                                          for gt in contrib_type.split(",")])
            thr_hypothetical_contribs = {"merged": np.concatenate([mean(d['hyp_contrib'][task][gt])[dist_filter]
                                                                   for task in tasks
                                                                   for gt in contrib_type.split(",")])}

            thr_contrib_scores = {"merged": thr_hypothetical_contribs['merged'] * thr_one_hot}
            task_names = ['merged']
        else:
            thr_one_hot = one_hot[dist_filter]
            thr_hypothetical_contribs = {f"{task}/{gt}": mean(d['hyp_contrib'][task][gt])[dist_filter]
                                         for task in tasks
                                         for gt in contrib_type.split(",")}
            thr_contrib_scores = {f"{task}/{gt}": thr_hypothetical_contribs[f"{task}/{gt}"] * thr_one_hot
                                  for task in tasks
                                  for gt in contrib_type.split(",")}
            task_names = [f"{task}/{gt}" for task in tasks for gt in contrib_type.split(",")]

    if null_contrib_scores is not None:
        logger.info(f"Using null_contrib_scores: {null_contrib_scores}")
        null_isf = ContribScoreFile(null_contrib_scores)
        null_per_pos_scores = {f"{task}/{gt}": v.sum(axis=-1) for gt in contrib_type.split(",")
                               for task, v in null_isf.get_contrib(contrib_score=gt).items() if task in tasks}
    else:
        # default Null distribution. Requires modisco 5.0
        logger.info(f"Using default null_contrib_scores")
        null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=10000)

    # -------------------------------------------------------------
    # run modisco
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Modisco defaults
        sliding_window_size=hp.sliding_window_size,
        flank_size=hp.flank_size,
        target_seqlet_fdr=hp.target_seqlet_fdr,
        min_passing_windows_frac=hp.min_passing_windows_frac,
        max_passing_windows_frac=hp.max_passing_windows_frac,
        min_metacluster_size=hp.min_metacluster_size,
        max_seqlets_per_metacluster=hp.max_seqlets_per_metacluster,
        seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=hp.trim_to_window_size,   # default: 30
            initial_flank_to_add=hp.initial_flank_to_add,  # default: 10
            kmer_len=hp.kmer_len,  # default: 8
            num_gaps=hp.num_gaps,  # default: 3
            num_mismatches=hp.num_mismatches,  # default: 2
            n_cores=num_workers,
            final_min_cluster_size=hp.final_min_cluster_size)  # default: 30
    )(
        task_names=task_names,
        contrib_scores=thr_contrib_scores,  # -> task score
        hypothetical_contribs=thr_hypothetical_contribs,
        one_hot=thr_one_hot,
        null_per_pos_scores=null_per_pos_scores)
    # -------------------------------------------------------------
    # save the results
    grp = h5py.File(output_path)
    tfmodisco_results.save_hdf5(grp)


def modisco_plot(modisco_dir,
                 output_dir,
                 # filter_npy=None,
                 # ignore_dist_filter=False,
                 figsize=(10, 10), contribsf=None):
    """Plot the results of a modisco run

    Args:
      modisco_dir: modisco directory
      output_dir: Output directory for writing the results
      figsize: Output figure size
      contribsf: [optional] modisco contribution score file (ContribScoreFile)
    """
    plt.switch_backend('agg')
    add_file_logging(output_dir, logger, 'modisco-plot')
    from bpnet.plot.vdom import write_heatmap_pngs
    from bpnet.plot.profiles import plot_profiles
    from bpnet.utils import flatten

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # load modisco
    mr = ModiscoResult(f"{modisco_dir}/modisco.h5")

    if contribsf is not None:
        d = contribsf
    else:
        d = ContribScoreFile.from_modisco_dir(modisco_dir)
        logger.info("Loading the contribution scores")
        d.cache()  # load all

    thr_one_hot = d.get_seq()
    # thr_hypothetical_contribs
    tracks = d.get_profiles()
    thr_hypothetical_contribs = dict()
    thr_contrib_scores = dict()
    # TODO - generalize this
    thr_hypothetical_contribs['weighted'] = d.get_hyp_contrib()
    thr_contrib_scores['weighted'] = d.get_contrib()

    tasks = d.get_tasks()

    # Count contribution (if it exists)
    if d.contains_contrib_score("counts/pre-act"):
        count_contrib_score = "counts/pre-act"
        thr_hypothetical_contribs['count'] = d.get_hyp_contrib(contrib_score=count_contrib_score)
        thr_contrib_scores['count'] = d.get_contrib(contrib_score=count_contrib_score)
    elif d.contains_contrib_score("count"):
        count_contrib_score = "count"
        thr_hypothetical_contribs['count'] = d.get_hyp_contrib(contrib_score=count_contrib_score)
        thr_contrib_scores['count'] = d.get_contrib(contrib_score=count_contrib_score)
    else:
        # Don't do anything
        pass

    thr_hypothetical_contribs = OrderedDict(flatten(thr_hypothetical_contribs, separator='/'))
    thr_contrib_scores = OrderedDict(flatten(thr_contrib_scores, separator='/'))

    #     # load contribution scores
    #     modisco_kwargs = read_json(f"{modisco_dir}/kwargs.json")
    #     d = HDF5Reader.load(modisco_kwargs['contrib_scores'])
    #     if 'hyp_contrib' not in d:
    #         # backcompatibility
    #         d['hyp_contrib'] = d['grads']
    #     tasks = list(d['targets']['profile'])

    #     if isinstance(d['inputs'], dict):
    #         one_hot = d['inputs']['seq']
    #     else:
    #         one_hot = d['inputs']

    #     # load used strand distance filter

    #     included_samples = load_included_samples(modisco_dir)

    #     contrib_type = "count,weighted"  # always plot both contribution scores

    #     thr_hypothetical_contribs = OrderedDict([(f"{gt}/{task}", mean(d['hyp_contrib'][task][gt])[included_samples])
    #                                              for task in tasks
    #                                              for gt in contrib_type.split(",")])
    #     thr_one_hot = one_hot[included_samples]
    #     thr_contrib_scores = OrderedDict([(f"{gt}/{task}", thr_hypothetical_contribs[f"{gt}/{task}"] * thr_one_hot)
    #                                       for task in tasks
    #                                       for gt in contrib_type.split(",")])
    #     tracks = OrderedDict([(task, d['targets']['profile'][task][included_samples]) for task in tasks])
    # -------------------------------------------------

    all_seqlets = mr.seqlets()
    all_patterns = mr.patterns()
    if len(all_patterns) == 0:
        print("No patterns found")
        return

    # 1. Plots with tracks and contrib scores
    print("Writing results for contribution scores")
    plot_profiles(all_seqlets,
                  thr_one_hot,
                  tracks=tracks,
                  contribution_scores=thr_contrib_scores,
                  legend=False,
                  flip_neg=True,
                  rotate_y=0,
                  seq_height=.5,
                  patterns=all_patterns,
                  n_bootstrap=100,
                  fpath_template=str(output_dir / "{pattern}/agg_profile_contribcores"),
                  mkdir=True,
                  figsize=figsize)

    # 2. Plots only with hypothetical contrib scores
    print("Writing results for hypothetical contribution scores")
    plot_profiles(all_seqlets,
                  thr_one_hot,
                  tracks={},
                  contribution_scores=thr_hypothetical_contribs,
                  legend=False,
                  flip_neg=True,
                  rotate_y=0,
                  seq_height=1,
                  patterns=all_patterns,
                  n_bootstrap=100,
                  fpath_template=str(output_dir / "{pattern}/agg_profile_hypcontribscores"),
                  figsize=figsize)

    print("Plotting heatmaps")
    for pattern in tqdm(all_patterns):
        write_heatmap_pngs(all_seqlets[pattern],
                           d,
                           tasks,
                           pattern,
                           output_dir=str(output_dir / pattern))

    mr.close()


def load_modisco_results(modisco_dir):
    """Load modisco_result - return

    Args:
      modisco_dir: directory path `output_dir` in `bpnet.cli.modisco.modisco_run`
        contains: modisco.h5, strand_distances.h5, kwargs.json

    Returns:
      TfModiscoResults object containing original track_set
    """
    import modisco
    from modisco.tfmodisco_workflow import workflow
    modisco_kwargs = read_json(f"{modisco_dir}/kwargs.json")
    contrib_type = modisco_kwargs['contrib_type']

    # load used strand distance filter
    included_samples = HDF5Reader.load(f"{modisco_dir}/strand_distances.h5")['included_samples']

    # load contribution scores
    d = HDF5Reader.load(modisco_kwargs['contrib_scores'])
    if 'hyp_contrib' not in d:
        # backcompatibility
        d['hyp_contrib'] = d['grads']

    tasks = list(d['targets']['profile'])
    if isinstance(d['inputs'], dict):
        one_hot = d['inputs']['seq']
    else:
        one_hot = d['inputs']
    thr_hypothetical_contribs = {f"{task}/{gt}": mean(d['hyp_contrib'][task][gt])[included_samples]
                                 for task in tasks
                                 for gt in contrib_type.split(",")}
    thr_one_hot = one_hot[included_samples]
    thr_contrib_scores = {f"{task}/{gt}": thr_hypothetical_contribs[f"{task}/{gt}"] * thr_one_hot
                          for task in tasks
                          for gt in contrib_type.split(",")}

    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
        task_names=tasks,
        contrib_scores=thr_contrib_scores,
        hypothetical_contribs=thr_hypothetical_contribs,
        one_hot=thr_one_hot)

    with h5py.File(os.path.join(modisco_dir, "modisco.h5"), "r") as grp:
        mr = workflow.TfModiscoResults.from_hdf5(grp, track_set=track_set)
    return mr, tasks, contrib_type


def modisco_centroid_seqlet_matches(modisco_dir, contrib_scores, output_dir,
                                    trim_frac=0.08, n_jobs=1,
                                    contribsf=None):
    """Write pattern matches to .csv
    """
    from bpnet.modisco.table import ModiscoData
    assert os.path.exists(output_dir)
    add_file_logging(output_dir, logger, 'centroid-seqlet-matches')

    logger.info("Loading required data")
    data = ModiscoData.load(modisco_dir, contrib_scores, contribsf=contribsf)

    logger.info("Generating the table")
    df = data.get_centroid_seqlet_matches(trim_frac=trim_frac, n_jobs=n_jobs)
    df.to_csv(os.path.join(output_dir, 'centroid_seqlet_matches.csv'))


def modisco_score2(modisco_dir,
                   output_file,
                   trim_frac=0.08,
                   contrib_scores=None,
                   contribution=None,
                   ignore_filter=False,
                   n_jobs=20):
    """Modisco score instances

    Args:
      modisco_dir: modisco directory - used to obtain centroid_seqlet_matches.csv and modisco.h5
      output_file: output file path for the tsv file. If the suffix is
        tsv.gz, then also gzip the file
      trim_frac: how much to trim the pattern when scanning
      contrib_scores: hdf5 file of contribution scores (contains `contribution` score)
        if None, then load the default contribution scores from modisco
      contribution: which contribution scores to use
      n_jobs: number of parallel jobs to use

    Writes a gzipped tsv file(tsv.gz)
    """
    add_file_logging(os.path.dirname(output_file), logger, 'modisco-score2')
    modisco_dir = Path(modisco_dir)
    modisco_kwargs = read_json(f"{modisco_dir}/kwargs.json")
    if contribution is None:
        contribution = modisco_kwargs['contrib_type']

    # Centroid matches
    cm_path = modisco_dir / 'centroid_seqlet_matches.csv'
    if not cm_path.exists():
        logger.info(f"Generating centroid matches to {cm_path.resolve()}")
        modisco_centroid_seqlet_matches(modisco_dir,
                                        contrib_scores,
                                        modisco_dir,
                                        trim_frac=trim_frac,
                                        n_jobs=n_jobs)
    logger.info(f"Loading centroid matches from {cm_path.resolve()}")
    dfm_norm = pd.read_csv(cm_path)

    mr = ModiscoResult(modisco_dir / "modisco.h5")
    mr.open()
    tasks = mr.tasks()

    # HACK prune the tasks of contribution (in case it's present)
    tasks = [t.replace(f"/{contribution}", "") for t in tasks]

    logger.info(f"Using tasks: {tasks}")

    if contrib_scores is not None:
        logger.info(f"Loading the contribution scores from: {contrib_scores}")
        contrib = ContribScoreFile(contrib_scores, default_contrib_score=contribution)
    else:
        contrib = ContribScoreFile.from_modisco_dir(modisco_dir, ignore_include_samples=ignore_filter)

    seq, contrib, hyp_contrib, profile, ranges = contrib.get_all()

    logger.info("Scanning for patterns")
    dfl = []
    for pattern_name in tqdm(mr.patterns()):
        pattern = mr.get_pattern(pattern_name).trim_seq_ic(trim_frac)
        match, contribution = pattern.scan_contribution(contrib, hyp_contrib, tasks,
                                                        n_jobs=n_jobs, verbose=False)
        seq_match = pattern.scan_seq(seq, n_jobs=n_jobs, verbose=False)
        dfm = pattern.get_instances(tasks, match, contribution, seq_match,
                                    norm_df=dfm_norm[dfm_norm.pattern == pattern_name],
                                    verbose=False, plot=False)
        dfl.append(dfm)

    logger.info("Merging")
    # merge and write the results
    dfp = pd.concat(dfl)

    # append the ranges
    logger.info("Append ranges")
    ranges.columns = ["example_" + v for v in ranges.columns]
    dfp = dfp.merge(ranges, on="example_idx", how='left')

    logger.info("Table info")
    dfp.info()
    logger.info(f"Writing the resuling pd.DataFrame of shape {dfp.shape} to {output_file}")
    # write to a parquet file
    dfp.to_parquet(output_file, partition_on=['pattern'], engine='fastparquet')
    logger.info("Done!")
    # except:
    #    import pdb
    #    pdb.set_trace()
    # dfp.to_csv(output_file, index=False, sep='\t', compression='gzip')


def modisco_report(modisco_dir, output_dir):
    render_ipynb(os.path.join(this_path, "../templates/modisco.ipynb"),
                 os.path.join(output_dir, "results.ipynb"),
                 params=dict(modisco_dir=modisco_dir))


def modisco_cluster_patterns(modisco_dir, output_dir):
    render_ipynb(os.path.join(this_path, "../modisco/cluster-patterns.ipynb"),
                 os.path.join(output_dir, "cluster-patterns.ipynb"),
                 params=dict(modisco_dir=str(modisco_dir),
                             output_dir=str(output_dir)))


# TODO - reimplement or remove
#         - just make it part of the main export function (?)
# def modisco_instances_to_bed(modisco_h5, instances_parq, contrib_score_h5, output_dir, trim_frac=0.08):
#     from bpnet.modisco.pattern_instances import load_instances

#     add_file_logging(output_dir, logger, 'modisco-instances-to-bed')
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     mr = ModiscoResult(modisco_h5)
#     mr.open()

#     print("load task_id")
#     d = HDF5Reader(contrib_score_h5)
#     d.open()
#     if 'hyp_contrib' not in d.f.keys():
#         # backcompatibility
#         d['hyp_contrib'] = d['grads']

#     id_hash = pd.DataFrame({"peak_id": d.f['/metadata/interval_from_task'][:],
#                             "example_idx": np.arange(d.f['/metadata/interval_from_task'].shape[0])})

#     # load the instances data frame
#     print("load all instances")
#     df = load_instances(instances_parq, motifs=None, dedup=True)
#     # import pdb
#     # pdb.set_trace()
#     df = df.merge(id_hash, on="example_idx")  # append peak_id

#     patterns = df.pattern.unique().tolist()
#     pattern_pssms = {pattern: mr.get_pssm(*pattern.split("/"))
#                      for pattern in patterns}
#     # present in basepair.modisco.core.scan
#     append_pattern_loc(df, pattern_pssms, trim_frac=trim_frac)

#     # write out the results
#     example_cols = ['example_chr', 'example_start', 'example_end', 'example_id', 'peak_id']
#     df_examples = df[example_cols].drop_duplicates().sort_values(["example_chr", "example_start"])
#     df_examples.to_csv(output_dir / "scored_regions.bed", sep='\t', header=False, index=False)

#     df["pattern_start_rel"] = df.pattern_start + df.example_start
#     df["pattern_end_rel"] = df.pattern_end + df.example_start
#     df["strand"] = df.revcomp.astype(bool).map({True: "-", False: "+"})

#     # TODO - update this - ?
#     pattern_cols = ['example_chr', 'pattern_start_rel', 'pattern_end_rel',
#                     'example_id', 'percnormed_score', 'strand', 'peak_id', 'seqlet_score']

#     (output_dir / "README").write_text("score_regions.bed columns: " +
#                                        ", ".join(example_cols) + "\n" +
#                                        "metacluster_<>/pattern_<>.bed columns: " +
#                                        ", ".join(pattern_cols))
#     df_pattern = df[pattern_cols]
#     for pattern in df.pattern.unique():
#         out_path = output_dir / (pattern + ".bed.gz")
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         dfp = df_pattern[df.pattern == pattern].drop_duplicates().sort_values(["example_chr",
#                                                                                "pattern_start_rel"])
#         dfp.to_csv(out_path, compression='gzip', sep='\t', header=False, index=False)


def modisco2bed(modisco_dir, output_dir, trim_frac=0.08):
    from pybedtools import Interval
    from bpnet.modisco.results import ModiscoResult
    add_file_logging(output_dir, logger, 'modisco2bed')
    ranges = load_ranges(modisco_dir)
    example_intervals = [Interval(row.chrom, row.start, row.end)
                         for i, row in ranges.iterrows()]

    r = ModiscoResult(os.path.join(modisco_dir, "modisco.h5"))
    r.export_seqlets_bed(output_dir,
                         example_intervals=example_intervals,
                         position='absolute',
                         trim_frac=trim_frac)
    r.close()


def modisco_table(modisco_dir, contrib_scores, output_dir, report_url=None, contribsf=None):
    """Write the pattern table to as .html and .csv
    """
    plt.switch_backend('agg')
    from bpnet.modisco.table import ModiscoData, modisco_table, write_modisco_table
    from bpnet.modisco.motif_clustering import hirearchically_reorder_table
    add_file_logging(output_dir, logger, 'modisco-table')
    print("Loading required data")
    data = ModiscoData.load(modisco_dir, contrib_scores, contribsf=contribsf)

    print("Generating the table")
    df = modisco_table(data)

    print("Writing the results")
    write_modisco_table(df, output_dir, report_url, 'pattern_table')

    print("Writing clustered table")
    write_modisco_table(hirearchically_reorder_table(df, data.tasks),
                        output_dir, report_url, 'pattern_table.sorted')

    print("Writing footprints")
    profiles = OrderedDict([(pattern, {task: data.get_profile_wide(pattern, task).mean(axis=0)
                                       for task in data.tasks})
                            for pattern in data.mr.patterns()])
    write_pkl(profiles, Path(output_dir) / 'footprints.pkl')
    print("Done!")


def modisco_centroid_seqlet_matches2(modisco_dir, contrib_scores, output_dir=None, trim_frac=0.08, n_jobs=1, data_class='profile'):
    """Write pattern matches to .csv
    """
    from bpnet.modisco.table import ModiscoData, ModiscoDataSingleBinary
    if(output_dir is None):
        output_dir = modisco_dir
    add_file_logging(output_dir, logger, 'centroid-seqlet-matches')

    logger.info("Loading required data")
    if(data_class == 'profile'):
        data = ModiscoData.load(modisco_dir, contrib_scores)
    else:
        data = ModiscoDataSingleBinary.load(modisco_dir)

    logger.info("Generating the table")
    df = data.get_centroid_seqlet_matches(trim_frac=trim_frac, n_jobs=n_jobs)

    df.to_csv(os.path.join(output_dir, 'centroid_seqlet_matches.csv'))

    return None


def modisco_enrich_patterns(patterns_pkl_file, modisco_dir, output_file, contribsf=None):
    """Add stacked_seqlet_contrib to pattern `attrs`

    Args:
      patterns_pkl: patterns.pkl file path
      modisco_dir: modisco directory containing
      output_file: output file path for patterns.pkl
    """
    from bpnet.utils import read_pkl, write_pkl
    from bpnet.cli.contrib import ContribScoreFile
    from bpnet.modisco.core import StackedSeqletContrib

    logger.info("Loading patterns")
    modisco_dir = Path(modisco_dir)
    patterns = read_pkl(patterns_pkl_file)

    mr = ModiscoResult(modisco_dir / 'modisco.h5')
    mr.open()

    if contribsf is None:
        contrib_file = ContribScoreFile.from_modisco_dir(modisco_dir)
        logger.info("Loading ContribScoreFile into memory")
        contrib_file.cache()
    else:
        logger.info("Using the provided ContribScoreFile")
        contrib_file = contribsf

    logger.info("Extracting profile and contribution scores")
    extended_patterns = []
    for p in tqdm(patterns):
        p = p.copy()
        profile_width = p.len_profile()
        # get the shifted seqlets
        seqlets = [s.pattern_align(**p.attrs['align']) for s in mr._get_seqlets(p.name)]

        # keep only valid seqlets
        valid_seqlets = [s for s in seqlets
                         if s.valid_resize(profile_width, contrib_file.get_seqlen() + 1)]
        # extract the contribution scores
        p.attrs['stacked_seqlet_contrib'] = contrib_file.extract(valid_seqlets, profile_width=profile_width)

        p.attrs['n_seqlets'] = mr.n_seqlets(*p.name.split("/"))
        extended_patterns.append(p)

    write_pkl(extended_patterns, output_file)


def modisco_export_patterns(modisco_dir, output_file, contribsf=None):
    """Export patterns to a pkl file. Don't cluster them

    Adds `stacked_seqlet_contrib` and `n_seqlets` to pattern `attrs`

    Args:
      patterns_pkl: patterns.pkl file path
      modisco_dir: modisco directory containing
      output_file: output file path for patterns.pkl
    """
    from bpnet.utils import read_pkl, write_pkl
    from bpnet.cli.contrib import ContribScoreFile
    from bpnet.modisco.core import StackedSeqletContrib

    logger.info("Loading patterns")
    modisco_dir = Path(modisco_dir)

    mr = ModiscoResult(modisco_dir / 'modisco.h5')
    mr.open()
    patterns = [mr.get_pattern(pname)
                for pname in mr.patterns()]

    if contribsf is None:
        contrib_file = ContribScoreFile.from_modisco_dir(modisco_dir)
        logger.info("Loading ContribScoreFile into memory")
        contrib_file.cache()
    else:
        logger.info("Using the provided ContribScoreFile")
        contrib_file = contribsf

    logger.info("Extracting profile and contribution scores")
    extended_patterns = []
    for p in tqdm(patterns):
        p = p.copy()

        # get the shifted seqlets
        valid_seqlets = mr._get_seqlets(p.name)

        # extract the contribution scores
        sti = contrib_file.extract(valid_seqlets, profile_width=None)
        sti.dfi = mr.get_seqlet_intervals(p.name, as_df=True)
        p.attrs['stacked_seqlet_contrib'] = sti
        p.attrs['n_seqlets'] = mr.n_seqlets(*p.name.split("/"))
        extended_patterns.append(p)

    write_pkl(extended_patterns, output_file)


def modisco_report_all(modisco_dir, trim_frac=0.08, n_jobs=20, scan_instances=False, force=False):
    """Compute all the results for modisco. Runs:
    - modisco_plot
    - modisco_report
    - modisco_table
    - modisco_centroid_seqlet_matches
    - modisco_score2
    - modisco2bed
    - modisco_instances_to_bed

    Args:
      modisco_dir: directory path `output_dir` in `bpnet.cli.modisco.modisco_run`
        contains: modisco.h5, strand_distances.h5, kwargs.json
      trim_frac: how much to trim the pattern
      n_jobs: number of parallel jobs to use
      force: if True, commands will be re-run regardless of whether whey have already
        been computed

    Note:
      All the sub-commands are only executed if they have not been ran before. Use --force override this.
      Whether the commands have been run before is deterimined by checking if the following file exists:
        `{modisco_dir}/.modisco_report_all/{command}.done`.
    """
    plt.switch_backend('agg')
    from bpnet.utils import ConditionalRun

    modisco_dir = Path(modisco_dir)
    # figure out the contribution scores used
    kwargs = read_json(modisco_dir / "kwargs.json")
    contrib_scores = kwargs["contrib_scores"]

    mr = ModiscoResult(f"{modisco_dir}/modisco.h5")
    mr.open()
    all_patterns = mr.patterns()
    mr.close()
    if len(all_patterns) == 0:
        print("No patterns found.")
        # Touch results.html for snakemake
        open(modisco_dir / 'results.html', 'a').close()
        open(modisco_dir / 'seqlets/scored_regions.bed', 'a').close()
        return

    # class determining whether to run the command or not (poor-man's snakemake)
    cr = ConditionalRun("modisco_report_all", None, modisco_dir, force=force)

    sync = []
    # --------------------------------------------
    if (not cr.set_cmd('modisco_plot').done()
        or not cr.set_cmd('modisco_cluster_patterns').done()
            or not cr.set_cmd('modisco_enrich_patterns').done()):
        # load ContribScoreFile and pass it to all the functions
        logger.info("Loading ContribScoreFile")
        contribsf = ContribScoreFile.from_modisco_dir(modisco_dir)
        contribsf.cache()
    else:
        contribsf = None
    # --------------------------------------------
    # Basic reports
    if not cr.set_cmd('modisco_plot').done():
        modisco_plot(modisco_dir,
                     modisco_dir / 'plots',
                     figsize=(10, 10), contribsf=contribsf)
        cr.write()
    sync.append("plots")

    if not cr.set_cmd('modisco_report').done():
        modisco_report(str(modisco_dir), str(modisco_dir))
        cr.write()
    sync.append("results.html")

    if not cr.set_cmd('modisco_table').done():
        modisco_table(modisco_dir, contrib_scores, modisco_dir, report_url=None, contribsf=contribsf)
        cr.write()
    sync.append("footprints.pkl")
    sync.append("pattern_table.*")

    if not cr.set_cmd('modisco_cluster_patterns').done():
        modisco_cluster_patterns(modisco_dir, modisco_dir)
        cr.write()
    sync.append("patterns.pkl")
    sync.append("cluster-patterns.*")
    sync.append("motif_clustering")

    if not cr.set_cmd('modisco_enrich_patterns').done():
        modisco_enrich_patterns(modisco_dir / 'patterns.pkl',
                                modisco_dir,
                                modisco_dir / 'patterns.pkl', contribsf=contribsf)
        cr.write()
    # sync.append("patterns.pkl")

    # TODO - run modisco align
    # - [ ] add the motif clustering step (as ipynb) and export the aligned tables
    #   - save the final table as a result to CSV (ready to be imported in excel)
    # --------------------------------------------
    # Finding new instances
    if scan_instances:
        if not cr.set_cmd('modisco_centroid_seqlet_matches').done():
            modisco_centroid_seqlet_matches(modisco_dir, contrib_scores, modisco_dir,
                                            trim_frac=trim_frac,
                                            n_jobs=n_jobs,
                                            contribsf=contribsf)
            cr.write()

        # TODO - this would not work with the per-TF contribution score file....
        if not cr.set_cmd('modisco_score2').done():
            modisco_score2(modisco_dir,
                           modisco_dir / 'instances.parq',
                           trim_frac=trim_frac,
                           contrib_scores=None,  # Use the default one
                           contribution=None,  # Use the default one
                           n_jobs=n_jobs)
            cr.write()
    # TODO - update the pattern table -> compute the fraction of other motifs etc
    # --------------------------------------------
    # Export bed-files and bigwigs

    # Seqlets
    if not cr.set_cmd('modisco2bed').done():
        modisco2bed(str(modisco_dir), str(modisco_dir / 'seqlets'), trim_frac=trim_frac)
        cr.write()
    sync.append("seqlets")

    # Scanned instances
    # if not cr.set_cmd('modisco_instances_to_bed').done():
    #     modisco_instances_to_bed(str(modisco_dir / 'modisco.h5'),
    #                              instances_parq=str(modisco_dir / 'instances.parq'),
    #                              contrib_score_h5=contrib_scores,
    #                              output_dir=str(modisco_dir / 'instances_bed/'),
    #                              )
    #     cr.write()
    # sync.append("instances_bed")

    # print the rsync command to run in order to sync the output
    # directories to the webserver
    logger.info("Run the following command to sync files to the webserver")
    dirs = " ".join(sync)
    print(f"rsync -av --progress {dirs} <output_dir>/")
