import pandas as pd
from bpnet.preproc import resize_interval
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

from bpnet.plot.profiles import extract_signal
from bpnet.modisco.core import dfi2seqlets, resize_seqlets, resize_seqlets
from bpnet.simulate import profile_sim_metrics
from bpnet.stats import quantile_norm


def profile_split(profile, seqlets):
    """Split the profile to counts and profile probabilities
    """
    total_counts = profile.sum(axis=-1).sum(axis=-1)
    sort_idx = np.argsort(-total_counts)

    # probabilities
    p = profile[sort_idx] / profile[sort_idx].sum(axis=1, keepdims=True)

    # drop NA's
    notnan = ~np.any(np.any(np.isnan(p), axis=-1), axis=-1)
    total_counts = total_counts[sort_idx][notnan]
    p = p[notnan]

    seqlet_idx = np.array([s.seqname for s in seqlets])[notnan]
    return p, total_counts, seqlet_idx


def profile_features(seqlets, ref_seqlets, profile, profile_width=70):
    # tasks = list(profile)

    # resize
    seqlets = resize_seqlets(seqlets, profile_width, seqlen=profile.shape[1])
    seqlets_ref = resize_seqlets(ref_seqlets, profile_width, seqlen=profile.shape[1])

    # extract the profile
    seqlet_profile = extract_signal(profile, seqlets)
    seqlet_profile_ref = extract_signal(profile, seqlets_ref)

    # compute the average profile
    avg_profile = seqlet_profile_ref.mean(axis=0)

    metrics = pd.DataFrame([profile_sim_metrics(avg_profile, cp) for cp in seqlet_profile])
    metrics_ref = pd.DataFrame([profile_sim_metrics(avg_profile, cp) for cp in seqlet_profile_ref])

    assert len(metrics) == len(seqlets)  # needs to be the same length
    return pd.DataFrame(OrderedDict([
        ("profile_match", metrics.simmetric_kl),
        ("profile_match_p", quantile_norm(metrics.simmetric_kl, metrics_ref.simmetric_kl)),
        ("profile_counts", metrics['counts']),
        ("profile_counts_p", quantile_norm(metrics['counts'], metrics_ref['counts'])),
        ("profile_max", metrics['max']),
        ("profile_max_p", quantile_norm(metrics['max'], metrics_ref['max'])),
    ]))


def annotate_profile(dfi, mr, profiles, profile_width=70, trim_frac=0.08):
    """Append profile match columns to dfi
    """
    dfi = dfi.copy()
    dfp_list = []
    for pattern in tqdm(dfi.pattern.unique()):
        for task in profiles:
            dfp = profile_features(dfi2seqlets(dfi[dfi.pattern == pattern]),
                                   ref_seqlets=mr._get_seqlets(pattern, trim_frac=trim_frac),
                                   profile=profiles[task],
                                   profile_width=profile_width)
            dfp.columns = [f'{task}/{c}' for c in dfp.columns]  # prepend task
            dfp_list.append(dfp)
    return pd.concat(dfp_list + [dfi], axis=1)
