from typing import List
import os
import subprocess
import numpy as np
import pandas as pd
import attr
from tqdm import tqdm
from copy import deepcopy
from bpnet.modisco.utils import ic_scale, trim_pssm_idx
from concise.utils.pwm import DEFAULT_LETTER_TO_INDEX, DEFAULT_INDEX_TO_LETTER
from collections import OrderedDict
from bpnet.plot.tracks import plot_tracks, filter_tracks, rc_tracks, skip_nan_tracks, pad_track, pad_tracks
from bpnet.plot.profiles import extract_signal
from bpnet.utils import flatten, unflatten, halve
from bpnet.modisco.utils import shorten_pattern
from bpnet.modisco.sliding_similarities import sliding_similarity, pssm_scan, pad_same
from bpnet.stats import fdr_threshold_norm_right, quantile_norm, low_medium_high, get_metric
from bpnet.functions import mean

# TODO - add `stacked_seqlet_tracks` to Profile `attrs` and pickle it to `patterns.pkl`
#   - make sure you shift and rc the seqlet when extracting it


class Pattern:

    VOCAB = ["A", "C", "G", "T"]  # sequence vocabulary
    _track_list = ['seq', 'contrib', 'hyp_contrib', 'profile']

    def __init__(self, name, seq, contrib, hyp_contrib, profile=None, attrs={}):
        """Pattern container

        Args:
          name (str): name of the pattern
          seq (np.array): numpy array of shape
        """
        # name
        self.name = name

        # tracks
        self.seq = seq  # shall we call it PWM?
        self.contrib = contrib
        self.hyp_contrib = hyp_contrib
        self.profile = profile

        # other custom attributes
        self.attrs = attrs

        self._tasks = list(self.contrib.keys())

        # validate that all the scores are of the same length
        for kind in ['contrib', 'hyp_contrib', 'profile']:
            t = self._get_track(kind)
            if t is None:
                continue
            assert set(t.keys()) == set(self.tasks())
            for task in self.tasks():
                if t[task] is None:
                    continue
                assert isinstance(t[task], np.ndarray)
                if kind != 'profile':
                    assert len(t[task]) == len(self.seq)
                    assert t[task].ndim == 2

    def tasks(self):
        return self._tasks

    # properties. used with `to_summary_dict
    @property
    def seq_info_content(self):
        """Get the sequence information content
        """
        return self.get_seq_ic().sum()

    @property
    def short_name(self):
        return shorten_pattern(self.name)

    # --------------------------------

    def to_summary_dict(self, properties=['name', 'seq_info_content']):
        """Summarize the pattern to properties
        """
        return OrderedDict([(p, getattr(self, p)) for p in properties])

    def _validate_kind(self, kind):
        if kind not in self._track_list + ['seq_ic']:
            raise ValueError("kind needs to be from " + ",".join(self._track_list))

    def _get_track(self, kind='seq'):
        if "/" in kind:
            kind, task = kind.split("/")
        else:
            kind, task = kind, None

        def get_task(d, k):
            if k is None:
                return d
            elif k == 'mean':
                return mean(list(d.values()))
            elif k == 'sum':
                return sum(list(d.values()))
            # TODO - add weighted based on the contribution scores
            else:
                return d[k]

        self._validate_kind(kind)
        if kind == 'seq':
            return self.seq
        elif kind == 'contrib':
            return get_task(self.contrib, task)
        elif kind == 'hyp_contrib':
            return get_task(self.hyp_contrib, task)
        elif kind == 'profile':
            return get_task(self.profile, task)
        elif kind == 'seq_ic':
            return self.get_seq_ic()
        else:
            raise ValueError("kind needs to be from seq,contrib,hyp_contrib")

    def get_consensus(self, kind='seq'):
        """Get the consensus sequence

        Args:
          kind: one of 'seq', 'contrib' or 'hyp_contrib'
        """
        max_idx = self._get_track(kind).argmax(axis=1)
        index_to_letter = {i: l for i, l in enumerate(self.VOCAB)}
        return ''.join([index_to_letter[x] for x in max_idx])

    def __repr__(self):
        return f"Pattern('{self.name}', '{self.get_consensus(kind='seq')}' ...)"

    @classmethod
    def from_hdf5_grp(cls, grp, name):
        """Load the pattern from the hdf5 group
        """
        tasks = [t for t in grp.keys() if t not in ['sequence', 'seqlets_and_alnmts']]

        if ('_contrib_scores' in tasks[0] and '_hypothetical_contribs' in tasks[1]):
            # this is a signle task case
            task = tasks[0].split('_contrib_scores')[0]
            return Pattern(name,
                           seq=grp['sequence']['fwd'][:],
                           contrib={task: grp[tasks[0]]['fwd'][:]},
                           hyp_contrib={task: grp[tasks[1]]['fwd'][:]})

        possible_names = list(grp[tasks[0]].keys())
        if possible_names == ['fwd', 'rev']:
            return Pattern(name,
                           seq=grp['sequence']['fwd'][:],
                           contrib={t: grp[t]['fwd'][:] for t in tasks},
                           hyp_contrib={t: grp[t]['fwd'][:] for t in tasks})
        else:
            contrib_name = [pn for pn in possible_names if "contrib_scores" in pn]
            if len(contrib_name) == 1:
                contrib_name = contrib_name[0]
                hyp_contrib_name = [pn for pn in possible_names if "hypothetical_contribs" in pn][0]
                return Pattern(name,
                               seq=grp['sequence']['fwd'][:],
                               contrib={t: grp[t][contrib_name]['fwd'][:] for t in tasks},
                               hyp_contrib={t: grp[t][hyp_contrib_name]['fwd'][:] for t in tasks})
            else:
                # contrib_name contains a slash...
                assert len(possible_names) == 1
                grp_1 = possible_names[0]
                possible_names = list(grp[tasks[0]][grp_1].keys())

                contrib_name = [pn for pn in possible_names if "contrib_scores" in pn][0]
                hyp_contrib_name = [pn for pn in possible_names if "hypothetical_contribs" in pn][0]
                return Pattern(name,
                               seq=grp['sequence']['fwd'][:],
                               contrib={t: grp[t][grp_1][contrib_name]['fwd'][:] for t in tasks},
                               hyp_contrib={t: grp[t][grp_1][hyp_contrib_name]['fwd'][:] for t in tasks})

    def get_seq_ic(self):
        """Get the sequence on the information-content scale
        """
        return ic_scale(self.seq)

    def _trim_seq_ic_ij(self, trim_frac=0.0):
        return trim_pssm_idx(self.get_seq_ic(), frac=trim_frac)

    def trim_seq_ic(self, trim_frac=0.0):
        """Trim the pattern based on sequence information content
        """
        if trim_frac is None:
            return self.copy()
        # get the information content trim fraction
        i, j = self._trim_seq_ic_ij(trim_frac=trim_frac)
        return self.trim(i, j)

    def append_pfm_to_database(self, meme_db, trim_frac=0.08, freqs=[0.27, 0.23, 0.23, 0.27]):
        """write the pfm into a meme file to be used by motif search tools
        """
        motif_name = "PWM_{}".format(self.short_name)
        if(os.path.exists(meme_db)):
            with open(meme_db, "r") as fp:
                for line in fp:
                    if(motif_name in line):
                        print("{} already exists".format(motif_name))
                        return None
        else:
            with open(meme_db, "w") as fp:
                fp.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\n\n")
                fp.write("A {0} C {1} G {2} T {3}\n\n".format(freqs[0], freqs[1], freqs[2], freqs[3]))

        i, j = trim_pssm_idx(self.get_seq_ic(), frac=trim_frac)
        trimmed_pattern = self.trim(i, j)
        pfm = trimmed_pattern.seq
        with open(meme_db, 'a') as fp:
            fp.write("MOTIF {}\n".format(motif_name))
            fp.write("letter-probability matrix: alength= 4 w= {} nsites= 20 E= 0e+0\n".format(pfm.shape[0]))
            for line in pfm:
                fp.write('%.5f %.5f %.5f %.5f\n' % tuple(line))
            fp.write("\n")
        return None

    def append_pwm_to_database(self, pwm_db, trim_frac=0.08):
        """write a the pwm into a database to be used by motif search tools

        """
        pwm_name = "PWM_{}".format(self.short_name)
        if(os.path.exists(pwm_db)):  # search whether the pwm already exists in the database
            with open(pwm_db, 'r') as fp:
                for line in fp:
                    if(pwm_name in line):
                        print("{} already exists".format(pwm_name))
                        return None

        i, j = trim_pssm_idx(self.get_seq_ic(), frac=trim_frac)
        trimmed_pattern = self.trim(i, j)
        pssm = trimmed_pattern.get_seq_ic()
        with open(pwm_db, 'a') as fp:
            fp.write(">{}\n".format(pwm_name))
            for line in pssm:
                fp.write('%.5f %.5f %.5f %.5f\n' % tuple(line))

        return None

    def write_meme_file(self, bg, fname):
        """write a temporary meme file to be used by tomtom
        Args:
           bg: background
        """
        ppm = self.seq  # position probability matrix
        f = open(fname, 'w')
        f.write('MEME version 4\n\n')
        f.write('ALPHABET= ACGT\n\n')
        f.write('strands: + -\n\n')
        f.write('Background letter frequencies (from unknown source):\n')
        f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
        f.write('MOTIF 1 TEMP\n\n')
        f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
        for s in ppm:
            f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
        f.close()

        return None

    def fetch_tomtom_matches(self, background=[0.27, 0.23, 0.23, 0.27],
                             tomtom_exec_path='tomtom',
                             motifs_db='HOCOMOCOv11_full_HUMAN_mono_meme_format.meme',
                             save_report=False,
                             report_dir='./',
                             temp_dir='./',
                             trim_frac=0.08):
        """Fetches top matches from a motifs database using TomTom.
        Args:
            background: list with ACGT background probabilities
            tomtom_exec_path: path to TomTom executable
            motifs_db: path to motifs database in meme format
            n: number of top matches to return, ordered by p-value
            temp_dir: directory for storing temp files
            trim_threshold: the ppm is trimmed from left till first position for which
                probability for any base pair >= trim_threshold. Similarly from right.
        Returns:
            list: a list of up to n results returned by tomtom, each entry is a
                dictionary with keys 'Target ID', 'p-value', 'E-value', 'q-value'
        """
        fname = os.path.join(temp_dir, 'query_file')
        # trim and prepare meme file
        i, j = trim_pssm_idx(self.get_seq_ic(), frac=trim_frac)
        trimmed_pattern = self.trim(i, j)
        trimmed_pattern.write_meme_file(background, fname)

        # run tomtom
        if(save_report):
            cmd = '{0} -no-ssc -oc {1} -verbosity 1 -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 {2} {3}'.format(tomtom_exec_path, report_dir, fname, motifs_db)
            print(cmd)
            out = subprocess.check_output(cmd, shell=True)
            df = pd.read_table("{}/tomtom.tsv".format(report_dir))
            df = df[['Target_ID', 'p-value', 'E-value', 'q-value']]
            schema = list(df.columns)
            dat = df.get_values()
        else:
            cmd = "{0} -no-ssc -oc . -verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 {1} {2}".format(tomtom_exec_path, fname, motifs_db)
            print(cmd)
            out = subprocess.check_output(cmd, shell=True)
            dat = [x.split('\t') for x in out.strip().decode("utf-8").split('\n')]
            schema = dat[0]
            dat = dat[1:]

        tget_idx, pval_idx, eval_idx, qval_idx = schema.index('Target_ID'), schema.index('p-value'), schema.index('E-value'), schema.index('q-value')
        r = []
        for t in dat:
            if(len(t) < 4):
                break
            mtf = {}
            mtf['Target ID'] = t[tget_idx]
            mtf['p-value'] = float(t[pval_idx])
            mtf['E-value'] = float(t[eval_idx])
            mtf['q-value'] = float(t[qval_idx])
            # if(mtf['q-value']<0.001):
            #    break
            r.append(mtf)

        os.system('rm ' + fname)
        return r

    def resize(self, new_len, anchor='center'):
        """Note: profile is left intact
        """
        if anchor != 'center':
            raise NotImplementedError("Only anchor='center' is implemented at the moment")
        # if self.profile is not None:
        #    raise ValueError("resize() not possible with profile != None")
        if new_len < len(self):
            delta = len(self) - new_len
            i = delta // 2
            j = i + new_len
            return self.trim(i, j)
        else:
            return self.pad(new_len)

    def resize_profile(self, new_len, anchor='center'):
        """
        """
        if anchor != 'center':
            raise NotImplementedError("Only anchor='center' is implemented at the moment")
        # if self.profile is not None:
        #    raise ValueError("resize() not possible with profile != None")
        if new_len < self.len_profile():
            delta = self.len_profile() - new_len
            i = delta // 2
            j = i + new_len
            return self.trim_profile(i, j)
        else:
            return self.pad_profile(new_len)

    def trim(self, i, j):
        return Pattern(self.name,
                       self.seq[i:j],
                       filter_tracks(self.contrib, [i, j]),
                       filter_tracks(self.hyp_contrib, [i, j]),
                       self.profile,
                       self.attrs)

    def trim_profile(self, i, j):
        return Pattern(self.name,
                       self.seq,
                       self.contrib,
                       self.hyp_contrib,
                       filter_tracks(self.profile, [i, j]),
                       self.attrs)

    def pad(self, new_len, value=0):
        return Pattern(self.name,
                       pad_track(self.seq, new_len, value=value),
                       pad_tracks(self.contrib, new_len, value=value),
                       pad_tracks(self.hyp_contrib, new_len, value=value),
                       self.profile,
                       self.attrs)

    def pad_profile(self, new_len, value=0):
        return Pattern(self.name,
                       self.seq,
                       self.contrib,
                       self.hyp_contrib,
                       pad_tracks(self.profile, new_len, value=value),
                       self.attrs)

    def shift(self, offset=0, pad_value=0):
        """Shift the motif

        positive offset means the 'frame' is kept still
        while the motif is shifted to the right
        """
        padded = self.pad(len(self) + 2 * np.abs(offset), value=pad_value)
        padded = padded.pad_profile(padded.len_profile() + 2 * np.abs(offset))
        start = np.abs(offset) - offset
        return padded.trim(start, start + len(self)).trim_profile(start, start + self.len_profile())

    def rc(self):
        """Return the reverse-complemented version of the pattern
        """
        return Pattern(self.name,
                       self.seq[::-1, ::-1],
                       rc_tracks(self.contrib),
                       rc_tracks(self.hyp_contrib),
                       rc_tracks(self.profile),
                       self.attrs)

    def plot(self, kind='all', rotate_y=90, letter_width=0.2,
             height=0.8, ylab=True, **kwargs):
        if isinstance(kind, list):
            kind_list = kind
        else:
            if kind == 'all':
                kind_list = self._track_list
            else:
                self._validate_kind(kind)
                kind_list = [kind]

        tracks = OrderedDict([(kind, self._get_track(kind)) for kind in kind_list])
        tracks = skip_nan_tracks(flatten(tracks, "/"))
        if 'seq' in tracks:
            tracks['seq'] = self.get_seq_ic()  # override the sequence with information content
        if 'title' not in kwargs:
            kwargs['title'] = self.name
        return plot_tracks(tracks,
                           # title=self.name,
                           rotate_y=rotate_y,
                           fig_width=len(self) * letter_width,
                           fig_height_per_track=height,
                           ylab=ylab,
                           **kwargs)

    def vdom_plot(self, kind='seq', width=80, letter_width=0.2, letter_height=0.8, as_html=False):
        """Get the html
        """
        from bpnet.plot.vdom import vdom_pssm

        if kind == 'contrib':
            # summarize across tasks
            arr = mean(list(self.contrib.values()))  # average the contrib scores across tasks
        elif kind == 'hyp_contrib':
            # summarize across tasks
            arr = mean(list(self.hyp_contrib.values()))  # average the contrib scores across tasks
        elif kind == 'seq':
            # get the IC
            arr = self.get_seq_ic()
        else:
            self._validate_kind(kind)
            arr = self._get_track(kind)

        vdom_obj = vdom_pssm(arr,
                             letter_width=letter_width,
                             letter_height=letter_height)
        if as_html:
            return vdom_obj.to_html().replace("<img", f"<img width={width}")  # hack
        else:
            return vdom_obj

    def __len__(self):
        return len(self.seq)

    def len_profile(self):
        if self.profile is None:
            return 0
        else:
            return len(self.profile[self.tasks()[0]])

    def copy(self):
        return deepcopy(self)

    def aligned_distance_seq(self, pattern, metric='simmetric_kl', pseudo_p=1e-3):
        """Average per-base distribution distance
        """
        # introduce pseudo-counts
        sp1 = self.seq + pseudo_p
        sp1 = sp1 / sp1.sum(1, keepdims=True)
        sp2 = pattern.seq + pseudo_p
        sp2 = sp2 / sp1.sum(1, keepdims=True)
        m = get_metric(metric)
        return mean([m(sp1[i], sp2[i]) for i in range(len(sp1))])

    def aligned_distance_profile(self, pattern, metric='simmetric_kl', pseudo_p=1e-8):
        """Compare two profile distributions (average across strands)
        """
        m = get_metric(metric)

        # introduce pseudo-counts
        o = dict()
        for t in self.tasks():
            pp1 = self.profile[t] + pseudo_p
            pp1 = pp1 / pp1.sum(0, keepdims=True)
            pp2 = pattern.profile[t] + pseudo_p
            pp2 = pp2 / pp2.sum(0, keepdims=True)
            o[t] = mean([m(pp1[i], pp2[i]) for i in range(pp1.shape[1])])
        return o

    def similarity(self, pattern, track='seq', metric='continousjaccard', max_score=True):
        """Compute the similarity to another pattern

        Args:
          pattern: other Pattern or a list of patterns. This pattern is used to as the template that gets scanned by self
          track: which track to use
          metric: which metric to use
          max_score: if True, the maximum similarity is returned. If False, all similarity
            scores are returned

        Returns:
          (float) similarity score. The higher, the more similar the two patterns are
        """
        if track.startswith("profile"):
            motif_len = self.len_profile()
            if isinstance(pattern, list):
                pattern = [p.resize_profile(motif_len * 3 - 1)
                           for p in pattern]
            else:
                pattern = pattern.resize_profile(motif_len * 3 - 1)
        else:
            motif_len = len(self)
            if isinstance(pattern, list):
                pattern = [p.resize(motif_len * 3 - 1)
                           for p in pattern]
            else:
                pattern = pattern.resize(motif_len * 3 - 1)

        if isinstance(pattern, list):
            match_scores = sliding_similarity(self._get_track(track)[np.newaxis],
                                              np.concatenate([p._get_track(track)[np.newaxis]
                                                              for p in pattern]),
                                              metric=metric,
                                              verbose=False,
                                              n_jobs=1)
            if metric == 'continousjaccard':
                match_scores = np.concatenate(match_scores[0]).sum(axis=0)
            else:
                match_scores = match_scores.sum(axis=0)
        else:
            match_scores = sliding_similarity(self._get_track(track)[np.newaxis],
                                              pattern._get_track(track)[np.newaxis],
                                              metric=metric,
                                              verbose=False,
                                              n_jobs=1)
            if metric == 'continousjaccard':
                match_scores = match_scores[0][0][0]
            else:
                match_scores = match_scores[0]

        assert len(match_scores) == 2 * motif_len
        if max_score:
            return match_scores.max()
        else:
            return match_scores

    def align(self, pattern, track='seq', metric='continousjaccard', max_shift=None, pad_value=0,
              return_similarity=False):
        """Compute the similarity to another pattern

        Args:
          pattern: other Pattern
          track: which track to use
          metric: which metric to use
          max_shift: if specified the shift of the motif exceeds some threshold,
            then the alignemnt will not be performed

        Returns:
          (float) similarity score. The higher, the more similar the two patterns are
        """
        similarities_fwd = self.similarity(pattern, track=track, metric=metric, max_score=False)
        similarities_rev = self.rc().similarity(pattern, track=track, metric=metric, max_score=False)
        if similarities_fwd.max() > similarities_rev.max():
            sim = similarities_fwd
            obj = self
            use_rc = False
        else:
            sim = similarities_rev
            obj = self.rc()
            use_rc = True

        offset = sim.argmax() - len(sim) // 2 + 1
        if max_shift is not None and np.abs(offset) > max_shift:
            # shift was too large. Keep the original
            return self.add_attr("align", {"use_rc": False, "offset": 0})
        else:
            # shift the value and note the shift / rc used
            return (obj.shift(offset, pad_value=pad_value).
                    add_attr("align", {"use_rc": use_rc, "offset": offset}))

    def scan_raw(self, seq=None, contrib=None, hyp_contrib=None, profile=None, pad_mode='median',
                 n_jobs=8, verbose=True):
        """Scan the tracks using this pattern

        Args:
          seq
          contrib
          hyp_contrib
          profile
          n_jobs (int): number of cores to use

        Returns:
          a tuple containing scans for: seq, contrib, hyp_contrib, profile
        """
        seq_scan, contrib_scan, hyp_contrib_scan, profile_scan = None, None, None, None

        if seq is not None:
            seq_scan = pssm_scan(self.seq, seq, n_jobs=n_jobs, pad_mode=pad_mode, verbose=verbose)
        if contrib is not None:
            contrib_scan = {task: sliding_similarity(self.contrib[task], contrib[task],
                                                     metric='continousjaccard', pad_mode=pad_mode,
                                                     n_jobs=n_jobs, verbose=verbose)
                            for task in self.tasks()}
        if hyp_contrib is not None:
            hyp_contrib_scan = {task: sliding_similarity(self.hyp_contrib[task], hyp_contrib[task],
                                                         metric='continousjaccard', pad_mode=pad_mode,
                                                         n_jobs=n_jobs, verbose=verbose)
                                for task in self.tasks()}
        if profile is not None:
            profile_scan = {task: sliding_similarity(self.profile[task], profile[task],
                                                     pad_mode=pad_mode,
                                                     metric='dotproduct',  # TODO add a kl metric
                                                     n_jobs=n_jobs, verbose=verbose)
                            for task in self.tasks()}
        return seq_scan, contrib_scan, hyp_contrib_scan, profile_scan

    def scan_profile(self, profile, pad_mode='median', n_jobs=8, verbose=True):
        """Scan the DNA sequence (pwm scan)
        """
        _, _, _, profile_scan = self.scan_raw(None, None, None,
                                              profile, pad_mode, n_jobs, verbose=verbose)
        _, _, _, profile_scan_rc = self.rc().scan_raw(None, None, None,
                                                      profile, pad_mode, n_jobs, verbose=verbose)
        return np.stack([profile_scan, profile_scan_rc], axis=-1)

    def scan_seq(self, seq, pad_mode='median', n_jobs=8, verbose=True):
        """Scan the DNA sequence (pwm scan)
        """
        seq_scan = pssm_scan(self.seq, seq, n_jobs=n_jobs, pad_mode=pad_mode, verbose=verbose)
        seq_scan_rc = pssm_scan(self.rc().seq, seq, n_jobs=n_jobs, pad_mode=pad_mode, verbose=verbose)
        return np.stack([seq_scan, seq_scan_rc], axis=-1)

    def get_instances_seq_scan(self, seq, threshold=3, n_jobs=8, verbose=True):
        match = self.scan_seq(seq, n_jobs=n_jobs, verbose=verbose)

        # combine the example and position axis into one
        matchfw = match.reshape((-1, match.shape[2]))

        # keep track of the positions and the example idx
        positions = np.broadcast_to(np.arange(match.shape[1]).reshape((1, -1)),
                                    match.shape[:2]).reshape((-1,))
        example_idx = np.broadcast_to(np.arange(match.shape[0]).reshape((-1, 1)),
                                      match.shape[:2]).reshape((-1,))

        # choose the right strand by taking the max score across strands
        match_score = matchfw.max(axis=-1)   # score per position
        max_strand = np.argmax(matchfw, axis=-1)  # strand per position

        # which values to keep
        keep = match_score > threshold

        pattern_len = len(self)
        i, j = halve(pattern_len)
        return pd.DataFrame(OrderedDict([
            ("pattern", self.name),
            ("example_idx", example_idx[keep]),
            ("pattern_start", positions[keep] - j),
            # TODO - the central base is not at the same positions if you RC?!
            ("pattern_end", positions[keep] + i),
            ("strand", pd.Series(max_strand[keep]).map({0: "+", 1: "-"})),
            ("pattern_len", pattern_len),
            ("pattern_center", positions[keep]),
            ("seq_match_score", match_score[keep])]
        ))

    def scan_contribution(self, contrib, hyp_contrib, tasks, pad_mode='median', n_jobs=8, verbose=True):
        """Scan the tracks using this pattern

        Args:
          seq
          contrib
          hyp_contrib
          profile
          n_jobs (int): number of cores to use

        Returns:
          a tuple containing match, contribution scans, each with shape: (batch, seqlen, tasks, strand)
        """
        def merge_contrib(a, b):
            """Merge the contribution scores taking using the L1 norm (per sequence)
            """
            return a
            # return (a / np.abs(a).mean() + b / np.abs(b).mean()) / 2

        # fwd
        _, contrib_scan, _, _ = self.scan_raw(None, contrib, None,  # hyp_contrib,
                                              None, pad_mode, n_jobs, verbose=verbose)
        # stack by tasks
        contrib_scan_match = np.stack([contrib_scan[t][0] for t in tasks], axis=-1)
        contrib_scan_contribution = np.stack([contrib_scan[t][1] for t in tasks], axis=-1)

        # rev
        _, contrib_scan_rc, _, _ = self.rc().scan_raw(None, contrib, None,  # hyp_contrib,
                                                      None, pad_mode, n_jobs, verbose=verbose)
        # stack by tasks
        contrib_scan_match_rc = np.stack([contrib_scan_rc[t][0] for t in tasks], axis=-1)

        return (np.stack([contrib_scan_match, contrib_scan_match_rc], axis=-1),
                contrib_scan_contribution)  # contribution doesn't have the strand axis

    def get_task_contribution(self, kind='contrib'):
        tasks = self.tasks()
        task_contrib = np.array([np.abs(self._get_track(kind)[t]).mean() for t in tasks])
        task_contrib = task_contrib / task_contrib.sum()
        return {t: task_contrib[i] for i, t in enumerate(tasks)}

    def get_instances(self, tasks, match, contribution, seq_match=None, norm_df=None,
                      fdr=0.01, skip_percentile=99, verbose=False, plot=False):
        """Convert the match arrays produced by pattern.scan_contribution to motif instances

        At each position:
        1. Determine the task with maximal contribution
        2. Select the strand based on the match at task with maximal contribution
        3. Select instances with significantly high match
          (null distribution = Gaussian estimated on [0, `skip_percentile`] of the points (percentiles)
          - Benjamini-Hochberg correction is used to determine the P-value cuttof with fdr=`fdr`
        4. Throw away all the location not passing the signicance cuttof and format into a pd.DataFrame


        Args:
          tasks: same list as used in scan_contribution
          pattern (bpnet.modisco.core.Pattern)
          tasks: list of tasks
          match, contribution: returned by pattern.scan_contribution
          seq_match: optional. returned by pattern.scan_seq
          norm_df: match scores for the seqlets discovered by modisco.
            Obtained by `bpnet.cli.modisco.cwm_scan_seqlets` if not None, it will be used
            as normalization. all scores with match< min(norm_match) will be discarded
          fdr: fdr threshold to use when thresholding the contribution matches
          skip_percentile: points from the percentile > skip_percentile will be skipped when
            estimating the Gaussian null-distribution
          verbose: if True, qq-plots will be plotted

        Returns:
          pd.DataFrame with columns: pattern, example_idx, pattern_start, ...
        """
        pattern = self

        # combine the example and position axis into one
        matchf = match.reshape((-1, match.shape[2], match.shape[3]))
        contributionf = contribution.reshape((-1, contribution.shape[2]))
        assert matchf.shape[:2] == contributionf.shape[:2]
        positions = np.broadcast_to(np.arange(match.shape[1]).reshape((1, -1)),
                                    match.shape[:2]).reshape((-1,))
        example_idx = np.broadcast_to(np.arange(match.shape[0]).reshape((-1, 1)),
                                      match.shape[:2]).reshape((-1,))

        # aggregate matches across tasks
        idx = np.arange(len(contributionf))
        task_weights_dict = self.get_task_contribution('contrib')
        task_weights = np.array([task_weights_dict[t] for t in tasks])
        matchfw = np.tensordot(matchf, task_weights, axes=(1, 0))
        contributionfw = np.tensordot(contributionf, task_weights, axes=(1, 0))

        # choose the right strand
        match_score = matchfw.max(axis=-1)
        max_strand = np.argmax(matchfw, axis=-1)

        match_max_task = np.argmax(matchf[idx, :, max_strand], axis=-1)
        match_max = matchf[idx, match_max_task, max_strand]

        contrib_max_task = np.argmax(contributionf, axis=-1)
        contrib_max = contributionf[idx, contrib_max_task]

        # TODO - append also the background p-value by fitting a
        # gaussian to it using robust statistics as proposed by
        # Avanti. Formula:
        # mean = median
        # sigma = 1.4826*MAD
        if norm_df is None:
            threshold = fdr_threshold_norm_right(match_score,
                                                 skip_percentile=skip_percentile,
                                                 fdr=fdr)
            keep = match_score > threshold
            keep_idx = np.arange(len(keep))[keep]

            match_p = None
            match_cat = None
            contrib_p = None
            contrib_cat = None

            if verbose:
                print(f"Keeping {keep.sum()}/{len(keep)} ({keep.mean():.2%}) of the instances."
                      f" Match score cutoff={threshold:.3}")
            if plot:
                import matplotlib.pyplot as plt
                import scipy.stats as stats
                # subsample the data to make plotting faster
                idx_list = np.random.randint(0, len(match_score), 100000)
                idx_list2 = np.random.randint(0, len(match_score[~keep]), 100000)
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                axes[0].hist(match_score[idx_list], 100)
                axes[0].set_xlabel("Match score")
                axes[0].set_ylabel("Frequency")
                axes[0].set_title(pattern.name)
                stats.probplot(match_score[idx_list], dist='norm', plot=axes[1])
                axes[1].set_title("Original normal qq-plot")
                stats.probplot(match_score[~keep][idx_list2], dist='norm', plot=axes[2])
                axes[2].set_title("Points not passing the cutoff")
                plt.tight_layout()
        else:
            norm_match = norm_df['match_weighted']
            threshold = norm_match.min()
            keep = match_score > threshold
            keep_idx = np.arange(len(keep))[keep]

            match_p = quantile_norm(match_score[keep], norm_match)
            match_cat = low_medium_high(match_p)
            contrib_p = quantile_norm(contributionfw[keep], norm_df['contrib_weighted'])
            contrib_cat = low_medium_high(contrib_p)

        # optionally include the PWM match
        if seq_match is not None:
            seq_matchf = seq_match.reshape((-1, seq_match.shape[2]))
            assert seq_matchf.shape[0] == matchf.shape[0]
            score_seq_match = seq_matchf[keep_idx, max_strand[keep]]

            if norm_df is not None and 'seq_match' in norm_df:
                score_seq_match_p = quantile_norm(score_seq_match, norm_df['seq_match'])
                score_seq_match_cat = low_medium_high(score_seq_match_p)
            else:
                score_seq_match_p = None
                score_seq_match_cat = None

            sm = [("seq_match", score_seq_match),
                  ("seq_match_p", score_seq_match_p),
                  ("seq_match_cat", score_seq_match_cat),
                  ]

        else:
            score_seq_match = None
            sm = []

        # convert to pd.DataFrame
        i, j = halve(len(pattern))

        # Get the right pattern coordinates
        start = positions[keep] - j
        end = positions[keep] + i
        strand = pd.Series(max_strand[keep]).map({0: "+", 1: "-"})

        # get the correct center position
        #   >>>>
        #    .    -> uncorrected center = (start + end) // 2
        #     |   -> true center
        #
        #   <<<<
        #    .    -> uncorrected center
        #    |    -> true center
        # 1. uncorrected center
        uncorrected_center = (start + end) // 2
        # 2. add the possible 1bp offset
        delta = (start + end) % 2  # center not perfectly divisible by 2
        add_offset = strand.map({"+": 1, "-": 0})
        center = uncorrected_center + add_offset * delta

        df = pd.DataFrame(OrderedDict([
            ("pattern", pattern.name),
            ("example_idx", example_idx[keep]),
            ("pattern_start", start),
            ("pattern_end", end),
            ("strand", strand),
            ("pattern_center", center),
            ("pattern_len", len(pattern)),
            ("match_weighted", match_score[keep]),
            ("match_weighted_p", match_p),
            ("match_weighted_cat", match_cat),
            ("match_max", match_max[keep]),
            ("match_max_task", pd.Series(match_max_task[keep]).
             map({i: t for i, t in enumerate(tasks)})),
            ("contrib_weighted", contributionfw[keep]),
            ("contrib_weighted_p", contrib_p),
            ("contrib_weighted_cat", contrib_cat),
            ("contrib_max", contrib_max[keep]),
            ("contrib_max_task", pd.Series(contrib_max_task[keep]).
             map({i: t for i, t in enumerate(tasks)})),
        ] + sm +
            [("match/" + t, matchf[keep_idx, i, max_strand[keep]])
             for i, t in enumerate(tasks)] +
            [("contrib/" + t, contributionf[keep_idx, i])
             for i, t in enumerate(tasks)]
        ))

        # filter
        if "seq_match" in df:
            if verbose:
                print("Keeping only entries with seq_match > 0")
            df = df[df.seq_match > 0]
        return df

    def add_attr(self, key, value):
        obj = self.copy()
        obj.attrs[key] = value
        return obj

    def add_profile(self, profile):
        assert self.profile is None
        obj = self.copy()
        # Make sure that only the available tasks are added
        obj.profile = {t: profile[t] for t in obj.tasks()}
        return obj

    def _trim_center_shift(self, trim_frac):
        """Compute how much would the center of the pattern shift had
        we ran self.trim_seq_ic(trim_frac)

        You can get the new center using:

        old_center + self._trim_center_shift()
        """
        trim_i, trim_j = self._trim_seq_ic_ij(trim_frac)
        orig_len = len(self)
        trim_len = trim_j - trim_i
        # TODO - make sure this is correct
        return ((halve(trim_len)[0] + trim_i) - halve(orig_len)[0],
                (halve(trim_len)[1] + orig_len - trim_j) - halve(orig_len)[1])

    def rename(self, name):
        obj = self.copy()
        obj.name = name
        return obj


def patterns_to_df(patterns: List[Pattern], properties: List[str]) -> pd.DataFrame:
    """Convert a list of patterns to DataFrame
    """
    return pd.DataFrame([p.to_summary_dict(properties)
                         for p in patterns])


class StackedSeqletContrib:

    VOCAB = ["A", "C", "G", "T"]  # sequence vocabulary
    _track_list = ['seq', 'contrib', 'hyp_contrib', 'profile']

    def __init__(self, seq, contrib, hyp_contrib, profile=None,
                 name=None, dfi=None, attrs={}):
        """Stacked Seqlet container

        Args:
          name (str): name of the pattern
          seq (np.array): numpy array of shape
          dfi (pd.DataFrame): information about each seqlet instance
            stored as a pd.DataFrame
        """
        # tracks
        self.seq = seq  # shall we call it PWM?
        self.contrib = contrib
        self.hyp_contrib = hyp_contrib
        self.profile = profile
        self.dfi = dfi

        # other custom attributes
        self.name = name
        self.attrs = attrs

        self._tasks = list(self.contrib.keys())

        # validate that all the scores are of the same length
        for kind in ['contrib', 'hyp_contrib', 'profile']:
            t = self._get_track(kind)
            if t is None:
                continue
            assert set(t.keys()) == set(self.tasks())
            for task in self.tasks():
                if t[task] is None:
                    continue
                assert isinstance(t[task], np.ndarray)
                if kind != 'profile':
                    assert t[task].shape[0] == self.seq.shape[0]
                    assert t[task].shape[1] == self.seq.shape[1]
                    assert t[task].ndim == 3

        if self.dfi is not None:
            assert len(self.dfi) == len(self)

    def _validate_kind(self, kind):
        if kind not in self._track_list + ['seq_ic']:
            raise ValueError("kind needs to be from " + ",".join(self._track_list))

    def _get_track(self, kind='seq'):
        if "/" in kind:
            kind, task = kind.split("/")
        else:
            kind, task = kind, None

        def get_task(d, k):
            if k is None:
                return d
            elif k == 'mean':
                return mean(list(d.values()))
            elif k == 'sum':
                return sum(list(d.values()))
            # TODO - add weighted based on the contribution scores
            else:
                return d[k]

        self._validate_kind(kind)
        if kind == 'seq':
            return self.seq
        elif kind == 'contrib':
            return get_task(self.contrib, task)
        elif kind == 'hyp_contrib':
            return get_task(self.hyp_contrib, task)
        elif kind == 'profile':
            return get_task(self.profile, task)
        # elif kind == 'seq_ic':
        #     return self.get_seq_ic()
        else:
            raise ValueError("kind needs to be from seq,contrib,hyp_contrib")

    def tasks(self):
        return self._tasks

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) or isinstance(idx, pd.Series):
            # just subset it using the numpy array
            if self.dfi is not None:
                if idx.dtype == np.bool:
                    dfi_subset = self.dfi[idx]
                else:
                    dfi_subset = self.dfi.iloc[idx]
            else:
                dfi_subset = None
            return StackedSeqletContrib(name=self.name,
                                        seq=self.seq[idx],
                                        contrib={t: self.contrib[t][idx]
                                                 for t in self.tasks()},
                                        hyp_contrib={t: self.hyp_contrib[t][idx]
                                                     for t in self.tasks()},
                                        profile={t: self.profile[t][idx]
                                                 for t in self.tasks()},
                                        dfi=dfi_subset,
                                        attrs=self.attrs)
        else:
            dfi_row = self.dfi.iloc[idx] if self.dfi is not None else None
            return Pattern(name=self.name,
                           seq=self.seq[idx],
                           contrib={t: self.contrib[t][idx]
                                    for t in self.tasks()},
                           hyp_contrib={t: self.hyp_contrib[t][idx]
                                        for t in self.tasks()},
                           profile={t: self.profile[t][idx]
                                    for t in self.tasks()},
                           attrs={"dfi_row": dfi_row, **self.attrs})

    def aggregate(self, fn=np.mean, axis=0, idx=None):
        """Aggregate across all tracks

        Args:
          idx: subset index
        """
        def agg_fn(x):
            if idx is not None:
                x = x[idx]
            return fn(x, axis=axis)

        def dapply(d, cfn):
            return OrderedDict([(k, cfn(v)) for k, v in d.items()])

        return Pattern(name=self.name,
                       seq=agg_fn(self.seq),
                       contrib=dapply(self.contrib, agg_fn),
                       hyp_contrib=dapply(self.hyp_contrib, agg_fn),
                       profile=dapply(self.profile, agg_fn),
                       attrs=self.attrs)

    def __len__(self):
        return len(self.seq)

    def plot(self, kind='profile', **kwargs):
        """Plot the stacked seqlets

        Args:
          kind: which track to use for the plot
          kwargs: optional kwargs to pass for the profile plotting function:

        Used plotting functions (from `bpnet.plot.heatmaps`):
          profile: `multiple_heatmap_stranded_profile`
          (hyp_)contrib: `multiple_heatmap_contribution_profile`
          seq: `heatmaps.heatmap_sequence`
        """
        from bpnet.plot.heatmaps import (multiple_heatmap_contribution_profile,
                                         multiple_heatmap_stranded_profile,
                                         heatmap_sequence)
        from bpnet.plot.profiles import multiple_plot_stranded_profile

        if kind == 'profile_agg':
            t = self._get_track("profile")
        else:
            t = self._get_track(kind)
        sort_idx = np.arange(len(self))  # use the same ordering

        if kind == 'profile':
            return multiple_heatmap_stranded_profile(t, sort_idx=sort_idx, **kwargs)
        elif kind == 'profile_agg':
            return multiple_plot_stranded_profile(t, **kwargs)  # figsize_tmpl=(2.55,2)
        elif kind == 'contrib':
            return multiple_heatmap_contribution_profile(t, sort_idx=sort_idx, **kwargs)
        elif kind == 'hyp_contrib':
            return multiple_heatmap_contribution_profile(t, sort_idx=sort_idx, **kwargs)
        elif kind == 'seq':
            return heatmap_sequence(t, sort_idx=sort_idx, **kwargs)

    @classmethod
    def from_seqlet_contribs(cls, seqlet_contribs):
        from kipoi.data_utils import numpy_collate
        s1 = seqlet_contribs[0]
        # tasks = s1.tasks()
        return cls(
            seq=np.stack([s.seq for s in seqlet_contribs]),
            contrib=numpy_collate([s.contrib for s in seqlet_contribs]),
            hyp_contrib=numpy_collate([s.hyp_contrib for s in seqlet_contribs]),
            profile=numpy_collate([s.profile for s in seqlet_contribs]),
            name=s1.name,
            attrs=s1.attrs
        )

    @classmethod
    def concat(self, stacked_seqlet_list):
        out = stacked_seqlet_list[0]
        for s in stacked_seqlet_list[1:]:
            out += s
        return out

    def __add__(self, stacked_seqlets):
        s = stacked_seqlets
        # tasks = self.tasks()

        from kipoi.data_utils import numpy_collate_concat
        return StackedSeqletContrib(
            name=self.name,
            seq=np.concatenate([self.seq, s.seq]),
            contrib=numpy_collate_concat([self.contrib, s.contrib]),
            hyp_contrib=numpy_collate_concat([self.hyp_contrib, s.hyp_contrib]),
            profile=numpy_collate_concat([self.profile, s.profile]),
            dfi=(pd.concat([self.dfi, s.dfi])
                 if self.dfi is not None and s.dfi is not None
                 else None),
            attrs=self.attrs
        )

    def shuffle(self):
        """Permute the order of seqlets
        """
        idx = pd.Series(np.arange(len(self))).sample(frac=1).values
        return self[idx]

    def split(self, i):
        return self[np.arange(i)], self[np.arange(i, len(self))]


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


def shuffle_seqlets(s1, s2):
    """Shuffle the seqlets among two seqlet groups

    Args:
      s1, s2: StackedSeqletContrib

    Returns:
      tuple of 2 StackedSeqletContrib with the same length as before
      but with seqlets randomly chosen from one of the two groups
    """
    return s1.append(s2).shuffle().split(len(s1))
