"""Test pattern_instances.py

"""
import pytest
from pytest import fixture
import pandas as pd
import numpy as np
from bpnet.modisco.utils import longer_pattern
import bpnet
from bpnet.modisco.core import Pattern
from concise.preprocessing import encodeDNA

def scan_seq(p, seq):
    scanned = p.scan_seq(seqs_one_hot, n_jobs=1)
    strands = scanned.argmax(-1)
    positions = scanned.max(axis=-1).argmax(-1)
    strands = strands[np.arange(len(positions)), positions]

    return pd.DataFrame({"pattern": p.name,
                         "strand": strands,
                         "center": positions,
                         "seq_idx": np.arange(len(seqs_one_hot))})

# 'TTTACAATTT'  # seq1
# 'TTTACAATT'   # seq2
# '  AACAAA '  # m1
# ' AAACAA  '  # m1
# '   ACAAT '  # m2


seqs = ['TTTACAATTT',
        'TTTACAATT']
seqs_one_hot = encodeDNA(seqs)

motif_seqs_1 = ['TTTGTT',
                'AAACAA',
                'TTGTTT',
                'ACAATT',
                'TATTGT']

motif_seqs_2 = ['AACAAA',
                'AAACAA',
                'TTGTTT',
                'ACAATT',
                'TATTGT']


def create_patterns(motif_seqs):
    patterns = [Pattern(seq=encodeDNA([s])[0],
                        contrib=dict(a=encodeDNA([s])[0]),
                        hyp_contrib=dict(a=encodeDNA([s])[0]), name=str(i)) for i, s in enumerate(motif_seqs)]

    aligned_patterns = [p.align(patterns[0], pad_value=np.array([0.25] * 4)) for p in patterns]
    return patterns, aligned_patterns


@pytest.mark.parametrize("motif_seqs", [motif_seqs_1, motif_seqs_2])
def test_pattern_shift(motif_seqs):
    patterns, aligned_patterns = create_patterns(motif_seqs)
    dfi = pd.concat([scan_seq(p, seqs_one_hot) for p in patterns])
    dfi_aligned = pd.concat([scan_seq(p, seqs_one_hot) for p in aligned_patterns])
    np.all(dfi_aligned.center == 5)

    shift = {p.name: (p.attrs['align']['use_rc'] * 2 - 1) * p.attrs['align']['offset'] for p in aligned_patterns}
    # This works
    strand_shift = {p.name: p.attrs['align']['use_rc'] for p in aligned_patterns}

    assert np.all(dfi_aligned.center == dfi.center - (dfi.strand * 2 - 1) * dfi.pattern.map(shift))
    assert np.all(dfi_aligned.strand == np.where(dfi.pattern.map(strand_shift), 1 - dfi.strand, dfi.strand))
