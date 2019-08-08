"""Test ModiscoFile
"""
import pandas as pd
from bpnet.modisco.files import ModiscoFile, ModiscoFileGroup
from bpnet.modisco.core import Pattern, Seqlet


def test_modisco_file(mf, contrib_file):
    # contrib_file required for `mf.get_ranges()`
    assert len(mf.patterns()) > 0

    p = mf.get_pattern("metacluster_0/pattern_0")
    assert isinstance(p, Pattern)

    assert len(mf.patterns()) > 0
    assert isinstance(mf.patterns()[0], Pattern)

    assert len(mf.pattern_names()) > 0
    assert isinstance(mf.pattern_names()[0], str)

    assert mf.tasks() == ['Oct4/profile/wn']

    assert 'patterns' in mf.stats()

    assert isinstance(mf.seqlet_df_instances(), pd.DataFrame)

    assert mf.n_seqlets("metacluster_0/pattern_0") > 0

    assert isinstance(mf.load_ranges(), pd.DataFrame)

    assert isinstance(mf.seqlets()['metacluster_0/pattern_0'][0], Seqlet)


def test_modisco_file_group(mfg):
    p = mfg.get_pattern("Oct4/metacluster_0/pattern_0")
    assert isinstance(p, Pattern)
    assert len(mfg.patterns()) > 0
    assert isinstance(mfg.patterns()[0], Pattern)

    assert len(mfg.pattern_names()) > 0
    assert isinstance(mfg.pattern_names()[0], str)

    assert mfg.tasks() == ['Oct4/profile/wn']  # since we used two times the Oct4 task
    assert mfg.n_seqlets("Oct4/metacluster_0/pattern_0") > 0
