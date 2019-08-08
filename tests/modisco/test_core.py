"""Test core objects
"""
from pytest import fixture
from bpnet.modisco.files import ModiscoFile, ModiscoFileGroup
from bpnet.modisco.core import Seqlet


@fixture
def pattern(modisco_dir):
    mf = ModiscoFile(modisco_dir / 'modisco.h5')
    return mf.get_pattern("metacluster_0/pattern_0")


@fixture
def seqlet():
    return Seqlet(seqname='1',
                  start=10,
                  end=20,
                  name='m0_p0',
                  strand='-')


def test_pattern(pattern):
    assert len(pattern.seq.shape) == 2
