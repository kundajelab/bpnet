import numpy as np
from bpnet.plot.tracks import pad_track


def test_pad_track():
    assert pad_track(np.ones((10, 3)), 20).shape == (20, 3)
