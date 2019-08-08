from __future__ import absolute_import

__author__ = 'Ziga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.0.19'

try:
    from comet_ml import Experiment  # needs to be imported before keras/tensorflow
except Exception:
    pass

import pandas as pd  # need to import that first to prevent some install issues
from . import metrics
from . import trainers
from . import utils
from . import losses
from . import activations
from . import cli

from keras.utils.generic_utils import get_custom_objects
custom_objects_modules = [losses, activations]
for mod in custom_objects_modules:
    for f in mod.AVAILABLE:
        get_custom_objects()[f] = mod.get(f)

# remove variables from the scope
del get_custom_objects
