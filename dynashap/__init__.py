# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dynamic Shapley Value Computation Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basic usage:
    Refer to docs.
"""

from .__version__ import __title__, __description__, __version__
from .__version__ import __author__, __author_email__, __license__
from .__version__ import __copyright__

from . import utils
from .dynamic import (
    BaseShap, PivotShap, DeltaShap, YnShap, HeurShap,
    mc_shap, exact_shap
)
from .exceptions import (
    UnImpException, FlagError, ParamError, StepWarning
)
