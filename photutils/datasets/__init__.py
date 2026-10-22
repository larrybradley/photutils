# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Subpackage containing tools for loading datasets or making simulated
data.

These tools are typically used in examples, tutorials, and tests, but
can also be used for general data analysis or exploration.

.. deprecated:: 3.2
    The simulation tools in this subpackage are deprecated and this
    subpackage will be removed in version 4.0. Use the equivalent
    tools in `photutils.simulation` instead.
"""

from .images import *  # noqa: F401, F403
from .load import *  # noqa: F401, F403
from .model_params import *  # noqa: F401, F403
from .noise import *  # noqa: F401, F403
from .wcs import *  # noqa: F401, F403

# Prevent circular imports
# isort: off
from .examples import *  # noqa: F401, F403
