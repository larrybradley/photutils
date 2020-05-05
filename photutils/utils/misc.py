# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides general-purpose tools that do not have a clear home
module/package.
"""

import numpy as np

__all__ = ['mean_magnitude']


def mean_magnitude(magnitudes):
    """
    Calculate the mean magnitude.

    Parameters
    ----------
    magnitudes : array_like or `~astropy.units.Quantity`
        The input magnitudes whose mean is calculated.

    Returns
    -------
    mean_mag : float
        The mean magnitude.
    """

    magnitudes = np.asarray(magnitudes)
    return -2.5 * np.log10(np.mean(np.power(10., -0.4 * magnitudes)))
