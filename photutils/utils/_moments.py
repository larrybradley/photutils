# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provide tools for calculating image moments.
"""

import numpy as np

__all__ = ['image_moments']


def image_moments(data, center=(0, 0), order=1, weights=None, mask=None):
    """
    Calculate the weighted image moments up to the specified order.

    Raw moments correspond to center=(0, 0). A different center can be
    input to compute central moments.

    Parameters
    ----------
    data : 2D array_like
        The input 2D array.

    center : tuple of two floats or `None`, optional
        The ``(x, y)`` center position.

    order : int, optional
        The maximum order of the moments to calculate.

    weights : 2D array_like or None, optional
        An array of weights with the same shape as data.

    mask : 2D boolean array or None, optional
        A boolean array with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked pixels are ignored when computing the moments.

    Returns
    -------
    moments : 2D `~numpy.ndarray`
        The image moments.
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != data.shape:
            raise ValueError('weights must have the same shape as data.')

    bad_mask = ~np.isfinite(data)
    if weights is not None:
        bad_mask |= ~np.isfinite(weights)
    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError('mask must have the same shape as data.')
        bad_mask |= mask

    data = np.where(bad_mask, 0.0, data)
    if weights is not None:
        weights = np.where(bad_mask, 0.0, weights)
        data *= weights

    indices = np.ogrid[tuple(slice(0, i) for i in data.shape)]
    ypowers = (indices[0] - center[1]) ** np.arange(order + 1)
    xpowers = np.transpose(indices[1] - center[0]) ** np.arange(order + 1)

    return np.dot(np.dot(np.transpose(ypowers), data), xpowers)
