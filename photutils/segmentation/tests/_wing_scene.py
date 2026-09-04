# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Synthetic scenes shared by the spurious-detection tests.
"""

import numpy as np
from astropy.modeling.models import Gaussian2D

THRESHOLD = 5.0
N_PIXELS = 4

# A bright star with a faint blob inside its cleaning zone. The blob
# is well outside the star's isophote (its own segment) but the star's
# model wing at the blob exceeds the blob's n_pixels-th brightest
# pixel above the threshold.
STAR = (1000.0, 30.0, 70.0, 3.5)
NEAR_BLOB = (7.0, 30.0, 52.0, 2.5)

# A very faint blob within the near blob's cleaning zone and within
# the star's cleaning zone.
FAINT_BLOB = (5.3, 30.0, 40.0, 3.0)

# A faint blob far from everything.
FAR_BLOB = (7.0, 30.0, 12.0, 2.5)


def make_scene(sources, shape=(100, 60)):
    """
    Return a noiseless image of Gaussian sources.

    Parameters
    ----------
    sources : list of tuple
        The ``(amplitude, x, y, sigma)`` of each source.

    shape : tuple of int, optional
        The image shape.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    data = np.zeros(shape)
    for amplitude, x, y, sigma in sources:
        data += Gaussian2D(amplitude, x, y, sigma, sigma)(xx, yy)
    return data


def labels_at(segm, sources):
    """
    Return the segment label at the center of each source.

    Parameters
    ----------
    segm : `~photutils.segmentation.SegmentationImage`
        The segmentation image.

    sources : list of tuple
        The ``(amplitude, x, y, sigma)`` of each source.

    Returns
    -------
    labels : list of int
        The labels.
    """
    return [int(segm.data[int(y), int(x)]) for _, x, y, _ in sources]
