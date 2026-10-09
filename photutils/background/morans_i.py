# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making a source mask using the Moran's I spatial
autocorrelation statistic.
"""

import math
from functools import partial

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.units import Quantity
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           uniform_filter)

__all__ = ['make_morans_i_mask', 'morans_i_map']

# The minimum fraction of a full patch that must be usable pixels
_MIN_PATCH_FRACTION = 0.25

# The approximate number of pixels sampled to estimate image statistics
_N_SAMPLES = 1_000_000

# Patches whose variance is below this fraction of their mean squared
# value are treated as constant. This rejects rounding residuals in
# regions of constant data.
_VARIANCE_FLOOR = 1.0e-10


def _subsample(array):
    """
    Return a regularly-spaced subsample of a 2D array as a view.
    """
    step = max(1, math.ceil(math.sqrt(array.size / _N_SAMPLES)))
    return array[::step, ::step]


def _box_size(patch_size, n_box_passes):
    """
    Return the side length of the box whose repeated box filter has the
    same variance as a single box of ``patch_size`` pixels.

    The result is an odd integer of at least 3.
    """
    width = math.sqrt((patch_size**2 - 1) / n_box_passes + 1)
    return max(3, 2 * round((width - 1) / 2) + 1)


def _box_sum(array, size, *, n_passes=1):
    """
    Return the sum of a 2D array in a square box centered on each
    pixel, with zeros outside of the array.

    The box sum is applied ``n_passes`` times.
    """
    for _ in range(n_passes):
        array = uniform_filter(array, size=size, mode='constant') * size**2
    return array


def _neighbor_sum(array):
    """
    Return the sum of the eight nearest neighbors of each pixel, with
    zeros outside of the array.

    The center pixel is left out of the sum instead of being subtracted
    from it, so the sum of non-negative values is exactly zero only if
    all of the neighbors are zero.
    """
    ny, nx = array.shape
    padded = np.pad(array, 1)
    result = np.zeros(array.shape)
    for dy in range(3):
        for dx in range(3):
            if (dy, dx) != (1, 1):
                result += padded[dy:dy + ny, dx:dx + nx]
    return result


def morans_i_map(data, *, patch_size=21, n_box_passes=1, mask=None,
                 weights=None):
    """
    Calculate a map of the Moran's I spatial autocorrelation statistic.

    The value at each pixel is the global `Moran's I
    <https://en.wikipedia.org/wiki/Moran%27s_I>`_ statistic of the
    square patch of pixels centered on that pixel. The statistic
    measures how strongly each pixel value is correlated with the
    values of its eight nearest neighbors. Patches that contain only
    uncorrelated background noise have values close to zero, while
    patches that contain sources have large positive values.

    The statistic does not change if a constant is added to the data or
    if the data are multiplied by a positive constant. It therefore
    does not require the background level or the background noise to be
    known.

    Parameters
    ----------
    data : 2D array_like or `~astropy.units.Quantity`
        The 2D image. Non-finite values (NaN and inf) are excluded.

    patch_size : int, optional
        The side length in pixels of the square patch. It must be an
        odd integer larger than 3. Small patches give a noisier
        statistic, while large patches lose spatial resolution. The
        standard deviation of the statistic for uncorrelated noise is
        approximately ``0.5 / patch_size``.

    n_box_passes : int, optional
        The number of box-filter passes that make the patch kernel. For
        the default of 1, all of the pixels in the square patch
        contribute equally. For larger values, the pixels are weighted
        by a kernel that tapers smoothly toward its edges and
        approaches a Gaussian (see Notes). The kernel has the same
        effective width as the square patch of ``patch_size`` pixels,
        but it extends farther from its center.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded (e.g., bad pixels).

    weights : 2D array_like, optional
        The weights of the pixels, with the same shape as ``data``
        (e.g., an inverse-variance map). The patch means, the patch
        variance, and the neighbor means are then weighted. Pixels with
        a non-finite or non-positive weight are excluded. The statistic
        does not change if the weights are multiplied by a positive
        constant. If `None`, all of the pixels have the same weight.

    Returns
    -------
    result : 2D float32 `~numpy.ndarray`
        The Moran's I map. A pixel is NaN if it is excluded, if fewer
        than 25% of the pixels of a full patch are usable, or if its
        patch has zero variance.

    See Also
    --------
    make_morans_i_mask

    Notes
    -----
    For a patch with pixel values :math:`x_i` and pixel weights
    :math:`w_i`, the statistic is

    .. math::

        I = \\frac{\\sum_i k_i w_i (x_i - \\bar{x}) (n_i - \\bar{n})}
            {\\sum_i k_i w_i (x_i - \\bar{x})^2}

    where :math:`n_i` is the weighted mean of the usable pixels among
    the eight nearest neighbors of pixel :math:`i`, :math:`k_i` is the
    patch kernel, and :math:`\\bar{x}` and :math:`\\bar{n}` are the
    means of :math:`x_i` and :math:`n_i` over the patch, weighted by
    :math:`k_i w_i`. This is the global Moran's I statistic with
    row-standardized weights. The neighbor means use all of the image
    pixels, including those just outside of the patch. Patches are
    truncated at the image edges.

    The patch kernel :math:`k_i` is 1 for ``n_box_passes=1``. For
    larger values it is a smaller square box convolved with itself
    ``n_box_passes`` times, which is the same as box filtering each of
    the sums that many times before they are combined. The side length
    of the box is the odd integer nearest to ``sqrt((patch_size**2 - 1)
    / n_box_passes + 1)`` (but at least 3), which gives the kernel the
    same variance as the square patch. The kernel is ``n_box_passes``
    times the box side length, minus ``n_box_passes - 1``, pixels wide.
    For example, ``patch_size=21`` with ``n_box_passes=3`` uses a box
    of 13 pixels and a kernel that is 37 pixels wide. A compact
    bright source dominates every patch that it falls in, so a patch
    with equal weights gives the statistic a flat square footprint
    around the source. A tapered kernel gives a footprint with softer
    edges whose size increases with the brightness of the source.

    Noise that is correlated between adjacent pixels (e.g., in
    resampled or drizzled images) has a positive Moran's I even without
    sources. The statistic should then be compared to its typical value
    over the image instead of to zero, as is done by
    `make_morans_i_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.background import morans_i_map
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(5.0, 1.0, (101, 101))
    >>> data[45:56, 45:56] += 5.0
    >>> result = morans_i_map(data)
    >>> print(f'{result[50, 50]:.2f}')
    0.68
    >>> print(f'{result[10, 10]:.2f}')
    -0.01
    """
    if isinstance(data, Quantity):
        data = data.value
    data = np.asanyarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(float)

    if (isinstance(patch_size, bool)
            or not isinstance(patch_size, (int, np.integer))
            or patch_size <= 3 or patch_size % 2 == 0):
        msg = 'patch_size must be an odd integer larger than 3'
        raise ValueError(msg)

    if (isinstance(n_box_passes, bool)
            or not isinstance(n_box_passes, (int, np.integer))
            or n_box_passes < 1):
        msg = 'n_box_passes must be a positive integer'
        raise ValueError(msg)

    valid = np.isfinite(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)
        valid &= ~mask

    if weights is None:
        wts = valid.astype(float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != data.shape:
            msg = 'weights must have the same shape as data'
            raise ValueError(msg)
        with np.errstate(invalid='ignore'):
            valid &= np.isfinite(weights) & (weights > 0)
        wts = np.where(valid, weights, 0.0)

    # Subtract an approximate background level to limit the rounding
    # error of the box sums. The statistic does not depend on it.
    sample = _subsample(data)[_subsample(valid)]
    pedestal = float(np.median(sample)) if sample.size > 0 else 0.0
    values = np.zeros(data.shape)
    values[valid] = data[valid] - pedestal

    # The weighted mean of the valid pixels among the eight nearest
    # neighbors
    neighbor_wts = _neighbor_sum(wts)
    usable = valid & (neighbor_wts > 0)
    neighbors = np.zeros(data.shape)
    neighbors[usable] = (_neighbor_sum(wts * values)[usable]
                         / neighbor_wts[usable])
    wts[~usable] = 0.0

    box_size = _box_size(patch_size, n_box_passes)
    patch_sum = partial(_box_sum, size=box_size, n_passes=n_box_passes)

    count = np.rint(patch_sum(usable.astype(float)))
    sum_wts = patch_sum(wts)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = patch_sum(wts * values) / sum_wts
        mean_sq = patch_sum(wts * values**2) / sum_wts
        mean_neighbors = patch_sum(wts * neighbors) / sum_wts
        mean_product = patch_sum(wts * values * neighbors) / sum_wts
        var = mean_sq - mean**2
        result = (mean_product - mean * mean_neighbors) / var

    # The sum of the patch kernel is the number of pixels in a full
    # patch
    min_count = _MIN_PATCH_FRACTION * box_size**(2 * n_box_passes)
    good = usable & (count >= min_count) & (var > _VARIANCE_FLOOR * mean_sq)
    result[~good] = np.nan
    return result.astype(np.float32)


def _calc_threshold(stat, n_sigma, percent):
    """
    Calculate the Moran's I threshold above which pixels are sources.

    Parameters
    ----------
    stat : 2D `~numpy.ndarray`
        The Moran's I map.

    n_sigma : float
        The number of standard deviations above the background value.

    percent : float or `None`
        The percentage of pixels to flag. If not `None`, it is used
        instead of ``n_sigma``.

    Returns
    -------
    threshold : float or `None`
        The threshold. `None` if ``stat`` has no finite values.
    """
    sample = _subsample(stat)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None

    if percent is not None:
        return np.percentile(sample, 100.0 - percent)

    # Sources only increase the statistic, so the scatter of the
    # background values is measured from the values below the median
    _, median, _ = sigma_clipped_stats(sample, sigma=3.0)
    lower = sample[sample < median]
    std = np.sqrt(np.mean((lower - median)**2)) if lower.size > 0 else 0.0
    return median + n_sigma * std


def make_morans_i_mask(data, *, patch_size=21, n_box_passes=1, n_sigma=3.0,
                       percent=None, mask=None, weights=None,
                       opening_iterations=None):
    """
    Make a source mask using the Moran's I spatial autocorrelation
    statistic.

    Pixels are flagged as sources where the Moran's I statistic of the
    surrounding patch of pixels (see `morans_i_map`) is above a
    threshold. Holes in the flagged regions are then filled and small
    isolated regions are removed.

    The mask can be made before the background is known because the
    statistic does not depend on the background level or on the
    amplitude of the background noise. It is intended to be used as the
    ``mask`` input to `~photutils.background.Background2D`.

    Parameters
    ----------
    data : 2D array_like or `~astropy.units.Quantity`
        The 2D image. Non-finite values (NaN and inf) are excluded.

    patch_size : int, optional
        The side length in pixels of the square patch used to calculate
        the statistic. It must be an odd integer larger than 3. Sources
        are masked out to a distance of approximately half the patch
        size beyond where they are detected.

    n_box_passes : int, optional
        The number of box-filter passes that make the patch kernel (see
        `morans_i_map`). Values larger than 1 taper the patch toward
        its edges while keeping its effective width, which softens the
        square shape of the masked regions and masks bright sources
        out to larger distances.

    n_sigma : float, optional
        The number of standard deviations above the background value of
        the statistic at which pixels are flagged as sources. The
        background value and its standard deviation are estimated from
        the statistic map itself, so the threshold adapts to noise that
        is correlated between pixels. Ignored if ``percent`` is input.

    percent : float or `None`, optional
        The percentage of the pixels to flag as sources, selecting the
        pixels with the largest values of the statistic. It must be
        larger than 0 and smaller than 100. If input, it is used
        instead of ``n_sigma``. This percentage of the image is flagged
        even if it contains no sources.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded from the statistic
        (e.g., bad pixels). These pixels are not flagged as sources
        unless they are enclosed by flagged pixels.

    weights : 2D array_like, optional
        The weights of the pixels used to calculate the statistic,
        with the same shape as ``data`` (e.g., an inverse-variance
        map). Pixels with a non-finite or non-positive weight are
        excluded. If `None`, all of the pixels have the same weight.

    opening_iterations : int or `None`, optional
        The number of iterations of the binary opening that is applied
        to the mask. The opening removes flagged regions that are
        narrower than about twice this number of pixels and rounds the
        corners of the remaining regions, without enlarging them.
        Regions are not eroded from the image edges. If `None`,
        ``patch_size // 2 - 2`` iterations are used for
        ``n_box_passes=1`` and 2 iterations are used otherwise. For
        ``n_box_passes=1``, a detected source is flagged in a region at
        least as wide as the patch, so the default keeps the sources
        while rounding the square shape that the patch gives their
        regions. Larger values remove compact sources. A tapered patch
        already gives rounder regions, and it flags faint compact
        sources in regions that are too small to survive a large
        opening. Set to 0 to skip this step.

    Returns
    -------
    source_mask : 2D bool `~numpy.ndarray`
        The source mask, where `True` values indicate source pixels.

    See Also
    --------
    morans_i_map, photutils.segmentation.SegmentationImage.make_source_mask

    Examples
    --------
    >>> from photutils.background import Background2D, make_morans_i_mask
    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()
    >>> mask = make_morans_i_mask(data)
    >>> bkg = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3))
    """
    if not n_sigma > 0:
        msg = 'n_sigma must be a positive number'
        raise ValueError(msg)
    if percent is not None and not 0 < percent < 100:
        msg = 'percent must be between 0 and 100 (exclusive)'
        raise ValueError(msg)
    if (opening_iterations is not None
            and (isinstance(opening_iterations, bool)
                 or not isinstance(opening_iterations, (int, np.integer))
                 or opening_iterations < 0)):
        msg = 'opening_iterations must be a non-negative integer'
        raise ValueError(msg)

    stat = morans_i_map(data, patch_size=patch_size,
                        n_box_passes=n_box_passes, mask=mask,
                        weights=weights)
    threshold = _calc_threshold(stat, n_sigma, percent)
    if threshold is None:
        return np.zeros(stat.shape, dtype=bool)
    if opening_iterations is None:
        opening_iterations = patch_size // 2 - 2 if n_box_passes == 1 else 2

    source_mask = binary_fill_holes(stat > threshold)
    if opening_iterations > 0:
        # Pixels outside of the image are taken as flagged for the
        # erosion, so that the regions of sources at the image edges
        # are not eroded from the edges
        source_mask = binary_erosion(source_mask,
                                     iterations=opening_iterations,
                                     border_value=1)
        source_mask = binary_dilation(source_mask,
                                      iterations=opening_iterations)
    return source_mask
