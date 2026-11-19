# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making a source mask using the Moran's I spatial
autocorrelation statistic.
"""

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning

from photutils.background._morans_i import fill_holes, morans_i_rows, opening
from photutils.utils._misc import split_masked_array

__all__ = ['make_morans_i_map', 'make_morans_i_mask']

# The minimum fraction of a full patch that must be usable pixels
_MIN_PATCH_FRACTION = 0.25

# The approximate number of pixels sampled to estimate image statistics
_N_SAMPLES = 1_000_000

# The standard deviation of the statistic for uncorrelated noise is
# about 0.5 / patch_size. Noise that is strongly correlated between
# adjacent pixels increases it by up to about a factor of 2. A larger
# ratio than this means that the background could not be measured.
_MAX_STD_RATIO = 4.0

# The ratio of the standard deviation to the median absolute deviation
# of a Gaussian distribution
_MAD_TO_STD = 1.482602218505602

# The mean squared value of the image is limited to this multiple of
# its robust variance. A few extreme unmasked values (e.g., hot pixels)
# would otherwise make every patch look constant.
_MAX_MEAN_SQ_RATIO = 1.0e6


def _subsample(array):
    """
    Return a regularly spaced subsample of a 2D array as a view.
    """
    step = max(1, math.ceil(math.sqrt(array.size / _N_SAMPLES)))
    return array[::step, ::step]


def _image_mean_sq(values):
    """
    Return the mean squared value of a sample of the image, limited to
    a multiple of its robust variance.

    The robust variance is calculated from the median absolute value
    of the nonzero values. The zero values are left out because more
    than half of the values are zero if most of the image is a constant
    region.
    """
    mean_sq = float(np.mean(values**2))
    nonzero = np.abs(values[values != 0])
    if nonzero.size == 0:
        return mean_sq
    robust_std = _MAD_TO_STD * float(np.median(nonzero))
    return min(mean_sq, _MAX_MEAN_SQ_RATIO * robust_std**2)


def _validate_n_threads(n_threads):
    if (isinstance(n_threads, bool)
            or not isinstance(n_threads, (int, np.integer))
            or n_threads < 1):
        msg = 'n_threads must be a positive integer'
        raise ValueError(msg)
    return int(n_threads)


def make_morans_i_map(data, *, patch_size=21, mask=None, weights=None,
                      n_threads=1):
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
        The 2D image. Nonfinite values (NaN and inf) are excluded. If
        ``data`` is a `~numpy.ma.MaskedArray`, its mask is combined
        with the input ``mask``.

    patch_size : int, optional
        The side length in pixels of the square patch. It must be an
        odd integer larger than 3. Small patches give a noisier
        statistic, while large patches lose spatial resolution. The
        standard deviation of the statistic for uncorrelated noise is
        approximately ``0.5 / patch_size``.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded (e.g., bad pixels).

    weights : 2D array_like, optional
        The weights of the pixels, with the same shape as ``data``
        (e.g., an inverse-variance map). The patch means, the patch
        variance, and the neighbor means are then weighted. Pixels with
        a nonfinite or nonpositive weight are excluded, as are the
        masked pixels if ``weights`` is a `~numpy.ma.MaskedArray`. The
        statistic does not change if the weights are multiplied by a
        positive constant. If `None`, all of the pixels have the same
        weight.

    n_threads : int, optional
        The number of threads to use. The image rows are divided into
        ``n_threads`` bands that are processed concurrently. The
        results agree with the single-threaded results to within
        floating-point rounding. It must be a positive integer. The
        default is 1 (no multithreading).

    Returns
    -------
    result : 2D float32 `~numpy.ndarray`
        The Moran's I map. A pixel is NaN if it is excluded, if none of
        its eight nearest neighbors is usable, if the usable pixels make
        up less than 25% of a full patch, or if the variance of its
        patch is too small to be measured (see Notes).

    See Also
    --------
    make_morans_i_mask

    Notes
    -----
    For a patch with pixel values :math:`x_i` and pixel weights
    :math:`w_i`, the statistic is

    .. math::

        I = \\frac{\\sum_i w_i (x_i - \\bar{x}) (n_i - \\bar{n})}
            {\\sum_i w_i (x_i - \\bar{x})^2}

    where :math:`n_i` is the weighted mean of the usable pixels among
    the eight nearest neighbors of pixel :math:`i`, and :math:`\\bar{x}`
    and :math:`\\bar{n}` are the means of :math:`x_i` and :math:`n_i`
    over the patch, weighted by :math:`w_i`. This is the global Moran's
    I statistic with row-standardized weights. The neighbor means use
    all of the image pixels, including those just outside of the patch.
    Patches are truncated at the image edges.

    The variance of a patch is too small to be measured if it is
    smaller than about ``1e-10`` times the mean squared deviation from
    the median of the image, of either the patch or the whole image.
    Regions of constant data (e.g., blank regions that are filled with
    zeros) therefore have no value of the statistic. Neither do regions
    whose values differ from the median of the image by more than
    about ``1e5`` times their standard deviation.

    The image is processed in a single pass using running sums, so
    the run time is independent of ``patch_size`` and the working
    memory is small compared to the image.

    Noise that is correlated between adjacent pixels (e.g., in
    resampled or drizzled images) has a positive Moran's I even without
    sources. The statistic should then be compared to its typical value
    over the image instead of to zero, as is done by
    `make_morans_i_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.background import make_morans_i_map
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(5.0, 1.0, (101, 101))
    >>> data[45:56, 45:56] += 5.0
    >>> result = make_morans_i_map(data)
    >>> print(f'{result[50, 50]:.2f}')
    0.68
    >>> print(f'{result[10, 10]:.2f}')
    -0.01
    """
    data, mask = split_masked_array(data, mask)
    if isinstance(data, Quantity):
        data = data.value
    data = np.asarray(data)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(float)
    data = np.ascontiguousarray(data)

    if (isinstance(patch_size, bool)
            or not isinstance(patch_size, (int, np.integer))
            or patch_size <= 3 or patch_size % 2 == 0):
        msg = 'patch_size must be an odd integer larger than 3'
        raise ValueError(msg)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)

    if weights is not None:
        weights, weights_mask = split_masked_array(weights, None)

        # The kernel reads the weights with the data type of the data
        weights = np.ascontiguousarray(weights, dtype=data.dtype)
        if weights.shape != data.shape:
            msg = 'weights must have the same shape as data'
            raise ValueError(msg)
        if weights_mask is not None:
            mask = weights_mask if mask is None else mask | weights_mask

    if mask is not None:
        mask = np.ascontiguousarray(mask).view(np.uint8)

    n_threads = _validate_n_threads(n_threads)

    # Subtract an approximate background level to limit the rounding
    # error of the running sums. The statistic does not depend on it.
    # The mean squared value of the image sets the scale of that
    # rounding error.
    sample = _subsample(data)
    keep = np.isfinite(sample)
    if mask is not None:
        keep &= _subsample(mask) == 0
    if weights is not None:
        keep &= _subsample(weights) > 0
    sample = sample[keep].astype(float)
    pedestal = 0.0
    image_mean_sq = 0.0
    if sample.size > 0:
        pedestal = float(np.median(sample))
        image_mean_sq = _image_mean_sq(sample - pedestal)

    min_count = _MIN_PATCH_FRACTION * patch_size**2
    result = np.empty(data.shape, dtype=np.float32)
    ny = data.shape[0]
    n_chunks = max(1, min(n_threads, ny))
    if n_chunks == 1:
        morans_i_rows(data, mask, weights, pedestal, image_mean_sq,
                      patch_size, min_count, result, 0, ny)
        return result

    # Each thread writes a separate band of rows of the result
    edges = np.linspace(0, ny, n_chunks + 1).astype(int)
    with ThreadPoolExecutor(max_workers=n_chunks) as executor:
        list(executor.map(
            lambda rows: morans_i_rows(data, mask, weights, pedestal,
                                       image_mean_sq, patch_size,
                                       min_count, result, *rows),
            pairwise(edges)))
    return result


def _half_sample_mode(values):
    """
    Calculate the half-sample mode of a 1D array.

    The mode is found by repeatedly keeping the shortest interval that
    contains half of the values (Bickel & Fruhwirth 2006).

    Parameters
    ----------
    values : 1D `~numpy.ndarray`
        The values. Must not be empty.

    Returns
    -------
    mode : float
        The half-sample mode.
    """
    values = np.sort(values)
    while values.size > 3:
        n_half = (values.size + 1) // 2
        widths = values[n_half - 1:] - values[:values.size - n_half + 1]
        start = int(np.argmin(widths))
        values = values[start:start + n_half]
    return float(np.median(values))


def _lower_std(values, center):
    """
    Estimate the standard deviation of the values from the values that
    are not above ``center``.

    The estimate is the median deviation of those values from
    ``center``, scaled to the standard deviation of a Gaussian
    distribution. Unlike an RMS deviation, it changes only moderately
    with outliers (e.g., the extreme negative values of the statistic
    next to unmasked hot pixels) unless they make up at least half of
    the values not above ``center``.

    Parameters
    ----------
    values : 1D `~numpy.ndarray`
        The values. At least one value must not be above ``center``.

    center : float
        The center value.

    Returns
    -------
    std : float
        The estimated standard deviation.
    """
    lower = values[values <= center]
    return _MAD_TO_STD * float(np.median(center - lower))


def _calc_threshold(stat, *, n_sigma, percent, patch_size):
    """
    Calculate the Moran's I threshold above which pixels are sources.

    Parameters
    ----------
    stat : 2D `~numpy.ndarray`
        The Moran's I map.

    n_sigma : float
        The number of standard deviations above the background value.

    percent : float or `None`
        The percentage of the pixels with a finite value of ``stat`` to
        flag. If not `None`, it is used instead of ``n_sigma``.

    patch_size : int
        The side length in pixels of the patch used for ``stat``.

    Returns
    -------
    threshold : float or `None`
        The threshold. `None` if the subsample of ``stat`` used to
        estimate the threshold has no finite values.
    """
    sample = _subsample(stat)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return None

    if percent is not None:
        return np.percentile(sample, 100.0 - percent)

    # Sources only increase the statistic, so the background values
    # form the lowest peak of its distribution. The median is a source
    # value if most of the patches contain sources, but the values
    # below the median still peak at the background value unless
    # nearly all of the patches contain sources. The mode of those
    # values is then refined with the median of the values near it,
    # which removes the bias of the mode for pure noise. The scatter
    # of the background values is measured robustly from the values
    # below the background value, because those values also include
    # the extreme negative values next to any unmasked hot pixels. The
    # center is a median of some of the values, so at least one value
    # is not above it.
    center = _half_sample_mode(sample[sample <= np.median(sample)])
    std = _lower_std(sample, center)
    for _ in range(2):
        center = float(np.median(sample[np.abs(sample - center)
                                        <= 3.0 * std]))
        std = _lower_std(sample, center)

    if std > _MAX_STD_RATIO * 0.5 / patch_size:
        msg = ("The background value of the Moran's I statistic could "
               'not be estimated, probably because nearly all of the '
               'patches contain sources or the background changes '
               'strongly across each patch. The source mask may be '
               'incomplete. Use the percent keyword or a smaller '
               'patch_size instead.')
        warnings.warn(msg, AstropyUserWarning)

    return center + n_sigma * std


def make_morans_i_mask(data, *, patch_size=21, n_sigma=3.0, percent=None,
                       mask=None, weights=None, opening_iterations=None,
                       n_threads=1):
    """
    Make a source mask using the Moran's I spatial autocorrelation
    statistic.

    Pixels are flagged as sources where the Moran's I statistic of the
    surrounding patch of pixels (see `make_morans_i_map`) is above a
    threshold. Holes in the flagged regions are then filled and small
    isolated regions are removed.

    The mask can be made before the background is known because the
    statistic does not depend on the background level or on the
    amplitude of the background noise. It is intended to be used as the
    ``mask`` input to `~photutils.background.Background2D`.

    Parameters
    ----------
    data : 2D array_like or `~astropy.units.Quantity`
        The 2D image. Nonfinite values (NaN and inf) are excluded. If
        ``data`` is a `~numpy.ma.MaskedArray`, its mask is combined
        with the input ``mask``.

    patch_size : int, optional
        The side length in pixels of the square patch used to calculate
        the statistic. It must be an odd integer larger than 3. Sources
        are masked out to a distance of approximately half the patch
        size beyond where they are detected.

    n_sigma : float, optional
        The number of standard deviations above the background value of
        the statistic at which pixels are flagged as sources. The
        background value and its standard deviation are estimated from
        the statistic map itself, so the threshold adapts to noise that
        is correlated between pixels. The background value is taken
        from the lowest peak of the distribution of the statistic, so
        it can be estimated even if most of the patches contain
        sources. A warning is raised if the background value cannot be
        estimated, for example because nearly all of the patches
        contain sources or the background changes strongly across each
        patch. It must be positive, even if ``percent`` is input, in
        which case it is otherwise ignored.

    percent : float or `None`, optional
        The percentage of the pixels with a value of the statistic to
        flag as sources, selecting the pixels with the largest values.
        It must be larger than 0 and smaller than 100. If input, it is
        used instead of ``n_sigma``. This percentage of the pixels is
        flagged even if the image contains no sources. The final
        masked fraction differs somewhat because holes in the flagged
        regions are then filled and the binary opening removes small
        regions. For images with more than about one million pixels,
        the threshold is calculated from a regularly spaced subsample
        of the statistic, so the percentage is approximate.

    mask : 2D bool array_like, optional
        A boolean mask with the same shape as ``data``, where `True`
        values indicate pixels that are excluded from the statistic
        (e.g., bad pixels). These pixels are not flagged as sources
        unless they are enclosed by flagged pixels, so this mask
        should be combined with the returned source mask (e.g.,
        ``source_mask | mask``) when it is input to
        `~photutils.background.Background2D`. Bad pixels with
        extreme values (e.g., hot pixels) should be excluded, because
        such a pixel dominates the statistic of every patch that
        contains it, which hides the sources near it. Blank regions
        that are filled with a constant value instead of NaN (e.g.,
        the zeros outside of the footprint of a mosaic) should also be
        excluded. The step at their edges is otherwise flagged, and
        the hole filling then flags everything that those edges
        enclose.

    weights : 2D array_like, optional
        The weights of the pixels used to calculate the statistic,
        with the same shape as ``data`` (e.g., an inverse-variance
        map). Pixels with a nonfinite or nonpositive weight are
        excluded, as are the masked pixels if ``weights`` is a
        `~numpy.ma.MaskedArray`. If `None`, all of the pixels have the
        same weight.

    opening_iterations : int or `None`, optional
        The number of iterations of the binary opening that is applied
        to the mask. The opening removes flagged regions that are
        narrower than about twice this number of pixels and rounds the
        corners of the remaining regions, without enlarging them.
        Regions are not eroded from the image edges or from pixels
        where the statistic is undefined (e.g., masked pixels or
        regions with too few usable pixels). If `None`,
        ``patch_size // 2 - 2`` iterations are used. A detected source
        is flagged in a region at least as wide as the patch, so the
        default keeps the sources while rounding the square shape that
        the patch gives their regions. Larger values remove compact
        sources. Set to 0 to skip this step.

    n_threads : int, optional
        The number of threads used to calculate the statistic (see
        `make_morans_i_map`). It must be a positive integer. The default
        is 1 (no multithreading).

    Returns
    -------
    source_mask : 2D bool `~numpy.ndarray`
        The source mask, where `True` values indicate source pixels.

    See Also
    --------
    make_morans_i_map
    photutils.segmentation.SegmentationImage.make_source_mask

    Examples
    --------
    >>> from photutils.background import Background2D, make_morans_i_mask
    >>> from photutils.datasets import make_100gaussians_image
    >>> data = make_100gaussians_image()
    >>> mask = make_morans_i_mask(data)
    >>> bkg = Background2D(data, (50, 50), mask=mask, filter_size=(3, 3),
    ...                    exclude_percentile=50.0)

    The source mask covers a large fraction of the image, so
    ``exclude_percentile`` is increased to keep the boxes that are
    partly covered by the mask.

    Excluded pixels are not included in the returned source mask unless
    they are enclosed by flagged pixels, so a bad-pixel mask must be
    combined with the source mask when it is input to
    `~photutils.background.Background2D`.

    >>> import numpy as np
    >>> bad_mask = np.zeros(data.shape, dtype=bool)
    >>> bad_mask[:, 300] = True
    >>> source_mask = make_morans_i_mask(data, mask=bad_mask)
    >>> bkg = Background2D(data, (50, 50), mask=source_mask | bad_mask,
    ...                    filter_size=(3, 3), exclude_percentile=50.0)
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
        msg = 'opening_iterations must be a nonnegative integer'
        raise ValueError(msg)

    stat = make_morans_i_map(data, patch_size=patch_size, mask=mask,
                             weights=weights, n_threads=n_threads)
    threshold = _calc_threshold(stat, n_sigma=n_sigma, percent=percent,
                                patch_size=patch_size)
    if threshold is None:
        return np.zeros(stat.shape, dtype=bool)
    if opening_iterations is None:
        opening_iterations = patch_size // 2 - 2

    source_mask = stat > threshold
    excluded = np.isnan(stat).view(np.uint8)
    del stat
    work = source_mask.view(np.uint8)
    fill_holes(work)
    if opening_iterations > 0:
        opening(work, excluded, opening_iterations)
    return source_mask
