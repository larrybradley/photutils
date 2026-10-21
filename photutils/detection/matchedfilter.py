# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
MatchedFilterStarFinder class.
"""

import math
import warnings

import numpy as np
from astropy.stats import mad_std, sigma_clipped_stats
from astropy.table import QTable
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.signal import oaconvolve

from photutils.detection.core import StarFinderBase, _validate_n_brightest
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._repr import make_repr
from photutils.utils.cutouts import _make_cutouts
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['MatchedFilterStarFinder']

# Fraction of the weight sum of a full kernel footprint of typical
# weights below which a pixel has too little unmasked data in its kernel
# footprint to be fitted
_MIN_WEIGHT_FRACTION = 1.0e-8

# Number of image rows that are processed at a time to limit the memory
# use
_N_STRIP_ROWS = 256

# Weighted values and weights larger than this factor times their
# typical size are added directly instead of through the fast Fourier
# transform and the running sums of the box filter, whose rounding
# errors scale with the largest value they include
_DIRECT_FACTOR = 1.0e6

# Approximate number of pixels sampled to estimate the typical weight
_N_SAMPLES = 1_000_000


def _typical_weight(data, error, mask):
    """
    Return the median inverse-variance weight of the usable pixels.

    The weights are sampled on a regular grid of about ``_N_SAMPLES``
    pixels, or on all of the pixels if the sample has no usable pixels.
    A median, unlike the largest weight, is not changed by a few pixels
    with tiny uncertainties.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D image array.

    error : float or 2D `~numpy.ndarray`
        The 1-sigma uncertainty of ``data``.

    mask : 2D bool `~numpy.ndarray` or `None`
        The pixel mask.

    Returns
    -------
    weight : float or `None`
        The median weight, or `None` if no pixel is usable.
    """
    step = max(1, math.ceil(math.sqrt(data.size / _N_SAMPLES)))
    for stride in dict.fromkeys((step, 1)):
        sample = (slice(None, None, stride),) * 2
        usable = np.isfinite(data[sample])
        if mask is not None:
            usable &= ~mask[sample]
        if np.ndim(error) == 0:
            if np.any(usable):
                return 1.0 / error**2
            continue
        sample_error = np.asarray(error[sample], dtype=float)
        usable &= np.isfinite(sample_error) & (sample_error > 0)
        if np.any(usable):
            return 1.0 / np.median(sample_error[usable])**2
    return None


def _validate_error_scale(error_scale):
    """
    Validate the ``error_scale`` input.

    Parameters
    ----------
    error_scale : `None`, float, or ``'empirical'``
        The input value.

    Returns
    -------
    error_scale : `None`, float, or ``'empirical'``
        The validated value.
    """
    if error_scale is None or (isinstance(error_scale, str)
                               and error_scale == 'empirical'):
        return error_scale
    if (not isinstance(error_scale, str) and np.ndim(error_scale) == 0
            and np.isfinite(error_scale) and error_scale > 0):
        return float(error_scale)
    msg = "error_scale must be None, a positive number, or 'empirical'"
    raise ValueError(msg)


def _empirical_error_scale(snr, mask, source_mask):
    """
    Return the standard deviation of the S/N image of the noise.

    The S/N image is sampled on a regular grid of about ``_N_SAMPLES``
    pixels, or on all of the pixels if the sample has no usable pixels.
    The standard deviation is the sigma-clipped median absolute
    deviation, scaled to the standard deviation of a Gaussian.

    Parameters
    ----------
    snr : 2D `~numpy.ndarray`
        The S/N image, which is zero where there is too little unmasked
        data in the kernel footprint.

    mask : 2D bool `~numpy.ndarray` or `None`
        The pixel mask.

    source_mask : 2D bool `~numpy.ndarray` or `None`
        The source pixels to leave out.

    Returns
    -------
    scale : float
        The standard deviation.
    """
    step = max(1, math.ceil(math.sqrt(snr.size / _N_SAMPLES)))
    for stride in dict.fromkeys((step, 1)):
        sample = (slice(None, None, stride),) * 2
        values = snr[sample]
        usable = np.isfinite(values) & (values != 0)
        for excluded in (mask, source_mask):
            if excluded is not None:
                usable &= ~excluded[sample]
        if np.any(usable):
            _, _, scale = sigma_clipped_stats(values[usable], sigma=3.0,
                                              stdfunc='mad_std')
            if scale > 0:
                return float(scale)
    msg = ('The empirical error scale could not be estimated because there '
           'are no usable pixels with a nonzero S/N')
    raise ValueError(msg)


def _make_weights(data, error, mask):
    """
    Return the image and its inverse-variance weights as float64 arrays.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The image, a strip of it, or a stack of cutouts.

    error : float or `~numpy.ndarray`
        The 1-sigma uncertainty of ``data``, with the same shape as
        ``data`` if it is an array.

    mask : bool `~numpy.ndarray` or `None`
        The excluded pixels, with the same shape as ``data``.

    Returns
    -------
    data, weights : `~numpy.ndarray`
        Float64 copies of the data and the weights. Both are zero at the
        excluded pixels.
    """
    data = np.array(data, dtype=float)
    bad = ~np.isfinite(data)
    if mask is not None:
        bad |= mask
    if np.ndim(error) == 0:
        weights = np.full(data.shape, 1.0 / error**2)
    else:
        error = np.asarray(error, dtype=float)
        good_error = np.isfinite(error) & (error > 0)
        bad |= ~good_error
        weights = 1.0 / np.where(good_error, error, 1.0)**2

    # Nonfinite values must be replaced because they are not removed by
    # a zero weight
    data[bad] = 0.0
    weights[bad] = 0.0
    return data, weights


def _split_large(image, limit):
    """
    Remove the values of an image that are larger than a limit.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The 2D image array.

    limit : float
        The absolute value above which an image value is removed.

    Returns
    -------
    small : 2D `~numpy.ndarray`
        The image with the large values set to zero.

    yy, xx, values : 1D `~numpy.ndarray`
        The positions and values of the large values.
    """
    large = np.abs(image) > limit
    if not np.any(large):
        empty = np.array([], dtype=int)
        return image, empty, empty, np.array([])
    yy, xx = np.nonzero(large)
    return np.where(large, 0.0, image), yy, xx, image[yy, xx]


def _add_kernel_copies(result, yy, xx, values, flipped):
    """
    Add scaled copies of a kernel to an image, as a convolution of
    isolated values would.

    Parameters
    ----------
    result : 2D `~numpy.ndarray`
        The image, which is changed in place.

    yy, xx, values : 1D `~numpy.ndarray`
        The positions and values of the isolated image values.

    flipped : 2D `~numpy.ndarray`
        The flipped kernel, which must have odd dimensions.
    """
    n_rows, n_cols = result.shape
    yhalf = flipped.shape[0] // 2
    xhalf = flipped.shape[1] // 2
    for ypos, xpos, value in zip(yy, xx, values, strict=True):
        y0 = max(ypos - yhalf, 0)
        y1 = min(ypos + yhalf + 1, n_rows)
        x0 = max(xpos - xhalf, 0)
        x1 = min(xpos + xhalf + 1, n_cols)
        result[y0:y1, x0:x1] += value * flipped[
            y0 - ypos + yhalf:y1 - ypos + yhalf,
            x0 - xpos + xhalf:x1 - xpos + xhalf]


def _correlate(image, kernel, limit):
    """
    Correlate an image with a kernel, with zeros beyond the image.

    The values larger than ``limit`` are left out of the fast Fourier
    transform and added as scaled copies of the kernel instead, so they
    change only the pixels within the kernel footprint around them.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The 2D image array.

    kernel : 2D `~numpy.ndarray`
        The kernel array, which must have odd dimensions.

    limit : float
        The absolute value above which an image value is added
        directly.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The correlated image.
    """
    # Convolution with the flipped kernel is correlation
    flipped = kernel[::-1, ::-1]
    small, yy, xx, values = _split_large(image, limit)
    result = oaconvolve(small, flipped, mode='same')
    _add_kernel_copies(result, yy, xx, values, flipped)
    return result


def _box_sum(image, shape, limit):
    """
    Sum an image in a box centered on each pixel, with zeros beyond the
    image.

    The values larger than ``limit`` are left out of the running sums of
    the box filter and added directly instead, so they change only the
    pixels within the box around them.

    Parameters
    ----------
    image : 2D `~numpy.ndarray`
        The 2D image array.

    shape : tuple of int
        The odd ``(ny, nx)`` shape of the box.

    limit : float
        The absolute value above which an image value is added
        directly.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The box sums.
    """
    small, yy, xx, values = _split_large(image, limit)
    result = uniform_filter(small, shape, mode='constant') * math.prod(shape)
    _add_kernel_copies(result, yy, xx, values, np.ones(shape))
    return result


class MatchedFilterStarFinder(StarFinderBase):
    """
    Detect stars in an image using a matched filter.

    At every pixel, the flux of a point source centered on that pixel
    is fitted to the data by weighted linear least squares, using the
    input ``kernel`` as the point spread function (PSF) and the inverse
    variance of each pixel as its weight. Stars are the local maxima of
    the resulting signal-to-noise ratio (S/N) image that are greater
    than ``threshold``.

    The ``threshold`` is therefore the significance of the fitted flux.
    Its meaning does not depend on the width of the kernel or on the
    noise level of the image.

    Pixels that are masked, that are not finite, or that are beyond
    the image borders have zero weight and are excluded from the fit.
    No values are assumed for the pixels beyond the image borders.

    Parameters
    ----------
    threshold : float
        The minimum signal-to-noise ratio of the fitted flux for a
        source to be detected. It must be positive.

    kernel : 2D `~numpy.ndarray`
        The PSF of the image, with the star at the center of the array.
        The array must have odd dimensions, and its values must be
        finite with a positive sum. It is normalized to have a sum of
        1. The kernel should extend far enough to include most of the
        PSF flux.

    local_bkg : bool, optional
        Whether to fit a constant local background, along with the
        flux, to the pixels within the kernel array at every pixel.
        If `True` (default), the detection does not depend on the
        background level, at the cost of a small loss of sensitivity.
        If `False`, the image must be accurately
        background-subtracted. A background residual that is a small
        fraction of the pixel noise can then produce spurious sources.

    error_scale : `None`, float, or ``'empirical'``, optional
        The factor by which the uncertainty of the fitted flux is
        multiplied, for example to correct for noise that is correlated
        between pixels (see Notes). The S/N image is divided by the same
        factor before the threshold is applied. If `None` (default),
        the uncertainty is not scaled. If ``'empirical'``, the factor
        is the standard deviation of the S/N image, so that the S/N of
        the noise has a standard deviation of 1. It is estimated with
        sigma clipping from the pixels that are not excluded and not in
        the ``source_mask`` input to ``find_stars``, most of which must
        contain only noise.

    min_separation : `None` or float, optional
        The minimum separation (in pixels) for detected objects. If
        `None` (default), the minimum separation is the full width
        at half maximum of the kernel, estimated from its second
        moments. If 0, a local peak is required only within a region
        with the shape of the kernel.

    exclude_border : bool, optional
        Whether to exclude sources found within half the size of the
        kernel from the image borders. The positions and fluxes of
        sources that are truncated by the image border are less
        accurate.

    n_brightest : int, None, optional
        The number of brightest objects to keep after sorting the source
        list by flux. If ``n_brightest`` is set to `None`, all objects
        will be selected.

    max_peak_excess : float or `None`, optional
        The maximum allowed ``peak_excess`` of a source. This keyword
        may be used to reject single-pixel artifacts such as cosmic
        rays and hot pixels. If `None` (default), no sources are
        rejected. See ``find_stars`` for the definition of
        ``peak_excess``.

    See Also
    --------
    DAOStarFinder, IRAFStarFinder, StarFinder

    Notes
    -----
    For a kernel :math:`P` and pixel weights :math:`w = 1 /
    \\sigma^2`, the fitted flux at a pixel and its variance are

    .. math::
        F = \\frac{\\sum P \\, d \\, w}{\\sum P^2 \\, w}
        \\qquad
        \\sigma_F^2 = \\frac{1}{\\sum P^2 \\, w}

    where the sums are over the pixels of the kernel array centered on
    that pixel. With ``local_bkg=True``, a constant is fitted along
    with the flux.

    After the peaks are found, the subpixel position of each source is
    calculated from the peak of the S/N image. The flux and the local
    background of each source are then fitted again with the kernel
    shifted to that position.

    A single extreme value, such as a corrupt pixel or a pixel with a
    tiny uncertainty, changes only the S/N values within the kernel
    footprint around it.

    The S/N is correct only if the input ``error`` describes the noise
    of the image and the noise of different pixels is not correlated.
    For images with correlated noise (e.g., resampled images), the
    S/N is overestimated unless ``error_scale`` corrects it. A single
    factor is appropriate if the correlation is the same over the
    image.
    """

    def __init__(self, threshold, kernel, *, local_bkg=True,
                 error_scale=None, min_separation=None,
                 exclude_border=False, n_brightest=None,
                 max_peak_excess=None):

        if np.ndim(threshold) != 0 or threshold <= 0:
            msg = 'threshold must be a positive scalar value'
            raise ValueError(msg)

        kernel = np.array(kernel, dtype=float)
        if kernel.ndim != 2:
            msg = 'kernel must be a 2D array'
            raise ValueError(msg)
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            msg = 'kernel must have odd dimensions'
            raise ValueError(msg)
        if not np.all(np.isfinite(kernel)) or np.sum(kernel) <= 0:
            msg = 'kernel must have finite values and a positive sum'
            raise ValueError(msg)

        self.threshold = threshold
        self.kernel = kernel / np.sum(kernel)
        self.local_bkg = local_bkg
        self.error_scale = _validate_error_scale(error_scale)
        self.exclude_border = exclude_border
        self.n_brightest = _validate_n_brightest(n_brightest)
        self.max_peak_excess = max_peak_excess

        if min_separation is None:
            min_separation = float(self._kernel_fwhm())
        elif min_separation < 0:
            msg = 'min_separation must be >= 0'
            raise ValueError(msg)
        self.min_separation = min_separation

    def _kernel_fwhm(self):
        """
        Estimate the FWHM of the kernel from its second moments.
        """
        ny, nx = self.kernel.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        kernel = np.clip(self.kernel, 0.0, None)
        radius2 = (xx - nx // 2)**2 + (yy - ny // 2)**2
        sigma = np.sqrt(0.5 * np.sum(kernel * radius2) / np.sum(kernel))
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    def _repr_str_params(self):
        params = ('threshold', 'kernel', 'local_bkg', 'error_scale',
                  'min_separation', 'exclude_border', 'n_brightest',
                  'max_peak_excess')
        overrides = {'kernel': f'<array; shape={self.kernel.shape}>'}
        return params, overrides

    def __repr__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides)

    def __str__(self):
        params, overrides = self._repr_str_params()
        return make_repr(self, params, overrides=overrides, long=True)

    def __call__(self, data, *, error=None, mask=None, source_mask=None):
        """
        Find stars in an astronomical image.

        See ``find_stars`` for the parameters and the returned table.
        """
        return self.find_stars(data, error=error, mask=mask,
                               source_mask=source_mask)

    @staticmethod
    def _validate_error(data, error, mask):
        """
        Validate the uncertainty, estimating it if it is not input.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array.

        error : `None`, float, or 2D `~numpy.ndarray`
            The 1-sigma uncertainty of ``data``.

        mask : 2D bool `~numpy.ndarray` or `None`
            The pixel mask.

        Returns
        -------
        error : float or 2D `~numpy.ndarray`
            The uncertainty. An array is returned without a copy.
        """
        if error is None:
            bad = ~np.isfinite(data)
            if mask is not None:
                bad |= mask
            error = mad_std(np.asarray(data[~bad], dtype=float))
            if not error > 0:
                msg = ('The noise could not be estimated from data. Input '
                       'the error array.')
                raise ValueError(msg)

        if np.ndim(error) == 0:
            if not error > 0:
                msg = 'error must be positive'
                raise ValueError(msg)
            return float(error)

        error = np.asanyarray(error)
        if error.shape != data.shape:
            msg = 'error must have the same shape as data'
            raise ValueError(msg)
        return error

    def _make_snr_image(self, data, error, mask, typical_weight):
        """
        Make the image of the fitted-flux signal-to-noise ratio.

        The image is processed in strips of rows. Only one strip of the
        image and its weights is converted to float64 at a time, so that
        the S/N image is the only array with the size of the image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array.

        error : float or 2D `~numpy.ndarray`
            The 1-sigma uncertainty of ``data``.

        mask : 2D bool `~numpy.ndarray` or `None`
            The pixel mask.

        typical_weight : float
            The median weight of the usable pixels.

        Returns
        -------
        snr : 2D `~numpy.ndarray`
            The S/N image. Pixels with too little unmasked data in their
            kernel footprint are zero.
        """
        half = self.kernel.shape[0] // 2
        n_rows = data.shape[0]
        min_pp = (_MIN_WEIGHT_FRACTION * typical_weight
                  * np.sum(self.kernel**2))
        # A weighted value is the significance of the pixel times the
        # square root of its weight
        limits = (_DIRECT_FACTOR * math.sqrt(typical_weight),
                  _DIRECT_FACTOR * typical_weight)

        snr = np.empty(data.shape)
        for start in range(0, n_rows, _N_STRIP_ROWS):
            stop = min(start + _N_STRIP_ROWS, n_rows)
            # Rows beyond the strip that are within the kernel of it
            low = max(start - half, 0)
            high = min(stop + half, n_rows)
            rows = slice(low, high)
            strip_error = error if np.ndim(error) == 0 else error[rows]
            strip_mask = None if mask is None else mask[rows]
            strip, weights = _make_weights(data[rows], strip_error,
                                           strip_mask)
            strip = self._make_snr_strip(strip, weights, min_pp, limits)
            snr[start:stop] = strip[start - low:stop - low]
        return snr

    def _make_snr_strip(self, data, weights, min_pp, limits):
        """
        Make the S/N image of a strip of image rows.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The strip of the image, with the excluded pixels set to
            zero.

        weights : 2D `~numpy.ndarray`
            The inverse-variance weights of the strip.

        min_pp : float
            The weighted sum of the squared kernel below which a pixel
            has too little unmasked data to be fitted.

        limits : tuple of float
            The absolute values above which a weighted data value and a
            weight are added directly instead of through the fast
            Fourier transform or the box filter.

        Returns
        -------
        snr : 2D `~numpy.ndarray`
            The S/N image of the strip. The rows within half the kernel
            height of a strip edge that is not an image edge are not
            valid.
        """
        kernel = self.kernel
        data_limit, weight_limit = limits
        data_weights = data * weights
        sum_pd = _correlate(data_weights, kernel, data_limit)
        sum_pp = _correlate(weights, kernel**2, weight_limit)
        valid = sum_pp > min_pp

        if not self.local_bkg:
            numerator = sum_pd
            denominator = sum_pp
        else:
            shape = kernel.shape
            sum_w = _box_sum(weights, shape, weight_limit)
            sum_d = _box_sum(data_weights, shape, data_limit)
            sum_p = _correlate(weights, kernel, weight_limit)

            # Determinant of the normal equations for the flux and the
            # local background
            det = sum_pp * sum_w - sum_p**2
            valid &= det > _MIN_WEIGHT_FRACTION * sum_pp * sum_w
            numerator = sum_pd * sum_w - sum_p * sum_d
            denominator = sum_w * det

        denominator[~valid] = 1.0
        snr = numerator / np.sqrt(denominator)
        snr[~valid] = 0.0
        return snr

    @staticmethod
    def _subpixel_offsets(snr, xpeak, ypeak):
        """
        Calculate the offsets of the S/N peaks from the peak pixels.

        The peak along each axis is the vertex of the parabola through
        the logarithm of the three S/N values centered on the peak
        pixel. The logarithm of a Gaussian peak is a parabola.

        Parameters
        ----------
        snr : 2D `~numpy.ndarray`
            The S/N image.

        xpeak, ypeak : 1D `~numpy.ndarray` of int
            The pixel coordinates of the peaks.

        Returns
        -------
        dx, dy : 1D `~numpy.ndarray`
            The offsets, which are in the range -0.5 to 0.5. The offset
            is zero along an axis where the peak is at the image border
            or where a neighboring S/N value is not positive.
        """
        ny, nx = snr.shape
        offsets = []
        for axis, peak, size in ((1, xpeak, nx), (0, ypeak, ny)):
            offset = np.zeros(len(peak))
            inside = (peak > 0) & (peak < size - 1)
            xpk = xpeak[inside]
            ypk = ypeak[inside]
            step = np.array((0, 1)) if axis == 1 else np.array((1, 0))
            values = np.array([snr[ypk + i * step[0], xpk + i * step[1]]
                               for i in (-1, 0, 1)])
            positive = np.all(values > 0, axis=0)
            left, center, right = np.log(np.where(positive, values, 1.0))
            curvature = left - 2.0 * center + right
            good = positive & (curvature < 0)
            curvature[~good] = -1.0
            shift = np.where(good, 0.5 * (left - right) / curvature, 0.0)
            offset[inside] = np.clip(shift, -0.5, 0.5)
            offsets.append(offset)
        return offsets[0], offsets[1]

    def _fit_sources(self, data, error, mask, xpeak, ypeak, dx, dy):
        """
        Fit the flux and the local background of each source with the
        kernel shifted to the source position.

        Parameters
        ----------
        data : 2D `~numpy.ndarray`
            The 2D image array.

        error : float or 2D `~numpy.ndarray`
            The 1-sigma uncertainty of ``data``.

        mask : 2D bool `~numpy.ndarray` or `None`
            The pixel mask.

        xpeak, ypeak : 1D `~numpy.ndarray` of int
            The pixel coordinates of the peaks.

        dx, dy : 1D `~numpy.ndarray`
            The offsets of the source positions from the peak pixels.

        Returns
        -------
        result : dict
            The fitted ``flux``, ``flux_err``, ``local_bkg``,
            ``reduced_chi2``, and ``peak_excess`` of each source.
        """
        shape = self.kernel.shape
        ycen = shape[0] // 2
        xcen = shape[1] // 2
        cut_data, overlap = _make_cutouts(data, xpeak, ypeak, shape)
        cut_bad = ~overlap
        if mask is not None:
            cut_mask, _ = _make_cutouts(mask, xpeak, ypeak, shape)
            cut_bad |= cut_mask.astype(bool)
        cut_error = error
        if np.ndim(error) != 0:
            cut_error, _ = _make_cutouts(error, xpeak, ypeak, shape,
                                         fill_value=np.inf)
        cut_data, cut_weights = _make_weights(cut_data, cut_error, cut_bad)

        # Kernel shifted to the subpixel position of each source
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        coords = np.array((yy - dy[:, np.newaxis, np.newaxis],
                           xx - dx[:, np.newaxis, np.newaxis]))
        model = map_coordinates(self.kernel, coords, order=3,
                                mode='constant', cval=0.0)

        axis = (1, 2)
        sum_pp = np.sum(cut_weights * model**2, axis=axis)
        sum_pd = np.sum(cut_weights * model * cut_data, axis=axis)
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.local_bkg:
                sum_w = np.sum(cut_weights, axis=axis)
                sum_p = np.sum(cut_weights * model, axis=axis)
                sum_d = np.sum(cut_weights * cut_data, axis=axis)
                det = sum_pp * sum_w - sum_p**2
                flux = (sum_pd * sum_w - sum_p * sum_d) / det
                flux_err = np.sqrt(sum_w / det)
                bkg = (sum_d - flux * sum_p) / sum_w
            else:
                flux = sum_pd / sum_pp
                flux_err = 1.0 / np.sqrt(sum_pp)
                bkg = np.zeros(len(flux))

            resid = (cut_data - flux[:, np.newaxis, np.newaxis] * model
                     - bkg[:, np.newaxis, np.newaxis])
            n_params = 3 + int(self.local_bkg)
            n_good = np.count_nonzero(cut_weights, axis=axis)
            dof = np.maximum(n_good - n_params, 1)
            reduced_chi2 = np.sum(cut_weights * resid**2, axis=axis) / dof

        core = (slice(None), slice(ycen - 1, ycen + 2),
                slice(xcen - 1, xcen + 2))
        peak_excess = np.max(resid[core] * np.sqrt(cut_weights[core]),
                             axis=axis)

        return {'flux': flux, 'flux_err': flux_err, 'local_bkg': bkg,
                'reduced_chi2': reduced_chi2, 'peak_excess': peak_excess}

    def find_stars(self, data, *, error=None, mask=None, source_mask=None):
        """
        Find stars in an astronomical image.

        Parameters
        ----------
        data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
            The 2D image array. The image should be
            background-subtracted if ``local_bkg`` is `False`.

        error : `None`, float, or 2D `~numpy.ndarray`, optional
            The 1-sigma uncertainty of ``data``, either a scalar or
            an array with the same shape as ``data``. It should
            be the background noise, without the Poisson noise of the
            sources. Pixels with an uncertainty that is not finite
            or not positive are excluded. If `None` (default), a
            constant uncertainty is estimated from the unmasked ``data``
            values using `~astropy.stats.mad_std`. If ``data`` is a
            `~astropy.units.Quantity` array, then ``error`` must have
            the same units.

        mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates the corresponding element of ``data``
            is masked. Masked pixels are excluded from the fits. A
            source can be detected at a masked pixel from the unmasked
            pixels around it.

        source_mask : 2D bool array, optional
            A boolean mask with the same shape as ``data``, where a
            `True` value indicates a source pixel that is left out of
            the estimate of the empirical ``error_scale``. Because the
            S/N is affected within the kernel footprint of a source, the
            mask should extend beyond the sources by about half the
            kernel size. Source pixels are still used to find stars. It
            can be used only with ``error_scale='empirical'``.

        Returns
        -------
        table : `~astropy.table.QTable` or `None`
            A table of found stars. `None` is returned if no stars are
            found. The table contains the following parameters:

            * ``id``: unique object identification number.
            * ``x_centroid, y_centroid``: object position, calculated
              from the peak of the S/N image.
            * ``flux``: the fitted flux of the object. It is the total
              flux of the object if ``kernel`` includes all of the PSF.
            * ``flux_err``: the uncertainty of ``flux``.
            * ``snr``: the signal-to-noise ratio, ``flux / flux_err``.
              This value is from the fit at the object position. It
              can therefore be slightly smaller than ``threshold``,
              which is applied to the S/N image.
            * ``local_bkg``: the fitted local background. It is zero if
              ``local_bkg`` is `False`.
            * ``reduced_chi2``: the reduced chi-square of the fit over
              the kernel array.
            * ``peak_excess``: the largest positive residual of the fit
              within one pixel of the peak pixel, in units of the
              pixel uncertainty. Single-pixel artifacts have large
              values.
        """
        (data, error), unit = process_quantities((data, error),
                                                 ('data', 'error'))
        # The inputs are not converted to float64 here. Only strips of
        # rows and the cutouts of the sources are converted.
        data = np.asanyarray(data)
        if data.ndim != 2:
            msg = 'data must be a 2D array'
            raise ValueError(msg)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != data.shape:
                msg = 'mask must have the same shape as data'
                raise ValueError(msg)
        if source_mask is not None:
            if self.error_scale != 'empirical':
                msg = ('source_mask can be used only with '
                       "error_scale='empirical'")
                raise ValueError(msg)
            source_mask = np.asarray(source_mask, dtype=bool)
            if source_mask.shape != data.shape:
                msg = 'source_mask must have the same shape as data'
                raise ValueError(msg)

        error = self._validate_error(data, error, mask)
        typical_weight = _typical_weight(data, error, mask)
        if typical_weight is None:
            snr = np.zeros(data.shape)
        else:
            snr = self._make_snr_image(data, error, mask, typical_weight)

        error_scale = self.error_scale
        if error_scale == 'empirical':
            error_scale = (1.0 if typical_weight is None
                           else _empirical_error_scale(snr, mask,
                                                       source_mask))
        if error_scale is not None:
            snr /= error_scale

        xypos = self._find_stars(snr, self.kernel, self.threshold,
                                 min_separation=self.min_separation,
                                 exclude_border=self.exclude_border)
        if xypos is None:
            msg = 'No sources were found.'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        xpeak = xypos[:, 0]
        ypeak = xypos[:, 1]
        dx, dy = self._subpixel_offsets(snr, xpeak, ypeak)
        fit = self._fit_sources(data, error, mask, xpeak, ypeak, dx, dy)
        if error_scale is not None:
            fit['flux_err'] *= error_scale

        table = QTable()
        table.meta.update(_get_meta())
        table['x_centroid'] = xpeak + dx
        table['y_centroid'] = ypeak + dy
        table['flux'] = fit['flux']
        table['flux_err'] = fit['flux_err']
        with np.errstate(divide='ignore', invalid='ignore'):
            table['snr'] = fit['flux'] / fit['flux_err']
        table['local_bkg'] = fit['local_bkg']
        table['reduced_chi2'] = fit['reduced_chi2']
        table['peak_excess'] = fit['peak_excess']

        keep = np.isfinite(table['flux']) & np.isfinite(table['flux_err'])
        if self.max_peak_excess is not None:
            keep &= table['peak_excess'] <= self.max_peak_excess
        table = table[keep]
        if len(table) == 0:
            msg = 'Sources were found, but none pass the filtering criteria'
            warnings.warn(msg, NoDetectionsWarning)
            return None

        if self.n_brightest is not None:
            idx = np.argsort(table['flux'])[::-1][:self.n_brightest]
            table = table[idx]

        if unit is not None:
            for name in ('flux', 'flux_err', 'local_bkg'):
                table[name] <<= unit

        table.add_column(np.arange(1, len(table) + 1), name='id', index=0)
        return table
