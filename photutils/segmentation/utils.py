# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for image segmentation.
"""

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from scipy.ndimage import generate_binary_structure
from scipy.signal import oaconvolve

from photutils.utils._deprecation import deprecated_positional_kwargs
from photutils.utils._parameters import as_pair
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['make_2dgaussian_kernel', 'matched_filter']

# Number of image rows that are filtered at a time to limit the memory
# use
_N_STRIP_ROWS = 256

# Fraction of the largest weight sum below which a pixel has too little
# unmasked data in its kernel footprint to be filtered
_MIN_WEIGHT_FRACTION = 1.0e-8


@deprecated_positional_kwargs(since='3.0', until='4.0')
def make_2dgaussian_kernel(fwhm, size, mode='oversample', oversampling=10):
    """
    Make a normalized 2D circular Gaussian kernel.

    The kernel must have odd sizes in both X and Y, be centered in the
    central pixel, and normalized to sum to 1.

    Parameters
    ----------
    fwhm : float
        The full width at half maximum (FWHM) of the 2D circular
        Gaussian kernel, in pixels.

    size : int or (2,) int array_like
        The size of the kernel along each axis. If ``size`` is a scalar
        then a square size of ``size`` will be used. If ``size`` has
        two elements, they must be in ``(ny, nx)`` (i.e., array shape)
        order. ``size`` must have odd values for both axes.

    mode : {'oversample', 'center', 'linear_interp', 'integrate'}, optional
        The mode to use for discretizing the 2D Gaussian model:

        * 'oversample' (default):
          Discretize model by taking the average on an oversampled grid.

        * 'center':
          Discretize model by taking the value at the center of the bin.

        * 'linear_interp':
          Discretize model by performing a bilinear interpolation
          between the values at the corners of the bin.

        * 'integrate':
          Discretize model by integrating the model over the bin.

    oversampling : int, optional
        The oversampling factor used when ``mode='oversample'``.

    Returns
    -------
    kernel : `astropy.convolution.Kernel2D`
        The output smoothing kernel, normalized such that it sums to 1.
    """
    ysize, xsize = as_pair('size', size, lower_bound=(0, 0), check_odd=True)

    kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma,
                              x_size=xsize, y_size=ysize, mode=mode,
                              factor=oversampling)
    kernel.normalize(mode='integral')  # ensure kernel sums to 1

    return kernel


def matched_filter(data, kernel, error, *, mask=None):
    """
    Filter an image with a kernel, weighting each pixel by its inverse
    variance.

    The filtered image is an alternative to the convolved image that
    is input to `~photutils.segmentation.detect_sources`. Each pixel of
    the filtered image is the amplitude of the kernel, centered on that
    pixel, that best fits the data in a weighted least-squares sense.
    The ratio of the filtered image to its uncertainty is the
    signal-to-noise ratio (S/N) of that amplitude. A detection threshold
    applied to this ratio is therefore a significance, whose meaning
    does not depend on the kernel or on the noise level.

    Compared to a convolution, pixels with a larger uncertainty have a
    smaller weight. Pixels that are masked, that are not finite, or
    that are beyond the image borders have zero weight, so no values
    are assumed for them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D image array. The image should be background-subtracted.

    kernel : 2D `~numpy.ndarray` or `~astropy.convolution.Kernel2D`
        The filter kernel, which should have approximately the shape of
        the sources to detect (e.g., the point spread function for point
        sources). It must have odd dimensions, and its values must be
        finite with a positive sum. It is normalized to have a sum of
        1.

    error : float or 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 1-sigma uncertainty of ``data``, either a scalar or an
        array with the same shape as ``data``. It should be the
        background noise, without the Poisson noise of the sources.
        Pixels with an uncertainty that is not finite or not positive
        are excluded. If ``data`` is a `~astropy.units.Quantity` array,
        then ``error`` must have the same units.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is
        masked. Masked pixels are excluded.

    Returns
    -------
    filtered_data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The filtered image, in the units of ``data``. For a uniform
        ``error`` and no excluded pixels, it is equal to the
        correlation of ``data`` with the normalized ``kernel``, except
        near the image borders. It is zero where there are no included
        pixels within the kernel footprint.

    filtered_error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 1-sigma uncertainty of ``filtered_data``. It is infinite
        where there are no included pixels within the kernel footprint.

    See Also
    --------
    photutils.segmentation.detect_sources

    Notes
    -----
    For a kernel :math:`K` and pixel weights :math:`w = 1 /
    \\sigma^2`, the filtered image and its variance are

    .. math::
        f = \\frac{\\sum K^2 \\, \\sum K \\, d \\, w}{\\sum K^2 \\, w}
        \\qquad
        \\sigma_f^2 = \\frac{(\\sum K^2)^2}{\\sum K^2 \\, w}

    where the sums are over the pixels of the kernel array centered on
    each pixel.

    Because the amplitude of the kernel is fitted, the filtered value
    at the center of a source with the shape of the kernel does not
    depend on which of its pixels are excluded. This is not true for
    sources with other shapes, including a constant background.

    The uncertainty is correct only if ``error`` describes the noise
    of the image and the noise of different pixels is not correlated.
    For images with correlated noise (e.g., resampled images), the S/N
    is overestimated.

    The S/N of neighboring pixels is correlated by the filter. The
    number of connected pixels required by
    `~photutils.segmentation.detect_sources` should be chosen with
    that in mind.

    Examples
    --------
    Detect the sources that have at least 5 connected pixels with a
    S/N greater than 3:

    >>> import numpy as np
    >>> from photutils.segmentation import (detect_sources,
    ...                                     make_2dgaussian_kernel,
    ...                                     matched_filter)
    >>> rng = np.random.default_rng(0)
    >>> yy, xx = np.mgrid[0:51, 0:51]
    >>> error = np.where(xx > 30, 4.0, 1.0)
    >>> data = rng.normal(0.0, 1.0, (51, 51)) * error
    >>> data += 10.0 * np.exp(-((xx - 20)**2 + (yy - 25)**2) / 4.0)
    >>> kernel = make_2dgaussian_kernel(3.0, size=5)
    >>> filtered_data, filtered_error = matched_filter(data, kernel, error)
    >>> snr = filtered_data / filtered_error
    >>> segment_map = detect_sources(snr, 3.0, n_pixels=5)
    >>> segment_map.n_labels
    1
    """
    (data, error), unit = process_quantities((data, error),
                                             ('data', 'error'))
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        msg = 'data must be a 2D array'
        raise ValueError(msg)

    kernel = np.array(getattr(kernel, 'array', kernel), dtype=float)
    if kernel.ndim != 2:
        msg = 'kernel must be a 2D array'
        raise ValueError(msg)
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        msg = 'kernel must have odd dimensions'
        raise ValueError(msg)
    if not np.all(np.isfinite(kernel)) or np.sum(kernel) <= 0:
        msg = 'kernel must have finite values and a positive sum'
        raise ValueError(msg)
    kernel /= np.sum(kernel)

    bad = ~np.isfinite(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            msg = 'mask must have the same shape as data'
            raise ValueError(msg)
        bad |= mask

    error = np.asarray(error, dtype=float)
    if error.ndim != 0 and error.shape != data.shape:
        msg = 'error must be a scalar or have the same shape as data'
        raise ValueError(msg)
    good_error = np.isfinite(error) & (error > 0)
    if not np.any(good_error):
        msg = 'error must have finite and positive values'
        raise ValueError(msg)
    weights = np.broadcast_to(1.0 / np.where(good_error, error, 1.0)**2,
                              data.shape).copy()
    bad |= ~good_error
    weights[bad] = 0.0
    data = np.where(bad, 0.0, data)

    # Convolution with the flipped kernel is correlation
    flipped = kernel[::-1, ::-1]
    flipped2 = flipped**2
    norm = np.sum(flipped2)
    min_kk = _MIN_WEIGHT_FRACTION * np.max(weights) * norm

    # The image is filtered in strips of rows to limit the memory used
    # by the fast Fourier transforms
    half = kernel.shape[0] // 2
    n_rows = data.shape[0]
    filtered_data = np.empty(data.shape)
    filtered_error = np.empty(data.shape)
    for start in range(0, n_rows, _N_STRIP_ROWS):
        stop = min(start + _N_STRIP_ROWS, n_rows)
        # Rows beyond the strip that are within the kernel of it
        low = max(start - half, 0)
        high = min(stop + half, n_rows)
        rows = slice(start - low, stop - low)
        strip_weights = weights[low:high]
        sum_kd = oaconvolve(data[low:high] * strip_weights, flipped,
                            mode='same')[rows]
        sum_kk = oaconvolve(strip_weights, flipped2, mode='same')[rows]

        valid = sum_kk > min_kk
        sum_kk[~valid] = 1.0
        filtered_data[start:stop] = np.where(valid, norm * sum_kd / sum_kk,
                                             0.0)
        filtered_error[start:stop] = np.where(valid, norm / np.sqrt(sum_kk),
                                              np.inf)

    if unit is not None:
        filtered_data <<= unit
        filtered_error <<= unit

    return filtered_data, filtered_error


def _make_binary_structure(ndim, connectivity):
    """
    Make a binary structure element.

    Parameters
    ----------
    ndim : int
        The number of array dimensions.

    connectivity : {4, 8}
        For the case of ``ndim=2``, the type of pixel connectivity used
        in determining how pixels are grouped into a detected source.
        The options are 4 or 8. 4-connected pixels touch along their
        edges. 8-connected pixels touch along their edges or corners.
        For reference, SourceExtractor uses 8-connected pixels.

    Returns
    -------
    array : `~numpy.ndarray`
        The binary structure element. If ``ndim <= 2`` an array of int
        is returned, otherwise an array of bool is returned.
    """
    if ndim == 1:
        footprint = np.array((1, 1, 1))
    elif ndim == 2:
        if connectivity == 4:
            footprint = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
        elif connectivity == 8:
            footprint = np.ones((3, 3), dtype=int)
        else:
            msg = f'Invalid connectivity={connectivity}. Options are 4 or 8'
            raise ValueError(msg)
    else:
        footprint = generate_binary_structure(ndim, 1)

    return footprint


def _mask_to_mirrored_value(data, replace_mask, xycenter, *, mask=None,
                            return_uncorrected=False):
    """
    Replace masked pixels with the value of the pixel mirrored across a
    given center position.

    If the mirror pixel is unavailable (i.e., it is outside the image or
    masked), then the masked pixel value is set to zero.

    Note that this function is not used in production. The
    `~photutils.segmentation.SourceCatalog` batch kernels apply the
    mirroring correction in compiled code. It is kept as the pure-Python
    mirror of that correction and must track the compiled semantics
    exactly, which is enforced by the batch cross-implementation tests.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        A 2D array.

    replace_mask : 2D bool `~numpy.ndarray`
        A boolean mask where `True` values indicate the pixels that
        should be replaced, if possible, by mirrored pixel values. It
        must have the same shape as ``data``.

    xycenter : tuple of two float
        The (x, y) center coordinates around which masked pixels will be
        mirrored.

    mask : 2D bool `~numpy.ndarray`, optional
        A boolean mask where `True` values indicate ``replace_mask``
        *mirrored* pixels that should never be used to fix
        ``replace_mask`` pixels. In other words, if a pixel in
        ``replace_mask`` has a mirror pixel in this ``mask``, then the
        mirrored value is set to zero. Using this keyword prevents
        potential spreading of known nonfinite or bad pixel values.

    return_uncorrected : bool, optional
        If `True`, also return a boolean mask of the ``replace_mask``
        pixels that could not be replaced by a mirrored value (i.e., the
        mirror pixel was outside the image, in ``mask``, or itself in
        ``replace_mask``) and were set to zero instead.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        A 2D array with replaced masked pixels. If
        ``return_uncorrected`` is `True`, then a tuple of ``(result,
        uncorrected)`` is returned instead, where ``uncorrected`` is a
        2D bool `~numpy.ndarray` that is `True` for the ``replace_mask``
        pixels that were set to zero because their mirrored value was
        unavailable.
    """
    outdata = np.copy(data)
    uncorrected = (np.zeros(data.shape, dtype=bool) if return_uncorrected
                   else None)

    ymasked, xmasked = np.nonzero(replace_mask)
    xmirror = 2 * int(xycenter[0] + 0.5) - xmasked
    ymirror = 2 * int(xycenter[1] + 0.5) - ymasked

    # Find mirrored pixels that are outside the image
    badmask = ((xmirror < 0) | (ymirror < 0) | (xmirror >= data.shape[1])
               | (ymirror >= data.shape[0]))

    # Remove them from the set of replace_mask pixels and set them to
    # zero
    if np.any(badmask):
        youtside = ymasked[badmask]
        xoutside = xmasked[badmask]
        outdata[youtside, xoutside] = 0.0
        if return_uncorrected:
            uncorrected[youtside, xoutside] = True
        # Remove the badmask pixels from pixels to be replaced
        goodmask = ~badmask
        ymasked = ymasked[goodmask]
        xmasked = xmasked[goodmask]
        xmirror = xmirror[goodmask]
        ymirror = ymirror[goodmask]

    outdata[ymasked, xmasked] = outdata[ymirror, xmirror]

    # Find mirrored pixels that are masked and replace_mask pixels that are
    # mirrored to other replace_mask pixels. Set them both to zero.
    mirror_mask = replace_mask[ymirror, xmirror]
    if mask is not None:
        mirror_mask |= mask[ymirror, xmirror]
    xbad = xmasked[mirror_mask]
    ybad = ymasked[mirror_mask]
    outdata[ybad, xbad] = 0.0

    if return_uncorrected:
        uncorrected[ybad, xbad] = True
        return outdata, uncorrected

    return outdata
