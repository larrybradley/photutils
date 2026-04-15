# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for convolving images with a kernel.
"""

import warnings

import numpy as np
from astropy.convolution import Kernel2D
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve as ndi_convolve


def _nanconvolve(data, kernel, mask=None, mode='constant', fill_value=0.0,
                 mask_output=False, max_invalid_fraction=None):
    """
    Convolve a 2D array with a 2D kernel, ignoring NaN and masked
    pixels.

    This function handles NaN values and an optional boolean mask by
    decomposing the kernel into its positive and negative parts and
    independently renormalizing each part over the valid (non-NaN,
    non-masked) pixels. This approach works correctly for any kernel,
    including zero-sum kernels (e.g., DAOStarFinder detection kernels),
    unlike `astropy.convolution.convolve` with
    ``nan_treatment='interpolate'``, which requires the kernel to be
    normalizable.

    When all pixels are valid, the result is identical to
    `scipy.ndimage.convolve`.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The input 2D array. May contain NaN values.

    kernel : 2D `~numpy.ndarray`
        The 2D convolution kernel.

    mask : 2D bool `~numpy.ndarray` or `None`, optional
        A boolean mask with the same shape as ``data``, where `True`
        indicates pixels to ignore during convolution.

    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, \
optional
        The mode parameter determines how the array borders are
        handled. For the ``'constant'`` mode, values outside the array
        borders are set to ``fill_value``. The default is
        ``'constant'``.

    fill_value : float, optional
        Value to fill data values beyond the array borders if ``mode``
        is ``'constant'``. The default is ``0.0``.

    mask_output : bool, optional
        If `True`, originally-invalid pixels (NaN in ``data`` or
        `True` in ``mask``) are set to NaN in the output. This is
        useful when the interpolated values at invalid locations are
        not wanted. The default is `False`.

    max_invalid_fraction : float or `None`, optional
        If not `None`, output pixels that have more than this fraction
        of their kernel footprint overlapping with invalid pixels are
        set to NaN. This marks unreliable boundary pixels around
        large NaN/masked regions. For example,
        ``max_invalid_fraction=0.25`` means that if more than 25% of
        the kernel footprint is invalid, the output pixel is set to
        NaN. A value of 0.0 sets NaN for any pixel whose kernel
        footprint includes *any* invalid pixel. Requires
        ``mask_output=True`` to have an effect (when `True`,
        ``mask_output`` is implicitly set to `True`). The default is
        `None`.

    Returns
    -------
    result : 2D `~numpy.ndarray`
        The convolved array. If ``mask_output`` is `False` (default),
        NaN/masked pixels are interpolated based on valid neighbors
        and the result only contains NaN if all kernel-overlapping
        pixels are invalid. If ``mask_output`` is `True`,
        originally-invalid pixels are set to NaN. If
        ``max_invalid_fraction`` is set, boundary pixels with
        insufficient valid coverage are also NaN.

    Notes
    -----
    Astropy's `~astropy.convolution.convolve` with
    ``nan_treatment='interpolate'`` computes, for each output pixel:

    .. math::

        R = \\frac{\\sum_{i \\in V} K_i \\, d_i}
                  {\\sum_{i \\in V} K_i}

    where :math:`V` is the set of valid (non-NaN) pixels, :math:`K`
    is the kernel, and :math:`d` is the data. This fails for zero-sum
    kernels because :math:`\\sum K_i \\to 0`.

    This function instead decomposes the kernel into positive
    (:math:`K^+`) and negative (:math:`K^-`) parts and renormalizes
    each independently:

    .. math::

        R = S^+ \\frac{\\sum_{i \\in V} K^+_i \\, d_i}
                      {\\sum_{i \\in V} K^+_i}
          + S^- \\frac{\\sum_{i \\in V} K^-_i \\, d_i}
                      {\\sum_{i \\in V} K^-_i}

    where :math:`S^+ = \\sum K^+` and :math:`S^- = \\sum K^-`. Each
    lobe of the kernel is independently renormalized to compensate for
    missing pixels, avoiding the division-by-zero problem.

    The convolution sums are computed efficiently using
    `scipy.ndimage.convolve` on ``data`` (with invalid pixels set to
    0) and on a validity array (1 where valid, 0 where invalid).
    Border pixels are treated as "valid" in the validity convolution
    (``cval=1``) since they contribute known ``fill_value`` data, not
    missing data.
    """
    kernel = np.asarray(kernel, dtype=float)

    if max_invalid_fraction is not None:
        mask_output = True

    # Identify invalid pixels (NaN in data or True in mask)
    invalid = np.isnan(data)
    if mask is not None:
        invalid = invalid | mask

    # Fast path: no invalid pixels, fall back to scipy
    if not np.any(invalid):
        return ndi_convolve(data, kernel, mode=mode, cval=fill_value)

    # Replace invalid pixels with 0 for the convolution
    data_filled = data.copy()
    data_filled[invalid] = 0.0

    # Validity array: 1.0 where valid, 0.0 where invalid
    valid = (~invalid).astype(float)

    # Decompose kernel into positive and negative parts
    kernel_pos = np.maximum(kernel, 0.0)
    kernel_neg = np.minimum(kernel, 0.0)

    sum_pos = kernel_pos.sum()
    sum_neg = kernel_neg.sum()  # always <= 0

    result = np.zeros(data.shape, dtype=float)

    # Renormalize the positive kernel lobe
    if sum_pos > 0:
        num = ndi_convolve(data_filled, kernel_pos, mode=mode,
                           cval=fill_value)
        # Use cval=1 for the validity convolution so that border pixels
        # (which contribute known fill_value, not missing data) are
        # not treated as invalid
        den = ndi_convolve(valid, kernel_pos, mode=mode, cval=1.0)
        good = den > 0
        result[good] += sum_pos * num[good] / den[good]

    # Renormalize the negative kernel lobe
    if sum_neg < 0:
        num = ndi_convolve(data_filled, kernel_neg, mode=mode,
                           cval=fill_value)
        den = ndi_convolve(valid, kernel_neg, mode=mode, cval=1.0)
        # den values are <= 0 (negative kernel weights on positive
        # validity values), so valid regions have den < 0
        good = den < 0
        result[good] += sum_neg * num[good] / den[good]

    if mask_output:
        # Reapply NaN at originally-invalid pixel locations
        result[invalid] = np.nan

        if max_invalid_fraction is not None:
            # Compute the fraction of invalid kernel-weighted pixels
            # for each output pixel. We use an all-ones kernel of the
            # same shape to count the total number of pixels in the
            # kernel footprint, and compare to the validity convolution.
            ones_kernel = np.ones_like(kernel)
            total_weight = ndi_convolve(np.ones_like(valid), ones_kernel,
                                        mode=mode, cval=1.0)
            valid_count = ndi_convolve(valid, ones_kernel,
                                       mode=mode, cval=1.0)
            invalid_fraction = 1.0 - valid_count / total_weight
            unreliable = invalid_fraction > max_invalid_fraction
            result[unreliable] = np.nan

    return result


def _filter_data(data, kernel, *, mode='constant', fill_value=0.0,
                 check_normalization=False, mask=None):
    """
    Convolve a 2D image with a 2D kernel.

    The kernel may either be a 2D `~numpy.ndarray` or a
    `~astropy.convolution.Kernel2D` object.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    kernel : array_like (2D) or `~astropy.convolution.Kernel2D`
        The 2D kernel used to filter the input ``data``. Filtering the
        ``data`` will smooth the noise and maximize detectability of
        objects with a shape similar to the kernel.

    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` determines how the array borders are handled. For
        the ``'constant'`` mode, values outside the array borders are
        set to ``fill_value``. The default is ``'constant'``.

    fill_value : scalar, optional
        Value to fill data values beyond the array borders if ``mode``
        is ``'constant'``. The default is ``0.0``. When ``data`` is a
        `~astropy.units.Quantity`, the result has the same unit; the
        numerical value of ``fill_value`` is used as-is (it is not
        converted to the data unit).

    check_normalization : bool, optional
        If `True` then a warning will be issued if the kernel is not
        normalized to 1.

    mask : 2D bool `~numpy.ndarray` or `None`, optional
        A boolean mask with the same shape as ``data``, where `True`
        indicates pixels to ignore during convolution. If `None` (the
        default), no masking is applied and NaN values in the data
        will propagate through the convolution. If provided, both
        masked and NaN pixels are ignored and their values are
        interpolated using `_nanconvolve`.

    Returns
    -------
    result : `~numpy.ndarray` or `~astropy.units.Quantity`
        The convolved image. A `~astropy.units.Quantity` is returned if
        ``data`` has units; otherwise a `~numpy.ndarray`.
    """
    if kernel is None:
        return data

    kernel_array = kernel.array if isinstance(kernel, Kernel2D) else kernel

    if check_normalization and not np.allclose(np.sum(kernel_array), 1.0):
        msg = 'The kernel is not normalized.'
        warnings.warn(msg, AstropyUserWarning)

    # scipy.ndimage.convolve currently strips units, but be explicit in
    # case that behavior changes
    unit = None
    if isinstance(data, Quantity):
        unit = data.unit
        data = data.value

    # NOTE: if data is int and kernel is float, ndimage.convolve will
    # return an int image. If the data dtype is int, we make the data
    # float so that a float image is always returned
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)

    if mask is not None:
        result = _nanconvolve(data, kernel_array, mask=mask, mode=mode,
                              fill_value=fill_value)
    else:
        result = ndi_convolve(data, kernel_array, mode=mode, cval=fill_value)

    # Reapply the input unit
    if unit is not None:
        result <<= unit

    return result
