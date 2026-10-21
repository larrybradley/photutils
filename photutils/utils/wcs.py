# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for computing the on-sky area of image pixels and the sky errors
of pixel positions from a WCS.
"""

import warnings

import astropy.units as u
import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.utils._moments import PSD_RTOL, pixel_to_sky_covariance
from photutils.utils._wcs_helpers import (_is_gwcs, _low_level_wcs,
                                          compute_pixel_to_sky_jacobians)

__all__ = ['compute_pixel_area_map', 'compute_pixel_areas',
           'pixel_to_sky_errors']


def compute_pixel_areas(wcs, x, y):
    """
    Compute the on-sky area of the pixels at the given positions.

    The area of a pixel is the area of the parallelogram spanned by
    its two edge vectors on the sky, which is the absolute determinant
    of the local forward WCS Jacobian ``d(sky_arcsec)/d(pixel)``. The
    Jacobian is evaluated by central finite differences half a pixel
    either side of each position, so the area is exact for a locally
    quadratic distortion in the small-angle limit. The finite pixel
    size adds a relative error of order ``h**2`` for a pixel that
    subtends an angle ``h`` in radians. The size of the error depends
    on the projection. It is below 1e-6 at 0.1 deg per pixel and
    negligible for imaging data. The area is well defined at
    the celestial poles and across the longitude wraparound.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    x, y : float or array_like
        The pixel coordinates. Both must have the same shape.

    Returns
    -------
    areas : float or `~numpy.ndarray`
        The pixel areas in arcsec\\ :sup:`2`, with the same shape as
        ``x`` and ``y``. A float is returned for scalar input. The area
        is NaN for a nonfinite position or a position where the WCS is
        undefined (e.g., outside the valid region of the projection).

    See Also
    --------
    compute_pixel_area_map

    Notes
    -----
    The pixel area of a distorted image typically varies by a few
    percent across the field. Multiplying a surface brightness image
    (e.g., in MJy/sr) by a single nominal pixel area therefore leaves
    a position-dependent error in the resulting fluxes. This function
    gives the area of each pixel from the WCS instead.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_wcs
    >>> from photutils.utils import compute_pixel_areas
    >>> wcs = make_wcs((100, 100))
    >>> areas = compute_pixel_areas(wcs, [10.0, 50.0], [20.0, 50.0])
    >>> print(np.round(areas, 6))  # arcsec**2
    [0.01 0.01]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        msg = 'x and y must have the same shape'
        raise ValueError(msg)

    jacobians = compute_pixel_to_sky_jacobians(wcs, x, y)

    # The explicit 2x2 determinant is much faster than np.linalg.det on
    # a stack of small matrices and propagates NaN without raising a
    # warning.
    areas = np.abs(jacobians[:, 0, 0] * jacobians[:, 1, 1]
                   - jacobians[:, 0, 1] * jacobians[:, 1, 0])

    # An infinite position gives coincident offset points and thus a
    # zero area. Report NaN for any nonfinite position instead.
    finite = (np.isfinite(x) & np.isfinite(y)).ravel()
    areas = np.where(finite, areas, np.nan).reshape(x.shape)
    if areas.ndim == 0:
        return float(areas)
    return areas


def _is_positive_int(value):
    """
    Return whether ``value`` is a positive integer (and not a bool).
    """
    return (isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_)) and value > 0)


def _compute_areas_at_every_pixel(wcs, ny, nx):
    """
    Evaluate `compute_pixel_areas` directly at every pixel of an image
    with the given ``(ny, nx)`` shape.

    The image is evaluated in blocks of rows so that the temporary
    arrays of the Jacobian computation, which take about 500 bytes per
    pixel at their peak, stay bounded for large images. The result is
    identical to a single evaluation of the whole image.
    """
    max_block_pixels = 2**18
    n_rows = max(1, max_block_pixels // nx)
    areas = np.empty((ny, nx))
    x = np.arange(nx, dtype=float)
    for y0 in range(0, ny, n_rows):
        y1 = min(y0 + n_rows, ny)
        xx, yy = np.meshgrid(x, np.arange(y0, y1, dtype=float))
        areas[y0:y1] = compute_pixel_areas(wcs, xx, yy)
    return areas


def compute_pixel_area_map(wcs, shape, *, interpolate=True, step=None):
    """
    Compute the on-sky area of every pixel in an image.

    By default, the areas are computed with `compute_pixel_areas` on
    a coarse grid of positions and then interpolated onto the full
    image grid with a bicubic spline. Distortions vary smoothly on
    scales far larger than the grid spacing, so the interpolation error
    is negligible compared with the variation in area that the map
    captures. Set ``interpolate=False`` to instead evaluate the area
    directly at every pixel.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    shape : 2-tuple of int
        The ``(ny, nx)`` shape of the image.

    interpolate : bool, optional
        Whether to interpolate the areas from a coarse grid. If
        `False`, the area is evaluated directly at every pixel with
        `compute_pixel_areas`. That is exact but slower (a few seconds
        for a 4096 x 4096 image). For any image more than a few dozen
        pixels on a side, interpolation is faster than direct evaluation
        at every step of 2 or more (about 3 times at a step of 2 and
        more than 10 times at a step of 4), so a small step is not a
        reason to disable it. Set ``interpolate=False`` only when the
        exact per-pixel areas are wanted.

    step : int or None, optional
        The spacing in pixels of the coarse grid on which the areas are
        evaluated. The grid extends two steps before the first pixel
        and at least one step beyond the last pixel, so the
        interpolation never extrapolates and the spline is well defined
        for any image shape. Steps from about 32 to 128 give equivalent
        results. Smaller steps only add runtime and larger steps lose
        accuracy on strongly distorted or very wide fields. The largest
        allowed step is ``min(shape) // 8`` (and at least 1), so that
        the coarse grid always has at least eight intervals across the
        image. If `None`, the step is 64 or the largest allowed step,
        whichever is smaller. A larger step than allowed is reduced to
        the largest allowed step with a warning. A step of 1 samples
        every pixel, so the area is then evaluated directly at every
        pixel without interpolation, as if ``interpolate`` were `False`.
        ``step`` must be `None` if ``interpolate`` is `False`.

    Returns
    -------
    areas : `~numpy.ndarray`
        The 2D array of pixel areas in arcsec\\ :sup:`2`, with the same
        shape as the image. The area is NaN for pixels where the WCS is
        undefined (e.g., outside the valid region of the projection).

    See Also
    --------
    compute_pixel_areas

    Notes
    -----
    If any node of the padded coarse grid lies outside the valid region
    of the projection, which can happen for an all-sky map even when
    every image pixel is valid, the areas are instead evaluated directly
    at every pixel with `compute_pixel_areas`, as if ``interpolate``
    were `False`.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.datasets import make_wcs
    >>> from photutils.utils import compute_pixel_area_map
    >>> shape = (100, 100)
    >>> wcs = make_wcs(shape)
    >>> areas = compute_pixel_area_map(wcs, shape)
    >>> areas.shape
    (100, 100)
    >>> print(np.round(areas[50, 50], 6))  # arcsec**2
    0.01
    """
    msg = 'shape must be two positive integers'
    try:
        ny, nx = shape
    except (TypeError, ValueError):
        raise ValueError(msg) from None
    if not (_is_positive_int(ny) and _is_positive_int(nx)):
        raise ValueError(msg)

    if not isinstance(interpolate, (bool, np.bool_)):
        msg = f'interpolate must be a boolean, got {interpolate!r}'
        raise TypeError(msg)
    if not interpolate:
        if step is not None:
            msg = 'step must be None when interpolate=False'
            raise ValueError(msg)
        return _compute_areas_at_every_pixel(wcs, ny, nx)

    # Ensure the image spans at least eight grid intervals so that large
    # steps on smaller images do not leave the spline with too few
    # knots.
    max_step = max(1, min(ny, nx) // 8)
    if step is None:
        step = min(64, max_step)
    elif not _is_positive_int(step):
        msg = 'step must be a positive integer or None'
        raise ValueError(msg)
    elif step > max_step:
        if max_step == 1:
            msg = (f'step={step} was reduced to 1, so the area is '
                   'evaluated directly at every pixel')
        else:
            msg = (f'step={step} was reduced to {max_step} so that the '
                   'coarse grid has at least eight intervals across the '
                   'image')
        warnings.warn(msg, AstropyUserWarning)
        step = max_step

    # A step of 1 samples every pixel, so evaluate the areas directly
    # instead of interpolating them from themselves. This check comes
    # after the step is validated and capped so that a capped step still
    # warns before taking the direct path.
    if step == 1:
        return _compute_areas_at_every_pixel(wcs, ny, nx)

    # Coarse grid padded by two steps beyond each edge. This gives the
    # spline at least five knots per axis and keeps the interpolation
    # inside the sampled region for every pixel.
    grid_y = np.arange(-2 * step, ny + 2 * step, step, dtype=float)
    grid_x = np.arange(-2 * step, nx + 2 * step, step, dtype=float)
    xx, yy = np.meshgrid(grid_x, grid_y)
    coarse = compute_pixel_areas(wcs, xx, yy)

    # A NaN anywhere on the coarse grid would make the spline NaN at
    # every pixel. That happens when a node lies outside the valid
    # region of the projection, which the padding can do even when every
    # image pixel is valid (e.g., an all-sky map). Evaluate the areas
    # directly at every pixel in that case.
    if not np.all(np.isfinite(coarse)):
        return _compute_areas_at_every_pixel(wcs, ny, nx)

    # scipy.interpolate is slow to import, so it is imported only when needed
    from scipy.interpolate import RectBivariateSpline

    spline = RectBivariateSpline(grid_y, grid_x, coarse)
    return spline(np.arange(ny, dtype=float), np.arange(nx, dtype=float))


def _as_pixel_array(value, name, unit):
    """
    Return an input as a float array in pixel units.

    A `~astropy.units.Quantity` is converted to ``unit``. The masked
    values of a masked array are replaced with NaN.
    """
    if isinstance(value, u.Quantity):
        try:
            value = value.to_value(unit)
        except u.UnitConversionError:
            msg = f'{name} must be in units of {unit}'
            raise ValueError(msg) from None
    if np.ma.isMaskedArray(value):
        return value.astype(float).filled(np.nan)
    return np.asarray(value, dtype=float)


def pixel_to_sky_errors(wcs, x, y, x_err, y_err, *, xy_cov=0.0):
    """
    Convert pixel position errors to sky position errors.

    The pixel error covariance of each position is transported to the
    plane tangent to the sky at that position with the local forward WCS
    Jacobian ``F`` (``sky_covariance = F pixel_covariance F^T``). The
    Jacobian is evaluated by central finite differences half a pixel
    on either side of each position (see `compute_pixel_areas`), so
    the errors follow the distortion across the field and are well
    defined at the celestial poles and across the longitude wraparound.
    The finite differences ignore any gwcs bounding box, so the errors
    are defined for positions within half a pixel of the bounding-box
    edge.

    The returned errors are the square roots of the diagonal elements of
    the sky covariance. The propagation is linear, so it assumes that
    the errors are small compared with the scale over which the WCS
    distortion changes.

    Parameters
    ----------
    wcs : WCS object
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

    x, y : float, array_like, or `~astropy.units.Quantity`
        The pixel coordinates. Both must have the same shape. A
        `~astropy.units.Quantity` must be in pixel units.

    x_err, y_err : float, array_like, or `~astropy.units.Quantity`
        The nonnegative 1-sigma errors of the ``x`` and ``y`` pixel
        coordinates. They must be broadcastable to the shape of ``x``
        and ``y``. A `~astropy.units.Quantity` must be in pixel units.

    xy_cov : float, array_like, or `~astropy.units.Quantity`, optional
        The covariance of the ``x`` and ``y`` pixel coordinates in
        pixel\\ :sup:`2`. It must be broadcastable to the shape of ``x``
        and ``y`` and must not exceed ``x_err * y_err`` in absolute
        value. A `~astropy.units.Quantity` must be in pixel\\ :sup:`2`
        units. The default of zero treats the ``x`` and ``y`` errors as
        independent.

    Returns
    -------
    east_err, north_err : `~astropy.units.Quantity`
        The 1-sigma sky position errors in arcsec along East and North,
        with the same shape as ``x`` and ``y``. The East error is a
        great-circle angle (i.e., it includes the cos(latitude) factor).
        The directions are those of the longitude and latitude axes of
        the celestial frame of the ``wcs``. They are Right Ascension
        and Declination only for an equatorial frame (e.g., ICRS). The
        errors are NaN where the position or any of its errors is not
        finite or is masked, where the position is outside the bounding
        box of a gwcs, or where the WCS is undefined (e.g., outside the
        valid region of the projection).

    Raises
    ------
    ValueError
        If ``x_err`` or ``y_err`` is negative, if the absolute value
        of ``xy_cov`` exceeds ``x_err * y_err``, or if a
        `~astropy.units.Quantity` input is not in pixel units.

    See Also
    --------
    compute_pixel_areas

    Examples
    --------
    >>> from photutils.datasets import make_wcs
    >>> from photutils.utils import pixel_to_sky_errors
    >>> wcs = make_wcs((100, 100))  # 0.1 arcsec pixels
    >>> x_err = [0.1, 0.2]
    >>> y_err = [0.1, 0.2]
    >>> east_err, north_err = pixel_to_sky_errors(
    ...     wcs, [10.0, 50.0], [20.0, 50.0], x_err, y_err)
    >>> east_err  # doctest: +FLOAT_CMP
    <Quantity [0.01, 0.02] arcsec>
    >>> north_err  # doctest: +FLOAT_CMP
    <Quantity [0.01, 0.02] arcsec>
    """
    x = _as_pixel_array(x, 'x', u.pix)
    y = _as_pixel_array(y, 'y', u.pix)
    if x.shape != y.shape:
        msg = 'x and y must have the same shape'
        raise ValueError(msg)

    x_err = _as_pixel_array(x_err, 'x_err', u.pix)
    y_err = _as_pixel_array(y_err, 'y_err', u.pix)
    xy_cov = _as_pixel_array(xy_cov, 'xy_cov', u.pix**2)
    try:
        x_err, y_err, xy_cov = (np.broadcast_to(value, x.shape)
                                for value in (x_err, y_err, xy_cov))
    except ValueError:
        msg = ('x_err, y_err, and xy_cov must be broadcastable to the '
               'shape of x and y')
        raise ValueError(msg) from None

    if np.any(x_err < 0) or np.any(y_err < 0):
        msg = 'x_err and y_err must be nonnegative'
        raise ValueError(msg)

    # Ignore floating-point errors from nonfinite or huge errors, which
    # give NaN sky errors below. The tolerance allows for the rounding
    # of an xy_cov computed from a correlation of +1 or -1.
    with np.errstate(over='ignore', invalid='ignore'):
        if np.any(np.abs(xy_cov) > x_err * y_err * (1.0 + PSD_RTOL)):
            msg = 'abs(xy_cov) must not exceed x_err * y_err'
            raise ValueError(msg)

        pixel_covariance = np.empty((x.size, 2, 2))
        pixel_covariance[:, 0, 0] = x_err.ravel()**2
        pixel_covariance[:, 1, 1] = y_err.ravel()**2
        pixel_covariance[:, 0, 1] = xy_cov.ravel()
        pixel_covariance[:, 1, 0] = xy_cov.ravel()

    xycen = np.column_stack((x.ravel(), y.ravel()))
    sky_covariance = pixel_to_sky_covariance(wcs, pixel_covariance, xycen)

    # A covariance on the boundary of validity (abs(xy_cov) equal to
    # x_err * y_err) can give a variance that is slightly negative from
    # rounding. The clip keeps NaN variances.
    errors = np.sqrt(np.maximum(sky_covariance[:, (0, 1), (0, 1)], 0.0))

    # The Jacobians ignore any gwcs bounding box (see above). The
    # positions are evaluated again with the bounding box applied so
    # that the errors are NaN wherever pixel_to_world returns NaN.
    low_level_wcs = _low_level_wcs(wcs)
    if _is_gwcs(low_level_wcs):
        world = low_level_wcs.pixel_to_world_values(xycen[:, 0],
                                                    xycen[:, 1])
        errors[~np.all(np.isfinite(world), axis=0)] = np.nan
    east_err, north_err = errors.T.reshape(2, *x.shape)
    return east_err << u.arcsec, north_err << u.arcsec
