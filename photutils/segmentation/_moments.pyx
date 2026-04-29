# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
"""
Cython kernels used by `~photutils.segmentation.SourceCatalog`.

Includes
* `raw_moments_3rd_order` and `central_moments_3rd_order` which
  replace the per-source ``yp.T @ arr @ xp`` matrix-multiply
  formulation with a single C-level pass over each cutout's pixels.
* `centroid_win_step` which performs one iteration of the windowed
  (Gaussian-weighted) centroid update used by `centroid_win`.
"""

import numpy as np

cimport numpy as np


cdef extern from "math.h":
    double exp(double x)
    double sqrt(double x)
    int isfinite(double x)

__all__ = ['raw_moments_3rd_order', 'central_moments_3rd_order',
           'centroid_win_step', 'aperture_weighted_sum',
           'kron_radius_sums', 'build_aperture_cutout_mask']

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def raw_moments_3rd_order(list cutouts):
    """
    Compute raw spatial moments up to 3rd order for each cutout.

    Parameters
    ----------
    cutouts : list of 2D `~numpy.ndarray` (float64)
        Per-source cutouts already zero-filled outside the source
        segment / mask / on invalid pixels (i.e., the
        ``_moment_data_cutouts`` produced by SourceCatalog).

    Returns
    -------
    moments : `~numpy.ndarray` of shape ``(n, 4, 4)``, float64
        ``moments[i, p, q] = sum_{y,x} y**p * x**q * cutout_i[y, x]``
        for ``p, q in {0, 1, 2, 3}``.  Entries with ``p + q > 3``
        are set to zero (matching the previous Vandermonde-based
        result, which also leaves them populated as numerical
        cross-terms; downstream code only reads ``p + q <= 3``).
    """
    cdef Py_ssize_t n = len(cutouts)
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((n, 4, 4), dtype=DTYPE)
    cdef DTYPE_t[:, :, :] out_v = out

    cdef Py_ssize_t i, yi, xi, ny, nx
    cdef DTYPE_t v, y, x
    cdef DTYPE_t y2, y3, x2, x3
    # Per-row x-power partial sums
    cdef DTYPE_t s0, s1, s2, s3
    cdef DTYPE_t[:, :] arr_v

    for i in range(n):
        arr = cutouts[i]
        if arr.dtype != DTYPE:
            arr = np.ascontiguousarray(arr, dtype=DTYPE)
        elif not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        arr_v = arr
        ny = arr_v.shape[0]
        nx = arr_v.shape[1]

        for yi in range(ny):
            # Accumulate row sums weighted by x-powers; this is what
            # makes the moment computation O(ny*nx) with very small
            # constant factors.
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for xi in range(nx):
                v = arr_v[yi, xi]
                if v == 0.0:
                    continue
                x = <DTYPE_t>xi
                x2 = x * x
                x3 = x2 * x
                s0 += v
                s1 += v * x
                s2 += v * x2
                s3 += v * x3

            if s0 == 0.0 and s1 == 0.0 and s2 == 0.0 and s3 == 0.0:
                continue

            y = <DTYPE_t>yi
            y2 = y * y
            y3 = y2 * y

            # m[p, q] += y**p * (sum_x v * x**q)
            out_v[i, 0, 0] += s0
            out_v[i, 0, 1] += s1
            out_v[i, 0, 2] += s2
            out_v[i, 0, 3] += s3

            out_v[i, 1, 0] += y * s0
            out_v[i, 1, 1] += y * s1
            out_v[i, 1, 2] += y * s2
            out_v[i, 1, 3] += y * s3

            out_v[i, 2, 0] += y2 * s0
            out_v[i, 2, 1] += y2 * s1
            out_v[i, 2, 2] += y2 * s2
            out_v[i, 2, 3] += y2 * s3

            out_v[i, 3, 0] += y3 * s0
            out_v[i, 3, 1] += y3 * s1
            out_v[i, 3, 2] += y3 * s2
            out_v[i, 3, 3] += y3 * s3

    return out


def central_moments_3rd_order(list cutouts, np.ndarray[DTYPE_t, ndim=2] centroids):
    """
    Compute central spatial moments up to 3rd order for each cutout.

    Parameters
    ----------
    cutouts : list of 2D `~numpy.ndarray` (float64)
        Per-source zero-filled cutouts (same as ``raw_moments_3rd_order``).
    centroids : 2D `~numpy.ndarray` of shape ``(n, 2)``, float64
        Per-source ``(x_centroid, y_centroid)`` in cutout-local
        coordinates.

    Returns
    -------
    moments : `~numpy.ndarray` of shape ``(n, 4, 4)``, float64
        ``moments[i, p, q] = sum_{y,x} (y - y_c)**p * (x - x_c)**q
        * cutout_i[y, x]``.
    """
    cdef Py_ssize_t n = len(cutouts)
    cdef np.ndarray[DTYPE_t, ndim=3] out = np.zeros((n, 4, 4), dtype=DTYPE)
    cdef DTYPE_t[:, :, :] out_v = out
    cdef DTYPE_t[:, :] cen_v = centroids

    cdef Py_ssize_t i, yi, xi, ny, nx
    cdef DTYPE_t v, x, y, x2, x3, y2, y3, xc, yc
    cdef DTYPE_t s0, s1, s2, s3
    cdef DTYPE_t[:, :] arr_v

    for i in range(n):
        arr = cutouts[i]
        if arr.dtype != DTYPE:
            arr = np.ascontiguousarray(arr, dtype=DTYPE)
        elif not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        arr_v = arr
        ny = arr_v.shape[0]
        nx = arr_v.shape[1]
        xc = cen_v[i, 0]
        yc = cen_v[i, 1]

        # Propagate a NaN centroid (e.g., for a completely-masked
        # source whose raw moments are all zero) into all output
        # entries so that downstream shape properties evaluate to
        # NaN, matching the previous BLAS-based formulation where
        # NaN * 0 = NaN.
        if xc != xc or yc != yc:
            for yi in range(4):
                for xi in range(4):
                    out_v[i, yi, xi] = float('nan')
            continue

        for yi in range(ny):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            for xi in range(nx):
                v = arr_v[yi, xi]
                if v == 0.0:
                    continue
                x = <DTYPE_t>xi - xc
                x2 = x * x
                x3 = x2 * x
                s0 += v
                s1 += v * x
                s2 += v * x2
                s3 += v * x3

            if s0 == 0.0 and s1 == 0.0 and s2 == 0.0 and s3 == 0.0:
                continue

            y = <DTYPE_t>yi - yc
            y2 = y * y
            y3 = y2 * y

            out_v[i, 0, 0] += s0
            out_v[i, 0, 1] += s1
            out_v[i, 0, 2] += s2
            out_v[i, 0, 3] += s3

            out_v[i, 1, 0] += y * s0
            out_v[i, 1, 1] += y * s1
            out_v[i, 1, 2] += y * s2
            out_v[i, 1, 3] += y * s3

            out_v[i, 2, 0] += y2 * s0
            out_v[i, 2, 1] += y2 * s1
            out_v[i, 2, 2] += y2 * s2
            out_v[i, 2, 3] += y2 * s3

            out_v[i, 3, 0] += y3 * s0
            out_v[i, 3, 1] += y3 * s1
            out_v[i, 3, 2] += y3 * s2
            out_v[i, 3, 3] += y3 * s3

    return out


def centroid_win_step(np.ndarray[DTYPE_t, ndim=2] data,
                      np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                      double cx, double cy,
                      double radius_sq, double inv_2sigma2):
    """
    Perform one iteration of the windowed (SExtractor XWIN/YWIN)
    centroid update.

    Computes Gaussian-weighted moments of ``data`` over the unmasked
    pixels lying inside the circular aperture of radius
    ``sqrt(radius_sq)`` centered at ``(cx, cy)``.

    Parameters
    ----------
    data : 2D ndarray (float64)
        Cutout data; masked pixels may already be replaced by mirrored
        values when ``aperture_mask_method='correct'``.

    mask : 2D ndarray (bool)
        Boolean mask; pixels where ``mask`` is `True` are excluded.

    cx : float
        Cutout-relative ``x`` centroid position.

    cy : float
        Cutout-relative ``y`` centroid position.

    radius_sq : float
        Squared aperture radius (``(4 sigma)^2``); pixels with
        ``dx*dx + dy*dy > radius_sq`` are excluded.

    inv_2sigma2 : float
        ``-1 / (2 sigma^2)`` used in the Gaussian weight.

    Returns
    -------
    total : float
        Gaussian-weighted flux inside the aperture.

    mx : float
        First-moment in ``x`` (relative to ``cx``).

    my : float
        First-moment in ``y`` (relative to ``cy``).
    """
    cdef DTYPE_t[:, :] data_v = data
    cdef np.uint8_t[:, :] mask_v = mask
    cdef Py_ssize_t ny = data_v.shape[0]
    cdef Py_ssize_t nx = data_v.shape[1]
    cdef Py_ssize_t yi, xi
    cdef double dy, dx, r2, w
    cdef double total = 0.0
    cdef double mx = 0.0
    cdef double my = 0.0

    for yi in range(ny):
        dy = <double>yi - cy
        for xi in range(nx):
            if mask_v[yi, xi]:
                continue
            dx = <double>xi - cx
            r2 = dx * dx + dy * dy
            if r2 > radius_sq:
                continue
            w = data_v[yi, xi] * exp(r2 * inv_2sigma2)
            total += w
            mx += w * dx
            my += w * dy

    return total, mx, my


def aperture_weighted_sum(np.ndarray[DTYPE_t, ndim=2] weights,
                          np.ndarray[DTYPE_t, ndim=2] data,
                          np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                          error):
    """
    Compute aperture-weighted flux (and optional flux error) over the
    good pixels of a cutout.

    A pixel is considered good when its aperture weight is positive
    *and* the corresponding ``mask`` value is `False`.  This matches
    the per-source inner loop used by aperture photometry in
    `~photutils.segmentation.SourceCatalog`.

    Parameters
    ----------
    weights : 2D `~numpy.ndarray` of float64
        The aperture overlap weights.

    data : 2D `~numpy.ndarray` of float64
        The (background-subtracted) cutout data.

    mask : 2D `~numpy.ndarray` of bool
        Boolean mask.  Pixels where ``mask`` is `True` are excluded.

    error : 2D `~numpy.ndarray` of float64 or None
        Optional per-pixel error array.  When `None`, ``flux_err``
        is returned as ``NaN``.

    Returns
    -------
    flux : float
        ``sum_{good} weights * data``, or ``NaN`` if no good pixels.

    flux_err : float
        ``sqrt(sum_{good} weights * error**2)``, ``NaN`` if no good
        pixels or if ``error`` is `None`.
    """
    cdef DTYPE_t[:, :] w_v = weights
    cdef DTYPE_t[:, :] d_v = data
    cdef np.uint8_t[:, :] m_v = mask
    cdef DTYPE_t[:, :] e_v
    cdef bint has_error = error is not None
    cdef Py_ssize_t ny = w_v.shape[0]
    cdef Py_ssize_t nx = w_v.shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t n_good = 0
    cdef double w
    cdef double flux = 0.0
    cdef double err_sq = 0.0
    cdef double flux_err

    if has_error:
        e_v = error
        for j in range(ny):
            for i in range(nx):
                w = w_v[j, i]
                if w > 0.0 and not m_v[j, i]:
                    flux += w * d_v[j, i]
                    err_sq += w * e_v[j, i] * e_v[j, i]
                    n_good += 1
    else:
        for j in range(ny):
            for i in range(nx):
                w = w_v[j, i]
                if w > 0.0 and not m_v[j, i]:
                    flux += w * d_v[j, i]
                    n_good += 1

    if n_good == 0:
        return float('nan'), float('nan')

    if has_error:
        flux_err = err_sq ** 0.5
    else:
        flux_err = float('nan')
    return flux, flux_err


def kron_radius_sums(np.ndarray[DTYPE_t, ndim=2] data,
                     np.ndarray[np.uint8_t, ndim=2, cast=True] mask,
                     double xc, double yc,
                     double cxx, double cxy, double cyy,
                     double scale_sq):
    """
    Compute the unscaled Kron-radius numerator and denominator for one
    source.

    Replaces the per-source NumPy block::

        xx = np.arange(nx) - xc
        yy = np.arange(ny) - yc
        rr2 = cxx*xx*xx + cxy*xx*yy + cyy*yy*yy
        rr = sqrt(max(rr2, 0))
        pixel_mask = (rr <= scale) & ~mask
        flux_numer = sum(data[pixel_mask] * rr[pixel_mask])
        flux_denom = sum(data[pixel_mask])

    with a single C loop, eliminating the per-source allocation of the
    ``rr2``, ``rr`` and ``pixel_mask`` arrays.

    Parameters
    ----------
    data : 2D ndarray (float64)
        Cutout data (already mirror-corrected when needed).

    mask : 2D ndarray (bool)
        Boolean mask; pixels with ``mask`` True are excluded.

    xc : float
        Cutout-relative source ``x`` centroid.

    yc : float
        Cutout-relative source ``y`` centroid.

    cxx : float
        Elliptical-radius coefficient ``cxx``.

    cxy : float
        Elliptical-radius coefficient ``cxy``.

    cyy : float
        Elliptical-radius coefficient ``cyy``.

    scale_sq : float
        Squared aperture scale (``scale**2``); pixels with
        ``rr2 > scale_sq`` are excluded.

    Returns
    -------
    flux_numer : float
        ``sum(data * sqrt(max(rr2, 0)))`` over the aperture's good
        pixels.

    flux_denom : float
        ``sum(data)`` over the aperture's good pixels.
    """
    cdef DTYPE_t[:, :] d_v = data
    cdef np.uint8_t[:, :] m_v = mask
    cdef Py_ssize_t ny = d_v.shape[0]
    cdef Py_ssize_t nx = d_v.shape[1]
    cdef Py_ssize_t i, j
    cdef double dy, dx, r2, r, v
    cdef double number = 0.0
    cdef double denom = 0.0

    for j in range(ny):
        dy = <double>j - yc
        for i in range(nx):
            if m_v[j, i]:
                continue
            dx = <double>i - xc
            r2 = cxx * dx * dx + cxy * dx * dy + cyy * dy * dy
            if r2 > scale_sq:
                continue
            v = d_v[j, i]
            if r2 < 0.0:
                r2 = 0.0
            r = sqrt(r2)
            number += v * r
            denom += v

    return number, denom


def build_aperture_cutout_mask(np.ndarray[DTYPE_t, ndim=2, mode='c'] data,
                               user_mask,
                               segm_mask,
                               double xcen, double ycen,
                               double bkg, int method,
                               int zero_masked):
    """
    In-place build of the per-source aperture cutout used by the
    Kron / flux_radius / centroid_win / aperture-photometry paths.

    Performs in a single C pass:

    1. ``data -= bkg`` (so callers can pass the raw cutout).
    2. ``data_mask = ~isfinite(data) | user_mask``.
    3. If ``method == 1`` ('mask'), ``data_mask |= segm_mask``.
    4. If ``method == 2`` ('correct'), replace each pixel where
       ``segm_mask`` is `True` with the value of the pixel mirrored
       across ``(int(xcen + 0.5), int(ycen + 0.5))``.  Pixels whose
       mirror falls outside the cutout, or whose mirror is itself
       masked (in either ``data_mask`` or ``segm_mask``), are set to
       zero.  This matches the behaviour of
       ``photutils.segmentation.utils._mask_to_mirrored_value``.
    5. If ``zero_masked`` is non-zero, ``data[data_mask] = 0`` (used
       by ``flux_radius``).

    Parameters
    ----------
    data : 2D float64 `~numpy.ndarray` (C-contiguous)
        Modified in place.

    user_mask : 2D bool `~numpy.ndarray` or `None`
        Per-pixel user mask (`True` = excluded).

    segm_mask : 2D bool `~numpy.ndarray` or `None`
        Per-pixel neighbor-segment mask (`True` = neighboring source).
        Required when ``method != 0``.

    xcen, ycen : float
        Cutout-relative source centroid (used for mirror replacement
        when ``method == 2``).

    bkg : float
        Local background to subtract from ``data``.

    method : int
        ``0`` = no segm masking, ``1`` = include segm_mask in
        ``data_mask`` ('mask'), ``2`` = mirror-replace ('correct').

    zero_masked : int
        If non-zero, also zero out masked pixels in ``data`` so that a
        plain ``sum`` (no boolean indexing) returns the right value.

    Returns
    -------
    data_mask : 2D `~numpy.ndarray` of `~numpy.uint8`
        Per-pixel boolean (0 / 1) mask of pixels excluded from
        photometry.
    """
    cdef DTYPE_t[:, ::1] d_v = data
    cdef Py_ssize_t ny = d_v.shape[0]
    cdef Py_ssize_t nx = d_v.shape[1]
    cdef Py_ssize_t i, j, mi, mj
    cdef int has_user_mask = user_mask is not None
    cdef int has_segm_mask = segm_mask is not None
    cdef np.uint8_t[:, ::1] um_v
    cdef np.uint8_t[:, ::1] sm_v

    cdef np.ndarray[np.uint8_t, ndim=2] out_mask = np.zeros(
        (ny, nx), dtype=np.uint8)
    cdef np.uint8_t[:, ::1] m_v = out_mask

    cdef double v

    if has_user_mask:
        um_v = np.ascontiguousarray(user_mask).view(np.uint8)
    if has_segm_mask:
        sm_v = np.ascontiguousarray(segm_mask).view(np.uint8)

    # Pass 1: in-place bkg subtract; build data_mask
    for j in range(ny):
        for i in range(nx):
            v = d_v[j, i] - bkg
            d_v[j, i] = v
            if not isfinite(v):
                m_v[j, i] = 1
            elif has_user_mask and um_v[j, i]:
                m_v[j, i] = 1

    if method == 1 and has_segm_mask:
        # 'mask': fold segm_mask into data_mask
        for j in range(ny):
            for i in range(nx):
                if sm_v[j, i]:
                    m_v[j, i] = 1
    elif method == 2 and has_segm_mask:
        # 'correct': mirror-replace each segm_mask pixel
        cxc = int(xcen + 0.5)
        cyc = int(ycen + 0.5)
        for j in range(ny):
            for i in range(nx):
                if not sm_v[j, i]:
                    continue
                mi = 2 * cxc - i
                mj = 2 * cyc - j
                if mi < 0 or mj < 0 or mi >= nx or mj >= ny:
                    d_v[j, i] = 0.0
                elif m_v[mj, mi] or sm_v[mj, mi]:
                    d_v[j, i] = 0.0
                else:
                    d_v[j, i] = d_v[mj, mi]

    if zero_masked:
        for j in range(ny):
            for i in range(nx):
                if m_v[j, i]:
                    d_v[j, i] = 0.0

    return out_mask
