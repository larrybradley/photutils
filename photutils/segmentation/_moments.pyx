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

__all__ = ['raw_moments_3rd_order', 'central_moments_3rd_order',
           'centroid_win_step', 'aperture_weighted_sum']

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
