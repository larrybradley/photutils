# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
"""
Cython kernel for computing raw image moments (up to 3rd order)
for a list of 2D source cutouts.

This replaces the per-source ``yp.T @ arr @ xp`` matrix-multiply
formulation in `SourceCatalog.moments` with a single C-level pass
over each cutout's pixels, eliminating the Python/BLAS dispatch
overhead that dominates runtime when there are many small
(~10x10) sources.
"""

import numpy as np
cimport numpy as np

__all__ = ['raw_moments_3rd_order', 'central_moments_3rd_order']

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
