# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel cimport prange

ctypedef np.float64_t DTYPE_t


cdef inline double hermite(
    double t,
    double y0,
    double y1,
    double m0,
    double m1,
    double h
) nogil:
    cdef double t2 = t * t
    cdef double t3 = t2 * t

    return (
        (2*t3 - 3*t2 + 1) * y0 +
        (t3 - 2*t2 + t) * h * m0 +
        (-2*t3 + 3*t2) * y1 +
        (t3 - t2) * h * m1
    )


cdef void pchip_slopes_uniform(
    double[:] y,
    double inv_dx,
    double[:] m
) nogil:
    cdef Py_ssize_t n = y.shape[0]
    cdef Py_ssize_t i
    cdef double d1, d2

    m[0] = (y[1] - y[0]) * inv_dx
    m[n - 1] = (y[n - 1] - y[n - 2]) * inv_dx

    for i in range(1, n - 1):
        d1 = (y[i] - y[i - 1]) * inv_dx
        d2 = (y[i + 1] - y[i]) * inv_dx

        if d1 * d2 <= 0.0:
            m[i] = 0.0
        else:
            m[i] = 2.0 * d1 * d2 / (d1 + d2)


def pchip2d_resample(
    np.ndarray[DTYPE_t, ndim=2] z,
    double dx,
    double dy,
    Py_ssize_t nx_out,
    Py_ssize_t ny_out
):
    """
    Resample a 2D uniform-grid image using tensor-product PCHIP.

    Parameters
    ----------
    z : (ny, nx) ndarray
        Input image.
    dx, dy : float
        Input pixel spacing.
    nx_out, ny_out : int
        Output image shape.

    Returns
    -------
    zout : (ny_out, nx_out) ndarray
    """
    cdef Py_ssize_t ny, nx
    ny, nx = z.shape

    cdef double inv_dx = 1.0 / dx
    cdef double inv_dy = 1.0 / dy

    cdef double sx = (n
