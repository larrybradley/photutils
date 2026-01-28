# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np
from cython.parallel cimport prange

ctypedef np.float64_t DTYPE_t


cdef inline double pchip_eval(
    double x,
    double x0,
    double inv_dx,
    double y0,
    double y1,
    double m0,
    double m1
) nogil:
    cdef double t = (x - x0) * inv_dx
    cdef double t2 = t * t
    cdef double t3 = t2 * t
    cdef double h = 1.0 / inv_dx

    return (
        (2*t3 - 3*t2 + 1) * y0 +
        (t3 - 2*t2 + t) * h * m0 +
        (-2*t3 + 3*t2) * y1 +
        (t3 - t2) * h * m1
    )


cdef void compute_pchip_slopes_uniform(
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
            # Uniform-grid PCHIP simplifies nicely
            m[i] = 2.0 * d1 * d2 / (d1 + d2)


def pchip2d_uniform(
    np.ndarray[DTYPE_t, ndim=1] x,
    np.ndarray[DTYPE_t, ndim=1] y,
    np.ndarray[DTYPE_t, ndim=2] z,
    np.ndarray[DTYPE_t, ndim=1] xi,
    np.ndarray[DTYPE_t, ndim=1] yi
):
    """
    Fast 2D PCHIP interpolation for uniform grids.
    """
    cdef Py_ssize_t nx = x.shape[0]
    cdef Py_ssize_t ny = y.shape[0]
    cdef Py_ssize_t ni = xi.shape[0]

    cdef double x0 = x[0]
    cdef double y0 = y[0]
    cdef double inv_dx = 1.0 / (x[1] - x[0])
    cdef double inv_dy = 1.0 / (y[1] - y[0])

    cdef np.ndarray[DTYPE_t, ndim=2] zx = np.empty((ny, ni))
    cdef np.ndarray[DTYPE_t, ndim=1] zi = np.empty(ni)

    cdef np.ndarray[DTYPE_t, ndim=1] mx = np.empty(nx)
    cdef np.ndarray[DTYPE_t, ndim=1] my = np.empty(ny)
    cdef np.ndarray[DTYPE_t, ndim=1] co_]()
