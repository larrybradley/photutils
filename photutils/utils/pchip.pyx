# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t


cdef inline double pchip_eval(
    double x,
    double x0,
    double x1,
    double y0,
    double y1,
    double m0,
    double m1
) nogil:
    cdef double h = x1 - x0
    cdef double t = (x - x0) / h
    cdef double t2 = t * t
    cdef double t3 = t2 * t

    return (
        (2*t3 - 3*t2 + 1) * y0 +
        (t3 - 2*t2 + t) * h * m0 +
        (-2*t3 + 3*t2) * y1 +
        (t3 - t2) * h * m1
    )


cdef void compute_pchip_slopes(
    double[:] x,
    double[:] y,
    double[:] m
) nogil:
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef double dx1, dx2, dy1, dy2, w1, w2

    # Endpoints: one-sided
    m[0] = (y[1] - y[0]) / (x[1] - x[0])
    m[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

    for i in range(1, n - 1):
        dx1 = x[i] - x[i - 1]
        dx2 = x[i + 1] - x[i]
        dy1 = (y[i] - y[i - 1]) / dx1
        dy2 = (y[i + 1] - y[i]) / dx2

        if dy1 * dy2 <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * dx2 + dx1
            w2 = dx2 + 2.0 * dx1
            m[i] = (w1 + w2) / (w1 / dy1 + w2 / dy2)


def pchip2d(
    np.ndarray[DTYPE_t, ndim=1] x,
    np.ndarray[DTYPE_t, ndim=1] y,
    np.ndarray[DTYPE_t, ndim=2] z,
    np.ndarray[DTYPE_t, ndim=1] xi,
    np.ndarray[DTYPE_t, ndim=1] yi
):
    """
    Fast 2D tensor-product PCHIP interpolation.
    """
    cdef Py_ssize_t nx = x.shape[0]
    cdef Py_ssize_t ny = y.shape[0]
    cdef Py_ssize_t ni = xi.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] zx = np.empty((ny, ni))
    cdef np.ndarray[DTYPE_t, ndim=1] zi = np.empty(ni)

    cdef np.ndarray[DTYPE_t, ndim=1] m = np.empty(nx)
    cdef Py_ssize_t i, j, k

    # X-direction interpolation
    for j in range(ny):
        compute_pchip_slopes(x, z[j], m)
        for i in range(ni):
            k = np.searchsorted(x, xi[i]) - 1
            if k < 0:
                k = 0
            elif k >= nx - 1:
                k = nx - 2
            zx[j, i] = pchip_eval(
                xi[i],
                x[k], x[k + 1],
                z[j, k], z[j, k + 1],
                m[k], m[k + 1]
            )

    # Y-direction interpolation
    cdef np.ndarray[DTYPE_t, ndim=1] col = np.empty(ny)
    cdef np.ndarray[DTYPE_t, ndim=1] my = np.empty(ny)

    for i in range(ni):
        for j in range(ny):
            col[j] = zx[j, i]
        compute_pchip_slopes(y, col, my)

        k = np.searchsorted(y, yi[i]) - 1
        if k < 0:
            k = 0
        elif k >= ny - 1:
            k = ny - 2

        zi[i] = pchip_eval(
            yi[i],
            y[k], y[k + 1],
            col[k], col[k + 1],
            my[k], my[k + 1]
        )

    return zi
