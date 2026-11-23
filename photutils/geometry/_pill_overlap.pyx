# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Tools to calculate the area of overlap between a pill (stadium) and a
pixel grid.

A pill is the set of points within a given radius of a line segment.
It is a rectangle capped by two half disks.

The cdef functions are not intended to be called from Python code.
They are pure C math functions declared ``noexcept nogil`` so they can
be called without the GIL, including from multiple threads on
free-threaded Python builds. The signatures of the single-pixel
functions are exported via _pill_overlap.pxd for reuse elsewhere.
"""

import numpy as np

cimport numpy as np

__all__ = ['pill_overlap_grid']


cdef extern from "math.h" nogil:
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double atan2(double y, double x)
    double floor(double x)
    double ceil(double x)


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


def pill_overlap_grid(double xmin, double xmax, double ymin, double ymax,
                      int nx, int ny, double length, double width,
                      double theta, int use_exact, int subpixels):
    """
    Calculate the fractional overlap between a pill and a pixel grid.

    The pill is centered on the origin. It is the set of points within
    ``width / 2`` of a line segment of length ``length - width``, so it
    is a rectangle capped by two half disks.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        The extent of the grid in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax).

    nx, ny : int
        The grid dimensions in the x and y direction. The grid is
        defined by the rectangle with corners (xmin, ymin) and (xmax,
        ymax) and is divided into nx and ny pixels in the x and y
        direction, respectively.

    length : float
        The full extent of the pill along its long axis, from tip to
        tip. It must be greater than or equal to ``width``.

    width : float
        The full extent of the pill perpendicular to its long axis,
        which is also the diameter of the pill ends.

    theta : float
        The rotation angle of the long axis in radians, measured
        counterclockwise from the positive x-axis.

    use_exact : 0 or 1
        Set to ``1`` to use an exact method to calculate the overlap
        between the pill and each pixel. Set to ``0`` to use a
        subpixel sampling method to calculate the overlap, where each
        pixel is divided into ``subpixels ** 2`` subpixels and the
        fraction of subpixels that are within the pill is used to
        estimate the overlap.

    subpixels : int
        The number of subpixels to use in each dimension when using
        the subpixel sampling method (``use_exact=0``). It must be at
        least 1 in that case. It is ignored when ``use_exact=1``. Each
        pixel is resampled by this factor in each dimension, so each
        pixel is divided into ``subpixels ** 2`` subpixels.

        A subpixel is included only if its center lies strictly inside
        the pill. Subpixel centers lying exactly on the pill boundary
        are excluded (weight 0).

    Returns
    -------
    result : `~numpy.ndarray` (float)
        A 2D array of shape (ny, nx) giving the fraction of each
        pixel's area that overlaps with the pill, ranging from 0 to
        1. The element at index (j, i) corresponds to the pixel with
        corners at (xmin + i * dx, ymin + j * dy) and (xmin + (i + 1)
        * dx, ymin + (j + 1) * dy), where dx and dy are the width of
        each pixel in the x and y direction, respectively.

    Raises
    ------
    ValueError
        If ``use_exact`` is not 0 or 1, or if ``subpixels`` is less
        than 1 when ``use_exact=0``.
    """
    if use_exact != 0 and use_exact != 1:
        msg = 'use_exact must be 0 or 1'
        raise ValueError(msg)
    if use_exact == 0 and subpixels < 1:
        msg = 'subpixels must be a strictly positive integer'
        raise ValueError(msg)

    cdef unsigned int i, j
    cdef int i_min, i_max, j_min, j_max
    cdef double pxmin, pxmax, pymin, pymax
    cdef double dx, dy, margin
    cdef double half_straight = 0.5 * (length - width)
    cdef double radius = 0.5 * width
    cdef double cos_theta = cos(theta)
    cdef double sin_theta = sin(theta)
    cdef double bbox_dx, bbox_dy

    cdef np.ndarray[DTYPE_t, ndim=2] frac = np.zeros([ny, nx], dtype=DTYPE)
    cdef double[:, ::1] frac_view = frac

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    margin = 0.5 * sqrt(dx * dx + dy * dy)

    # Axis-aligned bounding box of the rotated pill, used to restrict
    # the pixel loops to the bounding-box index range (pixels outside
    # it have zero overlap). The clamping to [0, nx] and [0, ny] is done
    # in floating point to avoid integer overflow for pills far outside
    # the grid.
    bbox_dx = half_straight * fabs(cos_theta) + radius
    bbox_dy = half_straight * fabs(sin_theta) + radius

    i_min = <int>fmax(0.0, fmin(<double>nx,
                                floor((-bbox_dx - xmin) / dx)))
    i_max = <int>fmax(0.0, fmin(<double>nx,
                                ceil((bbox_dx - xmin) / dx)))
    j_min = <int>fmax(0.0, fmin(<double>ny,
                                floor((-bbox_dy - ymin) / dy)))
    j_max = <int>fmax(0.0, fmin(<double>ny,
                                ceil((bbox_dy - ymin) / dy)))

    with nogil:
        for i in range(i_min, i_max):
            pxmin = xmin + i * dx
            pxmax = pxmin + dx
            for j in range(j_min, j_max):
                pymin = ymin + j * dy
                pymax = pymin + dy
                frac_view[j, i] = pill_frac_core(
                    pxmin, pymin, pxmax, pymax, dx, dy, margin,
                    half_straight, radius, cos_theta, sin_theta,
                    use_exact, subpixels)

    return frac


cdef double pill_overlap_single_subpixel(double x0, double y0,
                                         double x1, double y1,
                                         double half_straight,
                                         double radius,
                                         double cos_theta,
                                         double sin_theta,
                                         int subpixels) noexcept nogil:
    """
    Fraction of overlap between a pill centered on the origin and a
    single pixel, using subpixel sampling.

    Parameters
    ----------
    x0, y0, x1, y1 : double
        The lower and upper edges of the pixel.

    half_straight : double
        Half the length of the straight section of the pill.

    radius : double
        The radius of the pill ends.

    cos_theta, sin_theta : double
        The cosine and sine of the pill rotation angle.

    subpixels : int
        The number of subpixels to sample in each dimension.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        pill.
    """
    cdef unsigned int _i, _j
    cdef double x, y, x_tr, y_tr, along
    cdef double frac = 0.0
    cdef double r2 = radius * radius
    cdef double dx = (x1 - x0) / subpixels
    cdef double dy = (y1 - y0) / subpixels

    x = x0 - 0.5 * dx
    for _i in range(subpixels):
        x += dx
        y = y0 - 0.5 * dy
        for _j in range(subpixels):
            y += dy
            x_tr = x * cos_theta + y * sin_theta
            y_tr = -x * sin_theta + y * cos_theta
            along = fmax(fabs(x_tr) - half_straight, 0.0)
            if along * along + y_tr * y_tr < r2:
                frac += 1.0

    return frac / (subpixels * subpixels)


cdef double pill_junction_overlap_exact(double pxmin, double pymin,
                                        double pxmax, double pymax,
                                        double half_straight,
                                        double radius,
                                        double cos_theta,
                                        double sin_theta) noexcept nogil:
    """
    Exact area of overlap between a pill centered on the origin and a
    single pixel that may straddle a junction between the straight
    section and an end.

    The area is the sum of three terms. The rectangle term is the
    overlap of the pixel with the straight section. Each half-disk term
    is the overlap of an end disk with the part of the pixel beyond
    that end's junction line. Both half-disk terms are always evaluated
    because a pixel can straddle both junction lines when the straight
    section is shorter than the pixel diagonal.

    Parameters
    ----------
    pxmin, pymin, pxmax, pymax : double
        The pixel edges, relative to the pill center.

    half_straight : double
        Half the length of the straight section of the pill.

    radius : double
        The radius of the pill ends.

    cos_theta, sin_theta : double
        The cosine and sine of the pill rotation angle.

    Returns
    -------
    area : double
        The area of overlap between the pill and the pixel.
    """
    cdef int k, n_clip
    cdef double area = 0.0
    cdef double corner_x[4]
    cdef double corner_y[4]
    cdef double quad_x[4]
    cdef double quad_y[4]
    cdef double clip_x[5]
    cdef double clip_y[5]
    cdef double poly_x[4]
    cdef double poly_y[4]
    cdef double buf_a_x[32]
    cdef double buf_a_y[32]
    cdef double buf_b_x[32]
    cdef double buf_b_y[32]

    if half_straight > 0.0:
        rect_vertices(half_straight, radius, cos_theta, sin_theta,
                      poly_x, poly_y)
        area += polygon_pixel_overlap(pxmin, pymin, pxmax, pymax, poly_x,
                                      poly_y, 4, buf_a_x, buf_a_y,
                                      buf_b_x, buf_b_y, 32)

    # Rotate the counterclockwise pixel corners into the pill frame.
    corner_x[0] = pxmin
    corner_y[0] = pymin
    corner_x[1] = pxmax
    corner_y[1] = pymin
    corner_x[2] = pxmax
    corner_y[2] = pymax
    corner_x[3] = pxmin
    corner_y[3] = pymax
    for k in range(4):
        quad_x[k] = corner_x[k] * cos_theta + corner_y[k] * sin_theta
        quad_y[k] = -corner_x[k] * sin_theta + corner_y[k] * cos_theta

    n_clip = clip_half_plane_x(quad_x, quad_y, 4, half_straight, 1.0,
                               clip_x, clip_y)
    area += disk_convex_polygon_overlap(clip_x, clip_y, n_clip,
                                        half_straight, radius)

    n_clip = clip_half_plane_x(quad_x, quad_y, 4, -half_straight, -1.0,
                               clip_x, clip_y)
    area += disk_convex_polygon_overlap(clip_x, clip_y, n_clip,
                                        -half_straight, radius)

    return area


cdef int clip_half_plane_x(double *in_x, double *in_y, int n_in,
                           double x0, double side, double *out_x,
                           double *out_y) noexcept nogil:
    """
    Clip a convex polygon to a half plane bounded by a vertical line,
    using the Sutherland-Hodgman algorithm.

    Parameters
    ----------
    in_x, in_y : double *
        The x and y coordinates of the input polygon vertices.

    n_in : int
        The number of input vertices.

    x0 : double
        The x position of the clipping line.

    side : double
        Set to 1 to keep the half plane ``x >= x0``. Set to -1 to keep
        the half plane ``x <= x0``.

    out_x, out_y : double *
        Output. The x and y coordinates of the clipped polygon
        vertices. Each must point to a buffer of at least ``n_in + 1``
        doubles.

    Returns
    -------
    n_out : int
        The number of clipped polygon vertices.
    """
    cdef int k
    cdef int n_out = 0
    cdef double prev_x, prev_y, cur_x, cur_y, t
    cdef bint prev_in, cur_in

    prev_x = in_x[n_in - 1]
    prev_y = in_y[n_in - 1]
    prev_in = side * (prev_x - x0) >= 0.0
    for k in range(n_in):
        cur_x = in_x[k]
        cur_y = in_y[k]
        cur_in = side * (cur_x - x0) >= 0.0
        if cur_in != prev_in:
            t = (x0 - prev_x) / (cur_x - prev_x)
            out_x[n_out] = x0
            out_y[n_out] = prev_y + t * (cur_y - prev_y)
            n_out += 1
        if cur_in:
            out_x[n_out] = cur_x
            out_y[n_out] = cur_y
            n_out += 1
        prev_x = cur_x
        prev_y = cur_y
        prev_in = cur_in

    return n_out


cdef double disk_convex_polygon_overlap(double *poly_x, double *poly_y,
                                        int n_poly, double center_x,
                                        double radius) noexcept nogil:
    """
    Exact area of overlap between a disk centered on the x-axis and a
    convex polygon.

    The polygon is shifted to the disk center and scaled by ``1 /
    radius``. For each polygon edge, the signed area of the unit disk
    inside the triangle formed by the disk center and the edge is
    added. The absolute value of the sum is the overlap area in units
    of ``radius**2``. The sum needs no special cases for duplicate
    vertices, zero-length edges, or vertices on the circle or at its
    center.

    Parameters
    ----------
    poly_x, poly_y : double *
        The x and y coordinates of the polygon vertices.

    n_poly : int
        The number of polygon vertices.

    center_x : double
        The x position of the disk center. Its y position is 0.

    radius : double
        The radius of the disk.

    Returns
    -------
    area : double
        The area of overlap between the disk and the polygon.
    """
    cdef int k
    cdef double inv_radius, x0, y0, x1, y1
    cdef double area = 0.0

    if n_poly < 3:
        return 0.0

    inv_radius = 1.0 / radius
    x0 = (poly_x[n_poly - 1] - center_x) * inv_radius
    y0 = poly_y[n_poly - 1] * inv_radius
    for k in range(n_poly):
        x1 = (poly_x[k] - center_x) * inv_radius
        y1 = poly_y[k] * inv_radius
        area += unit_disk_triangle_signed_area(x0, y0, x1, y1)
        x0 = x1
        y0 = y1

    return fabs(area) * radius * radius


cdef double unit_disk_triangle_signed_area(double x0, double y0,
                                           double x1, double y1
                                           ) noexcept nogil:
    """
    Signed area of the overlap between the unit disk and the triangle
    formed by the origin and the points (x0, y0) and (x1, y1).

    The edge from (x0, y0) to (x1, y1) is split where it crosses the
    unit circle. The part of the edge inside the circle contributes
    the area of its triangle with the origin, and each part outside
    the circle contributes the area of the circular sector it
    subtends. The result is positive when the triangle is
    counterclockwise.

    Parameters
    ----------
    x0, y0 : double
        The start point of the edge.

    x1, y1 : double
        The end point of the edge.

    Returns
    -------
    area : double
        The signed area of overlap.
    """
    cdef double dx = x1 - x0
    cdef double dy = y1 - y0
    cdef double a = dx * dx + dy * dy
    cdef double b, cross, disc, root, t_in, t_out
    cdef double xin, yin, xout, yout
    cdef double area

    if a == 0.0:
        return 0.0

    # The edge points at parameter t are inside the circle for t
    # between the two roots of a * t**2 + 2 * b * t + c = 0, where c
    # is x0**2 + y0**2 - 1. The discriminant b**2 - a * c is written
    # in terms of the cross product, which avoids the cancellation
    # between its two terms when the edge is far from the circle.
    b = x0 * dx + y0 * dy
    cross = x0 * dy - y0 * dx
    disc = a - cross * cross
    t_in = 1.0
    t_out = 0.0
    if disc > 0.0:
        root = sqrt(disc)
        t_in = fmax((-b - root) / a, 0.0)
        t_out = fmin((-b + root) / a, 1.0)

    # The edge is wholly outside the circle or tangent to it.
    if t_in >= t_out:
        return 0.5 * atan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1)

    xin = x0 + t_in * dx
    yin = y0 + t_in * dy
    xout = x0 + t_out * dx
    yout = y0 + t_out * dy
    area = 0.5 * (xin * yout - yin * xout)
    if t_in > 0.0:
        area += 0.5 * atan2(x0 * yin - y0 * xin, x0 * xin + y0 * yin)
    if t_out < 1.0:
        area += 0.5 * atan2(xout * y1 - yout * x1, xout * x1 + yout * y1)

    return area
