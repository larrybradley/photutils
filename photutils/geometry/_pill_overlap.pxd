# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, cdivision=True
"""
Declarations needed to cimport the pill overlap functions into other
Cython files. These single-pixel functions are pure C math functions
that are safe to call without the GIL.
"""

from ._polygon_overlap cimport polygon_pixel_overlap
from .circle_overlap cimport circle_overlap_single_exact
from .rectangle_overlap cimport rect_vertices


cdef extern from "math.h" nogil:
    double fabs(double x)
    double fmax(double x, double y)
    double fmin(double x, double y)


cdef double pill_overlap_single_subpixel(double x0, double y0,
                                         double x1, double y1,
                                         double half_straight,
                                         double radius,
                                         double cos_theta,
                                         double sin_theta,
                                         int subpixels) noexcept nogil
cdef double pill_junction_overlap_exact(double pxmin, double pymin,
                                        double pxmax, double pymax,
                                        double half_straight,
                                        double radius,
                                        double cos_theta,
                                        double sin_theta) noexcept nogil


cdef inline double pill_frac_core(double pxmin, double pymin,
                                  double pxmax, double pymax,
                                  double dx, double dy, double margin,
                                  double half_straight, double radius,
                                  double cos_theta, double sin_theta,
                                  int use_exact,
                                  int subpixels) noexcept nogil:
    """
    Fraction of a single pixel that overlaps a pill centered on the
    origin.

    The pill is the set of points within ``radius`` of the segment
    from ``(-half_straight, 0)`` to ``(half_straight, 0)`` in the pill
    frame, which is rotated by ``theta`` from the pixel frame.

    This is the per-pixel decision core for the pill overlap, shared
    by ``pill_overlap_grid`` and the batch photometry driver. It
    applies the interior and exterior fast paths, then dispatches to
    subpixel sampling or to the exact overlap. The exact overlap uses
    the rectangle clip when the pixel touches only the straight
    section, the exact circle kernel when it touches only one end, and
    the three-term sum otherwise. It uses only local stack buffers, so
    it is safe to call from multiple threads.

    Parameters
    ----------
    pxmin, pymin, pxmax, pymax : double
        The pixel edges, relative to the pill center.

    dx, dy : double
        The pixel width and height.

    margin : double
        Half the pixel diagonal.

    half_straight : double
        Half the length of the straight section, ``(length - width)
        / 2``.

    radius : double
        The radius of the pill ends, ``width / 2``.

    cos_theta, sin_theta : double
        The cosine and sine of the pill rotation angle.

    use_exact : int
        Set to 1 to use the exact geometric overlap calculation. Set
        to 0 to use subpixel sampling instead.

    subpixels : int
        The number of subpixels (per dimension) used when
        ``use_exact`` is 0.

    Returns
    -------
    frac : double
        The fraction (0 to 1) of the pixel's area that overlaps the
        pill.
    """
    cdef double xcen = 0.5 * (pxmin + pxmax)
    cdef double ycen = 0.5 * (pymin + pymax)
    cdef double xrot = xcen * cos_theta + ycen * sin_theta
    cdef double yrot = -xcen * sin_theta + ycen * cos_theta
    cdef double axrot = fabs(xrot)
    cdef double along = fmax(axrot - half_straight, 0.0)
    cdef double d2 = along * along + yrot * yrot
    cdef double r_outer = radius + margin
    cdef double r_inner = radius - margin
    cdef double end_x, end_y, area
    cdef double poly_x[4]
    cdef double poly_y[4]
    cdef double buf_a_x[32]
    cdef double buf_a_y[32]
    cdef double buf_b_x[32]
    cdef double buf_b_y[32]

    # Every point of the pixel lies within ``margin`` of its center,
    # and the distance to the trail segment changes no faster than the
    # position, so the pixel is wholly outside or wholly inside the
    # pill when the center distance clears the radius by ``margin``.
    if d2 >= r_outer * r_outer:
        return 0.0
    if r_inner > 0.0 and d2 <= r_inner * r_inner:
        return 1.0

    if not use_exact:
        return pill_overlap_single_subpixel(pxmin, pymin, pxmax, pymax,
                                            half_straight, radius,
                                            cos_theta, sin_theta,
                                            subpixels)

    # The pixel touches only the straight section.
    if axrot <= half_straight - margin:
        rect_vertices(half_straight, radius, cos_theta, sin_theta,
                      poly_x, poly_y)
        return (polygon_pixel_overlap(pxmin, pymin, pxmax, pymax,
                                      poly_x, poly_y, 4, buf_a_x,
                                      buf_a_y, buf_b_x, buf_b_y, 32)
                / (dx * dy))

    # The pixel touches only one end, so shift it to that end's center.
    if axrot >= half_straight + margin:
        if xrot > 0.0:
            end_x = half_straight * cos_theta
            end_y = half_straight * sin_theta
        else:
            end_x = -half_straight * cos_theta
            end_y = -half_straight * sin_theta
        return (circle_overlap_single_exact(pxmin - end_x, pymin - end_y,
                                            pxmax - end_x, pymax - end_y,
                                            radius)
                / (dx * dy))

    # The pixel straddles a junction line. The clamp removes rounding
    # outside the valid range of the area.
    area = pill_junction_overlap_exact(pxmin, pymin, pxmax, pymax,
                                       half_straight, radius, cos_theta,
                                       sin_theta)
    return fmin(fmax(area, 0.0), dx * dy) / (dx * dy)
