# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _pill_overlap module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from photutils.geometry import circular_overlap_grid, rectangular_overlap_grid
from photutils.geometry._pill_overlap import pill_overlap_grid

CASES = [(12.0, 5.0, 0.0), (12.0, 5.0, 0.6), (9.3, 2.2, 2.1),
         (6.0, 5.0, -1.0), (30.0, 1.0, 0.3), (3.0, 0.4, 1.3),
         (0.9, 0.3, 0.7), (1.0001, 1.0, 0.3), (20.0, 7.5, np.pi / 2)]


def pill_area(length, width):
    return (length - width) * width + np.pi * (0.5 * width) ** 2


def grid_edges(length):
    half = np.ceil(0.5 * length) + 2
    return (-half - 0.37, half - 0.37, -half + 0.21, half + 0.21,
            int(2 * half), int(2 * half))


@pytest.mark.parametrize(('length', 'width', 'theta'), CASES)
def test_exact_area(length, width, theta):
    frac = pill_overlap_grid(*grid_edges(length), length, width, theta,
                             1, 1)
    assert_allclose(frac.sum(), pill_area(length, width), rtol=1e-10)
    assert frac.min() >= 0.0
    assert frac.max() <= 1.0 + 1e-14


def test_circle_limit():
    edges = (-5.3, 4.7, -4.6, 5.4, 10, 10)
    pill = pill_overlap_grid(*edges, 6.0, 6.0, 0.4, 1, 1)
    circ = circular_overlap_grid(*edges, 3.0, 1, 1)
    assert_allclose(pill, circ, atol=1e-12)


def test_rectangle_limit():
    edges = (-10.3, 9.7, -9.6, 10.4, 20, 20)
    pill = pill_overlap_grid(*edges, 60.0, 4.0, 0.5, 1, 1)
    rect = rectangular_overlap_grid(*edges, 60.0, 4.0, 0.5, 1, 1)
    assert_allclose(pill, rect, atol=1e-12)


@pytest.mark.parametrize(('length', 'width', 'theta'), CASES[:4])
def test_subpixel_converges(length, width, theta):
    frac = pill_overlap_grid(*grid_edges(length), length, width, theta,
                             0, 50)
    assert_allclose(frac.sum(), pill_area(length, width), rtol=1e-3)


def test_center_matches_point_test():
    length, width, theta = 12.0, 5.0, 0.6
    xmin, xmax, ymin, ymax, nx, ny = grid_edges(length)
    frac = pill_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, length,
                             width, theta, 0, 1)
    x = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    y = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    xx, yy = np.meshgrid(x, y)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    along = np.maximum(np.abs(xr) - 0.5 * (length - width), 0.0)
    expected = (along**2 + yr**2 < (0.5 * width) ** 2).astype(float)
    assert_array_equal(frac, expected)


def test_outside_grid_is_zero():
    frac = pill_overlap_grid(50.0, 60.0, 50.0, 60.0, 10, 10, 12.0, 5.0,
                             0.6, 1, 1)
    assert_array_equal(frac, 0.0)


def integer_grid(xcen, ycen, length):
    """
    Return the grid edges, relative to the pill center, of a grid with
    pixel centers at integer positions around a pill centered at
    ``(xcen, ycen)``.
    """
    half = int(np.ceil(0.5 * length)) + 2
    ixmin = int(np.floor(xcen)) - half
    iymin = int(np.floor(ycen)) - half
    n_pixels = 2 * half + 1
    xmin = ixmin - 0.5 - xcen
    ymin = iymin - 0.5 - ycen
    return (xmin, xmin + n_pixels, ymin, ymin + n_pixels, n_pixels,
            n_pixels)


def check_exact_sum(xcen, ycen, length, width, theta):
    frac = pill_overlap_grid(*integer_grid(xcen, ycen, length), length,
                             width, theta, 1, 1)
    assert_allclose(frac.sum(), pill_area(length, width), rtol=1e-12)
    assert frac.min() >= 0.0
    assert frac.max() <= 1.0 + 1e-14


def trail_params(x0, y0, dx, dy, width):
    """
    Return the center, length, and theta of a pill whose trail runs
    from ``(x0, y0)`` to ``(x0 + dx, y0 + dy)``.
    """
    return (x0 + 0.5 * dx, y0 + 0.5 * dy, np.hypot(dx, dy) + width,
            np.arctan2(dy, dx))


@pytest.mark.parametrize('theta_deg', [45.0, 135.0, 225.0, -45.0])
@pytest.mark.parametrize('center', [(0.0, 0.0), (0.5, 0.5)])
def test_exact_circle_junction_on_corners(theta_deg, center):
    # With length equal to width, the junction line passes through the
    # disk center. At 45 degrees it runs along a pixel diagonal, so it
    # contains pixel corners when the center is on a pixel center or
    # corner.
    theta = np.deg2rad(theta_deg)
    for width in np.arange(0.25, 8.01, 0.25):
        check_exact_sum(*center, width, width, theta)


@pytest.mark.parametrize('width', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
def test_exact_trail_between_pixel_centers(width):
    # The trail ends, and so the junction lines, pass through pixel
    # centers. When dx and dy are both odd, the junction lines also
    # pass through pixel corners.
    for dx in range(7):
        for dy in range(-7, 8):
            if dx == 0 and dy <= 0:
                continue
            xcen, ycen, length, theta = trail_params(0, 0, dx, dy, width)
            check_exact_sum(xcen, ycen, length, width, theta)


@pytest.mark.parametrize('slope', [(1, 2), (2, 1), (1, 3), (3, 2),
                                   (-1, 2), (-3, 1), (2, -5), (4, 3)])
@pytest.mark.parametrize('width', [0.5, 1.0, 2.5, 5.0])
def test_exact_rational_slope_junction_on_corners(slope, width):
    # The pill center is on a pixel corner and the junction line passes
    # through the pixel corner (a, b) at a rational slope, so it also
    # passes through every corner (a - k * q, b + k * p).
    p, q = slope
    theta = np.arctan2(q, p)
    for a, b in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 3)):
        half_straight = (a * p + b * q) / np.hypot(p, q)
        if half_straight <= 0:
            continue
        length = 2.0 * half_straight + width
        check_exact_sum(0.5, 0.5, length, width, theta)


# The center, length, width, and theta of pills for which the triangle
# fan formulation of the half-disk terms returned up to the whole disk
# area for a single pixel.
JUNCTION_REGRESSION_CASES = [
    (20.0, 20.0, 6.0, 6.0, np.deg2rad(-45.0)),
    (20.0, 20.0, 1.0, 1.0, np.deg2rad(45.0)),
    (20.0, 20.0, np.hypot(6, 6) + 6, 6.0, np.arctan2(6, 6)),
    (20.0, 20.0, np.hypot(2, 2) + 1, 1.0, np.arctan2(2, 2)),
    (20.5, 21.5, np.hypot(1, 3) + 1.5, 1.5, np.arctan2(3, 1)),
    (21.5, 23.5, np.hypot(3, 7) + 6, 6.0, np.arctan2(7, 3)),
]


@pytest.mark.parametrize(('xcen', 'ycen', 'length', 'width', 'theta'),
                         JUNCTION_REGRESSION_CASES)
def test_exact_junction_regression(xcen, ycen, length, width, theta):
    check_exact_sum(xcen, ycen, length, width, theta)


def test_invalid_inputs():
    args = (-5.0, 5.0, -5.0, 5.0, 10, 10, 6.0, 3.0, 0.0)
    with pytest.raises(ValueError, match='use_exact must be 0 or 1'):
        pill_overlap_grid(*args, 2, 1)
    match = 'subpixels must be a strictly positive integer'
    with pytest.raises(ValueError, match=match):
        pill_overlap_grid(*args, 0, 0)
