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


def test_invalid_inputs():
    args = (-5.0, 5.0, -5.0, 5.0, 10, 10, 6.0, 3.0, 0.0)
    with pytest.raises(ValueError, match='use_exact must be 0 or 1'):
        pill_overlap_grid(*args, 2, 1)
    match = 'subpixels must be a strictly positive integer'
    with pytest.raises(ValueError, match=match):
        pill_overlap_grid(*args, 0, 0)
