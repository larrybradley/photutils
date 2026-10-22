# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the private cutouts module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Polynomial2D, Sersic2D
from numpy.testing import assert_allclose, assert_array_equal

from photutils.simulation import _cutouts
from photutils.simulation._cutouts import (integrate_cells, pixel_nodes,
                                           render_cells, validate_n_nodes)


def brute_force_render(model, shape, x_0, y_0, n_nodes=24, n_core=64):
    """
    Render a model with a brute-force integration.

    This reference is independent of the module under test. Every
    pixel is integrated with ``n_nodes`` Gauss-Legendre nodes per axis.
    The 3 x 3 pixels around the pixel containing the center are then
    split ``n_core`` x ``n_core`` and integrated again.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    offsets = 0.5 * nodes
    weights = 0.5 * weights

    def mean(xc, yc, size):
        xs = xc[..., None, None] + size * offsets[None, :]
        ys = yc[..., None, None] + size * offsets[:, None]
        values = model(xs, ys) * (weights[:, None] * weights[None, :])
        return values.sum(axis=(-2, -1))

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    result = mean(xx, yy, 1.0)

    cx = int(np.floor(x_0 + 0.5))
    cy = int(np.floor(y_0 + 0.5))
    sub = (np.arange(n_core) + 0.5) / n_core - 0.5
    for jy in range(cy - 1, cy + 2):
        for jx in range(cx - 1, cx + 2):
            if not (0 <= jy < shape[0] and 0 <= jx < shape[1]):
                continue
            xs = jx + sub[None, :] + np.zeros((n_core, 1))
            ys = jy + sub[:, None] + np.zeros((1, n_core))
            result[jy, jx] = mean(xs, ys, 1.0 / n_core).mean()
    return result


def make_grid(shape):
    """
    Return the pixel-center coordinate grids for an image shape.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    return xx, yy


def test_pixel_nodes():
    """
    Test the quadrature nodes and weights.
    """
    offsets, weights = pixel_nodes(4)
    assert_allclose(weights.sum(), 1.0, rtol=1e-14)
    assert_allclose(offsets, -offsets[::-1], atol=1e-15)
    assert np.all(np.abs(offsets) < 0.5)
    assert not offsets.flags.writeable
    assert not weights.flags.writeable


@pytest.mark.parametrize('value', [None, 1, 4, np.int64(8)])
def test_validate_n_nodes(value):
    """
    Test that valid n_nodes values are accepted.
    """
    assert validate_n_nodes(value) == value


@pytest.mark.parametrize('value', [0, -1, 2.5, True, '4'])
def test_validate_n_nodes_invalid(value):
    """
    Test that invalid n_nodes values raise a ValueError.
    """
    match = 'n_nodes must be None or a positive integer'
    with pytest.raises(ValueError, match=match):
        validate_n_nodes(value)


def test_render_cells_center_matches_point_evaluation():
    """
    Test that n_nodes=None is exactly the point evaluation.
    """
    model = Sersic2D(amplitude=1.0, r_eff=3.0, n=2.0, x_0=10.3, y_0=9.6)
    x, y = make_grid((21, 21))
    result = render_cells(model, x, y, 10.3, 9.6, n_nodes=None)
    assert_array_equal(result, model(x, y))


def test_integrate_cells_polynomial_exact():
    """
    Test that two nodes per axis integrate a cubic exactly.
    """
    coeffs = {'c0_0': 1.5, 'c1_0': -0.7, 'c2_0': 0.3, 'c3_0': 0.11,
              'c0_1': 0.9, 'c0_2': -0.4, 'c0_3': 0.07, 'c1_1': 0.6,
              'c2_1': -0.2, 'c1_2': 0.15}
    model = Polynomial2D(degree=3, **coeffs)
    x, y = make_grid((7, 9))
    x += 0.2
    y -= 0.35
    size = 0.8

    def moment(center, power):
        """
        Mean of u**power over a cell of side ``size``.
        """
        return (1.0, center, center**2 + size**2 / 12.0,
                center**3 + center * size**2 / 4.0)[power]

    expected = np.zeros_like(x)
    for name, coeff in coeffs.items():
        i, j = int(name[1]), int(name[3])
        expected += coeff * moment(x, i) * moment(y, j)

    result = integrate_cells(model, x, y, size, 2)
    assert result.shape == x.shape
    assert_allclose(result, expected, rtol=1e-14)


def test_integrate_cells_row_by_row(monkeypatch):
    """
    Test that large inputs, evaluated one row of nodes at a time, match.
    """
    model = Sersic2D(amplitude=1.0, r_eff=3.0, n=2.0, x_0=10.3, y_0=9.6)
    x, y = make_grid((21, 21))
    expected = integrate_cells(model, x, y, 1.0, 4)
    monkeypatch.setattr(_cutouts, '_MAX_QUADRATURE_ELEMENTS', 10)
    result = integrate_cells(model, x, y, 1.0, 4)
    assert_allclose(result, expected, rtol=1e-14)


@pytest.mark.parametrize('center', [(20.0, 20.0), (20.5, 20.0),
                                    (20.5, 20.5)])
def test_core_refinement_accuracy(center):
    """
    Test the accuracy for a Sersic profile centered on a pixel, an edge,
    and a corner.
    """
    x_0, y_0 = center
    model = Sersic2D(amplitude=1.0, r_eff=3.0, n=4, x_0=x_0, y_0=y_0,
                     ellip=0.3, theta=0.5)
    shape = (41, 41)
    x, y = make_grid(shape)
    result = render_cells(model, x, y, x_0, y_0, n_nodes=4)
    reference = brute_force_render(model, shape, x_0, y_0)
    error = np.abs(result - reference).max() / reference.max()
    assert error < 1e-3


def test_core_refinement_extreme():
    """
    Test the accuracy for a very compact profile on a pixel corner.
    """
    x_0, y_0 = 20.5, 20.5
    model = Sersic2D(amplitude=1.0, r_eff=0.2, n=4, x_0=x_0, y_0=y_0,
                     ellip=0.3, theta=0.5)
    shape = (41, 41)
    x, y = make_grid(shape)
    result = render_cells(model, x, y, x_0, y_0, n_nodes=8)
    reference = brute_force_render(model, shape, x_0, y_0)
    error = np.abs(result - reference).max() / reference.max()
    assert error < 2e-3


def test_core_refinement_sub_pixel_cells():
    """
    Test that cells smaller than a pixel are refined and conserve flux.
    """
    x_0, y_0 = 10.5, 10.5
    model = Sersic2D(amplitude=1.0, r_eff=1.0, n=4, x_0=x_0, y_0=y_0)
    size = 0.25
    edge = (np.arange(80) + 0.5) * size - 0.5
    x, y = np.meshgrid(edge, edge)
    result = render_cells(model, x, y, x_0, y_0, cell_size=size, n_nodes=4)
    reference = integrate_cells(model, x, y, size, 24)
    assert np.abs(result - reference).max() / reference.max() < 1e-3
    # Refinement leaves cells outside the core unchanged
    outside = (np.abs(x - 11) >= 1.5) | (np.abs(y - 11) >= 1.5)
    assert_array_equal(result[outside],
                       integrate_cells(model, x, y, size, 4)[outside])


def test_core_off_cutout():
    """
    Test a center outside the grid, which has no core cells to refine.
    """
    x_0, y_0 = 60.0, 60.0
    model = Sersic2D(amplitude=1.0, r_eff=3.0, n=1, x_0=x_0, y_0=y_0)
    x, y = make_grid((41, 41))
    result = render_cells(model, x, y, x_0, y_0, n_nodes=4)
    assert_array_equal(result, integrate_cells(model, x, y, 1.0, 4))
