# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the private cutouts module.
"""

import numpy as np
import pytest
from astropy.modeling import custom_model
from astropy.modeling.models import Polynomial2D, Sersic2D
from numpy.testing import assert_allclose, assert_array_equal

from photutils.simulation import _cutouts
from photutils.simulation._cutouts import (integrate_cells, pixel_nodes,
                                           render_cells, validate_n_nodes)


def cusp_nodes(lower, upper, center):
    """
    Return the nodes and weights for the mean over an interval, with
    the interval split at the source center and graded toward it.

    The interval is split into 16 equal pieces per unit length and at
    ``center`` and ``center +/- 2**-j`` for j from 0 to 30. Each piece
    is integrated with 8 Gauss-Legendre nodes.
    """
    n_uniform = max(1, int(np.ceil(16 * (upper - lower))))
    grade = 2.0 ** -np.arange(31)
    cuts = np.concatenate((
        lower + (upper - lower) * np.arange(n_uniform + 1) / n_uniform,
        [center], center - grade, center + grade))
    cuts = np.unique(np.clip(cuts, lower, upper))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    half = 0.5 * np.diff(cuts)[:, None]
    middle = 0.5 * (cuts[:-1] + cuts[1:])[:, None]
    return ((middle + half * nodes).ravel(),
            (half * weights).ravel() / (upper - lower))


def reference_render(model, x, y, x_0, y_0, cell_size=1.0):
    """
    Render a model with a converged reference integration.

    This reference is independent of the module under test. Every cell
    is integrated with 24 Gauss-Legendre nodes per axis. The cells
    whose centers are within 2.5 pixels of the source center along
    both axes are then integrated again with pieces that are split at
    the source center and graded toward it, which resolves the cusp
    of a Sersic profile at any position.
    """
    nodes, weights = np.polynomial.legendre.leggauss(24)
    offsets = 0.5 * cell_size * nodes
    weights = 0.5 * weights
    xs = x[..., None, None] + offsets[None, :]
    ys = y[..., None, None] + offsets[:, None]
    values = model(xs, ys) * (weights[:, None] * weights[None, :])
    result = values.sum(axis=(-2, -1))

    half = 0.5 * cell_size
    near = (np.abs(x - x_0) < 2.5) & (np.abs(y - y_0) < 2.5)
    for idx in zip(*np.nonzero(near), strict=True):
        x_nodes, x_weights = cusp_nodes(x[idx] - half, x[idx] + half, x_0)
        y_nodes, y_weights = cusp_nodes(y[idx] - half, y[idx] + half, y_0)
        xx, yy = np.meshgrid(x_nodes, y_nodes)
        result[idx] = np.sum(model(xx, yy) * np.outer(y_weights, x_weights))
    return result


def max_error(model, x_0, y_0, n_nodes, shape=(41, 41)):
    """
    Return the largest error of the rendered model relative to the peak.
    """
    x, y = make_grid(shape)
    result = render_cells(model, x, y, x_0, y_0, n_nodes=n_nodes)
    reference = reference_render(model, x, y, x_0, y_0)
    return np.abs(result - reference).max() / reference.max()


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
    assert max_error(model, x_0, y_0, 4) < 1e-3


def test_core_refinement_extreme():
    """
    Test the accuracy for a very compact profile on a pixel corner.
    """
    x_0, y_0 = 20.5, 20.5
    model = Sersic2D(amplitude=1.0, r_eff=0.2, n=4, x_0=x_0, y_0=y_0,
                     ellip=0.3, theta=0.5)
    assert max_error(model, x_0, y_0, 8) < 2e-3


@pytest.mark.parametrize('center', [(20.3, 19.8), (19.63, 20.17)])
@pytest.mark.parametrize(('r_eff', 'n_nodes', 'threshold'),
                         [(3.0, 4, 1e-3), (0.2, 8, 2e-3)])
def test_core_refinement_off_grid(center, r_eff, n_nodes, threshold):
    """
    Test the accuracy for Sersic profiles centered away from the pixel
    centers, edges, and corners.
    """
    x_0, y_0 = center
    model = Sersic2D(amplitude=1.0, r_eff=r_eff, n=4, x_0=x_0, y_0=y_0,
                     ellip=0.3, theta=0.5)
    assert max_error(model, x_0, y_0, n_nodes) < threshold


@pytest.mark.parametrize('center', [(10.5, 10.5), (10.37, 10.81)])
def test_core_refinement_sub_pixel_cells(center):
    """
    Test the accuracy for cells smaller than a pixel, with the source
    centered on a pixel corner and away from the pixel grid.
    """
    x_0, y_0 = center
    model = Sersic2D(amplitude=1.0, r_eff=1.0, n=4, x_0=x_0, y_0=y_0)
    size = 0.25
    edge = (np.arange(80) + 0.5) * size - 0.5
    x, y = np.meshgrid(edge, edge)
    result = render_cells(model, x, y, x_0, y_0, cell_size=size, n_nodes=4)
    reference = reference_render(model, x, y, x_0, y_0, cell_size=size)
    assert np.abs(result - reference).max() / reference.max() < 1e-3
    # Refinement leaves cells outside the core unchanged
    outside = ((np.abs(x - np.floor(x_0 + 0.5)) >= 1.5)
               | (np.abs(y - np.floor(y_0 + 0.5)) >= 1.5))
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


def test_core_clipped_at_edge():
    """
    Test a source whose core is partly outside the grid.
    """
    x_0, y_0 = 0.2, 0.3
    model = Sersic2D(amplitude=1.0, r_eff=3.0, n=4, x_0=x_0, y_0=y_0,
                     ellip=0.3, theta=0.5)
    x, y = make_grid((21, 21))
    result = render_cells(model, x, y, x_0, y_0, n_nodes=4)
    reference = reference_render(model, x, y, x_0, y_0)
    assert np.abs(result - reference).max() / reference.max() < 1e-3
    # Refinement leaves cells outside the core unchanged
    outside = (x >= 1.5) | (y >= 1.5)
    assert_array_equal(result[outside],
                       integrate_cells(model, x, y, 1.0, 4)[outside])


@pytest.mark.parametrize(('x_0', 'y_0'), [(20.5, 20.5), (20.5, 20.0),
                                          (20.0, 20.0), (20.3, 19.8)])
def test_core_never_evaluates_center(x_0, y_0):
    """
    Test that no core node lies exactly on the source center, so a
    model that is infinite there but integrable renders finite values.

    A center on a cell edge or corner gives core pieces of zero width,
    whose nodes must not be placed at the center.
    """
    @custom_model
    def inverse_radius(x, y, x_0=0.0, y_0=0.0):
        return 1.0 / np.hypot(x - x_0, y - y_0)

    model = inverse_radius(x_0=x_0, y_0=y_0)
    x, y = make_grid((41, 41))
    result = render_cells(model, x, y, x_0, y_0, n_nodes=4)
    assert np.all(np.isfinite(result))
