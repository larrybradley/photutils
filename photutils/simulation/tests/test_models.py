# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the models module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Sersic2D
from astropy.table import QTable
from numpy.testing import assert_allclose
from scipy.special import gamma, gammainc, gammaincinv

from photutils.simulation import SersicSource, make_model_image


def _amplitude(flux, r_eff, n, ellip):
    """
    Return the Sersic2D amplitude for the given total flux.
    """
    b_n = gammaincinv(2 * n, 0.5)
    return flux / (2 * np.pi * n * gamma(2 * n) * np.exp(b_n)
                   * b_n**(-2 * n) * r_eff**2 * (1 - ellip))


def _render_total(model, half_width=30):
    """
    Render a model at the center of a square image and return its sum.
    """
    size = 2 * half_width + 1
    params = QTable()
    params['x_0'] = [half_width]
    params['y_0'] = [half_width]
    image = make_model_image((size, size), model, params,
                             model_shape=(size, size), n_nodes=8)
    return image.sum()


@pytest.mark.parametrize('n', [0.5, 1, 2, 4, 6])
def test_sersic_total_flux(n):
    """
    Test that the rendered total flux lies between the enclosed flux
    of the inscribed circle and that of the circumscribed circle.
    """
    half_width = 30
    r_eff = 3
    flux = 100
    model = SersicSource(flux=flux, r_eff=r_eff, n=n)
    total = _render_total(model, half_width)

    b_n = gammaincinv(2 * n, 0.5)

    def enclosed(radius):
        return gammainc(2 * n, b_n * (radius / r_eff)**(1 / n))

    lower = flux * enclosed(half_width) * (1 - 1e-3)
    upper = flux * enclosed(half_width * np.sqrt(2)) * (1 + 1e-3)
    assert lower <= total <= upper


def test_sersic_matches_sersic2d():
    """
    Test that the model equals Sersic2D with the computed amplitude.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, 100)
    y = rng.uniform(-10, 10, 100)
    kwargs = {'x_0': 0.7, 'y_0': -0.4, 'r_eff': 2.5, 'n': 3.2,
              'ellip': 0.3, 'theta': 0.6}
    model = SersicSource(flux=50, **kwargs)
    amplitude = _amplitude(50, kwargs['r_eff'], kwargs['n'],
                           kwargs['ellip'])
    expected = Sersic2D(amplitude=amplitude, **kwargs)(x, y)
    assert_allclose(model(x, y), expected, rtol=1e-14)


def test_sersic_ellip_flux():
    """
    Test that an elliptical profile has the same total flux as the
    circular case.
    """
    circular = _render_total(SersicSource(flux=100, r_eff=3, n=1, ellip=0))
    elliptical = _render_total(SersicSource(flux=100, r_eff=3, n=1,
                                            ellip=0.5, theta=0.4))
    assert_allclose(elliptical, circular, rtol=1e-3)


def test_sersic_bounding_box():
    """
    Test the default and keyword bounding box half-widths.
    """
    model = SersicSource(x_0=5, y_0=-2, r_eff=2.5)
    ((y_min, y_max), (x_min, x_max)) = model.bounding_box.bounding_box()
    assert_allclose([y_min, y_max], [-2 - 25, -2 + 25])
    assert_allclose([x_min, x_max], [5 - 25, 5 + 25])

    ((y_min, y_max), (x_min, x_max)) = model.bounding_box(factor=5)
    assert_allclose([y_min, y_max], [-2 - 12.5, -2 + 12.5])
    assert_allclose([x_min, x_max], [5 - 12.5, 5 + 12.5])


def test_sersic_bbox_factor_in_make_model_image():
    """
    Test that bbox_factor reaches the model bounding box.
    """
    model = SersicSource(r_eff=2)
    params = QTable()
    params['x_0'] = [30.0]
    params['y_0'] = [30.0]
    image = make_model_image((61, 61), model, params, bbox_factor=5)
    assert image[30, 30 + 11] == 0.0
    assert image[30, 30 + 9] > 0.0


def test_sersic_units():
    """
    Test that the units on flux propagate to the output.
    """
    model = SersicSource(flux=10 * u.nJy, r_eff=2, n=2)
    result = model(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.0, 0.0]))
    assert result.unit == u.nJy
    plain = SersicSource(flux=10, r_eff=2, n=2)(np.array([0.0, 1.0, 2.0]),
                                                np.array([0.0, 0.0, 0.0]))
    assert_allclose(result.value, plain)
