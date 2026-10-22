# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the snr module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_array_equal

from photutils.psf import CircularGaussianPRF, CircularGaussianPSF
from photutils.simulation import (SersicSource, make_convolved_model_image,
                                  make_model_image, optimal_snr)


def make_error_image(shape, seed=0):
    """
    Make a positive error image that varies from pixel to pixel.
    """
    return np.random.default_rng(seed).uniform(1.0, 3.0, size=shape)


def make_sersic_params():
    """
    Make a table with three Sersic sources.
    """
    params = QTable()
    params['x_0'] = [20.3, 41.6, 30.1]
    params['y_0'] = [25.8, 34.2, 20.7]
    params['flux'] = [100.0, 250.0, 60.0]
    params['r_eff'] = [1.5, 2.5, 3.0]
    params['n'] = [1.0, 2.5, 1.5]
    params['ellip'] = [0.2, 0.4, 0.1]
    params['theta'] = [0.3, 1.2, 2.0]
    return params


def test_optimal_snr_gaussian_analytic():
    """
    Test the SNR of a Gaussian PRF against the analytic matched-filter
    value.

    The sum of the squared pixel fluxes of a Gaussian of variance
    sigma**2 is flux**2 / (4 pi (sigma**2 + 1 / 12)), where the 1 / 12
    is the variance of the pixel box.
    """
    sigma = 3.0
    model = CircularGaussianPRF(fwhm=sigma * 2.0 * np.sqrt(2.0 * np.log(2.0)))
    params = QTable()
    params['x_0'] = [40.3]
    params['y_0'] = [38.8]
    params['flux'] = [100.0]
    error = np.full((80, 80), 2.0)
    result = optimal_snr(model, params, error)
    expected = 100.0 / (2.0 * np.sqrt(4.0 * np.pi * (sigma**2 + 1.0 / 12.0)))
    assert result.shape == (1,)
    assert_allclose(result, expected, rtol=1e-4)


def test_optimal_snr_scales():
    """
    Test that doubling the flux doubles the SNR and doubling the error
    halves it.
    """
    model = CircularGaussianPRF(fwhm=3.0)
    params = QTable()
    params['x_0'] = [20.3, 30.7]
    params['y_0'] = [25.8, 12.2]
    params['flux'] = [100.0, 300.0]
    error = make_error_image((50, 50))
    snr = optimal_snr(model, params, error)

    params2 = params.copy()
    params2['flux'] *= 2
    assert_allclose(optimal_snr(model, params2, error), 2 * snr, rtol=1e-12)
    assert_allclose(optimal_snr(model, params, 2 * error), snr / 2,
                    rtol=1e-12)


@pytest.mark.parametrize('oversample', [3, 4])
def test_optimal_snr_convolved_matches_image(oversample):
    """
    Test that the SNR with a PSF model is the matched-filter sum of an
    image of the single source made by make_convolved_model_image.
    """
    params = make_sersic_params()
    shape = (60, 64)
    error = make_error_image(shape)
    source = SersicSource()
    psf_model = CircularGaussianPRF(fwhm=3.0)
    kwargs = {'oversample': oversample, 'n_nodes': 4}
    result = optimal_snr(source, params, error, psf_model=psf_model, **kwargs)
    assert result.shape == (3,)
    for i in range(len(params)):
        img = make_convolved_model_image(shape, source, params[i:i + 1],
                                         psf_model, **kwargs)
        expected = np.sqrt(np.sum(img**2 / error**2))
        assert_allclose(result[i], expected, rtol=1e-12)


def test_optimal_snr_convolved_default_n_nodes():
    """
    Test that the default n_nodes of None evaluates the source at the
    cell centers, as make_convolved_model_image does.
    """
    params = make_sersic_params()[:1]
    shape = (60, 64)
    error = make_error_image(shape)
    source = SersicSource()
    psf_model = CircularGaussianPRF(fwhm=3.0)
    result = optimal_snr(source, params, error, psf_model=psf_model)
    img = make_convolved_model_image(shape, source, params, psf_model,
                                     n_nodes=None)
    assert_allclose(result[0], np.sqrt(np.sum(img**2 / error**2)),
                    rtol=1e-12)


@pytest.mark.parametrize('n_nodes', [None, 4])
def test_optimal_snr_matches_model_image(n_nodes):
    """
    Test that the SNR without a PSF model is the matched-filter sum of
    an image of the single source made by make_model_image.
    """
    params = make_sersic_params()
    shape = (60, 64)
    error = make_error_image(shape)
    source = SersicSource()
    result = optimal_snr(source, params, error, n_nodes=n_nodes)
    for i in range(len(params)):
        img = make_model_image(shape, source, params[i:i + 1],
                               n_nodes=n_nodes)
        expected = np.sqrt(np.sum(img**2 / error**2))
        assert_allclose(result[i], expected, rtol=1e-12)


def test_optimal_snr_model_shape_column():
    """
    Test that a model_shape column limits the cutout of each source as
    it does for the image renderers.
    """
    params = make_sersic_params()
    params['model_shape'] = [9, 15, 21]
    shape = (60, 64)
    error = make_error_image(shape)
    source = SersicSource()
    result = optimal_snr(source, params, error, n_nodes=4)
    for i in range(len(params)):
        img = make_model_image(shape, source, params[i:i + 1], n_nodes=4)
        expected = np.sqrt(np.sum(img**2 / error**2))
        assert_allclose(result[i], expected, rtol=1e-12)

    params['model_shape'] = [9, 9, 9]
    smaller = optimal_snr(source, params, error, n_nodes=4)
    assert_array_equal(smaller[0], result[0])
    assert np.all(smaller[1:] < result[1:])


def test_optimal_snr_position_parameters():
    """
    Test that a model without x_0 and y_0 parameters raises an error.
    """
    params = QTable()
    params['x_mean'] = [20.3]
    params['y_mean'] = [25.8]
    params['amplitude'] = [10.0]
    match = 'model must have parameters named x_0 and y_0'
    with pytest.raises(ValueError, match=match):
        optimal_snr(Gaussian2D(), params, np.ones((50, 50)))


def test_optimal_snr_off_image_nan():
    """
    Test that a source with no overlap with the image has an SNR of NaN.
    """
    model = CircularGaussianPRF(fwhm=3.0)
    params = QTable()
    params['x_0'] = [-100.0, 20.0, 20.0, 500.0]
    params['y_0'] = [20.0, 20.0, -90.0, 20.0]
    params['flux'] = [100.0, 100.0, 100.0, 100.0]
    error = np.full((40, 50), 2.0)
    result = optimal_snr(model, params, error)
    assert_array_equal(np.isnan(result), [True, False, True, True])
    assert result[1] > 0


def test_optimal_snr_off_image_convolved():
    """
    Test the sources that are partly or wholly outside the image when a
    PSF model is given.
    """
    source = CircularGaussianPRF(fwhm=1.0)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    params = QTable()
    params['x_0'] = [-100.0, -4.0, -0.5, 25.0]
    params['y_0'] = [20.0, 20.0, 20.0, 20.0]
    params['flux'] = [100.0] * 4
    error = np.full((40, 50), 2.0)
    result = optimal_snr(source, params, error, psf_model=psf_model,
                         n_nodes=4)

    # No overlap, light that reaches the image through the PSF, half
    # of the source, and the full source
    assert np.isnan(result[0])
    assert 0 < result[1] < 0.1 * result[3]
    assert_allclose(result[2], result[3] / np.sqrt(2), rtol=0.02)


def test_optimal_snr_nonfinite_row_nan():
    """
    Test that a row with a nonfinite parameter has an SNR of NaN and
    does not change the other rows.
    """
    params = make_sersic_params()
    error = make_error_image((60, 64))
    source = SersicSource()
    psf_model = CircularGaussianPRF(fwhm=3.0)
    for kwargs in ({}, {'psf_model': psf_model}):
        expected = optimal_snr(source, params, error, **kwargs)
        bad = params.copy()
        bad['flux'][1] = np.nan
        bad['r_eff'][2] = np.inf
        result = optimal_snr(source, bad, error, **kwargs)
        assert_array_equal(np.isnan(result), [False, True, True])
        assert_array_equal(result[0], expected[0])


@pytest.mark.parametrize('kwargs', [{}, {'psf_model': CircularGaussianPRF()}])
def test_optimal_snr_bad_error_pixels(kwargs):
    """
    Test that pixels whose error is zero, negative, or nonfinite are
    excluded.
    """
    model = CircularGaussianPRF(fwhm=3.0)
    params = QTable()
    params['x_0'] = [25.3]
    params['y_0'] = [24.8]
    params['flux'] = [100.0]
    shape = (50, 50)
    error = np.full(shape, 2.0)
    if kwargs:
        img = make_convolved_model_image(shape, model, params,
                                         kwargs['psf_model'], n_nodes=None)
    else:
        img = make_model_image(shape, model, params)

    bad = np.zeros(shape, dtype=bool)
    bad[24, 25] = bad[25, 24] = bad[25, 26] = bad[23, 23] = True
    error[24, 25] = 0.0
    error[25, 24] = np.nan
    error[25, 26] = -1.0
    error[23, 23] = np.inf
    assert bad.sum() == 4

    result = optimal_snr(model, params, error, **kwargs)
    expected = np.sqrt(np.sum(img[~bad]**2 / error[~bad]**2))
    assert np.isfinite(result[0])
    assert_allclose(result[0], expected, rtol=1e-12)

    # A source whose pixels are all excluded has no signal to measure
    error[:] = 0.0
    assert_array_equal(optimal_snr(model, params, error, **kwargs), 0.0)


def test_optimal_snr_empty_table():
    """
    Test that an empty table gives an empty array.
    """
    params = make_sersic_params()[:0]
    result = optimal_snr(SersicSource(), params, np.ones((20, 20)))
    assert result.shape == (0,)


def test_optimal_snr_units_error():
    """
    Test that units in the table, the error, or the model raise an
    error.
    """
    model = CircularGaussianPRF(fwhm=3.0)
    params = QTable()
    params['x_0'] = [20.3]
    params['y_0'] = [25.8]
    params['flux'] = [100.0] * u.Jy
    error = np.full((50, 50), 2.0)
    match = 'optimal_snr does not support units'
    with pytest.raises(ValueError, match=match):
        optimal_snr(model, params, error)
    psf_model = CircularGaussianPRF()
    with pytest.raises(ValueError, match=match):
        optimal_snr(model, params, error, psf_model=psf_model)

    params['flux'] = [100.0]
    with pytest.raises(ValueError, match='error must not have units'):
        optimal_snr(model, params, error * u.Jy)

    model = CircularGaussianPRF(flux=1.0 * u.Jy)
    with pytest.raises(ValueError, match=match):
        optimal_snr(model, params[['x_0', 'y_0']], error)


def test_optimal_snr_local_bkg_error():
    """
    Test that a local_bkg column raises an error.
    """
    params = make_sersic_params()
    params['local_bkg'] = [1.0, 2.0, 3.0]
    match = 'optimal_snr does not support a local_bkg column'
    with pytest.raises(ValueError, match=match):
        optimal_snr(SersicSource(), params, np.ones((60, 64)))


@pytest.mark.parametrize('shape', [(5,), (2, 5, 5)])
def test_optimal_snr_error_shape(shape):
    """
    Test that the error must be a 2D array.
    """
    match = 'error must be a 2D array'
    with pytest.raises(ValueError, match=match):
        optimal_snr(SersicSource(), make_sersic_params(), np.ones(shape))


def test_optimal_snr_psf_model_rejected():
    """
    Test that a PSF model that is not integrated over the pixels is
    rejected.
    """
    match = 'not integrated over the pixels'
    with pytest.raises(TypeError, match=match):
        optimal_snr(SersicSource(), make_sersic_params(),
                    np.ones((60, 64)), psf_model=CircularGaussianPSF())


def test_optimal_snr_invalid_inputs():
    """
    Test that invalid n_nodes and oversample values raise errors.
    """
    params = make_sersic_params()
    error = np.ones((60, 64))
    source = SersicSource()
    with pytest.raises(ValueError, match='n_nodes must be'):
        optimal_snr(source, params, error, n_nodes=0)
    psf_model = CircularGaussianPRF()
    match = 'oversample must be a positive integer'
    with pytest.raises(ValueError, match=match):
        optimal_snr(source, params, error, psf_model=psf_model,
                    oversample=0)
    with pytest.raises(TypeError, match='params_table must be'):
        optimal_snr(source, {'x_0': [1]}, error)
