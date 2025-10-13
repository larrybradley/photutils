# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the utils module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.table import QTable
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf import CircularGaussianPRF, make_psf_model_image
from photutils.psf.utils import (ModelImageGenerator,
                                 _get_psf_model_main_params,
                                 _interpolate_missing_data,
                                 _validate_psf_model, fit_2dgaussian, fit_fwhm)


@pytest.fixture(name='test_data')
def fixture_test_data():
    psf_model = CircularGaussianPRF()
    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700), fwhm=(2.7, 2.7),
                                             min_separation=10, seed=0)
    return data, true_params


@pytest.mark.parametrize('fix_fwhm', [False, True])
def test_fit_2dgaussian_single(fix_fwhm):
    yy, xx = np.mgrid[:51, :51]
    fwhm = 3.123
    model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=fwhm)
    data = model(xx, yy)

    fit = fit_2dgaussian(data, fwhm=3, fix_fwhm=fix_fwhm)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == 1
    if fix_fwhm:
        assert 'fwhm_fit' not in fit_tbl.colnames
    else:
        assert 'fwhm_fit' in fit_tbl.colnames
        assert_allclose(fit_tbl['fwhm_fit'], fwhm)

    # test with NaNs
    data[22, 29] = np.nan
    match = 'Input data contains non-finite values'
    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        fit = fit_2dgaussian(data, fwhm=3, fix_fwhm=fix_fwhm)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == 1

    # test with NaNs and mask
    data[22, 29] = np.nan
    mask = np.isnan(data)
    fit = fit_2dgaussian(data, fwhm=3, fix_fwhm=fix_fwhm, mask=mask)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == 1


@pytest.mark.parametrize(('fix_fwhm', 'with_units'),
                         [(False, True), (True, False)])
def test_fit_2dgaussian_multiple(test_data, fix_fwhm, with_units):
    data, sources = test_data

    unit = u.nJy
    if with_units:
        data = data * unit

    xypos = list(zip(sources['x_0'], sources['y_0'], strict=True))
    fit = fit_2dgaussian(data, xypos=xypos, fit_shape=(5, 5),
                         fix_fwhm=fix_fwhm)
    fit_tbl = fit.results
    assert isinstance(fit_tbl, QTable)
    assert len(fit_tbl) == len(sources)
    if fix_fwhm:
        assert 'fwhm_fit' not in fit_tbl.colnames
    else:
        assert 'fwhm_fit' in fit_tbl.colnames
        assert_allclose(fit_tbl['fwhm_fit'], sources['fwhm'])

    if with_units:
        for column in fit_tbl.colnames:
            if 'flux' in column:
                assert fit_tbl['flux_fit'].unit == unit


def test_fit_fwhm_single():
    yy, xx = np.mgrid[:51, :51]
    fwhm0 = 3.123
    model = CircularGaussianPRF(x_0=22.17, y_0=28.87, fwhm=fwhm0)
    data = model(xx, yy)

    fwhm = fit_fwhm(data, fwhm=3)
    assert isinstance(fwhm, np.ndarray)
    assert len(fwhm) == 1
    assert_allclose(fwhm, fwhm0)

    # test warning message
    match = 'may not have converged. Please carefully check your results'
    with pytest.warns(AstropyUserWarning, match=match):
        fwhm = fit_fwhm(np.zeros(data.shape) + 1)
    assert len(fwhm) == 1


@pytest.mark.parametrize('with_units', [False, True])
def test_fit_fwhm_multiple(test_data, with_units):
    data, sources = test_data

    unit = u.nJy
    if with_units:
        data = data * unit

    xypos = list(zip(sources['x_0'], sources['y_0'], strict=True))
    fwhms = fit_fwhm(data, xypos=xypos, fit_shape=(5, 5))
    assert isinstance(fwhms, np.ndarray)
    assert len(fwhms) == len(sources)
    assert_allclose(fwhms, sources['fwhm'])


def test_interpolate_missing_data():
    data = np.arange(100).reshape(10, 10)
    mask = np.zeros_like(data, dtype=bool)
    mask[5, 5] = True

    data_int = _interpolate_missing_data(data, mask, method='nearest')
    assert 54 <= data_int[5, 5] <= 56

    data_int = _interpolate_missing_data(data, mask, method='cubic')
    assert 54 <= data_int[5, 5] <= 56

    match = 'data must be a 2D array'
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(np.arange(10), mask)

    match = 'mask and data must have the same shape'
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask[1:, :])

    match = 'Unsupported interpolation method'
    with pytest.raises(ValueError, match=match):
        _interpolate_missing_data(data, mask, method='invalid')


def test_validate_psf_model():
    model = np.arange(10)

    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        _validate_psf_model(model)

    match = 'psf_model must be two-dimensional'
    model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _validate_psf_model(model)

    match = 'psf_model must be two-dimensional'
    model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _validate_psf_model(model)


def test_get_psf_model_main_params():
    model = CircularGaussianPRF(fwhm=1.0)
    params = _get_psf_model_main_params(model)
    assert len(params) == 3
    assert params == ('x_0', 'y_0', 'flux')

    match = 'Invalid PSF model - could not find PSF parameter names'
    model = Gaussian2D()
    with pytest.raises(ValueError, match=match):
        _get_psf_model_main_params(model)

    set_params = ('x_mean', 'y_mean', 'amplitude')
    model.x_name = set_params[0]
    model.y_name = set_params[1]
    model.flux_name = set_params[2]
    params = _get_psf_model_main_params(model)
    assert len(params) == 3
    assert params == set_params


class TestModelImageGenerator:
    """
    Tests for the ModelImageGenerator class.
    """

    def setup_class(self):
        """
        Set up test data.
        """
        # Create a simple PSF model and parameters
        self.psf_model = CircularGaussianPRF(fwhm=3.0)
        self.shape = (50, 50)

        # Create model parameters table
        self.model_params = QTable()
        self.model_params['x_0'] = [15.0, 35.0]
        self.model_params['y_0'] = [20.0, 30.0]
        self.model_params['flux'] = [100.0, 150.0]

        # Create local background values
        self.local_bkg = np.array([5.0, 8.0])

    def test_initialization(self):
        """
        Test ModelImageGenerator initialization.
        """
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            self.local_bkg,
            progress_bar=False,
        )
        assert gen.psf_model == self.psf_model
        assert gen.model_params is self.model_params
        assert_allclose(gen.local_bkg, self.local_bkg)
        assert gen.progress_bar is False

    def test_make_model_image(self):
        """
        Test model image generation without local background.
        """
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            self.local_bkg,
        )
        model_image = gen.make_model_image(self.shape)
        assert model_image.shape == self.shape
        assert np.all(np.isfinite(model_image))
        assert np.sum(model_image) > 0

    def test_make_model_image_with_localbkg(self):
        """
        Test model image generation with local background.
        """
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            self.local_bkg,
        )
        model_image = gen.make_model_image(self.shape, include_localbkg=True)
        assert model_image.shape == self.shape
        assert np.all(np.isfinite(model_image))
        # With local background, the image should have higher values
        model_image_no_bkg = gen.make_model_image(
            self.shape, include_localbkg=False)
        assert np.sum(model_image) > np.sum(model_image_no_bkg)

    def test_make_model_image_nonfinite_localbkg(self):
        """
        Test model image with non-finite local background values.
        """
        # Create local background with NaN and inf
        local_bkg_bad = np.array([np.nan, np.inf])
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            local_bkg_bad,
        )
        # Should not raise an error and should treat non-finite as zero
        model_image = gen.make_model_image(self.shape, include_localbkg=True)
        assert model_image.shape == self.shape
        assert np.all(np.isfinite(model_image))

        # Should be same as without local background
        model_image_no_bkg = gen.make_model_image(
            self.shape, include_localbkg=False)
        assert_allclose(model_image, model_image_no_bkg)

    def test_make_residual_image(self):
        """
        Test residual image generation.
        """
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            self.local_bkg,
        )
        # Create a data array
        data = np.ones(self.shape) * 10.0
        residual = gen.make_residual_image(data)
        assert residual.shape == self.shape
        assert np.all(np.isfinite(residual))

    def test_make_residual_image_with_localbkg(self):
        """
        Test residual image with local background.
        """
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            self.local_bkg,
        )
        data = np.ones(self.shape) * 10.0
        residual_with_bkg = gen.make_residual_image(
            data, include_localbkg=True)
        residual_no_bkg = gen.make_residual_image(
            data, include_localbkg=False)
        assert residual_with_bkg.shape == self.shape
        assert residual_no_bkg.shape == self.shape
        assert np.all(np.isfinite(residual_with_bkg))
        assert np.all(np.isfinite(residual_no_bkg))

    def test_make_residual_image_nonfinite_localbkg(self):
        """
        Test residual image with non-finite local background.
        """
        local_bkg_bad = np.array([np.nan, np.inf])
        gen = ModelImageGenerator(
            self.psf_model,
            self.model_params,
            local_bkg_bad,
        )
        data = np.ones(self.shape) * 10.0
        residual = gen.make_residual_image(data, include_localbkg=True)
        assert residual.shape == self.shape
        assert np.all(np.isfinite(residual))

        # Should be same as without local background
        residual_no_bkg = gen.make_residual_image(
            data, include_localbkg=False)
        assert_allclose(residual, residual_no_bkg)
