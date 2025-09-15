# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the model_helpers module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D, Moffat2D
from astropy.nddata import NDData
from astropy.table import Table
from astropy.utils.exceptions import AstropyDeprecationWarning
from numpy.testing import assert_allclose, assert_equal
from scipy.integrate import dblquad

from photutils import datasets
from photutils.detection import find_peaks
from photutils.psf import (EPSFBuilder, PRFAdapter, extract_stars,
                           grid_from_epsfs, make_psf_model)
from photutils.psf.model_helpers import _integrate_model, _InverseShift


def test_inverse_shift():
    model = _InverseShift(10)
    assert model(1) == -9.0
    assert model(-10) == -20.0
    assert model.fit_deriv(10, 1)[0] == -1.0


def test_integrate_model():
    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(0.0)
    integral = _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0')
    assert integral == 0.0

    integral = _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                                use_dblquad=True)
    assert integral == 0.0

    match = 'dx and dy must be > 0'
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         dx=-10, dy=10)
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         dx=10, dy=-10)

    match = 'subsample must be >= 1'
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean_0', y_name='y_mean_0',
                         subsample=-1)

    match = 'model x and y positions must be finite'
    model = Gaussian2D(1, np.inf, 5, 1, 1)
    with pytest.raises(ValueError, match=match):
        _integrate_model(model, x_name='x_mean', y_name='y_mean')


@pytest.fixture(name='moffat_source', scope='module')
def fixture_moffat_source():
    model = Moffat2D(alpha=4.8)

    # this is the analytic value needed to get a total flux of 1
    model.amplitude = (model.alpha - 1.0) / (np.pi * model.gamma**2)

    xx, yy = np.meshgrid(*([np.linspace(-2, 2, 100)] * 2))

    return model, (xx, yy, model(xx, yy))


def test_moffat_fitting(moffat_source):
    """
    Test fitting with a Moffat2D model.
    """
    model, (xx, yy, data) = moffat_source

    # initial Moffat2D model close to the original
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.05,
                            amplitude=model.amplitude * 1.06, alpha=4.75)

    fitter = TRFLSQFitter()
    fit = fitter(guess_moffat, xx, yy, data)
    assert_allclose(fit.parameters, model.parameters, rtol=0.01, atol=0.0005)


# we set the tolerances in flux to be 2-3% because the guessed model
# parameters are known to be wrong
@pytest.mark.parametrize(('kwargs', 'tols'),
                         [({'x_name': 'x_0', 'y_name': 'y_0',
                            'flux_name': None, 'normalize': True},
                           (1e-3, 0.02)),
                          ({'x_name': None, 'y_name': None, 'flux_name': None,
                            'normalize': True}, (1e-3, 0.02)),
                          ({'x_name': None, 'y_name': None, 'flux_name': None,
                            'normalize': False}, (1e-3, 0.03)),
                          ({'x_name': 'x_0', 'y_name': 'y_0',
                            'flux_name': 'amplitude', 'normalize': False},
                           (1e-3, None))])
def test_make_psf_model(moffat_source, kwargs, tols):
    model, (xx, yy, data) = moffat_source

    # a close-but-wrong "guessed Moffat"
    guess_moffat = Moffat2D(x_0=0.1, y_0=-0.05, gamma=1.01,
                            amplitude=model.amplitude * 1.01, alpha=4.79)
    if kwargs['normalize']:
        # definitely very wrong, so this ensures the renormalization
        # works
        guess_moffat.amplitude = 5.0

    if kwargs['x_name'] is None:
        guess_moffat.x_0 = 0
    if kwargs['y_name'] is None:
        guess_moffat.y_0 = 0

    psf_model = make_psf_model(guess_moffat, **kwargs)
    fitter = TRFLSQFitter()
    fit_model = fitter(psf_model, xx, yy, data)
    xytol, fluxtol = tols

    if xytol is not None:
        assert np.abs(getattr(fit_model, fit_model.x_name)) < xytol
        assert np.abs(getattr(fit_model, fit_model.y_name)) < xytol
    if fluxtol is not None:
        assert np.abs(1.0 - getattr(fit_model, fit_model.flux_name)) < fluxtol

    # ensure the model parameters did not change
    assert fit_model[2].gamma == guess_moffat.gamma
    assert fit_model[2].alpha == guess_moffat.alpha
    if kwargs['flux_name'] is None:
        assert fit_model[2].amplitude == guess_moffat.amplitude


def test_make_psf_model_units():
    model = Moffat2D(amplitude=1.0 * u.Jy, x_0=25, y_0=25, alpha=4.8,
                     gamma=3.1)
    model.amplitude = (model.amplitude.unit * (model.alpha - 1.0)
                       / (np.pi * model.gamma**2))  # normalize to flux=1

    psf_model = make_psf_model(model, x_name='x_0', y_name='y_0',
                               normalize=True)
    yy, xx = np.mgrid[:51, :51]
    data1 = model(xx, yy)
    data2 = psf_model(xx, yy)
    assert_allclose(data1, data2)


def test_make_psf_model_compound():
    model = (Const2D(0.0) + Const2D(1.0) + Gaussian2D(1, 5, 5, 1, 1)
             * Const2D(1.0) * Const2D(1.0))
    psf_model = make_psf_model(model, x_name='x_mean_2', y_name='y_mean_2',
                               normalize=True)
    assert psf_model.x_name == 'x_mean_4'
    assert psf_model.y_name == 'y_mean_4'
    assert psf_model.flux_name == 'amplitude_7'


def test_make_psf_model_inputs():
    model = Gaussian2D(1, 5, 5, 1, 1)
    match = 'parameter name not found in the input model'
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean')
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean', y_name='y_mean_10')


def test_make_psf_model_integral():
    model = Gaussian2D(1, 5, 5, 1, 1) * Const2D(0.0)
    match = 'Cannot normalize the model because the integrated flux is zero'
    with pytest.raises(ValueError, match=match):
        make_psf_model(model, x_name='x_mean_0', y_name='y_mean_0',
                       normalize=True)


def test_make_psf_model_offset():
    """
    Test to ensure the offset is in the correct direction.
    """
    moffat = Moffat2D(x_0=0, y_0=0, alpha=4.8)
    psfmod1 = make_psf_model(moffat.copy(), x_name='x_0', y_name='y_0',
                             normalize=False)
    psfmod2 = make_psf_model(moffat.copy(), normalize=False)
    moffat.x_0 = 10
    psfmod1.x_0_2 = 10
    psfmod2.offset_0 = 10

    assert moffat(10, 0) == psfmod1(10, 0) == psfmod2(10, 0) == 1.0


class TestGridFromEPSFsComprehensive:
    """
    Comprehensive tests for grid_from_epsfs without remote data.
    """

    def _make_mock_epsf(self, data_shape=(25, 25), x_0=0.0, y_0=0.0,
                        oversampling=4, flux=1.0, fill_value=0.0,
                        origin=None, data_unit=None):
        """
        Helper to create a mock ImagePSF for testing.
        """
        from photutils.psf import ImagePSF

        # Create a simple Gaussian-like PSF data
        y_idx, x_idx = np.indices(data_shape)
        cy, cx = (np.array(data_shape) - 1) / 2.0
        sigma = 2.0
        data = np.exp(-((x_idx - cx)**2 + (y_idx - cy)**2) / (2 * sigma**2))
        data /= np.sum(data)  # Normalize

        if data_unit is not None:
            data = data * data_unit

        if origin is None:
            origin = (cx, cy)

        return ImagePSF(data, x_0=x_0, y_0=y_0, flux=flux,
                        oversampling=oversampling, origin=origin,
                        fill_value=fill_value)

    def test_empty_list(self):
        """
        Test with empty EPSFs list.
        """
        with pytest.raises(ValueError, match='epsfs list cannot be empty'):
            grid_from_epsfs([])

    def test_single_epsf(self):
        """
        Test with single EPSF.
        """
        epsf = self._make_mock_epsf(x_0=10.0, y_0=20.0)
        result = grid_from_epsfs([epsf])

        assert result.data.shape == (1, 25, 25)
        assert result.meta['grid_xypos'] == [(10.0, 20.0)]
        assert np.array_equal(result.meta['oversampling'], [4, 4])
        assert result.meta['fill_value'] == 0.0

    def test_four_epsfs_with_positions(self):
        """
        Test with four EPSFs using x_0, y_0 from EPSFs.

        Uses rectangular grid positions: (0,0), (1,0), (0,1), (1,1).
        """
        epsf1 = self._make_mock_epsf(x_0=0.0, y_0=0.0)
        epsf2 = self._make_mock_epsf(x_0=1.0, y_0=0.0)
        epsf3 = self._make_mock_epsf(x_0=0.0, y_0=1.0)
        epsf4 = self._make_mock_epsf(x_0=1.0, y_0=1.0)

        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

        assert result.data.shape == (4, 25, 25)
        expected_positions = [(0.0, 0.0), (1.0, 0.0),
                              (0.0, 1.0), (1.0, 1.0)]
        assert result.meta['grid_xypos'] == expected_positions

    def test_custom_grid_xypos(self):
        """
        Test with custom grid_xypos parameter.

        Uses rectangular grid positions forming a 2x2 grid.
        """
        epsf1 = self._make_mock_epsf()
        epsf2 = self._make_mock_epsf()
        epsf3 = self._make_mock_epsf()
        epsf4 = self._make_mock_epsf()

        custom_positions = [(100.0, 200.0), (300.0, 200.0),
                            (100.0, 400.0), (300.0, 400.0)]
        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4],
                                 grid_xypos=custom_positions)

        assert result.meta['grid_xypos'] == custom_positions

    def test_grid_xypos_length_mismatch(self):
        """
        Test error when grid_xypos length doesn't match EPSFs.
        """
        epsf1 = self._make_mock_epsf()
        epsf2 = self._make_mock_epsf()

        with pytest.raises(ValueError,
                           match='grid_xypos must be the same length'):
            grid_from_epsfs([epsf1, epsf2], grid_xypos=[(0, 0)])

    def test_non_imagepsf_type(self):
        """
        Test error with non-ImagePSF objects.
        """
        fake_epsf = 'not_an_epsf'
        with pytest.raises(TypeError,
                           match='All input epsfs must be of type ImagePSF'):
            grid_from_epsfs([fake_epsf])

    def test_mixed_types(self):
        """
        Test error with mix of ImagePSF and other objects.
        """
        epsf = self._make_mock_epsf()
        fake_epsf = 'not_an_epsf'

        with pytest.raises(TypeError,
                           match=r'got str at index 1'):
            grid_from_epsfs([epsf, fake_epsf])

    def test_inconsistent_oversampling(self):
        """
        Test error when EPSFs have different oversampling.
        """
        epsf1 = self._make_mock_epsf(oversampling=1)  # Changed from 2 to 1
        epsf2 = self._make_mock_epsf(oversampling=4)
        epsf3 = self._make_mock_epsf(oversampling=4)
        epsf4 = self._make_mock_epsf(oversampling=4)

        with pytest.raises(ValueError,
                           match='same value for oversampling'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_inconsistent_fill_value(self):
        """
        Test error when EPSFs have different fill_value.
        """
        epsf1 = self._make_mock_epsf(fill_value=0.0)
        epsf2 = self._make_mock_epsf(fill_value=-999.0)
        epsf3 = self._make_mock_epsf(fill_value=0.0)
        epsf4 = self._make_mock_epsf(fill_value=0.0)

        with pytest.raises(ValueError,
                           match='same value for fill_value'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_inconsistent_data_dimensions(self):
        """
        Test error when EPSFs have different oversampling values.
        """
        epsf1 = self._make_mock_epsf(data_shape=(25, 25))

        # Create ImagePSF with different oversampling
        from photutils.psf import ImagePSF
        data_2d_diff = np.ones((10, 10))
        # Use different oversampling to trigger the oversampling error
        epsf2 = ImagePSF(data_2d_diff, oversampling=(2, 2))

        # This should raise an error due to different oversampling values
        with pytest.raises(ValueError, match='same value for oversampling'):
            grid_from_epsfs([epsf1, epsf2])

    def test_inconsistent_data_shapes(self):
        """
        Test error when EPSFs have different data shapes.
        """
        from photutils.psf import ImagePSF

        # Create EPSFs with same oversampling but different shapes
        data1 = np.ones((10, 10))
        data2 = np.ones((15, 15))

        # Use consistent origin to avoid origin error before shape check
        origin = (0, 0)
        epsf1 = ImagePSF(data1, oversampling=(1, 1), origin=origin)
        epsf2 = ImagePSF(data2, oversampling=(1, 1), origin=origin)

        # This should raise an error due to different array shapes
        with pytest.raises(ValueError,
                           match='all input arrays must have the same shape'):
            grid_from_epsfs([epsf1, epsf2])

    def test_inconsistent_flux(self):
        """
        Test error when EPSFs have different flux.
        """
        epsf1 = self._make_mock_epsf(flux=1.0)
        epsf2 = self._make_mock_epsf(flux=1.5)  # Different flux
        epsf3 = self._make_mock_epsf(flux=1.0)
        epsf4 = self._make_mock_epsf(flux=1.0)

        with pytest.raises(ValueError,
                           match='same value for flux'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_inconsistent_origin(self):
        """
        Test error when EPSFs have different origins and using x_0, y_0.
        """
        epsf1 = self._make_mock_epsf(origin=(10, 10))
        epsf2 = self._make_mock_epsf(origin=(20, 20))
        epsf3 = self._make_mock_epsf(origin=(10, 10))
        epsf4 = self._make_mock_epsf(origin=(10, 10))

        with pytest.raises(ValueError,
                           match='origin must match'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_consistent_origin_ok(self):
        """
        Test that consistent origins work fine with rectangular grid
        positions.
        """
        origin = (12.0, 12.0)
        epsf1 = self._make_mock_epsf(x_0=0.0, y_0=0.0, origin=origin)
        epsf2 = self._make_mock_epsf(x_0=1.0, y_0=0.0, origin=origin)
        epsf3 = self._make_mock_epsf(x_0=0.0, y_0=1.0, origin=origin)
        epsf4 = self._make_mock_epsf(x_0=1.0, y_0=1.0, origin=origin)

        # Should not raise an error
        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])
        assert result.data.shape == (4, 25, 25)

    def test_inconsistent_data_units(self):
        """
        Test error when EPSFs have different data units.
        """
        epsf1 = self._make_mock_epsf(data_unit=u.count)
        epsf2 = self._make_mock_epsf(data_unit=u.photon)
        epsf3 = self._make_mock_epsf(data_unit=u.count)
        epsf4 = self._make_mock_epsf(data_unit=u.count)

        with pytest.raises(ValueError,
                           match='same unit'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_mixed_unit_none(self):
        """
        Test error when some EPSFs have units and others don't.
        """
        epsf1 = self._make_mock_epsf(data_unit=u.count)
        epsf2 = self._make_mock_epsf(data_unit=None)
        epsf3 = self._make_mock_epsf(data_unit=u.count)
        epsf4 = self._make_mock_epsf(data_unit=u.count)

        with pytest.raises(ValueError,
                           match='same unit'):
            grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

    def test_consistent_units_ok(self):
        """
        Test that consistent units work fine with rectangular grid
        positions.
        """
        unit = u.count
        epsf1 = self._make_mock_epsf(x_0=0.0, y_0=0.0, data_unit=unit)
        epsf2 = self._make_mock_epsf(x_0=1.0, y_0=0.0, data_unit=unit)
        epsf3 = self._make_mock_epsf(x_0=0.0, y_0=1.0, data_unit=unit)
        epsf4 = self._make_mock_epsf(x_0=1.0, y_0=1.0, data_unit=unit)

        # Should not raise an error
        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])
        assert result.data.shape == (4, 25, 25)

    def test_meta_handling(self):
        """
        Test metadata handling with rectangular grid positions.
        """
        epsf1 = self._make_mock_epsf(x_0=0.0, y_0=0.0)
        epsf2 = self._make_mock_epsf(x_0=1.0, y_0=0.0)
        epsf3 = self._make_mock_epsf(x_0=0.0, y_0=1.0)
        epsf4 = self._make_mock_epsf(x_0=1.0, y_0=1.0)

        # Test with custom meta
        custom_meta = {'my_key': 'my_value',
                       'grid_xypos': 'will_be_overridden'}
        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4],
                                 meta=custom_meta)

        # Check that our custom key is preserved
        assert result.meta['my_key'] == 'my_value'

        # Check that required keys are overridden
        expected_positions = [(0.0, 0.0), (1.0, 0.0),
                              (0.0, 1.0), (1.0, 1.0)]
        assert result.meta['grid_xypos'] == expected_positions
        assert np.array_equal(result.meta['oversampling'], [4, 4])
        assert result.meta['fill_value'] == 0.0

    def test_meta_original_not_modified(self):
        """
        Test that original meta dict is not modified.
        """
        epsf = self._make_mock_epsf()

        original_meta = {'my_key': 'original_value'}
        grid_from_epsfs([epsf], meta=original_meta)

        # Original meta should be unchanged
        assert original_meta == {'my_key': 'original_value'}

    def test_data_stacking(self):
        """
        Test that data is correctly stacked with rectangular grid
        positions.
        """
        # Create EPSFs with identifiable data patterns
        data1 = np.ones((10, 10))
        data2 = np.ones((10, 10)) * 2
        data3 = np.ones((10, 10)) * 3
        data4 = np.ones((10, 10)) * 4

        from photutils.psf import ImagePSF
        epsf1 = ImagePSF(data1, x_0=0.0, y_0=0.0)
        epsf2 = ImagePSF(data2, x_0=1.0, y_0=0.0)
        epsf3 = ImagePSF(data3, x_0=0.0, y_0=1.0)
        epsf4 = ImagePSF(data4, x_0=1.0, y_0=1.0)

        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4])

        assert result.data.shape == (4, 10, 10)
        assert np.allclose(result.data[0], 1.0)
        assert np.allclose(result.data[1], 2.0)
        assert np.allclose(result.data[2], 3.0)
        assert np.allclose(result.data[3], 4.0)

    def test_different_data_shapes_with_consistent_origins(self):
        """
        Test EPSFs with different data shapes fail during stacking.
        """
        from photutils.psf import ImagePSF

        origin = (5, 5)  # Same origin for both
        data1 = np.ones((10, 15))
        data2 = np.ones((12, 18))

        epsf1 = ImagePSF(data1, x_0=0.0, y_0=0.0, origin=origin)
        epsf2 = ImagePSF(data2, x_0=1.0, y_0=0.0, origin=origin)

        # This should raise an error due to different array shapes
        with pytest.raises(ValueError,
                           match='all input arrays must have the same shape'):
            grid_from_epsfs([epsf1, epsf2])

    def test_origin_not_checked_with_custom_grid_xypos(self):
        """
        Test origin consistency not checked with custom grid_xypos.
        """
        epsf1 = self._make_mock_epsf(origin=(10, 10))
        epsf2 = self._make_mock_epsf(origin=(20, 20))  # Different origins
        epsf3 = self._make_mock_epsf(origin=(30, 30))
        epsf4 = self._make_mock_epsf(origin=(40, 40))

        custom_positions = [(0.0, 0.0), (1.0, 0.0),
                            (0.0, 1.0), (1.0, 1.0)]

        # Should not raise an error when using custom grid_xypos
        result = grid_from_epsfs([epsf1, epsf2, epsf3, epsf4],
                                 grid_xypos=custom_positions)
        assert result.meta['grid_xypos'] == custom_positions


class TestPRFAdapter:
    """
    Tests for PRFAdapter.
    """

    def normalize_moffat(self, mof):
        # this is the analytic value needed to get a total flux of 1
        mof = mof.copy()
        mof.amplitude = (mof.alpha - 1) / (np.pi * mof.gamma**2)
        return mof

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False},
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': 'amplitude',
         'renormalize_psf': False}])
    def test_create_eval_prfadapter(self, adapterkwargs):
        mof = Moffat2D(gamma=1, alpha=4.8)
        with pytest.warns(AstropyDeprecationWarning):
            prf = PRFAdapter(mof, **adapterkwargs)

        # test that these work without errors
        prf.x_0 = 0.5
        prf.y_0 = -0.5
        prf.flux = 1.2
        prf(0, 0)

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': True},
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False}])
    def test_prfadapter_integrates(self, adapterkwargs):
        mof = Moffat2D(gamma=1.5, alpha=4.8)
        if not adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)
        with pytest.warns(AstropyDeprecationWarning):
            prf1 = PRFAdapter(mof, **adapterkwargs)

        # first check that the PRF over a central grid ends up summing to the
        # integrand over the whole PSF
        xg, yg = np.meshgrid(*([(-1, 0, 1)] * 2))
        evalmod = prf1(xg, yg)

        if adapterkwargs['renormalize_psf']:
            mof = self.normalize_moffat(mof)

        integrand, itol = dblquad(mof, -1.5, 1.5, lambda _: -1.5,
                                  lambda _: 1.5)
        assert_allclose(np.sum(evalmod), integrand, atol=itol * 10)

    @pytest.mark.parametrize('adapterkwargs', [
        {'xname': 'x_0', 'yname': 'y_0', 'fluxname': None,
         'renormalize_psf': False},
        {'xname': None, 'yname': None, 'fluxname': None,
         'renormalize_psf': False}])
    def test_prfadapter_sizematch(self, adapterkwargs):
        mof1 = self.normalize_moffat(Moffat2D(gamma=1, alpha=4.8))
        with pytest.warns(AstropyDeprecationWarning):
            prf1 = PRFAdapter(mof1, **adapterkwargs)

        # now try integrating over differently-sampled PRFs
        # and check that they match
        mof2 = self.normalize_moffat(Moffat2D(gamma=2, alpha=4.8))
        with pytest.warns(AstropyDeprecationWarning):
            prf2 = PRFAdapter(mof2, **adapterkwargs)

        xg1, yg1 = np.meshgrid(*([(-0.5, 0.5)] * 2))
        xg2, yg2 = np.meshgrid(*([(-1.5, -0.5, 0.5, 1.5)] * 2))

        eval11 = prf1(xg1, yg1)
        eval22 = prf2(xg2, yg2)

        _, itol = dblquad(mof1, -2, 2, lambda _: -2, lambda _: 2)
        # it's a bit of a guess that the above itol is appropriate, but
        # it should be close
        assert_allclose(np.sum(eval11), np.sum(eval22), atol=itol * 100)
