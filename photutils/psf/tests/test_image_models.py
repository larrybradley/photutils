# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the image_models module.
"""

import numpy as np
import pytest
from astropy.modeling.fitting import TRFLSQFitter
from numpy.testing import assert_allclose, assert_equal

from photutils.psf import CircularGaussianPSF, ImagePRF, ImagePSF


@pytest.fixture(name='gaussian_psf')
def fixture_gaussian_psf():
    return CircularGaussianPSF(fwhm=2.1)


@pytest.fixture(name='image_psf')
def fixture_image_psf(gaussian_psf):
    yy, xx = np.mgrid[-10:11, -10:11]
    psf_data = gaussian_psf(xx, yy)
    psf_data /= np.sum(psf_data)
    return ImagePSF(psf_data, interpolation='cubic')


class TestImagePSF:

    def test_imagepsf(self, gaussian_psf):
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)
        model = ImagePSF(psf_data, interpolation='cubic')

        # At integer grid positions, model should match normalized Gaussian
        expected = gaussian_psf(xx, yy)
        expected /= expected.sum()
        assert_allclose(model(xx, yy), expected, atol=1e-6)

        # Subpixel shifts should give similar shape (verify peak moves)
        for dx, dy in [(0.5, 0.5), (-0.5, 1.75)]:
            model.x_0 = dx
            model.y_0 = dy
            result = model(xx, yy)
            # The result should still be normalized (with tolerance for
            # small numerical errors from cubic interpolation)
            assert_allclose(result.sum(), 1.0, rtol=1e-5)
            # Peak should be near the shifted position
            peak_idx = np.unravel_index(np.argmax(result), result.shape)
            # Grid center is at index (10, 10) corresponding to (0, 0)
            # When model is at (dx, dy), peak in output should be at (dx, dy)
            # which is index (10 + dy, 10 + dx) since yy, xx = mgrid[-10:11]
            expected_y_idx = 10 + dy
            expected_x_idx = 10 + dx
            assert abs(peak_idx[0] - expected_y_idx) <= 1
            assert abs(peak_idx[1] - expected_x_idx) <= 1

    @pytest.mark.parametrize('oversampling', [1, 2, (2, 3)])
    @pytest.mark.parametrize('origin', [None, (11.0, 13.0)])
    def test_fit_deriv(self, oversampling, origin):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=12, y_0=12, fwhm=3.5)
        yy, xx = np.mgrid[0:25, 0:25].astype(float)
        psf_data = gaussian_psf(xx, yy)

        flux, x_0, y_0 = 2.0, 12.3, 11.7
        model = ImagePSF(psf_data, flux=flux, x_0=x_0, y_0=y_0,
                         oversampling=oversampling, origin=origin)

        # Evaluation points covering the valid model domain, with a
        # margin so that the central differences below do not cross
        # the fill_value boundary
        ny, nx = psf_data.shape
        margin = 0.5
        x_lo = x_0 - model.origin[0] / model.oversampling[1] + margin
        x_hi = (x_0 + (nx - 1 - model.origin[0]) / model.oversampling[1]
                - margin)
        y_lo = y_0 - model.origin[1] / model.oversampling[0] + margin
        y_hi = (y_0 + (ny - 1 - model.origin[1]) / model.oversampling[0]
                - margin)
        x, y = np.meshgrid(np.linspace(x_lo, x_hi, 15),
                           np.linspace(y_lo, y_hi, 15))
        x = x.ravel()
        y = y.ravel()

        d_flux, d_x_0, d_y_0 = model.fit_deriv(x, y, flux, x_0, y_0)

        eps = 1e-6

        def ev(f, a, b):
            return model.evaluate(x, y, f, a, b)

        num_flux = (ev(flux + eps, x_0, y_0)
                    - ev(flux - eps, x_0, y_0)) / (2 * eps)
        num_x_0 = (ev(flux, x_0 + eps, y_0)
                   - ev(flux, x_0 - eps, y_0)) / (2 * eps)
        num_y_0 = (ev(flux, x_0, y_0 + eps)
                   - ev(flux, x_0, y_0 - eps)) / (2 * eps)

        assert_allclose(d_flux, num_flux, atol=1e-8)
        assert_allclose(d_x_0, num_x_0, atol=1e-7)
        assert_allclose(d_y_0, num_y_0, atol=1e-7)

    def test_fit_deriv_out_of_bounds(self):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=5, y_0=5, fwhm=2.0)
        yy, xx = np.mgrid[0:11, 0:11].astype(float)
        psf_data = gaussian_psf(xx, yy)
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0)

        # Include positions that map outside the input pixel grid
        x = np.array([5.0, -50.0, 100.0])
        y = np.array([5.0, 5.0, 5.0])
        derivs = model.fit_deriv(x, y, 1.0, 5.0, 5.0)

        # All derivatives must be zero outside the input pixel grid
        for deriv in derivs:
            assert_equal(deriv[1:], 0.0)
        # The flux derivative is nonzero at the in-bounds peak position
        assert derivs[0][0] > 0.0

        # With fill_value=None, out-of-bounds derivatives are
        # extrapolated from the spline fit instead of being zeroed
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0,
                         fill_value=None)
        derivs = model.fit_deriv(np.array([12.0]), np.array([5.0]),
                                 1.0, 5.0, 5.0)
        assert np.all(np.isfinite([deriv[0] for deriv in derivs]))
        assert derivs[1][0] != 0.0

    def test_fit_deriv_scalar(self):
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=5, y_0=5, fwhm=2.0)
        yy, xx = np.mgrid[0:11, 0:11].astype(float)
        psf_data = gaussian_psf(xx, yy)
        model = ImagePSF(psf_data, flux=1.0, x_0=5.0, y_0=5.0)

        # Scalar inputs are promoted to 1D arrays, matching evaluate
        derivs = model.fit_deriv(4.5, 5.5, 1.0, 5.0, 5.0)
        expected = model.fit_deriv(np.array([4.5]), np.array([5.5]),
                                   1.0, 5.0, 5.0)
        for deriv, exp in zip(derivs, expected, strict=True):
            assert deriv.shape == (1,)
            assert_allclose(deriv, exp)

        # Scalar out-of-bounds inputs give zero derivatives
        derivs = model.fit_deriv(-50.0, 5.0, 1.0, 5.0, 5.0)
        for deriv in derivs:
            assert_equal(deriv, 0.0)

    def test_fit_deriv_fitting(self):
        """
        Test that fitting with the analytic Jacobian recovers the true
        parameters and matches the finite-difference approximation.
        """
        gaussian_psf = CircularGaussianPSF(flux=1, x_0=12, y_0=12, fwhm=3.5)
        yy, xx = np.mgrid[0:25, 0:25].astype(float)
        psf_data = gaussian_psf(xx, yy)

        truth = ImagePSF(psf_data, flux=250.0, x_0=11.6, y_0=12.4)
        rng = np.random.default_rng(0)
        data = truth(xx, yy) + rng.normal(0.0, 0.02, xx.shape)

        assert ImagePSF.fit_deriv is not None
        fit_params = []
        for estimate_jacobian in (False, True):
            init = ImagePSF(psf_data, flux=200.0, x_0=12.0, y_0=12.0)
            fitter = TRFLSQFitter()
            fit = fitter(init, xx.ravel(), yy.ravel(), data.ravel(),
                         estimate_jacobian=estimate_jacobian)
            fit_params.append(fit.parameters)

        assert_allclose(fit_params[0], fit_params[1], rtol=1e-5)
        assert_allclose(fit_params[0], (250.0, 11.6, 12.4), rtol=1e-2)

    def test_imagepsf_oversampling(self, gaussian_psf):
        oversamp = 3
        yy, xx = np.mgrid[-3:3.00001:(1 / oversamp), -3:3.00001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)

        model = ImagePSF(psf_data, oversampling=oversamp,
                         interpolation='cubic')

        # Test evaluation on a grid at various positions
        yy_out, xx_out = np.mgrid[-3:4, -3:4]

        # At origin (0, 0), result should be well-shaped Gaussian
        result = model(xx_out, yy_out)
        expected = gaussian_psf(xx_out, yy_out)
        # Normalize both for shape comparison
        assert_allclose(result / result.max(), expected / expected.max(),
                        atol=0.01)

        # Test with small position shift that keeps PSF within bounds
        model.x_0 = 0.5
        model.y_0 = 0.5
        result = model(xx_out, yy_out)
        # Peak should shift by approximately (0.5, 0.5)
        peak_idx = np.unravel_index(np.argmax(result), result.shape)
        # Center of grid (-3:4) is at index 3, so shifted peak should be
        # around (3.5, 3.5), meaning index 3 or 4
        assert abs(peak_idx[0] - 3.5) <= 1
        assert abs(peak_idx[1] - 3.5) <= 1

        # Without oversampling, the model should still work but won't
        # interpolate subpixel positions as accurately
        model_no_os = ImagePSF(psf_data, interpolation='cubic')
        result_no_os = model_no_os(xx_out, yy_out)
        # Result should be valid (non-negative max, reasonable shape)
        assert result_no_os.max() > 0

    def test_origin(self):
        # Create PSF data centered at (2, 2) with origin at (0, 0)
        yy, xx = np.mgrid[:5, :5]
        gaussian_psf = CircularGaussianPSF(x_0=2, y_0=2, fwhm=2.1)
        psf_data = gaussian_psf(xx, yy)
        origin = (0, 0)

        # Model with origin=(0, 0) means PSF data index [0,0] corresponds
        # to world coordinate (0, 0). The PSF peak is at data index [2, 2].
        model = ImagePSF(psf_data, x_0=0, y_0=0, origin=origin,
                         interpolation='cubic')
        assert_equal(model.origin, origin)

        # Evaluate on grid [0:5, 0:5] with model at (0, 0)
        # Peak should be at (2, 2) since that's where the PSF data has
        # its maximum
        yy_out, xx_out = np.mgrid[:5, :5]
        result = model(xx_out, yy_out)

        # Peak should be at (2, 2)
        peak_idx = np.unravel_index(np.argmax(result), result.shape)
        assert peak_idx == (2, 2)

        # Now shift model to (2, 2) - peak should move to (4, 4)
        model.x_0 = 2
        model.y_0 = 2
        result = model(xx_out, yy_out)
        peak_idx = np.unravel_index(np.argmax(result), result.shape)
        assert peak_idx == (4, 4)

    def test_bounding_box(self):
        psf_data = np.arange(30, dtype=float).reshape(5, 6)
        psf_data /= np.sum(psf_data)
        model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0)
        assert_equal(model.bounding_box.bounding_box(), ((-2.5, 2.5),
                                                         (-3.0, 3.0)))

        model = ImagePSF(psf_data, flux=1, x_0=0, y_0=0, oversampling=2)
        assert_equal(model.bounding_box.bounding_box(), ((-1.25, 1.25),
                                                         (-1.5, 1.5)))

    def test_data_inputs(self):
        match = 'Input data must be a 2D numpy array'
        with pytest.raises(TypeError, match=match):
            ImagePSF(42)

        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones(10))

        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10, 10)))

        match = 'The length of the x and y axes must both be at least 4'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((3, 4)))

        data = np.ones((10, 10))
        data[0, 0] = np.nan
        match = 'All elements of input data must be finite'
        with pytest.raises(ValueError, match=match):
            ImagePSF(data)

    def test_oversampling_inputs(self):
        data = np.arange(30).reshape(5, 6)

        for oversampling in [4, (3, 3), (3, 4)]:
            model = ImagePSF(data, oversampling=oversampling)
            if np.ndim(oversampling) == 0:
                assert_equal(model.oversampling, (oversampling, oversampling))
            else:
                assert_equal(model.oversampling, oversampling)

        match = 'oversampling must be > 0'
        for oversampling in [-1, [-2, 4]]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must have 1 or 2 elements'
        oversampling = (1, 4, 8)
        with pytest.raises(ValueError, match=match):
            ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must be 1D'
        for oversampling in [((1, 2), (3, 4)), np.ones((2, 2, 2))]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

        match = 'oversampling must have integer values'
        with pytest.raises(ValueError, match=match):
            ImagePSF(data, oversampling=2.1)

        match = 'oversampling must be a finite value'
        for oversampling in [np.nan, (1, np.inf)]:
            with pytest.raises(ValueError, match=match):
                ImagePSF(data, oversampling=oversampling)

    def test_shape(self, image_psf):
        assert image_psf.shape == image_psf.data.shape
        assert image_psf.shape == (21, 21)

    def test_evaluate_scalar_coords(self, image_psf):
        """
        Test that evaluate accepts scalar coordinates when called
        directly.
        """
        value = image_psf.evaluate(0.5, 0.5, 1.0, 0.0, 0.0)
        assert np.isfinite(value)

    def test_data_setter(self):
        yy, xx = np.mgrid[0:25, 0:25]
        data1 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=3.0)(xx, yy)
        data2 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=8.0)(xx, yy)

        model = ImagePSF(data1, x_0=12, y_0=12)
        assert_allclose(model(12.0, 12.0), data1[12, 12])

        # The cached interpolator must be discarded when data is set
        model.data = data2
        assert_allclose(model(12.0, 12.0), data2[12, 12])

    def test_data_setter_validation(self):
        model = ImagePSF(np.ones((10, 10)))

        match = 'Input data must be a 2D numpy array'
        with pytest.raises(TypeError, match=match):
            model.data = 42
        with pytest.raises(ValueError, match=match):
            model.data = np.ones(10)

        match = 'The length of the x and y axes must both be at least 4'
        with pytest.raises(ValueError, match=match):
            model.data = np.ones((3, 4))

    def test_data_setter_copy_independence(self, gaussian_psf):
        yy, xx = np.mgrid[0:25, 0:25]
        data1 = gaussian_psf(xx, yy)
        data2 = CircularGaussianPSF(x_0=12, y_0=12, fwhm=8.0)(xx, yy)

        model = ImagePSF(data1, x_0=12, y_0=12)
        value = model(12.0, 12.0)  # populate the interpolator cache

        model_copy = model.copy()
        model_copy.data = data2
        assert_allclose(model(12.0, 12.0), value)
        assert_allclose(model_copy(12.0, 12.0), data2[12, 12])

    def test_oversampling_setter(self):
        model = ImagePSF(np.ones((10, 10)))
        model.oversampling = 4
        assert_equal(model.oversampling, (4, 4))

        match = 'oversampling must be > 0'
        with pytest.raises(ValueError, match=match):
            model.oversampling = -3
        assert_equal(model.oversampling, (4, 4))

    def test_origin_inputs(self):
        match = 'origin must be 1D and have 2-elements'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=(1, 2, 3))
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=np.ones((2, 2)))

        match = 'All elements of origin must be finite'
        with pytest.raises(ValueError, match=match):
            ImagePSF(np.ones((10, 10)), origin=(np.nan, 1))

    @pytest.mark.parametrize('deepcopy', [False, True])
    def test_copy(self, deepcopy):
        data = np.arange(30).reshape(5, 6)
        model = ImagePSF(data, flux=1, x_0=0, y_0=0)
        model_copy = model.deepcopy() if deepcopy else model.copy()

        assert_equal(model.data, model_copy.data)
        assert_equal(model.flux, model_copy.flux)
        assert_equal(model.x_0, model_copy.x_0)
        assert_equal(model.y_0, model_copy.y_0)
        assert_equal(model.oversampling, model_copy.oversampling)
        assert_equal(model.origin, model_copy.origin)

        model_copy.data[0, 0] = 42
        if deepcopy:
            assert model.data[0, 0] != model_copy.data[0, 0]
        else:
            assert model.data[0, 0] == model_copy.data[0, 0]

        model_copy.flux = 2
        assert model.flux != model_copy.flux

        model_copy.x_0.fixed = True
        model_copy.y_0.fixed = True
        model_copy2 = model_copy.copy()
        assert model_copy2.x_0.fixed
        assert model_copy2.fixed == model_copy.fixed

    def test_repr(self, image_psf):
        model_repr = repr(image_psf)
        expected = ('<ImagePSF(flux=1., x_0=0., y_0=0., origin=[10.0, 10.0], '
                    'oversampling=[1, 1], fill_value=0.0, '
                    "interpolation='cubic')>")
        assert model_repr == expected
        for param in image_psf.param_names:
            assert param in model_repr

    def test_str(self, image_psf):
        model_str = str(image_psf)
        keys = ('PSF shape', 'Origin', 'Oversampling', 'Fill Value',
                'Interpolation')
        for key in keys:
            assert key in model_str
        for param in image_psf.param_names:
            assert param in model_str

    def test_interpolation_inputs(self, gaussian_psf):
        """Test interpolation parameter validation."""
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)

        # Valid inputs
        for interp in ('cubic', 'bilinear', 'pchip'):
            model = ImagePSF(psf_data, interpolation=interp)
            assert model.interpolation == interp

        # Invalid input
        match = 'interpolation must be one of'
        with pytest.raises(ValueError, match=match):
            ImagePSF(psf_data, interpolation='invalid')

    def test_bilinear_interpolation(self, gaussian_psf):
        """Test bilinear interpolation preserves non-negativity."""
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)

        model = ImagePSF(psf_data, interpolation='bilinear')
        shifts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.75)]
        for dx, dy in shifts:
            model.x_0 = dx
            model.y_0 = dy
            result = model(xx, yy)
            # Bilinear should never produce negative values
            assert result.min() >= 0

    def test_pchip_interpolation(self):
        """
        Test PCHIP interpolation with well-sampled PSF.

        PCHIP provides shape-preserving interpolation that should:
        1. Preserve non-negativity (like bilinear)
        2. Be smoother than bilinear (C1 continuous)
        3. Avoid ringing artifacts of cubic splines

        Note: PCHIP requires sufficient data points to work well.
        For very small PSFs (< 10 pixels), bilinear may be better.
        """
        # Use a well-sampled PSF (21x21)
        gaussian_psf = CircularGaussianPSF(fwhm=3.0)
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)

        model = ImagePSF(psf_data, interpolation='pchip')
        assert model.interpolation == 'pchip'

        # Evaluate at fractional positions
        shifts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.25, 0.75)]
        for dx, dy in shifts:
            model.x_0 = dx
            model.y_0 = dy
            result = model(xx, yy)

            # PCHIP should preserve non-negativity for non-negative input
            assert result.min() >= 0, f'Negative value at shift ({dx}, {dy})'

            # Flux should be reasonably conserved
            assert_allclose(result.sum(), 1.0, rtol=0.02)

    def test_bilinear_flux_conservation(self):
        """
        Test bilinear interpolation with small PSF has better flux
        conservation than cubic.
        """
        from photutils.psf import GaussianPSF
        model = GaussianPSF(x_0=2, y_0=2)
        yy, xx = np.mgrid[:5, :5]
        psf_data = model(xx, yy)
        psf_data /= np.sum(psf_data)

        yy_out, xx_out = np.mgrid[:25, :25]

        # Cubic has significant flux loss at fractional positions
        psf_cubic = ImagePSF(psf_data, interpolation='cubic')
        result_cubic = psf_cubic.evaluate(xx_out, yy_out, 1,
                                          x_0=10.5, y_0=10.5)

        # Bilinear has much better flux conservation
        psf_bilinear = ImagePSF(psf_data, interpolation='bilinear')
        result_bilinear = psf_bilinear.evaluate(xx_out, yy_out, 1,
                                                x_0=10.5, y_0=10.5)

        # Bilinear should be much closer to 1.0
        assert abs(result_bilinear.sum() - 1.0) < 0.01
        # Cubic has large error for this small PSF
        assert abs(result_cubic.sum() - 1.0) > 0.5

        # Bilinear should have no negative values
        assert result_bilinear.min() >= 0
        # Cubic may have negative values
        assert result_cubic.min() < 0

    @pytest.mark.parametrize('shift', [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0),
                                       (0.5, 0.5), (0.3, 0.7)])
    def test_flux_conserve(self, gaussian_psf, shift):
        """
        Test that flux is reasonably conserved during fractional pixel
        shifts with cubic spline interpolation.
        """
        # Use a larger PSF grid to ensure the PSF stays within bounds
        # for all fractional shifts tested
        yy, xx = np.mgrid[-12:13, -12:13]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)  # Normalize to sum=1

        flux = 10.0
        model = ImagePSF(psf_data, flux=flux)
        model.x_0 = shift[0]
        model.y_0 = shift[1]

        # Evaluate on a grid that keeps PSF fully within bounds
        yy_out, xx_out = np.mgrid[-11:12, -11:12]
        result = model(xx_out, yy_out)
        # Cubic spline interpolation conserves flux well for well-sampled PSFs
        assert_allclose(np.sum(result), flux, rtol=1e-4)

    def test_flux_conserve_narrow_psf(self):
        """
        Test flux conservation with narrow PSF.

        Narrow PSFs have more interpolation error, so we use a
        looser tolerance.
        """
        gaussian_psf = CircularGaussianPSF(fwhm=1.2)
        # Use a larger grid to ensure PSF stays within bounds
        yy, xx = np.mgrid[-12:13, -12:13]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)

        flux = 5.0
        model = ImagePSF(psf_data, flux=flux)

        # Evaluate on a smaller grid that keeps PSF fully within bounds
        yy_out, xx_out = np.mgrid[-11:12, -11:12]
        shifts = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.75)]
        for dx, dy in shifts:
            model.x_0 = dx
            model.y_0 = dy
            result = model(xx_out, yy_out)
            # Narrow PSF has more interpolation error
            assert_allclose(np.sum(result), flux, rtol=0.01)

    def test_flux_conserve_with_oversampling(self, gaussian_psf):
        """
        Test flux conservation with oversampled PSF.

        Oversampled PSFs should maintain their normalization
        (sum = oversampling^2 for a unit-flux PSF).
        """
        oversamp = 3
        # Use a larger grid to ensure PSF stays within bounds
        yy, xx = np.mgrid[-5:5.00001:(1 / oversamp), -5:5.00001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)
        # Don't normalize - keep natural oversampled normalization

        flux = 7.5
        model = ImagePSF(psf_data, flux=flux, oversampling=oversamp)

        # Evaluate on output grid that keeps PSF fully within bounds
        yy_out, xx_out = np.mgrid[-4:5, -4:5]
        shifts = [(0.0, 0.0), (0.5, 0.5), (0.33, 0.66)]
        for dx, dy in shifts:
            model.x_0 = dx
            model.y_0 = dy
            result = model(xx_out, yy_out)
            # Result sum should be close to flux * original_sum / oversamp^2
            # For well-sampled Gaussians, interpolation conserves flux well
            assert_allclose(np.sum(result), flux, rtol=0.02)

    def test_flux_at_edge(self, gaussian_psf):
        """
        Test that flux is reduced when PSF is at the image edge.

        When the PSF is placed at the image edge and some pixels are
        clipped to fill_value=0, the total flux should be reduced
        compared to when the PSF is fully within bounds.
        """
        yy, xx = np.mgrid[-10:11, -10:11]
        psf_data = gaussian_psf(xx, yy)
        psf_data /= np.sum(psf_data)  # Normalize to sum=1

        flux = 10.0
        model = ImagePSF(psf_data, flux=flux)

        # Evaluate on a grid that extends beyond the PSF bounds
        # This will cause some edge pixels to be clipped
        yy_out, xx_out = np.mgrid[-15:16, -15:16]

        # At shift (0, 0), some evaluation points will be outside valid bounds
        model.x_0 = 0.0
        model.y_0 = 0.0
        result = model(xx_out, yy_out)

        # With fill_value=0, edge pixels outside the valid PSF region
        # are set to 0, reducing the total flux. The sum should be
        # close to but less than the requested flux.
        assert result.sum() <= flux * 1.001  # Should not be boosted above flux
        assert result.sum() > flux * 0.9  # But should still have most of it

    def test_boundary_extension(self):
        """
        Test that extended bounds [-0.5, N-0.5] recover edge pixels.

        The valid coordinate range is extended by 0.5 pixels beyond pixel
        centers to ensure fractional shifts don't cause boundary clipping.
        """
        from photutils.psf import GaussianPSF

        # Create a small 5x5 PSF
        model = GaussianPSF(x_0=2, y_0=2)
        yy, xx = np.mgrid[:5, :5]
        psf_data = model(xx, yy)
        psf_data /= np.sum(psf_data)

        psf = ImagePSF(psf_data, interpolation='bilinear')

        # Output grid larger than PSF
        yy_out, xx_out = np.mgrid[:25, :25]

        # At integer position, should use all 5x5 = 25 pixels within bounds
        result_int = psf.evaluate(xx_out, yy_out, 1, x_0=10.0, y_0=10.0)
        nonzero_int = np.count_nonzero(result_int)

        # At half-pixel position, extended bounds should allow 6x6 = 36 pixels
        # (the extra pixels come from extrapolation at the edges)
        result_half = psf.evaluate(xx_out, yy_out, 1, x_0=10.5, y_0=10.5)
        nonzero_half = np.count_nonzero(result_half)

        # With extended bounds, half-pixel shift should have MORE valid pixels
        # than integer shift (36 vs 25) because it can extrapolate to edges
        assert nonzero_half >= nonzero_int

        # Flux should be very well conserved with bilinear + extended bounds
        assert_allclose(result_int.sum(), 1.0, rtol=1e-3)
        assert_allclose(result_half.sum(), 1.0, rtol=1e-3)

    def test_boundary_extension_no_clipping(self):
        """
        Test that fractional shifts don't cause boundary clipping.

        Previously, a 0.5 pixel shift would cause coordinates like -0.5
        to be clipped (set to fill_value=0). With extended bounds, these
        coordinates should now be valid and interpolated.
        """
        from photutils.psf import GaussianPSF

        model = GaussianPSF(x_0=2, y_0=2)
        yy, xx = np.mgrid[:5, :5]
        psf_data = model(xx, yy)
        psf_data /= np.sum(psf_data)

        psf = ImagePSF(psf_data, interpolation='bilinear')
        yy_out, xx_out = np.mgrid[:25, :25]

        # Test various fractional shifts
        shifts = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        fluxes = []
        for shift in shifts:
            result = psf.evaluate(xx_out, yy_out, 1, x_0=10 + shift,
                                  y_0=10 + shift)
            fluxes.append(result.sum())

        # All shifts should give approximately the same flux
        # (within 1% for bilinear interpolation)
        fluxes = np.array(fluxes)
        assert_allclose(fluxes, 1.0, rtol=0.01)

    @pytest.mark.parametrize('shift', [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_boundary_symmetry(self, shift):
        """
        Test that positive and negative fractional shifts behave symmetrically.
        """
        from photutils.psf import GaussianPSF

        model = GaussianPSF(x_0=2, y_0=2)
        yy, xx = np.mgrid[:5, :5]
        psf_data = model(xx, yy)
        psf_data /= np.sum(psf_data)

        psf = ImagePSF(psf_data, interpolation='bilinear')
        yy_out, xx_out = np.mgrid[:25, :25]

        # Positive shift
        result_pos = psf.evaluate(xx_out, yy_out, 1, x_0=10 + shift,
                                  y_0=10 + shift)

        # Negative shift (using a different center position)
        result_neg = psf.evaluate(xx_out, yy_out, 1, x_0=14 - shift,
                                  y_0=14 - shift)

        # Both should have similar total flux
        assert_allclose(result_pos.sum(), result_neg.sum(), rtol=1e-10)


class TestImagePRF:
    """Tests for the ImagePRF class."""

    @pytest.fixture
    def oversampled_psf_data(self):
        """Create oversampled Gaussian PSF data."""
        oversamp = 4
        gaussian_psf = CircularGaussianPSF(fwhm=2.5)
        yy, xx = np.mgrid[-5:5.001:(1 / oversamp), -5:5.001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)
        return psf_data, oversamp

    def test_imageprf_basic(self, oversampled_psf_data):
        """Test basic ImagePRF functionality."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp)

        # Evaluate on non-oversampled grid
        yy, xx = np.mgrid[-5:6, -5:6]
        result = model(xx, yy)

        # Result should have same shape as input grid
        assert result.shape == xx.shape

        # Result should be non-negative (for non-negative input)
        assert result.min() >= -1e-10  # allow small numerical errors

    def test_imageprf_inherits_from_imagepsf(self, oversampled_psf_data):
        """Test that ImagePRF inherits all ImagePSF features."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp)

        # Check inherited attributes
        assert hasattr(model, 'data')
        assert hasattr(model, 'origin')
        assert hasattr(model, 'oversampling')
        assert hasattr(model, 'fill_value')
        assert hasattr(model, 'interpolation')
        assert hasattr(model, 'interpolator')

        # Check parameters
        assert 'flux' in model.param_names
        assert 'x_0' in model.param_names
        assert 'y_0' in model.param_names

    def test_imageprf_interpolation_options(self, oversampled_psf_data):
        """Test that ImagePRF supports all interpolation methods."""
        psf_data, oversamp = oversampled_psf_data

        for interp in ('cubic', 'bilinear', 'pchip'):
            model = ImagePRF(psf_data, oversampling=oversamp,
                             interpolation=interp)
            assert model.interpolation == interp

            yy, xx = np.mgrid[-3:4, -3:4]
            result = model(xx, yy)
            assert result.shape == xx.shape

    def test_imageprf_flux_scaling(self, oversampled_psf_data):
        """Test that flux parameter properly scales the output."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp, flux=1.0)

        yy, xx = np.mgrid[-5:6, -5:6]
        result_flux1 = model(xx, yy)

        model.flux = 10.0
        result_flux10 = model(xx, yy)

        # Flux 10 should be 10x flux 1
        assert_allclose(result_flux10, result_flux1 * 10, rtol=1e-14)

    def test_imageprf_position_shift(self, oversampled_psf_data):
        """Test that x_0, y_0 properly shift the model position."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp)

        yy, xx = np.mgrid[-5:6, -5:6]

        # Centered at origin
        model.x_0 = 0
        model.y_0 = 0
        result_center = model(xx, yy)

        # Find peak position
        peak_idx = np.unravel_index(np.argmax(result_center),
                                    result_center.shape)
        # Peak should be near center
        assert abs(peak_idx[0] - 5) <= 1
        assert abs(peak_idx[1] - 5) <= 1

        # Shifted by 2 pixels
        model.x_0 = 2
        model.y_0 = 2
        result_shifted = model(xx, yy)

        # Peak should move
        peak_idx_shifted = np.unravel_index(np.argmax(result_shifted),
                                            result_shifted.shape)
        assert abs(peak_idx_shifted[0] - 7) <= 1
        assert abs(peak_idx_shifted[1] - 7) <= 1

    def test_prf_vs_psf_integer_positions(self, oversampled_psf_data):
        """
        Test that ImagePRF and ImagePSF give similar results at integer
        positions for well-sampled PSFs.
        """
        psf_data, oversamp = oversampled_psf_data

        prf = ImagePRF(psf_data, oversampling=oversamp)
        psf = ImagePSF(psf_data, oversampling=oversamp)

        yy, xx = np.mgrid[-3:4, -3:4]

        result_prf = prf(xx, yy)
        result_psf = psf(xx, yy)

        # With the oversampling area normalization, PRF and PSF should
        # give similar total flux for well-sampled PSFs
        assert_allclose(result_prf.sum(), result_psf.sum(), rtol=0.1)

    def test_imageprf_flux_conservation_fractional_shift(self):
        """
        Test that ImagePRF conserves flux better than ImagePSF
        at fractional pixel positions.
        """
        oversamp = 4
        gaussian_psf = CircularGaussianPSF(fwhm=2.0)
        yy, xx = np.mgrid[-5:5.001:(1 / oversamp), -5:5.001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)

        prf = ImagePRF(psf_data, oversampling=oversamp,
                       interpolation='bilinear')

        yy_out, xx_out = np.mgrid[-5:6, -5:6]

        # Test flux at various fractional positions
        fluxes = []
        for shift in [0.0, 0.25, 0.5, 0.75]:
            prf.x_0 = shift
            prf.y_0 = shift
            result = prf(xx_out, yy_out)
            fluxes.append(result.sum())

        # All fluxes should be similar (good flux conservation)
        fluxes = np.array(fluxes)
        # PRF should have small flux variation (< 5%)
        assert (fluxes.max() - fluxes.min()) / fluxes.mean() < 0.05

    def test_imageprf_flux_conservation(self, oversampled_psf_data):
        """Test that flux is always conserved in ImagePRF."""
        psf_data, oversamp = oversampled_psf_data

        flux = 100.0
        prf = ImagePRF(psf_data, oversampling=oversamp, flux=flux)

        # Use a smaller evaluation grid to keep PSF fully within bounds.
        # The PSF data is on grid [-5:5.001] with oversamp=4 (shape 41x41).
        # The evaluation grid must be small enough that all subpixel
        # coordinates stay within the valid range [-0.5, 40.5].
        yy, xx = np.mgrid[-4:5, -4:5]

        # Test at various positions
        for x_0, y_0 in [(0, 0), (0.5, 0.5), (0.25, 0.75)]:
            prf.x_0 = x_0
            prf.y_0 = y_0
            result = prf(xx, yy)
            assert_allclose(result.sum(), flux, rtol=1e-14)

    def test_imageprf_subpixel_integration(self):
        """
        Test that ImagePRF properly integrates over subpixels.

        For a constant input, the PRF should return the same constant
        (since summing N^2 subpixels each with value C gives N^2 * C,
        which is the expected normalization for an oversampled PSF).
        """
        oversamp = 3
        # Create a flat PSF (all ones)
        psf_data = np.ones((21, 21), dtype=float)

        prf = ImagePRF(psf_data, oversampling=oversamp)

        # Evaluate at a single pixel
        result = prf.evaluate(np.array([0.0]), np.array([0.0]),
                              flux=1.0, x_0=0.0, y_0=0.0)

        # For a flat PSF with all 1s, each subpixel has value 1.
        # Summing oversamp^2 subpixels gives oversamp^2, then dividing
        # by oversamp^2 (the oversampling area) gives 1.0
        expected = 1.0
        assert_allclose(result[0], expected, rtol=1e-10)

    def test_imageprf_reproduces_gaussian_integral(self):
        """
        Test that ImagePRF properly integrates a Gaussian PSF.

        For a well-sampled Gaussian, the PRF (which integrates) and PSF
        (which samples) should give similar shapes and relative values.
        The differences are expected since PRF integrates over each pixel
        while PSF samples at the center.
        """
        oversamp = 8
        gaussian_psf = CircularGaussianPSF(fwhm=2.5)  # well-sampled PSF
        yy, xx = np.mgrid[-5:5.001:(1 / oversamp), -5:5.001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)

        prf = ImagePRF(psf_data, oversampling=oversamp,
                       interpolation='bilinear')
        psf = ImagePSF(psf_data, oversampling=oversamp,
                       interpolation='bilinear')

        # Evaluate on a small grid
        yy_out, xx_out = np.mgrid[-2:3, -2:3]
        result_prf = prf(xx_out, yy_out)
        result_psf = psf(xx_out, yy_out)

        # Both should have similar shapes (peak at center)
        # PRF gives slightly different values due to integration vs sampling
        # Normalize and compare with moderate tolerance
        assert_allclose(result_prf / result_prf.max(),
                        result_psf / result_psf.max(), rtol=0.35)

    def test_imageprf_different_xy_oversampling(self):
        """Test ImagePRF with different oversampling in x and y."""
        oversamp_y = 3
        oversamp_x = 5
        gaussian_psf = CircularGaussianPSF(fwhm=2.5)

        # Create PSF data with different sampling in x and y
        yy, xx = np.mgrid[-5:5.001:(1 / oversamp_y),
                          -5:5.001:(1 / oversamp_x)]
        psf_data = gaussian_psf(xx, yy)

        prf = ImagePRF(psf_data, oversampling=(oversamp_y, oversamp_x))

        yy_out, xx_out = np.mgrid[-3:4, -3:4]
        result = prf(xx_out, yy_out)

        # Should work without error and produce reasonable output
        assert result.shape == xx_out.shape
        assert result.sum() > 0

    def test_imageprf_copy(self, oversampled_psf_data):
        """Test that copy() works for ImagePRF."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp, flux=5.0,
                         x_0=1.0, y_0=2.0)
        model_copy = model.copy()

        assert_equal(model.flux.value, model_copy.flux.value)
        assert_equal(model.x_0.value, model_copy.x_0.value)
        assert_equal(model.y_0.value, model_copy.y_0.value)

    def test_imageprf_deepcopy(self, oversampled_psf_data):
        """Test that deepcopy() works for ImagePRF."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp)
        model_copy = model.deepcopy()

        # Modify original
        model.data[0, 0] = 999.0

        # Deep copy should not be affected
        assert model_copy.data[0, 0] != 999.0

    def test_imageprf_fill_value(self, oversampled_psf_data):
        """Test that fill_value is applied for out-of-bounds pixels."""
        psf_data, oversamp = oversampled_psf_data
        fill_val = -99.0
        prf = ImagePRF(psf_data, oversampling=oversamp, fill_value=fill_val)

        # Evaluate far outside the PSF extent
        result = prf.evaluate(np.array([100.0]), np.array([100.0]),
                              flux=1.0, x_0=0.0, y_0=0.0)

        # All subpixels are out of bounds, so each gets fill_value.
        # Summing oversamp^2 fill_values then dividing by oversamp^2
        # gives just fill_value.
        expected = fill_val
        assert_allclose(result[0], expected, rtol=1e-10)

    def test_imageprf_str_repr(self, oversampled_psf_data):
        """Test string representations."""
        psf_data, oversamp = oversampled_psf_data
        model = ImagePRF(psf_data, oversampling=oversamp)

        # Should have string representations that work
        str_repr = str(model)
        repr_str = repr(model)

        assert 'ImagePRF' in repr_str
        assert 'PSF shape' in str_repr

    @pytest.mark.parametrize('shift', [0.0, 0.25, 0.5, 0.75])
    def test_imageprf_shift_symmetry(self, shift):
        """Test that shifts are symmetric."""
        oversamp = 4
        gaussian_psf = CircularGaussianPSF(fwhm=2.5)
        yy, xx = np.mgrid[-5:5.001:(1 / oversamp), -5:5.001:(1 / oversamp)]
        psf_data = gaussian_psf(xx, yy)

        prf = ImagePRF(psf_data, oversampling=oversamp,
                       interpolation='bilinear')

        yy_out, xx_out = np.mgrid[-5:6, -5:6]

        # Positive shift
        prf.x_0 = shift
        prf.y_0 = shift
        result_pos = prf(xx_out, yy_out)

        # Negative shift from different reference
        prf.x_0 = 1.0 - shift
        prf.y_0 = 1.0 - shift
        result_neg = prf(xx_out, yy_out)

        # Total flux should be the same (within small numerical tolerance)
        assert_allclose(result_pos.sum(), result_neg.sum(), rtol=1e-4)
