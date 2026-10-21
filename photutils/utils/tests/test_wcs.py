# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the wcs module.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from numpy.testing import assert_allclose, assert_equal

from photutils.datasets import make_gwcs
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.tests.wcs_test_helpers import (WCS_CDELT_ARCSEC,
                                                    WCS_CENTER, make_sip_wcs)
from photutils.utils.wcs import (compute_pixel_area_map, compute_pixel_areas,
                                 pixel_to_sky_errors)

UNIFORM_AREA = WCS_CDELT_ARCSEC**2


def _make_sip_wcs(shape, coeff=2e-5):
    """
    Build a TAN-SIP WCS whose pixel area varies smoothly across an array
    of the given shape.
    """
    return make_sip_wcs(shape, coeffs={'A_2_0': coeff, 'B_0_2': coeff})


def _make_wide_tan_wcs(shape, deg_per_pix=0.01):
    """
    Build a TAN WCS whose pixels are large enough that the pixel area
    varies non-polynomially across the field.

    A quadratic SIP distortion is reproduced exactly by the bicubic
    spline in ``compute_pixel_area_map``, so it cannot detect an interpolation
    error. The gnomonic area factor of a wide field can.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cdelt = [-deg_per_pix, deg_per_pix]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


ALLSKY_SHAPE = (180, 360)


def _make_allsky_wcs(projection):
    """
    Build an all-sky WCS with 1 degree pixels in the given
    projection (e.g., ``'CAR'`` or ``'AIT'``) for an array of shape
    ``ALLSKY_SHAPE``.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [ALLSKY_SHAPE[1] / 2 + 0.5, ALLSKY_SHAPE[0] / 2 + 0.5]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [-1.0, 1.0]
    wcs.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']
    return wcs


def _make_cd_wcs(cd_deg):
    """
    Build a TAN WCS with the given CD matrix in degrees per pixel.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [10.5, 10.5]
    wcs.wcs.crval = [WCS_CENTER.ra.deg, WCS_CENTER.dec.deg]
    wcs.wcs.cd = cd_deg
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    return wcs


class TestComputePixelAreas:
    def test_uniform_wcs(self, simple_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([1.5, 10.0, 4.0])
        areas = compute_pixel_areas(simple_wcs, x, y)
        assert areas.shape == (3,)
        assert_allclose(areas, UNIFORM_AREA, rtol=1e-6)

    def test_scalar_input(self, simple_wcs):
        area = compute_pixel_areas(simple_wcs, 10.0, 10.0)
        assert isinstance(area, float)
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    @pytest.mark.parametrize('shape', [(3, 7), (4, 3), (2, 3, 4)])
    def test_input_shape_preserved(self, shape):
        wcs = _make_sip_wcs((200, 200))
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 199, shape)
        y = rng.uniform(0, 199, shape)
        areas = compute_pixel_areas(wcs, x, y)
        assert areas.shape == shape
        expected = compute_pixel_areas(wcs, x.ravel(), y.ravel())
        assert_allclose(areas.ravel(), expected, rtol=1e-12)

    def test_shape_mismatch(self, simple_wcs):
        match = 'x and y must have the same shape'
        with pytest.raises(ValueError, match=match):
            compute_pixel_areas(simple_wcs, [1.0, 2.0], [1.0, 2.0, 3.0])

    def test_nonsquare_pixels(self, nonsquare_wcs):
        expected = (0.03 * 3600) * (0.05 * 3600)
        area = compute_pixel_areas(nonsquare_wcs, 10.0, 10.0)
        assert_allclose(area, expected, rtol=1e-6)

    def test_sheared_wcs(self):
        cd = np.array([[-2.0e-5, 0.5e-5], [0.3e-5, 3.0e-5]])
        wcs = _make_cd_wcs(cd)
        expected = np.abs(np.linalg.det(cd)) * 3600**2
        area = compute_pixel_areas(wcs, 10.0, 10.0)
        assert_allclose(area, expected, rtol=1e-6)

    def test_flipped_parity_is_positive(self, flipped_wcs):
        area = compute_pixel_areas(flipped_wcs, 10.0, 10.0)
        assert area > 0
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_nonfinite_position(self, simple_wcs, bad):
        x = np.array([10.0, bad, 10.0])
        y = np.array([10.0, 10.0, bad])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            areas = compute_pixel_areas(simple_wcs, x, y)
        assert_allclose(areas[0], UNIFORM_AREA, rtol=1e-6)
        assert np.all(np.isnan(areas[1:]))

    def test_outside_projection(self):
        # An all-sky Aitoff map has corners outside the projection
        wcs = _make_allsky_wcs('AIT')
        areas = compute_pixel_areas(wcs, [0.0, 180.0], [0.0, 90.0])
        assert np.isnan(areas[0])
        assert np.isfinite(areas[1])
        assert areas[1] > 0

    def test_varies_with_distortion(self):
        wcs = _make_sip_wcs((200, 200))
        areas = compute_pixel_areas(wcs, np.array([10.0, 190.0]),
                                    np.array([10.0, 190.0]))
        assert not np.isclose(areas[0], areas[1], rtol=1e-4)


class TestPixelAreaMap:
    def test_uniform_wcs(self, simple_wcs):
        shape = (20, 30)
        area = compute_pixel_area_map(simple_wcs, shape)
        assert isinstance(area, np.ndarray)
        assert area.shape == shape
        assert_allclose(area, UNIFORM_AREA, rtol=1e-6)

    def test_matches_point_values_with_distortion(self):
        shape = (300, 200)
        wcs = _make_sip_wcs(shape)
        area = compute_pixel_area_map(wcs, shape)

        # The distortion must actually produce a gradient
        assert np.ptp(area) / area.mean() > 1e-3

        rng = np.random.default_rng(0)
        y = rng.integers(0, shape[0], 50)
        x = rng.integers(0, shape[1], 50)
        expected = compute_pixel_areas(wcs, x, y)
        assert_allclose(area[y, x], expected, rtol=1e-6)

    @pytest.mark.parametrize('shape', [(1, 1), (2, 5), (10, 10), (65, 3),
                                       (3, 130)])
    def test_small_shapes(self, shape):
        wcs = _make_sip_wcs(shape)
        area = compute_pixel_area_map(wcs, shape)
        assert area.shape == shape
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        expected = compute_pixel_areas(wcs, xx.ravel(), yy.ravel())
        assert_allclose(area.ravel(), expected, rtol=1e-6)

    def test_matches_point_values_wide_field(self):
        shape = (300, 200)
        wcs = _make_wide_tan_wcs(shape)
        area = compute_pixel_area_map(wcs, shape)

        # The gnomonic projection must actually produce a gradient
        assert np.ptp(area) / area.mean() > 1e-3

        rng = np.random.default_rng(0)
        y = rng.integers(0, shape[0], 50)
        x = rng.integers(0, shape[1], 50)
        expected = compute_pixel_areas(wcs, x, y)
        assert_allclose(area[y, x], expected, rtol=1e-6)

    def test_step_independent(self):
        shape = (1040, 1040)
        wcs = _make_wide_tan_wcs(shape)
        fine = compute_pixel_area_map(wcs, shape, step=16)
        coarse = compute_pixel_area_map(wcs, shape, step=128)
        assert_allclose(fine, coarse, rtol=1e-6)

    @pytest.mark.parametrize('step', [7, 1000])
    def test_step_capped(self, step):
        """
        Test that a step larger than min(shape) // 8 is reduced to
        it with a warning, so the result is identical to that of the
        largest allowed step and differs from a smaller step.
        """
        shape = (64, 48)
        wcs = _make_wide_tan_wcs(shape, deg_per_pix=0.2)
        capped = compute_pixel_area_map(wcs, shape, step=6)
        match = f'step={step} was reduced to 6 '
        with pytest.warns(AstropyUserWarning, match=match):
            area = compute_pixel_area_map(wcs, shape, step=step)
        assert np.array_equal(area, capped)
        smaller = compute_pixel_area_map(wcs, shape, step=5)
        assert not np.array_equal(smaller, capped)

    def test_default_step_small_image(self):
        """
        Test that the default step is silently reduced on a small image
        and equals the largest allowed step.
        """
        shape = (64, 48)
        wcs = _make_wide_tan_wcs(shape, deg_per_pix=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape)
            expected = compute_pixel_area_map(wcs, shape, step=6)
        assert np.array_equal(area, expected)

    def test_default_step_large_image(self):
        shape = (600, 520)
        wcs = _make_wide_tan_wcs(shape)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape)
            expected = compute_pixel_area_map(wcs, shape, step=64)
        assert np.array_equal(area, expected)

    @pytest.mark.parametrize('projection', ['CAR', 'AIT'])
    def test_coarse_grid_outside_projection(self, projection):
        """
        Test all-sky maps whose padded coarse grid leaves the valid
        region of the projection.

        Every pixel of the CAR map is valid, while the AIT map has
        corners outside the projection. In both cases the map must
        match the directly evaluated areas pixel for pixel, including
        the NaN pattern.
        """
        wcs = _make_allsky_wcs(projection)
        yy, xx = np.mgrid[:ALLSKY_SHAPE[0], :ALLSKY_SHAPE[1]]
        expected = compute_pixel_areas(wcs, xx, yy)
        area = compute_pixel_area_map(wcs, ALLSKY_SHAPE)
        assert area.shape == ALLSKY_SHAPE
        assert_allclose(area, expected, rtol=1e-12, equal_nan=True)

        finite = np.isfinite(expected)
        if projection == 'CAR':
            assert np.all(finite)
        else:
            assert 0 < finite.mean() < 1

    @pytest.mark.parametrize('shape', [(1, 1), (2, 5), (65, 3), (300, 200),
                                       (600, 520)])
    def test_exact_matches_direct_evaluation(self, shape):
        """
        Test that ``interpolate=False`` returns the areas evaluated
        directly at every pixel, without any interpolation.

        The largest shape has more pixels than one evaluation block,
        with a partial final block, so it also checks that the blocks
        are stitched together correctly.
        """
        wcs = _make_wide_tan_wcs(shape)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape, interpolate=False)
        assert isinstance(area, np.ndarray)
        assert area.shape == shape
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        expected = compute_pixel_areas(wcs, xx, yy)
        assert np.array_equal(area, expected)

    def test_exact_agrees_with_interpolated(self):
        shape = (300, 200)
        wcs = _make_wide_tan_wcs(shape)
        exact = compute_pixel_area_map(wcs, shape, interpolate=False)
        interpolated = compute_pixel_area_map(wcs, shape)
        assert np.ptp(exact) / exact.mean() > 1e-3
        assert_allclose(interpolated, exact, rtol=1e-6)

    def test_exact_outside_projection(self):
        """
        Test that ``interpolate=False`` keeps the NaN pattern of the
        pixels outside the projection.
        """
        wcs = _make_allsky_wcs('AIT')
        area = compute_pixel_area_map(wcs, ALLSKY_SHAPE, interpolate=False)
        yy, xx = np.mgrid[:ALLSKY_SHAPE[0], :ALLSKY_SHAPE[1]]
        expected = compute_pixel_areas(wcs, xx, yy)
        assert np.array_equal(area, expected, equal_nan=True)
        assert 0 < np.isfinite(area).mean() < 1

    @pytest.mark.parametrize(('shape', 'step'),
                             [((300, 200), 1), ((12, 9), None),
                              ((12, 9), 1)])
    def test_step_one_is_exact(self, shape, step):
        """
        Test that a step of 1, explicit or from the default on a tiny
        image, evaluates every pixel directly instead of interpolating.
        """
        wcs = _make_wide_tan_wcs(shape)
        exact = compute_pixel_area_map(wcs, shape, interpolate=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            area = compute_pixel_area_map(wcs, shape, step=step)
        assert np.array_equal(area, exact)

    def test_step_capped_to_one(self):
        """
        Test that a step capped to 1 on a tiny image warns that every
        pixel is evaluated directly and gives the exact map.
        """
        shape = (12, 9)
        wcs = _make_wide_tan_wcs(shape)
        exact = compute_pixel_area_map(wcs, shape, interpolate=False)
        match = ('step=64 was reduced to 1, so the area is evaluated '
                 'directly at every pixel')
        with pytest.warns(AstropyUserWarning, match=match):
            area = compute_pixel_area_map(wcs, shape, step=64)
        assert np.array_equal(area, exact)

    @pytest.mark.parametrize('interpolate', ['no', None, 0, 1])
    def test_invalid_interpolate(self, simple_wcs, interpolate):
        match = f'interpolate must be a boolean, got {interpolate!r}'
        with pytest.raises(TypeError, match=match):
            compute_pixel_area_map(simple_wcs, (20, 20),
                                   interpolate=interpolate)

    @pytest.mark.parametrize('step', [1, 16, 1000])
    def test_exact_rejects_step(self, simple_wcs, step):
        match = 'step must be None when interpolate=False'
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, (20, 20), step=step,
                                   interpolate=False)

    @pytest.mark.parametrize('step', [0, -4, 2.5, True])
    def test_invalid_step(self, simple_wcs, step):
        match = 'step must be a positive integer or None'
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, (20, 20), step=step)

    @pytest.mark.parametrize('shape', [(20,), (20, 20, 20), (0, 20),
                                       (20, -1), (20.0, 20), (True, 20)])
    def test_invalid_shape(self, simple_wcs, shape):
        match = 'shape must be two positive integers'
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, shape)
        with pytest.raises(ValueError, match=match):
            compute_pixel_area_map(simple_wcs, shape, interpolate=False)

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs_bounding_box(self):
        shape = (40, 50)
        gwcs = make_gwcs(shape)
        gwcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
        area = compute_pixel_area_map(gwcs, shape)
        assert np.all(np.isfinite(area))
        # make_gwcs has a uniform 0.1 arcsec pixel scale
        assert_allclose(area, 0.01, rtol=1e-6)

    def test_matches_astropy_proj_plane_area(self, simple_wcs):
        expected = (proj_plane_pixel_area(simple_wcs) * u.deg**2).to_value(
            u.arcsec**2)
        area = compute_pixel_area_map(simple_wcs, (20, 20))
        assert_allclose(area, expected, rtol=1e-6)


def _sky_offsets(wcs, x, y, dx, dy):
    """
    Return the (East, North) offsets in arcsec of the sky position at
    pixel ``(x + dx, y + dy)`` from that at ``(x, y)``, measured with
    the astropy spherical offsets.
    """
    start = wcs.pixel_to_world(x, y)
    end = wcs.pixel_to_world(x + dx, y + dy)
    east, north = start.spherical_offsets_to(end)
    return east.to_value(u.arcsec), north.to_value(u.arcsec)


class TestPixelToSkyErrors:
    def test_uniform_wcs(self, simple_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([1.5, 10.0, 4.0])
        x_err = np.array([0.1, 0.2, 0.0])
        y_err = np.array([0.3, 0.05, 0.0])
        east_err, north_err = pixel_to_sky_errors(simple_wcs, x, y, x_err,
                                                  y_err)
        assert east_err.unit == u.arcsec
        assert north_err.unit == u.arcsec
        assert east_err.shape == (3,)
        assert_allclose(east_err.value, x_err * WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(north_err.value, y_err * WCS_CDELT_ARCSEC,
                        rtol=1e-6)

    @pytest.mark.parametrize('wcs_name', ['rotated_wcs', 'sip_wcs',
                                          'nonsquare_wcs', 'flipped_wcs',
                                          'swapped_wcs'])
    @pytest.mark.parametrize('rho', [-1.0, -0.3, 0.0, 0.3, 1.0])
    def test_matches_spherical_offsets(self, request, wcs_name, rho):
        """
        Test the errors against the propagation of the pixel errors
        through offsets measured independently with astropy.

        The steps are small enough that the offsets are linear in them.
        A correlation of +1 or -1 is the boundary of a valid
        covariance.
        """
        wcs = request.getfixturevalue(wcs_name)
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([15.5, 10.0, 4.0])
        x_err = np.array([0.02, 0.05, 0.01])
        y_err = np.array([0.04, 0.01, 0.03])
        xy_cov = rho * x_err * y_err
        east_err, north_err = pixel_to_sky_errors(wcs, x, y, x_err, y_err,
                                                  xy_cov=xy_cov)

        east_x, north_x = _sky_offsets(wcs, x, y, x_err, 0.0)
        east_y, north_y = _sky_offsets(wcs, x, y, 0.0, y_err)
        east_var = east_x**2 + east_y**2 + 2 * rho * east_x * east_y
        north_var = north_x**2 + north_y**2 + 2 * rho * north_x * north_y
        assert_allclose(east_err.value, np.sqrt(east_var), rtol=1e-5)
        assert_allclose(north_err.value, np.sqrt(north_var), rtol=1e-5)

    def test_correlation_changes_errors(self, rotated_wcs):
        kwargs = {'x_err': 0.1, 'y_err': 0.2}
        errs = [pixel_to_sky_errors(rotated_wcs, 10.0, 10.0, **kwargs,
                                    xy_cov=xy_cov)
                for xy_cov in (-0.02, 0.0, 0.02)]
        east = [err[0].value for err in errs]
        assert east[0] != pytest.approx(east[1])
        assert east[2] != pytest.approx(east[1])

    def test_axis_order_independent(self, simple_wcs, swapped_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([15.5, 10.0, 4.0])
        expected = pixel_to_sky_errors(simple_wcs, x, y, 0.1, 0.2,
                                       xy_cov=0.01)
        result = pixel_to_sky_errors(swapped_wcs, x, y, 0.1, 0.2,
                                     xy_cov=0.01)
        assert_allclose(result[0], expected[0], rtol=1e-6)
        assert_allclose(result[1], expected[1], rtol=1e-6)

    def test_scalar_input(self, simple_wcs):
        east_err, north_err = pixel_to_sky_errors(simple_wcs, 10.0, 10.0,
                                                  0.1, 0.2)
        assert isinstance(east_err, u.Quantity)
        assert east_err.isscalar
        assert north_err.isscalar
        assert_allclose(east_err.value, 0.1 * WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(north_err.value, 0.2 * WCS_CDELT_ARCSEC, rtol=1e-6)

    @pytest.mark.parametrize('shape', [(3, 7), (2, 3, 4)])
    def test_input_shape_preserved(self, shape):
        wcs = _make_sip_wcs((200, 200))
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 199, shape)
        y = rng.uniform(0, 199, shape)
        x_err = rng.uniform(0.01, 0.1, shape)
        y_err = rng.uniform(0.01, 0.1, shape)
        east_err, north_err = pixel_to_sky_errors(wcs, x, y, x_err, y_err)
        assert east_err.shape == shape
        assert north_err.shape == shape
        expected = pixel_to_sky_errors(wcs, x.ravel(), y.ravel(),
                                       x_err.ravel(), y_err.ravel())
        assert_allclose(east_err.ravel(), expected[0], rtol=1e-12)
        assert_allclose(north_err.ravel(), expected[1], rtol=1e-12)

    def test_errors_broadcast(self, simple_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([1.5, 10.0, 4.0])
        east_err, north_err = pixel_to_sky_errors(simple_wcs, x, y, 0.1,
                                                  [0.1, 0.2, 0.3])
        assert_allclose(east_err.value, 0.1 * WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(north_err.value,
                        np.array([0.1, 0.2, 0.3]) * WCS_CDELT_ARCSEC,
                        rtol=1e-6)

    def test_shape_mismatch(self, simple_wcs):
        match = 'x and y must have the same shape'
        with pytest.raises(ValueError, match=match):
            pixel_to_sky_errors(simple_wcs, [1.0, 2.0], [1.0, 2.0, 3.0],
                                0.1, 0.1)

    @pytest.mark.parametrize('kwargs', [
        {'x_err': [0.1, 0.2], 'y_err': 0.1},
        {'x_err': 0.1, 'y_err': [0.1, 0.2]},
        {'x_err': 0.1, 'y_err': 0.1, 'xy_cov': [0.0, 0.0]}])
    def test_errors_not_broadcastable(self, simple_wcs, kwargs):
        match = ('x_err, y_err, and xy_cov must be broadcastable to the '
                 'shape of x and y')
        with pytest.raises(ValueError, match=match):
            pixel_to_sky_errors(simple_wcs, [1.0, 2.0, 3.0],
                                [1.0, 2.0, 3.0], **kwargs)

    @pytest.mark.parametrize(('x_err', 'y_err'), [(-0.1, 0.1), (0.1, -0.1)])
    def test_negative_errors(self, simple_wcs, x_err, y_err):
        match = 'x_err and y_err must be nonnegative'
        with pytest.raises(ValueError, match=match):
            pixel_to_sky_errors(simple_wcs, 10.0, 10.0, x_err, y_err)

    @pytest.mark.parametrize(('x_err', 'y_err', 'xy_cov'), [
        (0.1, 0.2, 0.0201), (0.1, 0.2, -0.0201), (0.1, 0.2, np.inf),
        (0.1, 0.2, -np.inf), (1.0, 1e-7, 2e-7), (0.0, 0.1, 1e-3)])
    def test_invalid_covariance(self, simple_wcs, x_err, y_err, xy_cov):
        match = r'abs\(xy_cov\) must not exceed x_err \* y_err'
        with pytest.raises(ValueError, match=match):
            pixel_to_sky_errors(simple_wcs, 10.0, 10.0, x_err, y_err,
                                xy_cov=xy_cov)

    def test_huge_errors(self, simple_wcs):
        """
        Test that errors whose variances overflow and an infinite error
        times a zero error give nonfinite errors without a warning.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            east_err, north_err = pixel_to_sky_errors(
                simple_wcs, [10.0, 10.0], [10.0, 10.0], [1e200, np.inf],
                [1e200, 0.0])
        assert not np.any(np.isfinite(east_err))
        assert not np.any(np.isfinite(north_err))

    def test_quantity_input(self, rotated_wcs):
        x = np.array([2.0, 10.0, 17.3])
        y = np.array([15.5, 10.0, 4.0])
        expected = pixel_to_sky_errors(rotated_wcs, x, y, 0.1, 0.2,
                                       xy_cov=0.01)
        result = pixel_to_sky_errors(rotated_wcs, x * u.pix, y * u.pix,
                                     0.1 * u.pix, 0.2 * u.pix,
                                     xy_cov=0.01 * u.pix**2)
        assert_equal(result[0], expected[0])
        assert_equal(result[1], expected[1])

    @pytest.mark.parametrize(('name', 'kwargs'), [
        ('x', {'x': 10.0 * u.deg}),
        ('x_err', {'x_err': 0.1 * u.arcsec}),
        ('y_err', {'y_err': 0.1 * u.dimensionless_unscaled}),
        ('xy_cov', {'xy_cov': 0.001 * u.pix})])
    def test_quantity_invalid_unit(self, simple_wcs, name, kwargs):
        values = {'x': 10.0, 'y': 10.0, 'x_err': 0.1, 'y_err': 0.1}
        values.update(kwargs)
        with pytest.raises(ValueError, match=f'{name} must be in units of'):
            pixel_to_sky_errors(simple_wcs, **values)

    def test_masked_input(self, simple_wcs):
        """
        Test that masked positions and errors give NaN errors, even
        where the masked value is invalid.
        """
        x = np.ma.array([10.0, 10.0, 10.0, 10.0],
                        mask=[False, True, False, False])
        y = np.full(4, 10.0)
        x_err = np.ma.array([0.1, 0.1, -1.0, 0.1],
                            mask=[False, False, True, False])
        xy_cov = np.ma.array([0.0, 0.0, 0.0, 1.0],
                             mask=[False, False, False, True])
        east_err, north_err = pixel_to_sky_errors(simple_wcs, x, y, x_err,
                                                  0.1, xy_cov=xy_cov)
        for err in (east_err, north_err):
            assert not isinstance(err.value, np.ma.MaskedArray)
            assert_allclose(err.value[0], 0.1 * WCS_CDELT_ARCSEC,
                            rtol=1e-6)
            assert np.all(np.isnan(err[1:]))

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_nonfinite_input(self, simple_wcs, bad):
        """
        Test that a nonfinite position or error gives NaN errors for
        that source only, without a warning.

        A NaN covariance gives NaN errors. An infinite covariance with
        finite errors is invalid (see ``test_invalid_covariance``).
        """
        x = np.array([10.0, bad, 10.0, 10.0, 10.0])
        y = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        x_err = np.array([0.1, 0.1, abs(bad), 0.1, 0.1])
        y_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        xy_cov = np.array([0.0, 0.0, 0.0, 0.0, np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            east_err, north_err = pixel_to_sky_errors(
                simple_wcs, x, y, x_err, y_err, xy_cov=xy_cov)
        expected = 0.1 * WCS_CDELT_ARCSEC
        for err in (east_err, north_err):
            assert_allclose(err.value[[0, 3]], expected, rtol=1e-6)
            assert np.all(np.isnan(err[[1, 2, 4]]))

    def test_outside_projection(self):
        # An all-sky Aitoff map has corners outside the projection
        wcs = _make_allsky_wcs('AIT')
        east_err, north_err = pixel_to_sky_errors(wcs, [0.0, 180.0],
                                                  [0.0, 90.0], 0.1, 0.1)
        for err in (east_err, north_err):
            assert np.isnan(err[0])
            assert np.isfinite(err[1])

    def test_no_bounding_box_outside_image(self, simple_wcs):
        """
        Test that the errors are finite outside the image for a WCS
        without a bounding box.
        """
        east_err, north_err = pixel_to_sky_errors(simple_wcs, -5.0, 30.0,
                                                  0.1, 0.2)
        assert_allclose(east_err.value, 0.1 * WCS_CDELT_ARCSEC, rtol=1e-6)
        assert_allclose(north_err.value, 0.2 * WCS_CDELT_ARCSEC, rtol=1e-6)

    @pytest.mark.skipif(not HAS_GWCS, reason='gwcs is required')
    def test_gwcs_bounding_box(self):
        """
        Test the errors for a gwcs with a bounding box.

        The errors are NaN exactly where ``pixel_to_world`` returns NaN.
        Positions within half a pixel of the bounding-box edge are
        inside it, so their errors must be finite even though the finite
        differences step outside it.
        """
        shape = (40, 50)
        gwcs = make_gwcs(shape)
        gwcs.bounding_box = ((-0.5, shape[1] - 0.5), (-0.5, shape[0] - 0.5))
        x = np.array([0.0, 49.0, 25.0, -0.5, -0.3, 49.3, 49.5, -0.6, 49.6,
                      25.0, 25.0])
        y = np.array([0.0, 39.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
                      -0.6, 39.6])
        east_err, north_err = pixel_to_sky_errors(gwcs, x, y, 0.1, 0.2)

        outside = ~np.isfinite(gwcs.pixel_to_world(x, y).ra.deg)
        assert_equal(outside, [False] * 7 + [True] * 4)
        assert np.all(np.isnan(east_err[outside]))
        assert np.all(np.isnan(north_err[outside]))

        # make_gwcs has a uniform 0.1 arcsec pixel scale. It is rotated,
        # so only the total error is independent of the orientation.
        total = np.hypot(east_err[~outside], north_err[~outside])
        assert_allclose(total.to_value(u.arcsec), 0.1 * np.hypot(0.1, 0.2),
                        rtol=1e-6)

    @pytest.mark.parametrize('wcs_type', [
        'fits',
        pytest.param('gwcs', marks=pytest.mark.skipif(
            not HAS_GWCS, reason='gwcs is required'))])
    def test_empty_input(self, simple_wcs, wcs_type):
        wcs = simple_wcs if wcs_type == 'fits' else make_gwcs((10, 10))
        east_err, north_err = pixel_to_sky_errors(wcs, [], [], 0.1, 0.2)
        assert east_err.unit == u.arcsec
        assert east_err.shape == (0,)
        assert north_err.shape == (0,)
