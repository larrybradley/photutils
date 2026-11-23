# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the pill module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.aperture import (CircularAperture, RectangularAperture,
                                aperture_photometry)
from photutils.aperture.pill import PillAperture
from photutils.aperture.tests.test_aperture_common import (
    BaseTestRotatedPixelAperture)

POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RADII = (0.0, -1.0, -np.inf)


class TestPillAperture(BaseTestRotatedPixelAperture):
    aperture = PillAperture(POSITIONS, length=10.0, width=5.0,
                            theta=np.pi / 2.0)
    copy_param = 'length'
    copy_value = 20.0

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'length' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            PillAperture(POSITIONS, length=radius, width=5.0)
        match = "'width' must be a positive scalar"
        with pytest.raises(ValueError, match=match):
            PillAperture(POSITIONS, length=10.0, width=radius)

    def test_length_less_than_width(self):
        match = "'length' must be greater than or equal to 'width'"
        with pytest.raises(ValueError, match=match):
            PillAperture(POSITIONS, length=4.0, width=5.0)

    def test_ordering_on_mutation(self):
        aper = PillAperture(POSITIONS, length=10.0, width=5.0)
        match = "'length' must be greater than or equal to 'width'"
        with pytest.raises(ValueError, match=match):
            aper.width = 11.0
        assert aper.width == 5.0
        with pytest.raises(ValueError, match=match):
            aper.length = 4.0
        assert aper.length == 10.0
        aper.width = 10.0
        assert_allclose(aper.area, np.pi * 25.0)

    def test_area(self):
        aper = PillAperture((10, 20), length=12.0, width=5.0)
        assert_allclose(aper.area, 35.0 + 6.25 * np.pi)

    def test_circle_limit(self):
        pill = PillAperture((10.3, 20.6), length=6.0, width=6.0,
                            theta=0.4)
        circ = CircularAperture((10.3, 20.6), r=3.0)
        assert pill.bbox == circ.bbox
        assert_allclose(pill.to_mask().data, circ.to_mask().data,
                        atol=1e-12)

    @pytest.mark.parametrize('theta', [0.0, np.pi / 2.0])
    def test_bbox_matches_rectangle(self, theta):
        pill = PillAperture((10.3, 20.6), length=12.0, width=5.0,
                            theta=theta)
        rect = RectangularAperture((10.3, 20.6), w=12.0, h=5.0,
                                   theta=theta)
        assert pill.bbox == rect.bbox

    @pytest.mark.parametrize('theta', [0.0, 0.6, 2.1, -1.0])
    def test_exact_mask_sum(self, theta):
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=theta)
        assert_allclose(aper.to_mask().data.sum(), aper.area,
                        rtol=1e-10)

    @pytest.mark.parametrize('theta', [0.6 + np.pi, 0.6 - 2 * np.pi,
                                       np.rad2deg(0.6) * u.deg])
    def test_theta_wrapping(self, theta):
        ref = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                           theta=0.6)
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=theta)
        assert_allclose(aper.to_mask().data, ref.to_mask().data,
                        atol=1e-12)

    @pytest.mark.parametrize(('method', 'rtol'),
                             [('center', 0.05), ('subpixel', 0.01)])
    def test_approximate_methods(self, method, rtol):
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=0.6)
        mask = aper.to_mask(method=method, subpixels=5)
        assert_allclose(mask.data.sum(), aper.area, rtol=rtol)

    def test_photometry_on_ones(self):
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=0.6)
        tbl = aperture_photometry(np.ones((50, 50)), aper)
        assert_allclose(tbl['aperture_sum'][0], aper.area, rtol=1e-10)
