# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the pill module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose, assert_array_equal

from photutils.aperture import (AperturePhotometry, ApertureStats,
                                CircularAperture, RectangularAperture,
                                aperture_photometry)
from photutils.aperture.pill import PillAperture, SkyPillAperture
from photutils.aperture.tests.test_aperture_common import (
    BaseTestAperture, BaseTestRotatedPixelAperture)
from photutils.utils._optional_deps import HAS_MATPLOTLIB

POSITIONS = [(10, 20), (30, 40), (50, 60), (70, 80)]
RA, DEC = np.transpose(POSITIONS)
SKYCOORD = SkyCoord(ra=RA, dec=DEC, unit='deg')
UNIT = u.arcsec
RADII = (0.0, -1.0, -np.inf)


class NoBatchPillAperture(PillAperture):
    """
    A `PillAperture` subclass that does not opt in to the batch Cython
    driver (it is not decorated with ``_enable_batch_photometry``),
    forcing the mask-based code path.
    """


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
        area = aper.area
        bbox = aper.bbox
        aper.width = 10.0
        assert_allclose(aper.area, np.pi * 25.0)
        assert aper.area != area
        assert aper.bbox != bbox

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

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_patch_geometry(self):
        # The arc end points and the straight sides of the outline lie
        # at a distance of width / 2 from the trail segment.
        position = np.array((10.3, 20.6))
        origin = np.array((1.5, -2.0))
        length = 12.0
        width = 5.0
        theta = 0.6
        aper = PillAperture(position, length=length, width=width,
                            theta=theta)
        path = aper._to_patch(origin=origin).get_path()
        half_straight = 0.5 * (length - width)
        direction = np.array((np.cos(theta), np.sin(theta)))
        center = position - origin
        start = center - half_straight * direction
        vertices = [vertices[-2:] for vertices, code
                    in path.iter_segments(curves=True, simplify=False)
                    if code != path.CLOSEPOLY]
        assert len(vertices) > 4
        for vertex in vertices:
            along = np.clip(np.dot(vertex - start, direction), 0.0,
                            2.0 * half_straight)
            distance = np.hypot(*(vertex - start - along * direction))
            assert_allclose(distance, 0.5 * width, rtol=1e-12)

    @pytest.mark.parametrize(('method', 'rtol'),
                             [('center', 0.05), ('subpixel', 0.01)])
    def test_approximate_methods(self, method, rtol):
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=0.6)
        mask = aper.to_mask(method=method, subpixels=5)
        assert_allclose(mask.data.sum(), aper.area, rtol=rtol)

    @pytest.mark.parametrize(('length', 'width', 'theta'),
                             [(6.0, 6.0, -45 * u.deg),
                              (1.0, 1.0, 45 * u.deg),
                              (np.hypot(6, 6) + 6, 6.0, np.arctan2(6, 6))])
    def test_exact_junction_on_corners(self, length, width, theta):
        # Pixel corners lie on a junction line between the straight
        # section and an end.
        data = np.ones((41, 41))
        aper = PillAperture((20, 20), length=length, width=width,
                            theta=theta)
        mask = aper.to_mask()
        assert mask.data.max() <= 1.0
        assert_allclose(mask.data.sum(), aper.area, rtol=1e-12)
        tbl = aperture_photometry(data, aper)
        assert_allclose(tbl['aperture_sum'][0], aper.area, rtol=1e-12)
        stats = ApertureStats(data, aper)
        assert_allclose(stats.sum, aper.area, rtol=1e-12)

    def test_photometry_on_ones(self):
        aper = PillAperture((25.3, 24.6), length=12.0, width=5.0,
                            theta=0.6)
        tbl = aperture_photometry(np.ones((50, 50)), aper)
        assert_allclose(tbl['aperture_sum'][0], aper.area, rtol=1e-10)


class TestPillBatch:
    data = np.random.default_rng(0).normal(10.0, 1.0, (60, 60))
    positions = ((25.3, 24.6), (2.2, 30.1), (58.9, 58.2), (-20.0, 5.0))

    @pytest.mark.parametrize('theta', [0.5, np.pi / 2])
    @pytest.mark.parametrize('method', ['exact', 'center', 'subpixel'])
    def test_batch_matches_mask(self, method, theta):
        aper = PillAperture(self.positions, length=12.0, width=5.0,
                            theta=theta)
        assert aper._batch_shape_params() is not None
        tbl = aperture_photometry(self.data, aper, method=method)
        expected = [mask.multiply(self.data).sum()
                    if mask.cutout(self.data) is not None else np.nan
                    for mask in aper.to_mask(method=method)]
        assert_allclose(tbl['aperture_sum'], expected, rtol=1e-13,
                        equal_nan=True)

    def test_stats_match_mask(self):
        aper = PillAperture(self.positions[:3], length=12.0, width=5.0,
                            theta=0.5)
        stats = ApertureStats(self.data, aper)
        assert stats._batch_inputs is not None
        expected = [mask.multiply(self.data).sum()
                    for mask in aper.to_mask()]
        assert_allclose(stats.sum, expected, rtol=1e-13)

    def test_flags_match_rectangle(self):
        pill = PillAperture(self.positions, length=12.0, width=5.0)
        mask_pill = NoBatchPillAperture(self.positions, length=12.0,
                                        width=5.0)
        rect = RectangularAperture(self.positions, w=12.0, h=5.0)
        kwargs = {'error': None, 'mask': None, 'method': 'exact',
                  'subpixels': 5}
        assert pill._batch_photometry(self.data, **kwargs) is not None
        assert mask_pill._batch_photometry(self.data, **kwargs) is None
        flags_pill = AperturePhotometry(self.data, pill).flags
        flags_mask = AperturePhotometry(self.data, mask_pill).flags
        flags_rect = AperturePhotometry(self.data, rect).flags
        assert_array_equal(flags_pill, flags_mask)
        assert_array_equal(flags_pill, flags_rect)


class TestSkyPillAperture(BaseTestAperture):
    aperture = SkyPillAperture(SKYCOORD, length=10.0 * UNIT,
                               width=5.0 * UNIT, theta=30 * u.deg)
    copy_param = 'length'
    copy_value = 20.0 * UNIT

    @staticmethod
    @pytest.mark.parametrize('radius', RADII)
    def test_invalid_params(radius):
        match = "'length' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyPillAperture(SKYCOORD, length=radius * UNIT,
                            width=5.0 * UNIT, theta=30 * u.deg)

        match = "'width' must be greater than zero"
        with pytest.raises(ValueError, match=match):
            SkyPillAperture(SKYCOORD, length=10.0 * UNIT,
                            width=radius * UNIT, theta=30 * u.deg)

    def test_length_less_than_width(self):
        match = "'length' must be greater than or equal to 'width'"
        with pytest.raises(ValueError, match=match):
            SkyPillAperture(SKYCOORD, length=4 * UNIT, width=5 * UNIT)
