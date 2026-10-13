# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the matchedfilter module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from photutils.detection import MatchedFilterStarFinder
from photutils.utils.exceptions import NoDetectionsWarning

SIGMA = 2.0
NOISE = 3.0
SHAPE = (101, 121)
# (x, y, flux) of the stars. The last star is truncated by the border.
STARS = ((30.3, 40.7, 800.0), (80.6, 70.2, 300.0), (60.0, 20.5, 400.0),
         (100.2, 30.1, 1500.0), (1.2, 60.0, 600.0))


def make_psf_image(xcen, ycen, flux, shape=SHAPE):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (flux / (2.0 * np.pi * SIGMA**2)
            * np.exp(-((xx - xcen)**2 + (yy - ycen)**2) / (2.0 * SIGMA**2)))


@pytest.fixture(name='kernel')
def fixture_kernel():
    yy, xx = np.mgrid[-8:9, -8:9]
    return np.exp(-(xx**2 + yy**2) / (2.0 * SIGMA**2))


@pytest.fixture(name='data')
def fixture_data():
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, NOISE, SHAPE)
    for xcen, ycen, flux in STARS:
        data += make_psf_image(xcen, ycen, flux)
    return data


class TestMatchedFilterStarFinder:
    """
    Test the MatchedFilterStarFinder class.
    """

    @pytest.mark.parametrize('local_bkg', [True, False])
    def test_find(self, data, kernel, local_bkg):
        """
        Test that the stars are found with accurate positions, fluxes,
        and uncertainties.
        """
        finder = MatchedFilterStarFinder(5.0, kernel, local_bkg=local_bkg)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == len(STARS)
        assert_equal(tbl['id'], np.arange(len(STARS)) + 1)

        truth = np.array(sorted(STARS, key=lambda star: star[1]))
        tbl.sort('y_centroid')
        # Exclude the star that is truncated by the border
        inside = truth[:, 0] > 5
        # The position uncertainty of a Gaussian star along each axis
        # is sqrt(2) * sigma / snr
        pos_err = np.sqrt(2.0) * SIGMA / tbl['snr']
        xpull = (tbl['x_centroid'] - truth[:, 0]) / pos_err
        ypull = (tbl['y_centroid'] - truth[:, 1]) / pos_err
        assert np.all(np.abs(xpull[inside]) < 4.0)
        assert np.all(np.abs(ypull[inside]) < 4.0)
        pull = (tbl['flux'] - truth[:, 2]) / tbl['flux_err']
        assert np.all(np.abs(pull[inside]) < 4.0)
        assert_allclose(tbl['snr'], tbl['flux'] / tbl['flux_err'])
        assert np.all(tbl['snr'] > 5.0)
        assert_allclose(tbl['reduced_chi2'], 1.0, atol=0.3)
        if not local_bkg:
            assert_equal(tbl['local_bkg'], 0.0)

        # The truncated star is found near its true position
        assert_allclose(tbl['x_centroid'][~inside], 1.2, atol=0.5)
        assert_allclose(tbl['flux'][~inside], 600.0, rtol=0.2)

    def test_flux_err(self, kernel):
        """
        Test the flux uncertainty of an isolated star against the
        analytic matched-filter value.
        """
        data = make_psf_image(60.0, 50.0, 1000.0)
        finder = MatchedFilterStarFinder(5.0, kernel, local_bkg=False)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == 1
        norm = kernel / kernel.sum()
        assert_allclose(tbl['flux_err'], NOISE / np.sqrt(np.sum(norm**2)))
        assert_allclose(tbl['flux'], 1000.0, rtol=1.0e-3)
        assert_allclose(tbl['x_centroid'], 60.0, atol=1.0e-6)
        assert_allclose(tbl['y_centroid'], 50.0, atol=1.0e-6)

    @pytest.mark.parametrize('local_bkg', [True, False])
    def test_large_image(self, kernel, local_bkg):
        """
        Test stars at the boundaries of the row strips that are used
        for the convolutions.
        """
        shape = (600, 60)
        ycens = (150.0, 255.6, 256.0, 511.6, 590.0)
        xcens = (15.0, 45.0, 15.0, 45.0, 15.0)
        data = np.zeros(shape)
        for xcen, ycen in zip(xcens, ycens, strict=True):
            data += make_psf_image(xcen, ycen, 1000.0, shape=shape)
        finder = MatchedFilterStarFinder(5.0, kernel, local_bkg=local_bkg)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == len(ycens)
        tbl.sort('y_centroid')
        assert_allclose(tbl['y_centroid'], ycens, atol=0.01)
        assert_allclose(tbl['x_centroid'], xcens, atol=0.01)
        assert_allclose(tbl['flux'], 1000.0, rtol=0.01)
        assert_allclose(tbl['flux_err'], tbl['flux_err'][0], rtol=1.0e-3)

    def test_subpixel_position(self, kernel):
        """
        Test the position of a noiseless star at a subpixel offset.
        """
        data = make_psf_image(60.3, 49.6, 1000.0)
        finder = MatchedFilterStarFinder(5.0, kernel)
        tbl = finder(data, error=NOISE)
        assert_allclose(tbl['x_centroid'], 60.3, atol=0.01)
        assert_allclose(tbl['y_centroid'], 49.6, atol=0.01)
        assert_allclose(tbl['flux'], 1000.0, rtol=0.01)

    def test_error_none(self, data, kernel):
        """
        Test that the noise is estimated from the data.
        """
        finder = MatchedFilterStarFinder(5.0, kernel)
        tbl0 = finder(data)
        tbl1 = finder.find_stars(data, error=NOISE)
        assert len(tbl0) == len(tbl1)
        assert_allclose(tbl0['flux'], tbl1['flux'])
        assert_allclose(tbl0['flux_err'], tbl1['flux_err'], rtol=0.1)

    def test_error_array_and_mask(self, data, kernel):
        """
        Test that masked and infinite-error pixels are excluded from the
        fit.
        """
        finder = MatchedFilterStarFinder(5.0, kernel)
        tbl0 = finder(data, error=NOISE)

        bad = np.zeros(SHAPE, dtype=bool)
        bad[41, 30] = True  # peak pixel of a star
        bad[65:68, 78:83] = True
        data2 = data.copy()
        data2[bad] = 1.0e6
        error = np.full(SHAPE, NOISE)
        tbl1 = finder(data2, error=error, mask=bad)
        error[bad] = np.inf
        tbl2 = finder(data2, error=error)
        data2[bad] = np.nan
        tbl3 = finder(data2, error=NOISE)

        for tbl in (tbl1, tbl2, tbl3):
            assert len(tbl) == len(tbl0)
            assert_allclose(tbl['x_centroid'], tbl0['x_centroid'], atol=0.3)
            assert_allclose(tbl['flux'], tbl0['flux'], rtol=0.1)
            assert_allclose(tbl['flux'], tbl1['flux'])
        assert np.all(tbl1['flux_err'] > 0.999 * tbl0['flux_err'])
        assert np.any(tbl1['flux_err'] > 1.01 * tbl0['flux_err'])

    def test_variable_error(self, kernel):
        """
        Test that the threshold is relative to the local noise.
        """
        rng = np.random.default_rng(1)
        error = np.full(SHAPE, 1.0)
        error[:, 60:] = 10.0
        data = rng.normal(0.0, 1.0, SHAPE) * error
        data += make_psf_image(30.0, 50.0, 300.0)
        data += make_psf_image(95.0, 50.0, 300.0)
        tbl = MatchedFilterStarFinder(5.0, kernel)(data, error=error)
        assert len(tbl) == 1
        assert_allclose(tbl['x_centroid'], 30.0, atol=0.2)

    def test_local_bkg(self, data, kernel):
        """
        Test that the results do not depend on the background level
        when the local background is fitted.
        """
        finder = MatchedFilterStarFinder(5.0, kernel, exclude_border=True)
        tbl0 = finder(data, error=NOISE)
        tbl1 = finder(data + 50.0, error=NOISE)
        assert len(tbl0) == len(STARS) - 1
        assert len(tbl1) == len(tbl0)
        assert_allclose(tbl1['flux'], tbl0['flux'])
        assert_allclose(tbl1['x_centroid'], tbl0['x_centroid'])
        assert_allclose(tbl1['local_bkg'], tbl0['local_bkg'] + 50.0)

    def test_max_peak_excess(self, data, kernel):
        """
        Test that single-pixel artifacts are rejected.
        """
        data = data.copy()
        data[85, 20] += 40.0 * NOISE
        data[15, 100] += 60.0 * NOISE
        finder = MatchedFilterStarFinder(5.0, kernel)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == len(STARS) + 2
        assert np.count_nonzero(tbl['peak_excess'] > 10.0) == 2

        finder = MatchedFilterStarFinder(5.0, kernel, max_peak_excess=5.0)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == len(STARS)

        match = 'none pass the filtering criteria'
        finder = MatchedFilterStarFinder(5.0, kernel, max_peak_excess=-99.0)
        with pytest.warns(NoDetectionsWarning, match=match):
            assert finder(data, error=NOISE) is None

    def test_n_brightest(self, data, kernel):
        finder = MatchedFilterStarFinder(5.0, kernel, n_brightest=2)
        tbl = finder(data, error=NOISE)
        assert len(tbl) == 2
        assert_allclose(tbl['flux'], [1500.0, 800.0], rtol=0.1)
        assert_equal(tbl['id'], [1, 2])

    def test_min_separation(self, kernel):
        data = make_psf_image(50.0, 50.0, 1000.0)
        data += make_psf_image(62.0, 50.0, 500.0)
        finder = MatchedFilterStarFinder(5.0, kernel)
        assert_allclose(finder.min_separation, 4.7, atol=0.1)
        assert len(finder(data, error=NOISE)) == 2
        finder = MatchedFilterStarFinder(5.0, kernel, min_separation=0)
        assert len(finder(data, error=NOISE)) == 2
        finder = MatchedFilterStarFinder(5.0, kernel, min_separation=20.0)
        assert len(finder(data, error=NOISE)) == 1

    def test_units(self, data, kernel):
        finder = MatchedFilterStarFinder(5.0, kernel)
        tbl0 = finder(data, error=NOISE)
        tbl1 = finder(data << u.Jy, error=NOISE << u.Jy)
        for name in ('flux', 'flux_err', 'local_bkg'):
            assert tbl1[name].unit == u.Jy
            assert_allclose(tbl1[name].value, tbl0[name])
        assert_allclose(tbl1['snr'], tbl0['snr'])

    def test_nosources(self, kernel):
        data = np.random.default_rng(0).normal(0.0, 1.0, (50, 50))
        finder = MatchedFilterStarFinder(50.0, kernel)
        match = 'No sources were found'
        with pytest.warns(NoDetectionsWarning, match=match):
            assert finder(data, error=1.0) is None

    def test_all_masked_region(self, data, kernel):
        """
        Test that a region with no unmasked pixels gives no sources.
        """
        mask = np.zeros(SHAPE, dtype=bool)
        mask[:, :70] = True
        for local_bkg in (True, False):
            finder = MatchedFilterStarFinder(5.0, kernel,
                                             local_bkg=local_bkg)
            tbl = finder(data, error=NOISE, mask=mask)
            assert len(tbl) == 2
            assert np.all(tbl['x_centroid'] > 70)

    def test_inputs(self, data, kernel):
        match = 'threshold must be a positive scalar value'
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(0.0, kernel)
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(np.ones((3, 3)), kernel)
        match = 'kernel must be a 2D array'
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(5.0, np.ones(5))
        match = 'kernel must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(5.0, np.ones((4, 5)))
        match = 'kernel must have finite values and a positive sum'
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(5.0, np.zeros((5, 5)))
        match = 'min_separation must be >= 0'
        with pytest.raises(ValueError, match=match):
            MatchedFilterStarFinder(5.0, kernel, min_separation=-1)

        finder = MatchedFilterStarFinder(5.0, kernel)
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            finder(np.ones(10))
        match = 'mask must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            finder(data, mask=np.zeros((3, 3), dtype=bool))
        match = 'error must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            finder(data, error=np.ones((3, 3)))
        match = 'error must be positive'
        with pytest.raises(ValueError, match=match):
            finder(data, error=0.0)
        match = 'The noise could not be estimated from data'
        with pytest.raises(ValueError, match=match):
            finder(np.ones((20, 20)))

    def test_repr(self, kernel):
        finder = MatchedFilterStarFinder(5.0, kernel)
        assert repr(finder).startswith('MatchedFilterStarFinder(')
        assert 'shape=(17, 17)' in repr(finder)
        assert 'local_bkg: True' in str(finder)
