# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the morans_i module.
"""

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           uniform_filter)

from photutils.background import make_morans_i_mask, morans_i_map
from photutils.datasets import make_100gaussians_image


def _slow_box_sum(array, size):
    """
    Sum an array in a square box centered on each pixel.
    """
    half = size // 2
    result = np.zeros(array.shape)
    for iy in range(array.shape[0]):
        for ix in range(array.shape[1]):
            result[iy, ix] = array[max(iy - half, 0):iy + half + 1,
                                   max(ix - half, 0):ix + half + 1].sum()
    return result


def _reference_morans_i(data, patch_size, *, n_box_passes=1, mask=None,
                        weights=None):
    """
    Slow pure-NumPy reference for the sliding Moran's I statistic.
    """
    data = np.asarray(data, dtype=float)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= ~mask
    if weights is None:
        weights = np.ones(data.shape)
    with np.errstate(invalid='ignore'):
        valid &= np.isfinite(weights) & (weights > 0)
    wts = np.where(valid, weights, 0.0)
    values = np.where(valid, data, 0.0)
    ny, nx = data.shape

    neighbor_mean = np.zeros(data.shape)
    usable = np.zeros(data.shape, dtype=bool)
    for iy in range(ny):
        for ix in range(nx):
            if not valid[iy, ix]:
                continue
            slc = (slice(max(iy - 1, 0), iy + 2),
                   slice(max(ix - 1, 0), ix + 2))
            neighbor_wts = wts[slc].copy()
            neighbor_wts[iy - slc[0].start, ix - slc[1].start] = 0.0
            if neighbor_wts.sum() > 0:
                neighbor_mean[iy, ix] = ((neighbor_wts * values[slc]).sum()
                                         / neighbor_wts.sum())
                usable[iy, ix] = True
    wts[~usable] = 0.0

    sums = [usable.astype(float), wts, wts * values, wts * values**2,
            wts * neighbor_mean, wts * values * neighbor_mean]
    for _ in range(n_box_passes):
        sums = [_slow_box_sum(array, patch_size) for array in sums]
    count, sum_w, sum_x, sum_xx, sum_n, sum_xn = sums

    result = np.full(data.shape, np.nan)
    good = usable & (count >= 0.25 * patch_size**(2 * n_box_passes))
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sum_x / sum_w
        var = sum_xx / sum_w - mean**2
        good &= var > 1.0e-10 * sum_xx / sum_w
        stat = (sum_xn / sum_w - mean * sum_n / sum_w) / var
    result[good] = stat[good]
    return result


def _reference_opening(mask, iterations):
    """
    Open a mask with SciPy, with flagged pixels outside of the image
    for the erosion.
    """
    if iterations == 0:
        return mask
    eroded = binary_erosion(mask, iterations=iterations, border_value=1)
    return binary_dilation(eroded, iterations=iterations)


@pytest.fixture(name='noise')
def fixture_noise():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, (201, 251))


@pytest.fixture(name='small')
def fixture_small():
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, (40, 50))
    data[15:22, 20:30] += 4.0
    return data


class TestMoransIMap:
    def test_reference(self, small):
        """
        Test against a slow reference, including the image edges.
        """
        result = morans_i_map(small, patch_size=7)
        assert result.shape == small.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert_allclose(result, _reference_morans_i(small, 7), atol=1e-5)

    def test_reference_masked(self, small):
        """
        Test against a slow reference with masked and non-finite pixels.
        """
        data = small.copy()
        data[5, 5] = np.nan
        data[30, 40] = np.inf
        mask = np.zeros(data.shape, dtype=bool)
        mask[10:14, 10:14] = True
        mask[0:8, 44:] = True  # most of a corner patch

        result = morans_i_map(data, patch_size=7, mask=mask)
        expected = _reference_morans_i(data, 7, mask=mask)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)
        assert np.all(np.isnan(result[mask]))
        assert np.isnan(result[5, 5])
        assert np.isnan(result[30, 40])
        assert np.isnan(result[0, 49])

    def test_weights(self, small):
        """
        Test weighted data against a slow reference, including pixels
        that are excluded by their weights.
        """
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.1, 3.0, small.shape)
        weights[3, 4] = 0.0
        weights[20, 30] = -1.0
        weights[25, 10] = np.nan
        weights[30:34, 20:24] = 0.0
        weights[31, 21] = 0.3  # no neighbors with a positive weight
        mask = np.zeros(small.shape, dtype=bool)
        mask[10:14, 10:14] = True

        result = morans_i_map(small, patch_size=7, mask=mask,
                              weights=weights)
        expected = _reference_morans_i(small, 7, mask=mask, weights=weights)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)
        for idx in ((3, 4), (20, 30), (25, 10), (31, 21), (11, 11)):
            assert np.isnan(result[idx])
        assert not np.allclose(result, morans_i_map(small, patch_size=7),
                               atol=1e-3, equal_nan=True)

    def test_weights_scale(self, small):
        """
        Test that the result does not depend on the weight scale and
        that equal weights match the unweighted result.
        """
        expected = morans_i_map(small, patch_size=7)
        for scale in (1.0, 1.0e-12, 3.0e12):
            weights = np.full(small.shape, scale)
            result = morans_i_map(small, patch_size=7, weights=weights)
            assert_allclose(result, expected, atol=1e-5)

    @pytest.mark.parametrize(('n_box_passes', 'box_size'),
                             [(2, 7), (3, 5), (30, 3)])
    def test_n_box_passes(self, small, n_box_passes, box_size):
        """
        Test repeated box passes against a slow reference with masked
        pixels and weights.

        The passes use the box that gives the kernel the effective
        width of a patch of 9 pixels.
        """
        rng = np.random.default_rng(6)
        weights = rng.uniform(0.5, 2.0, small.shape)
        mask = np.zeros(small.shape, dtype=bool)
        mask[10:14, 10:14] = True
        mask[0:9, 42:] = True

        result = morans_i_map(small, patch_size=9,
                              n_box_passes=n_box_passes)
        expected = _reference_morans_i(small, box_size,
                                       n_box_passes=n_box_passes)
        assert np.any(np.isfinite(result))
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)

        result = morans_i_map(small, patch_size=9, mask=mask,
                              weights=weights, n_box_passes=n_box_passes)
        expected = _reference_morans_i(small, box_size, mask=mask,
                                       weights=weights,
                                       n_box_passes=n_box_passes)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)

    def test_n_box_passes_footprint(self, noise):
        """
        Test that repeated box passes extend the footprint of a bright
        source beyond the square patch.
        """
        yy, xx = np.mgrid[:201, :251]
        data = noise + 1000.0 * np.exp(-((xx - 120)**2 + (yy - 100)**2)
                                       / 4.5)
        single = morans_i_map(data, patch_size=11)
        triple = morans_i_map(data, patch_size=11, n_box_passes=3)
        assert_equal(morans_i_map(data, patch_size=11, n_box_passes=1),
                     single)

        # the single box ends 5 pixels from the source and the kernel
        # of three passes (of a box of 7 pixels) ends 9 pixels from it
        assert single[100, 120] > 0.5
        assert triple[100, 120] > 0.5
        assert abs(single[100, 132]) < 0.2
        assert triple[100, 132] > 0.5
        assert abs(triple[100, 140]) < 0.2

    def test_patch_larger_than_image(self, small):
        data = small[:12, :9]
        result = morans_i_map(data, patch_size=21)
        assert_allclose(result, _reference_morans_i(data, 21), atol=1e-5)

        result = morans_i_map(data, patch_size=21, n_box_passes=2)
        expected = _reference_morans_i(data, 15, n_box_passes=2)
        assert_equal(np.isnan(result), np.isnan(expected))
        assert_allclose(result, expected, atol=1e-5)

    def test_noise_and_source(self, noise):
        """
        Test that uncorrelated noise gives values near zero and that a
        source gives a large positive value.
        """
        result = morans_i_map(noise)
        assert_allclose(np.mean(result), -1.0 / 440.0, atol=0.003)
        assert_allclose(np.std(result), 0.023, atol=0.004)

        data = noise.copy()
        yy, xx = np.mgrid[:201, :251]
        data += 10.0 * np.exp(-((xx - 120)**2 + (yy - 100)**2) / 18.0)
        result = morans_i_map(data)
        assert result[100, 120] > 0.5

    def test_shift_and_scale_invariance(self, noise):
        expected = morans_i_map(noise)
        assert_allclose(morans_i_map(noise + 1.0e6), expected, atol=1e-5)
        assert_allclose(morans_i_map(noise * 1.0e-8 - 3.0), expected,
                        atol=1e-5)

    def test_float32(self, noise):
        expected = morans_i_map(noise.astype(np.float32).astype(float))
        result = morans_i_map(noise.astype(np.float32))
        assert_allclose(result, expected, atol=1e-5)

        # the weights are used with the precision of the data
        rng = np.random.default_rng(9)
        weights = rng.uniform(0.5, 2.0, noise.shape)
        expected = morans_i_map(noise, weights=weights)
        for dtypes in ((np.float32, np.float32), (np.float32, np.float64),
                       (np.float64, np.float32)):
            result = morans_i_map(noise.astype(dtypes[0]),
                                  weights=weights.astype(dtypes[1]))
            assert_allclose(result, expected, atol=1e-4)

    def test_non_contiguous_and_int(self, noise):
        expected = morans_i_map(np.ascontiguousarray(noise[::2, ::2]))
        assert_equal(morans_i_map(noise[::2, ::2]), expected)

        data = np.round(noise * 100).astype(int)
        assert_equal(morans_i_map(data), morans_i_map(data.astype(float)))

    def test_quantity(self, noise):
        result = morans_i_map(noise << u.Jy)
        assert not isinstance(result, u.Quantity)
        assert_equal(result, morans_i_map(noise))

    def test_n_threads(self, noise):
        data = noise.copy()
        data[50, 60] = np.nan
        rng = np.random.default_rng(8)
        weights = rng.uniform(0.5, 2.0, data.shape)
        for kwargs in ({}, {'weights': weights, 'n_box_passes': 3,
                            'patch_size': 9}):
            expected = morans_i_map(data, **kwargs)
            for n_threads in (2, 3, 500):
                result = morans_i_map(data, n_threads=n_threads, **kwargs)
                assert_equal(np.isnan(result), np.isnan(expected))
                assert_allclose(result, expected, atol=1e-6)

    def test_constant_data(self):
        """
        Test that patches with zero variance are NaN.
        """
        assert np.all(np.isnan(morans_i_map(np.full((30, 30), 5.0))))

        rng = np.random.default_rng(2)
        data = rng.normal(100.0, 1.0, (60, 60))
        data[:, 30:] = 0.0
        result = morans_i_map(data, patch_size=11)
        assert np.all(np.isnan(result[:, 36:]))
        assert np.all(np.isfinite(result[:, :30]))

    def test_all_masked(self, noise):
        assert np.all(np.isnan(morans_i_map(np.full((30, 30), np.nan))))
        mask = np.ones(noise.shape, dtype=bool)
        assert np.all(np.isnan(morans_i_map(noise, mask=mask)))

    def test_inputs(self, noise):
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            morans_i_map(np.ones(10))

        match = 'patch_size must be an odd integer'
        for patch_size in (20, 1, 3, -5, 21.0, True):
            with pytest.raises(ValueError, match=match):
                morans_i_map(noise, patch_size=patch_size)

        match = 'mask must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            morans_i_map(noise, mask=np.zeros((3, 3), dtype=bool))

        match = 'weights must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            morans_i_map(noise, weights=np.ones((3, 3)))

        match = 'n_box_passes must be a positive integer'
        for n_box_passes in (0, 1.5, True):
            with pytest.raises(ValueError, match=match):
                morans_i_map(noise, n_box_passes=n_box_passes)

        match = 'n_threads must be a positive integer'
        for n_threads in (0, 1.5, True):
            with pytest.raises(ValueError, match=match):
                morans_i_map(noise, n_threads=n_threads)

    def test_inputs_unchanged(self, noise):
        data = noise.copy()
        data[3, 3] = np.nan
        original = data.copy()
        morans_i_map(data)
        assert_equal(data, original)


class TestMakeMoransIMask:
    def test_noise(self, noise):
        """
        Test that almost none of a pure noise image is masked.
        """
        mask = make_morans_i_mask(noise)
        assert mask.shape == noise.shape
        assert mask.dtype == bool
        assert np.mean(mask) < 0.01
        assert not np.any(make_morans_i_mask(noise, n_sigma=5.0))

    def test_correlated_noise(self):
        """
        Test that correlated noise, which has a large positive Moran's
        I everywhere, is not masked with the default threshold.
        """
        rng = np.random.default_rng(3)
        data = uniform_filter(rng.normal(0.0, 1.0, (301, 301)), 2)
        assert np.median(morans_i_map(data)) > 0.3
        mask = make_morans_i_mask(data)
        assert np.mean(mask) < 0.01

    def test_sources(self):
        data = make_100gaussians_image()
        mask = make_morans_i_mask(data)
        assert mask.dtype == bool
        assert 0.05 < np.mean(mask) < 0.5

        # the brightest pixels are sources
        assert np.all(mask[data > np.percentile(data, 99.5)])

        # the unmasked pixels are consistent with the true background
        # (mean of 5 and standard deviation of 2)
        assert_allclose(np.mean(data[~mask]), 5.0, atol=0.2)
        assert np.mean(data[mask]) > 6.0

    def test_percent(self):
        data = make_100gaussians_image()
        mask = make_morans_i_mask(data, percent=35.0, opening_iterations=0)
        assert 0.34 < np.mean(mask) < 0.40

        # percent takes precedence over n_sigma
        mask2 = make_morans_i_mask(data, n_sigma=100.0, percent=35.0,
                                   opening_iterations=0)
        assert_equal(mask2, mask)

    @pytest.mark.parametrize('opening_iterations', [0, 1, 2, 3])
    @pytest.mark.parametrize('percent', [20.0, 50.0])
    @pytest.mark.parametrize('shape', [(300, 500), (30, 40)])
    def test_morphology(self, shape, percent, opening_iterations):
        """
        Test the hole filling and the opening against SciPy.
        """
        data = make_100gaussians_image()[:shape[0], :shape[1]]
        mask = make_morans_i_mask(data, percent=percent,
                                  opening_iterations=opening_iterations)

        stat = morans_i_map(data)
        threshold = np.percentile(stat[np.isfinite(stat)], 100.0 - percent)
        expected = _reference_opening(binary_fill_holes(stat > threshold),
                                      opening_iterations)
        assert np.any(expected)
        assert_equal(mask, expected)

    def test_fill_holes(self):
        """
        Test that the hole of a ring-shaped source is filled.
        """
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 1.0, (301, 301))
        yy, xx = np.mgrid[:301, :301]
        radius = np.hypot(xx - 150, yy - 150)
        ring = (radius > 90) & (radius < 110)
        data[ring] += rng.normal(0.0, 30.0, (301, 301))[ring] + 50.0
        mask = make_morans_i_mask(data, patch_size=5, percent=15.0,
                                  opening_iterations=0)
        assert np.all(mask[radius < 100])
        assert not mask[0, 0]

        stat = morans_i_map(data, patch_size=5)
        threshold = np.percentile(stat[np.isfinite(stat)], 85.0)
        assert not np.all((stat > threshold)[radius < 100])
        assert_equal(mask, binary_fill_holes(stat > threshold))

    def test_weights_and_n_box_passes(self):
        """
        Test that the weights and the number of box passes are used
        for the statistic.
        """
        data = make_100gaussians_image()
        rng = np.random.default_rng(7)
        weights = rng.uniform(0.2, 5.0, data.shape)
        weights[100:140, 200:260] = 0.0
        mask = make_morans_i_mask(data, patch_size=11, n_box_passes=2,
                                  percent=30.0, weights=weights)
        assert not np.any(mask[105:135, 205:255])

        stat = morans_i_map(data, patch_size=11, n_box_passes=2,
                            weights=weights)
        threshold = np.percentile(stat[np.isfinite(stat)], 70.0)
        expected = _reference_opening(binary_fill_holes(stat > threshold),
                                      2)
        assert_equal(mask, expected)
        assert not np.array_equal(
            mask, make_morans_i_mask(data, patch_size=11, percent=30.0))

    def test_n_sigma(self):
        data = make_100gaussians_image()
        low = make_morans_i_mask(data, n_sigma=3.0, opening_iterations=0)
        high = make_morans_i_mask(data, n_sigma=20.0, opening_iterations=0)
        assert np.sum(high) < np.sum(low)
        assert np.all(low[high])

    @pytest.mark.parametrize(('patch_size', 'n_box_passes', 'iterations'),
                             [(5, 1, 0), (11, 1, 3), (21, 1, 8), (21, 2, 2),
                              (11, 3, 2)])
    def test_opening_iterations_default(self, patch_size, n_box_passes,
                                        iterations):
        """
        Test that the default number of opening iterations is set by
        the patch size for a single box pass and is 2 otherwise.
        """
        data = make_100gaussians_image()
        kwargs = {'patch_size': patch_size, 'n_box_passes': n_box_passes,
                  'percent': 30.0}
        mask = make_morans_i_mask(data, **kwargs)
        expected = make_morans_i_mask(data, opening_iterations=iterations,
                                      **kwargs)
        assert_equal(mask, expected)
        other = make_morans_i_mask(data, opening_iterations=iterations + 2,
                                   **kwargs)
        assert not np.array_equal(mask, other)

    def test_opening_at_edges(self):
        """
        Test that the opening does not remove the flagged regions of
        sources at the image edges.
        """
        rng = np.random.default_rng(10)
        data = rng.normal(0.0, 1.0, (151, 201))
        yy, xx = np.mgrid[:151, :201]
        for ycen, xcen in ((0, 60), (75, 200), (150, 0), (75, 100)):
            data += 50.0 * np.exp(-((xx - xcen)**2 + (yy - ycen)**2) / 8.0)
        mask = make_morans_i_mask(data)
        for ycen, xcen in ((0, 60), (75, 200), (150, 0), (75, 100)):
            assert mask[ycen, xcen]
        assert np.mean(mask) < 0.15

    def test_opening_iterations(self):
        data = make_100gaussians_image()
        mask0 = make_morans_i_mask(data, n_sigma=2.0, opening_iterations=0)
        mask2 = make_morans_i_mask(data, n_sigma=2.0, opening_iterations=2)
        assert np.sum(mask2) < np.sum(mask0)
        assert np.all(mask0[mask2])

    def test_mask(self):
        """
        Test that the input mask is excluded from the statistic and is
        not itself flagged as a source.
        """
        data = make_100gaussians_image()
        bad = np.zeros(data.shape, dtype=bool)
        bad[:, 250:] = True
        data[bad] = 1.0e10
        mask = make_morans_i_mask(data, mask=bad)
        assert not np.any(mask[bad])
        assert np.any(mask[~bad])

    def test_no_valid_pixels(self):
        mask = make_morans_i_mask(np.full((50, 50), np.nan))
        assert mask.dtype == bool
        assert not np.any(mask)

    def test_n_threads(self):
        data = make_100gaussians_image()
        assert_equal(make_morans_i_mask(data, n_threads=4),
                     make_morans_i_mask(data))

    def test_inputs(self, noise):
        match = 'n_sigma must be a positive number'
        for n_sigma in (0.0, -1.0, np.nan):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, n_sigma=n_sigma)

        match = 'percent must be between 0 and 100'
        for percent in (-1.0, 0.0, 100.0, 150.0):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, percent=percent)

        match = 'opening_iterations must be a non-negative integer'
        for iterations in (-1, 1.5, True):
            with pytest.raises(ValueError, match=match):
                make_morans_i_mask(noise, opening_iterations=iterations)

        match = 'patch_size must be an odd integer'
        with pytest.raises(ValueError, match=match):
            make_morans_i_mask(noise, patch_size=20)
