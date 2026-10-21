# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _utils module.
"""

import warnings

import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.ndimage import correlate

from photutils.segmentation import detect_sources
from photutils.segmentation.utils import (_make_binary_structure,
                                          _mask_to_mirrored_value,
                                          make_2dgaussian_kernel,
                                          matched_filter)
from photutils.utils.exceptions import NoDetectionsWarning


def test_make_2dgaussian_kernel():
    """
    Test make 2dgaussian kernel.
    """
    kernel = make_2dgaussian_kernel(1.0, size=3)
    expected = np.array([[0.01411809, 0.0905834, 0.01411809],
                         [0.0905834, 0.58119403, 0.0905834],
                         [0.01411809, 0.0905834, 0.01411809]])
    assert_allclose(kernel.array, expected, atol=1.0e-6)
    assert_allclose(kernel.array.sum(), 1.0)


def test_make_2dgaussian_kernel_modes():
    """
    Test make 2dgaussian kernel modes.
    """
    kernel = make_2dgaussian_kernel(3.0, 5)
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='center')
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='linear_interp')
    assert_allclose(kernel.array.sum(), 1.0)

    kernel = make_2dgaussian_kernel(3.0, 5, mode='integrate')
    assert_allclose(kernel.array.sum(), 1.0)


def test_make_binary_structure():
    """
    Test make binary structure.
    """
    footprint = _make_binary_structure(1, 4)
    assert_allclose(footprint, np.array([1, 1, 1]))

    footprint = _make_binary_structure(3, 4)
    assert_equal(footprint[0, 0], np.array([False, False, False]))
    expected = np.array([[[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]],
                         [[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]],
                         [[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]]])
    assert_equal(footprint.astype(int), expected)


def test_mask_to_mirrored_value():
    """
    Test mask to mirrored value.
    """
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    data_ref = data.copy()
    data_ref[0, 0] = data[4, 4]
    data_ref[1, 1] = data[3, 3]
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_range():
    """
    Test mask_to_mirrored_value when mirrored pixels are outside the
    image.
    """
    center = (3.0, 3.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[2, 2] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.0
    data_ref[1, 1] = 0.0
    data_ref[2, 2] = data[4, 4]
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_masked():
    """
    Test mask_to_mirrored_value when mirrored pixels are also in the
    ``replace_mask``.
    """
    center = (2.0, 2.0)
    data = np.arange(25).reshape(5, 5)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    mask[3, 3] = True
    mask[4, 4] = True
    data_ref = data.copy()
    data_ref[0, 0] = 0.0
    data_ref[1, 1] = 0.0
    data_ref[3, 3] = 0.0
    data_ref[4, 4] = 0.0
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    mirror_data = _mask_to_mirrored_value(data, mask, center)
    assert_allclose(mirror_data, data_ref, rtol=0, atol=1.0e-6)


def test_mask_to_mirrored_value_mask_keyword():
    """
    Test mask_to_mirrored_value when mirrored pixels are masked (via the
    mask keyword).
    """
    center = (2.0, 2.0)
    data = np.arange(25.0).reshape(5, 5)
    replace_mask = np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 2] = True
    data[4, 2] = np.nan
    mask[4, 2] = True
    result = _mask_to_mirrored_value(data, replace_mask, center, mask=mask)
    assert result[0, 2] == 0


def test_mask_to_mirrored_value_return_uncorrected():
    """
    Test that return_uncorrected reports pixels whose mirror was outside
    the image.
    """
    center = (2.0, 2.0)
    data = np.arange(16.0).reshape(4, 4)
    replace_mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 0] = True  # mirror at (4, 4) is outside the image
    replace_mask[1, 2] = True  # mirror at (3, 2) is available
    result, uncorrected = _mask_to_mirrored_value(
        data, replace_mask, center, return_uncorrected=True)
    assert uncorrected[0, 0]
    assert not uncorrected[1, 2]
    assert np.count_nonzero(uncorrected) == 1
    assert result[0, 0] == 0
    assert result[1, 2] == data[3, 2]

    # The two-value return preserves the corrected data
    result_only = _mask_to_mirrored_value(data, replace_mask, center)
    assert_equal(result, result_only)


def test_mask_to_mirrored_value_return_uncorrected_masked():
    """
    Test that return_uncorrected reports pixels whose mirror is excluded
    by the ``mask`` keyword or is itself in ``replace_mask``.
    """
    center = (2.0, 2.0)
    data = np.arange(25.0).reshape(5, 5)
    replace_mask = np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    replace_mask[0, 2] = True  # mirror at (4, 2) is in mask
    replace_mask[1, 1] = True  # mirror at (3, 3) is in replace_mask
    replace_mask[3, 3] = True
    mask[4, 2] = True
    result, uncorrected = _mask_to_mirrored_value(
        data, replace_mask, center, mask=mask, return_uncorrected=True)
    assert_equal(np.nonzero(uncorrected),
                 (np.array([0, 1, 3]), np.array([2, 1, 3])))
    assert result[0, 2] == 0
    assert result[1, 1] == 0
    assert result[3, 3] == 0


class TestMatchedFilter:
    """
    Test the matched_filter function.
    """

    @pytest.fixture
    def kernel(self):
        # An asymmetric kernel distinguishes correlation from
        # convolution
        yy, xx = np.mgrid[-3:4, -2:3]
        kernel = np.exp(-((xx - 0.7)**2 / 2.0 + (yy + 0.4)**2 / 5.0))
        return kernel / kernel.sum()

    def test_uniform_error(self, kernel):
        """
        Test that the filtered image is the correlation with the kernel
        for uniform noise, including for images that are taller than
        the strips of rows used internally.
        """
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (600, 40))
        filtered, filtered_err = matched_filter(data, kernel, 2.5)
        expected = correlate(data, kernel, mode='constant')
        inside = (slice(3, -3), slice(2, -2))
        assert_allclose(filtered[inside], expected[inside], atol=1.0e-12)
        assert_allclose(filtered_err[inside],
                        2.5 * np.sqrt(np.sum(kernel**2)))

        # An array of errors and an unnormalized kernel give the same
        # result
        error = np.full(data.shape, 2.5)
        filtered2, filtered_err2 = matched_filter(data, 7.0 * kernel, error)
        assert_allclose(filtered2, filtered, atol=1.0e-12)
        assert_allclose(filtered_err2, filtered_err)

    def test_borders(self, kernel):
        """
        Test that no values are assumed beyond the image borders. The
        filtered value at the center of a source with the shape of the
        kernel does not depend on how much of it is in the image.
        """
        amplitude = 5.0
        expected = amplitude * np.sum(kernel**2)
        for ycen, xcen in ((15, 20), (0, 0), (1, 38), (29, 20)):
            data = np.zeros((30 + 6, 40 + 4))
            data[ycen:ycen + 7, xcen:xcen + 5] = amplitude * kernel
            data = data[3:-3, 2:-2]
            filtered, filtered_err = matched_filter(data, kernel, 1.0)
            assert_allclose(filtered[ycen, xcen], expected)
            assert np.all(np.isfinite(filtered_err))
        assert filtered_err[0, 0] > filtered_err[15, 20]

    def test_excluded_pixels(self, kernel):
        """
        Test that masked, nonfinite, and infinite-error pixels are
        excluded.
        """
        rng = np.random.default_rng(1)
        shape = (40, 50)
        data = rng.normal(10.0, 1.0, shape)
        bad = rng.uniform(size=shape) < 0.1
        data_bad = data.copy()
        data_bad[bad] = 1.0e9
        result0 = matched_filter(data_bad, kernel, 1.0, mask=bad)

        error = np.ones(shape)
        error[bad] = np.inf
        result1 = matched_filter(data_bad, kernel, error)
        error[bad] = -1.0
        result2 = matched_filter(data_bad, kernel, error)
        data_bad[bad] = np.nan
        result3 = matched_filter(data_bad, kernel, 1.0)
        for result in (result1, result2, result3):
            assert_allclose(result[0], result0[0])
            assert_allclose(result[1], result0[1])
        assert np.max(result0[0]) < 20.0

        # Excluding pixels increases the uncertainty
        _, filtered_err = matched_filter(data, kernel, 1.0)
        assert np.all(result0[1] > 0.999999 * filtered_err)
        assert np.any(result0[1] > 1.1 * filtered_err)

        # The filtered value at the center of a source with the shape
        # of the kernel does not depend on the excluded pixels
        source = np.zeros(shape)
        source[10:17, 20:25] = 4.0 * kernel
        source_bad = source.copy()
        source_bad[bad] = -50.0
        filtered0, _ = matched_filter(source, kernel, 1.0)
        filtered1, _ = matched_filter(source_bad, kernel, 1.0, mask=bad)
        assert_allclose(filtered0[13, 22], 4.0 * np.sum(kernel**2))
        assert_allclose(filtered1[13, 22], filtered0[13, 22])

    @pytest.mark.parametrize('value', [1.0e12, 1.0e20, 1.0e30])
    def test_extreme_value(self, kernel, value):
        """
        Test that one extreme pixel value changes only the pixels within
        the kernel footprint around it, where the result is exact.
        """
        rng = np.random.default_rng(2)
        data = rng.normal(0.0, 1.0, (300, 300))
        result0 = matched_filter(data, kernel, 1.0)
        data[100, 100] = value
        result1 = matched_filter(data, kernel, 1.0)

        footprint = np.zeros(data.shape, dtype=bool)
        footprint[97:104, 98:103] = True
        assert_allclose(result1[0][~footprint], result0[0][~footprint],
                        atol=1.0e-12)
        assert_equal(result1[1], result0[1])
        expected = correlate(data, kernel, mode='constant')
        assert_allclose(result1[0][footprint], expected[footprint],
                        rtol=1.0e-12)

    @pytest.mark.parametrize('value', [1.0e-5, 1.0e-10])
    def test_tiny_error(self, kernel, value):
        """
        Test that one pixel with a tiny uncertainty changes only the
        pixels within the kernel footprint around it.
        """
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, (300, 300))
        error = np.ones(data.shape)
        result0 = matched_filter(data, kernel, error)
        error[100, 100] = value
        result1 = matched_filter(data, kernel, error)

        footprint = np.zeros(data.shape, dtype=bool)
        footprint[97:104, 98:103] = True
        for value0, value1 in zip(result0, result1, strict=True):
            assert_allclose(value1[~footprint], value0[~footprint],
                            atol=1.0e-12)
        assert np.all(np.isfinite(result1[1]))

    def test_float32_input(self, kernel):
        """
        Test that float32 inputs give the float64 result.
        """
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 1.0, (600, 40)).astype(np.float32)
        error = rng.uniform(1.0, 2.0, data.shape).astype(np.float32)
        result0 = matched_filter(data.astype(float), kernel,
                                 error.astype(float))
        result1 = matched_filter(data, kernel, error)
        for value0, value1 in zip(result0, result1, strict=True):
            assert value1.dtype == np.float64
            assert_equal(value1, value0)

    @staticmethod
    def _correlated_noise(shape, seed):
        """
        Make noise with a per-pixel standard deviation of 1 that is
        correlated between neighboring pixels, as in a resampled image.
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0, shape)
        return correlate(noise, np.full((3, 3), 1.0 / 3.0), mode='wrap')

    def test_error_scale_number(self, kernel):
        rng = np.random.default_rng(5)
        data = rng.normal(0.0, 1.0, (60, 70))
        result0 = matched_filter(data, kernel, 1.0)
        result1 = matched_filter(data, kernel, 1.0, error_scale=2.5)
        assert_equal(result1[0], result0[0])
        assert_allclose(result1[1], 2.5 * result0[1])

    def test_error_scale_empirical(self, kernel):
        """
        Test that the empirical error scale corrects the S/N of
        correlated noise and does not change that of white noise.
        """
        data = self._correlated_noise((400, 400), 6)
        filtered, filtered_err = matched_filter(data, kernel, 1.0)
        assert np.std(filtered / filtered_err) > 1.5
        filtered, filtered_err = matched_filter(data, kernel, 1.0,
                                                error_scale='empirical')
        assert_allclose(np.std(filtered / filtered_err), 1.0, rtol=0.05)

        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, (400, 400))
        result0 = matched_filter(data, kernel, 1.0)
        result1 = matched_filter(data, kernel, 1.0, error_scale='empirical')
        assert_equal(result1[0], result0[0])
        assert_allclose(result1[1], result0[1], rtol=0.03)

    def test_error_scale_source_mask(self, kernel):
        """
        Test that source pixels can be left out of the empirical error
        scale, and that they are still filtered.
        """
        rng = np.random.default_rng(8)
        data = rng.normal(0.0, 1.0, (400, 400))
        source_mask = np.zeros(data.shape, dtype=bool)
        source_mask[:, :160] = True
        data[source_mask] += rng.uniform(0.0, 3.0, np.sum(source_mask))
        result0 = matched_filter(data, kernel, 1.0, error_scale='empirical')
        result1 = matched_filter(data, kernel, 1.0, error_scale='empirical',
                                 source_mask=source_mask)
        _, filtered_err = matched_filter(data, kernel, 1.0)
        assert np.all(result0[1] > 1.1 * filtered_err)
        assert_allclose(result1[1], filtered_err, rtol=0.03)
        assert_equal(result1[0], result0[0])

    def test_error_scale_inputs(self, kernel):
        data = np.random.default_rng(9).normal(0.0, 1.0, (30, 40))
        match = "error_scale must be None, a positive number, or 'empirical'"
        for error_scale in (0.0, -1.0, np.inf, 'auto', [1.0, 2.0]):
            with pytest.raises(ValueError, match=match):
                matched_filter(data, kernel, 1.0, error_scale=error_scale)
        source_mask = np.zeros(data.shape, dtype=bool)
        match = "source_mask can be used only with error_scale='empirical'"
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, 1.0, source_mask=source_mask)
        match = 'source_mask must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, 1.0, error_scale='empirical',
                           source_mask=np.zeros((5, 5), dtype=bool))
        match = 'The empirical error scale could not be estimated'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, 1.0, error_scale='empirical',
                           source_mask=~source_mask)
        with pytest.raises(ValueError, match=match):
            matched_filter(np.zeros(data.shape), kernel, 1.0,
                           error_scale='empirical')

    def test_no_included_pixels(self, kernel):
        data = np.ones((40, 50))
        mask = np.zeros(data.shape, dtype=bool)
        mask[:, :25] = True
        filtered, filtered_err = matched_filter(data, kernel, 1.0, mask=mask)
        assert_equal(filtered[:, :20], 0.0)
        assert np.all(np.isinf(filtered_err[:, :20]))
        assert_allclose(filtered[5:-5, 30:-5], 1.0)
        assert np.all(np.isfinite(filtered_err[:, 30:]))

    def test_variable_error(self):
        """
        Test a source next to a region with larger noise, which is
        found with the matched filter and not with a convolution.
        """
        size = 16
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        error = np.ones((size, size))
        error[yy > xx] = 4.0
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0],
                           [1.0, 2.0, 1.0]])
        rng = np.random.default_rng(5)
        n_matched = 0
        n_convolved = 0
        n_trials = 50
        for _ in range(n_trials):
            data = error * rng.normal(size=(size, size))
            data[6:9, 6:9] += 3.0 * kernel
            filtered, filtered_err = matched_filter(data, kernel, error)
            convolved = correlate(data, kernel / kernel.sum(),
                                  mode='constant')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', NoDetectionsWarning)
                segm = detect_sources(filtered / filtered_err, 3.0,
                                      n_pixels=5)
                n_matched += segm is not None and segm.data[7, 7] > 0
                segm = detect_sources(convolved / error, 3.0, n_pixels=5)
                n_convolved += segm is not None and segm.data[7, 7] > 0
        assert n_matched == n_trials
        assert n_convolved < 0.7 * n_trials

    def test_units(self, kernel):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (30, 40))
        result0 = matched_filter(data, kernel, 2.0)
        result1 = matched_filter(data << u.Jy, kernel, 2.0 << u.Jy)
        for value0, value1 in zip(result0, result1, strict=True):
            assert value1.unit == u.Jy
            assert_allclose(value1.value, value0)

    def test_kernel2d(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, (30, 40))
        kernel = make_2dgaussian_kernel(3.0, size=5)
        result0 = matched_filter(data, kernel, 1.0)
        result1 = matched_filter(data, kernel.array, 1.0)
        assert_equal(result0[0], result1[0])
        assert_equal(result0[1], result1[1])

    def test_inputs(self, kernel):
        data = np.ones((30, 40))
        match = 'data must be a 2D array'
        with pytest.raises(ValueError, match=match):
            matched_filter(np.ones(10), kernel, 1.0)
        match = 'kernel must be a 2D array'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, np.ones(5), 1.0)
        match = 'kernel must have odd dimensions'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, np.ones((4, 5)), 1.0)
        match = 'kernel must have finite values and a positive sum'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, np.zeros((5, 5)), 1.0)
        match = 'mask must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, 1.0, mask=np.zeros((3, 3), bool))
        match = 'error must be a scalar or have the same shape as data'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, np.ones((3, 3)))
        match = 'error must have finite and positive values'
        with pytest.raises(ValueError, match=match):
            matched_filter(data, kernel, 0.0)
