# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the noise module.
"""

import numpy as np
import pytest
from astropy.convolution import Box2DKernel, Gaussian2DKernel
from numpy.testing import assert_allclose, assert_array_equal

from photutils.datasets import make_noise_image as old_make_noise_image
from photutils.simulation import (apply_poisson_noise, make_noise_image,
                                  noise_correlation_factor)
from photutils.utils.exceptions import PhotutilsDeprecationWarning


def test_apply_poisson_noise():
    """
    Test if Poisson noise is applied correctly.
    """
    shape = (100, 100)
    data = np.ones(shape)
    result = apply_poisson_noise(data)
    assert result.shape == shape
    assert_allclose(result.mean(), 1.0, atol=1.0)


def test_apply_poisson_noise_dtype():
    """
    Test if the output data type is the same as the input data type.
    """
    data = np.ones((10, 10), dtype=float) * 5.0
    result = apply_poisson_noise(data, seed=0)
    assert result.dtype == data.dtype


def test_apply_poisson_noise_seed():
    """
    Test if Poisson noise is reproducible with a given seed.
    """
    shape = (100, 100)
    data = np.ones(shape) * 10.0
    result1 = apply_poisson_noise(data, seed=12345)
    result2 = apply_poisson_noise(data, seed=12345)
    assert_allclose(result1, result2)


def test_apply_poisson_noise_negative():
    """
    Test if negative image values raises ValueError.
    """
    shape = (100, 100)
    data = np.zeros(shape) - 1.0
    match = 'data must not contain any negative values'
    with pytest.raises(ValueError, match=match):
        apply_poisson_noise(data)


def test_apply_poisson_noise_nonfinite():
    """
    Test if nonfinite image values raise ValueError.
    """
    match = 'data must not contain any nonfinite values'
    for value in (np.nan, np.inf, -np.inf):
        data = np.array([[1.0, value], [2.0, 3.0]])
        with pytest.raises(ValueError, match=match):
            apply_poisson_noise(data)


def test_make_noise_image():
    """
    Test if noise image is generated correctly.
    """
    shape = (100, 100)
    image = make_noise_image(shape, distribution='gaussian', mean=0.0,
                             stddev=2.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 0.0, atol=1.0)


def test_make_noise_image_poisson():
    """
    Test noise image with Poisson noise.
    """
    shape = (100, 100)
    image = make_noise_image(shape, distribution='poisson', mean=1.0)
    assert image.shape == shape
    assert_allclose(image.mean(), 1.0, atol=1.0)


def test_make_noise_image_nomean():
    """
    Test invalid inputs to make_noise_image.
    """
    shape = (100, 100)
    match = 'Invalid distribution:'
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, distribution='invalid', mean=0, stddev=2.0)

    match = "'mean' must be input"
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, distribution='gaussian', stddev=2.0)

    match = "'stddev' must be input for Gaussian noise"
    with pytest.raises(ValueError, match=match):
        make_noise_image(shape, distribution='gaussian', mean=2.0)


def test_make_noise_image_seed():
    """
    Test if noise image is reproducible with a given seed.
    """
    shape = (100, 100)
    image1 = make_noise_image(shape, distribution='gaussian', mean=0.0,
                              stddev=2.0, seed=12345)
    image2 = make_noise_image(shape, distribution='gaussian', mean=0.0,
                              stddev=2.0, seed=12345)
    assert_allclose(image1, image2)


def test_kernel_stddev():
    """
    Test that correlated noise keeps the requested per-pixel mean and
    standard deviation, including at the image edges.
    """
    kernel = Gaussian2DKernel(2)
    image = make_noise_image((1024, 1024), mean=1.0, stddev=3.0,
                             kernel=kernel, seed=0)
    assert image.shape == (1024, 1024)
    assert_allclose(np.std(image), 3.0, rtol=0.01)
    assert_allclose(np.mean(image), 1.0, atol=0.1)

    edge = np.ones(image.shape, dtype=bool)
    edge[20:-20, 20:-20] = False
    assert_allclose(np.std(image[edge]), np.std(image[~edge]), rtol=0.05)


def test_kernel_smoothed_variance():
    """
    Test that the noise of block means matches the per-pixel standard
    deviation times the noise correlation factor.
    """
    kernel = Gaussian2DKernel(1)
    image = make_noise_image((1024, 1024), mean=0.0, stddev=2.0,
                             kernel=kernel, seed=0)
    block_means = image.reshape(32, 32, 32, 32).mean(axis=(1, 3))
    expected = 2.0 * noise_correlation_factor(kernel) / 32
    assert_allclose(np.std(block_means), expected, rtol=0.1)

    white = make_noise_image((1024, 1024), mean=0.0, stddev=2.0, seed=0)
    block_means = white.reshape(32, 32, 32, 32).mean(axis=(1, 3))
    assert_allclose(np.std(block_means), 2.0 / 32, rtol=0.1)


def test_noise_correlation_factor_values():
    """
    Test the noise correlation factor of simple kernels.
    """
    assert_allclose(noise_correlation_factor(np.ones((1, 1))), 1.0)
    assert_allclose(noise_correlation_factor(Box2DKernel(3)), 3.0)
    assert_allclose(noise_correlation_factor(np.full((3, 3), 5.0)), 3.0)
    assert_allclose(noise_correlation_factor(Gaussian2DKernel(3)),
                    2 * np.sqrt(np.pi) * 3, rtol=1e-3)


def test_kernel_poisson_error():
    """
    Test that a kernel with Poisson noise raises an error.
    """
    match = "'kernel' can be used only with Gaussian noise"
    with pytest.raises(ValueError, match=match):
        make_noise_image((10, 10), distribution='poisson', mean=5.0,
                         kernel=np.ones((3, 3)))


def test_kernel_not_2d_error():
    """
    Test that a kernel that is not a 2D array raises an error.
    """
    match = 'kernel must be a 2D array'
    for kernel in (np.ones(3), np.ones((3, 3, 3)), 2.0):
        with pytest.raises(ValueError, match=match):
            make_noise_image((10, 10), mean=0.0, stddev=1.0, kernel=kernel)
        with pytest.raises(ValueError, match=match):
            noise_correlation_factor(kernel)


def test_kernel_none_unchanged():
    """
    Test that the output without a kernel equals the deprecated
    ``photutils.datasets.make_noise_image`` output.
    """
    image = make_noise_image((50, 60), mean=1.0, stddev=2.0, kernel=None,
                             seed=3)
    with pytest.warns(PhotutilsDeprecationWarning):
        old = old_make_noise_image((50, 60), mean=1.0, stddev=2.0, seed=3)
    assert_array_equal(image, old)
