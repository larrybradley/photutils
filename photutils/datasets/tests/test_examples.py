# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the examples module.
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_path
from numpy.testing import assert_allclose, assert_array_equal

from photutils.datasets import make_4gaussians_image, make_100gaussians_image
from photutils.utils._overlap import overlap_slices

_DATA_DIR = get_pkg_data_path('datasets', 'data', package='photutils')


def test_make_4gaussians_image():
    """
    Test the make_4gaussians_image function.
    """
    shape = (100, 200)
    data_sum = 177189.58
    image = make_4gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-6)


def test_make_4gaussians_image_no_noise():
    """
    Test the make_4gaussians_image function with no noise.
    """
    shape = (100, 200)
    image = make_4gaussians_image(noise=False)
    assert image.shape == shape
    assert image.min() >= 0


def test_make_100gaussians_image():
    """
    Test the make_100gaussians_image function.
    """
    shape = (300, 500)
    data_sum = 826059.53
    image = make_100gaussians_image()
    assert image.shape == shape
    assert_allclose(image.sum(), data_sum, rtol=1.0e-6)


def test_make_100gaussians_image_no_noise():
    """
    Test the make_100gaussians_image function with no noise.
    """
    shape = (300, 500)
    image = make_100gaussians_image(noise=False)
    assert image.shape == shape
    assert image.min() >= 0


@pytest.mark.parametrize(('func', 'noise', 'shape', 'total', 'peak',
                          'center'), [
    (make_4gaussians_image, False, (100, 200), 176721.43514018974,
     215.0, 7.776858376253685),
    (make_4gaussians_image, True, (100, 200), 177189.57898447308,
     220.9835273485732, 4.84886671105143),
    (make_100gaussians_image, False, (300, 500), 826315.9085482477,
     111.39861659770074, 5.000927231666569),
    (make_100gaussians_image, True, (300, 500), 826059.5320334453,
     112.6499728001872, 3.5061252965673537)])
def test_examples_pinned(func, noise, shape, total, peak, center):
    """
    Test that the example images keep their exact pixel values.
    """
    data = func(noise=noise)
    assert data.shape == shape
    assert_allclose(data.sum(), total, rtol=1e-12)
    assert_allclose(data.max(), peak, rtol=1e-12)
    assert_allclose(data[shape[0] // 2, shape[1] // 2], center, rtol=1e-12)


def _reference_image(filename, shape, bbox_factor=None):
    """
    Make a noiseless example image independently of the library code.
    """
    params = QTable.read(f'{_DATA_DIR}/{filename}', format='ascii.ecsv')
    image = np.zeros(shape, dtype=float)
    names = [name for name in params.colnames
             if name in Gaussian2D.param_names]
    for row in params:
        model = Gaussian2D(**{name: row[name] for name in names})
        if bbox_factor is None:
            bbox = model.bounding_box.bounding_box()
        else:
            bbox = model.bounding_box(factor=bbox_factor)
        model_shape = (int(np.ceil(bbox[0][1] - bbox[0][0])),
                       int(np.ceil(bbox[1][1] - bbox[1][0])))
        slc, _ = overlap_slices(shape, model_shape,
                                (model.y_mean.value, model.x_mean.value),
                                mode='trim')
        yy, xx = np.mgrid[slc]
        image[slc] += model(xx, yy)
    image += 5.0
    return image


@pytest.mark.parametrize(('func', 'filename', 'shape', 'bbox_factor'), [
    (make_4gaussians_image, '4gaussians_params.ecsv', (100, 200), None),
    (make_100gaussians_image, '100gaussians_params.ecsv', (300, 500), 6.0)])
def test_examples_match_reference(func, filename, shape, bbox_factor):
    """
    Test that the noiseless example images match a test-local reference.
    """
    expected = _reference_image(filename, shape, bbox_factor=bbox_factor)
    assert_array_equal(func(noise=False), expected)
