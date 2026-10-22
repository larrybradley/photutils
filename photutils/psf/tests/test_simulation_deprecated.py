# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deprecation of photutils.psf.make_psf_model_image.
"""

import warnings

import pytest
from numpy.testing import assert_allclose, assert_array_equal

import photutils.simulation
from photutils.psf import CircularGaussianPRF, make_psf_model_image
from photutils.utils.exceptions import PhotutilsDeprecationWarning

# The warning itself is tested here, so it must not be an error
pytestmark = pytest.mark.filterwarnings(
    'ignore:This function was deprecated in version 3.2')


def test_psf_make_psf_model_image_deprecated():
    model = CircularGaussianPRF(fwhm=2.7)
    match = 'photutils.simulation.make_psf_model_image'
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        old_data, old_params = make_psf_model_image(
            (51, 61), model, 5, model_shape=(11, 11), seed=0)

    new_data, new_params = photutils.simulation.make_psf_model_image(
        (51, 61), model, 5, model_shape=(11, 11), seed=0)
    assert_array_equal(old_data, new_data)
    assert old_params.colnames == new_params.colnames
    for name in old_params.colnames:
        assert_array_equal(old_params[name], new_params[name])


def test_one_warning_per_call():
    # Set the filters here so the test does not depend on the pytest
    # configuration
    model = CircularGaussianPRF(fwhm=2.7)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        make_psf_model_image((51, 61), model, 5, model_shape=(11, 11))
    deprecations = [w for w in record
                    if issubclass(w.category, PhotutilsDeprecationWarning)]
    assert len(deprecations) == 1


def test_deprecated_output_pinned():
    """
    Test the deprecated output against values that do not depend on
    photutils.simulation.
    """
    model = CircularGaussianPRF(fwhm=2.7)
    with pytest.warns(PhotutilsDeprecationWarning):
        data, params = make_psf_model_image(
            (101, 101), model, 10, flux=(100, 200), min_separation=10,
            seed=0)

    assert data.shape == (101, 101)
    assert len(params) == 10
    assert_allclose(data.sum(), 1483.3508122243466, rtol=1e-12)
    assert_allclose(data.max(), 22.117569849307202, rtol=1e-12)
    assert_allclose(data[89, 72], 14.673146231963274, rtol=1e-12)
    assert_allclose(params['x_0'].sum(), 568.574738757422, rtol=1e-12)
    assert_allclose(params['y_0'].sum(), 554.90619211583, rtol=1e-12)
    assert_allclose(params['flux'].sum(), 1483.3509114877133, rtol=1e-12)
