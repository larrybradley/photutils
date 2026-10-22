# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the deprecation of the photutils.datasets functions.
"""

import warnings

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_array_equal

import photutils.datasets
import photutils.simulation
from photutils.utils._optional_deps import HAS_GWCS
from photutils.utils.exceptions import PhotutilsDeprecationWarning

# The warning itself is tested here, so it must not be an error
pytestmark = pytest.mark.filterwarnings(
    'ignore:This function was deprecated in version 3.2')


def _make_params_table():
    return QTable({'x_mean': [10.0], 'y_mean': [10.0], 'amplitude': [1.0]})


def _make_model_image():
    return photutils.datasets.make_model_image(
        (20, 20), Gaussian2D(), _make_params_table(), x_name='x_mean',
        y_name='y_mean')


CALLS = {
    'make_model_params': lambda: photutils.datasets.make_model_params(
        (50, 50), 3, flux=(1, 10)),
    'make_random_models_table': lambda:
        photutils.datasets.make_random_models_table(
            3, {'x_mean': [0, 100]}),
    'params_table_to_models': lambda:
        photutils.datasets.params_table_to_models(
            _make_params_table(), Gaussian2D()),
    'make_model_image': _make_model_image,
    'make_noise_image': lambda: photutils.datasets.make_noise_image(
        (5, 5), mean=0, stddev=1),
    'apply_poisson_noise': lambda: photutils.datasets.apply_poisson_noise(
        np.ones((5, 5))),
    'make_wcs': lambda: photutils.datasets.make_wcs((10, 10)),
    'make_gwcs': lambda: photutils.datasets.make_gwcs((10, 10)),
    'make_4gaussians_image': photutils.datasets.make_4gaussians_image,
    'make_100gaussians_image': photutils.datasets.make_100gaussians_image,
}


@pytest.mark.parametrize('name', list(CALLS))
def test_deprecated_warns(name):
    if name == 'make_gwcs' and not HAS_GWCS:
        pytest.skip('gwcs is required')
    match = (rf'deprecated in version 3\.2.*photutils\.simulation\.{name} '
             'instead')
    with pytest.warns(PhotutilsDeprecationWarning, match=match):
        CALLS[name]()


def test_deprecated_same_results():
    old = photutils.datasets.make_4gaussians_image()
    new = photutils.simulation.make_4gaussians_image()
    assert_array_equal(old, new)

    old = photutils.datasets.make_noise_image((5, 5), mean=0, stddev=1,
                                              seed=0)
    new = photutils.simulation.make_noise_image((5, 5), mean=0, stddev=1,
                                                seed=0)
    assert_array_equal(old, new)

    old = photutils.datasets.make_model_params((50, 50), 5, flux=(1, 10),
                                               seed=0)
    new = photutils.simulation.make_model_params((50, 50), 5, flux=(1, 10),
                                                 seed=0)
    assert old.colnames == new.colnames
    for name in old.colnames:
        assert_array_equal(old[name], new[name])

    old = photutils.datasets.make_wcs((10, 10))
    new = photutils.simulation.make_wcs((10, 10))
    assert old.to_header() == new.to_header()


def test_one_warning_per_call():
    # Set the filters here so the test does not depend on the pytest
    # configuration
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        photutils.datasets.make_4gaussians_image()
    deprecations = [w for w in record
                    if issubclass(w.category, PhotutilsDeprecationWarning)]
    assert len(deprecations) == 1


def test_old_submodule_paths():
    from photutils.datasets.images import make_model_image

    with pytest.warns(PhotutilsDeprecationWarning,
                      match='deprecated in version 3.2'):
        make_model_image((20, 20), Gaussian2D(), _make_params_table(),
                         x_name='x_mean', y_name='y_mean')
