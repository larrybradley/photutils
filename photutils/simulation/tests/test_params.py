# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the params module.
"""

import numpy as np
import pytest
from astropy.table import QTable, Table, vstack
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose

from photutils.psf import CircularGaussianPSF
from photutils.simulation import (make_model_params, make_random_models_table,
                                  params_table_to_models)


def test_make_model_params():
    """
    Test the basic functionality of ``make_model_params``.
    """
    shape = (100, 100)
    n_sources = 10
    flux = (100, 1000)
    params = make_model_params(shape, n_sources, flux=flux)
    assert isinstance(params, Table)
    assert len(params) == 10
    cols = ('id', 'x_0', 'y_0', 'flux')
    for col in cols:
        assert col in params.colnames
        assert np.min(params[col]) >= 0
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]

    # Test extra parameters
    sigma = (1, 2)
    alpha = (0, 1)
    params = make_model_params((120, 100), 5, flux=flux, sigma=sigma,
                               alpha=alpha, min_separation=3, border_size=10,
                               seed=0)
    cols = ('id', 'x_0', 'y_0', 'flux', 'sigma', 'alpha')
    for col in cols:
        assert col in params.colnames
        assert np.min(params[col]) >= 0
    assert np.min(params['flux']) >= flux[0]
    assert np.max(params['flux']) <= flux[1]
    assert np.min(params['sigma']) >= sigma[0]
    assert np.max(params['sigma']) <= sigma[1]
    assert np.min(params['alpha']) >= alpha[0]
    assert np.max(params['alpha']) <= alpha[1]

    match = 'flux must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=(1, 2, 3))

    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=5)

    match = 'alpha must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=(1, 2), alpha=(1, 2, 3))

    match = 'n_sources must be >= 0'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, -1, flux=(1, 2))


def test_make_model_params_meta():
    """
    Test that the output table contains version metadata.
    """
    params = make_model_params((50, 50), 2, seed=0)
    assert 'date' in params.meta
    assert 'version' in params.meta


def test_make_model_params_nsources():
    """
    Test case when the number of the possible sources is less than
    ``n_sources``.
    """
    match = r'Unable to produce .* coordinates within the given shape'
    shape = (200, 500)
    n_sources = 100
    with pytest.warns(AstropyUserWarning, match=match):
        params = make_model_params(shape, n_sources, min_separation=50,
                                   amplitude=(100, 500), x_stddev=(1, 5),
                                   y_stddev=(1, 5), theta=(0, np.pi))
    assert len(params) < 100


def test_make_model_params_border_size():
    """
    Test case when the border size is too large for the given shape.
    """
    shape = (10, 10)
    n_sources = 10
    flux = (100, 1000)
    match = 'border_size is too large for the given shape'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, n_sources, flux=flux, border_size=20)


def test_make_model_params_reserved_names():
    """
    Test that a keyword argument cannot replace the ID or position
    columns.
    """
    shape = (100, 100)
    for name in ('id', 'x_0', 'y_0'):
        match = f'{name} cannot be used as a keyword argument'
        with pytest.raises(ValueError, match=match):
            make_model_params(shape, 5, **{name: (1, 2)})

    match = 'x_mean cannot be used as a keyword argument'
    with pytest.raises(ValueError, match=match):
        make_model_params(shape, 5, x_name='x_mean', y_name='y_mean',
                          x_mean=(1, 2))

    # x_0 is an ordinary parameter name when x_name is different
    params = make_model_params(shape, 5, x_name='x_mean', y_name='y_mean',
                               x_0=(1, 2))
    assert params.colnames == ['id', 'x_mean', 'y_mean', 'x_0']


def test_log_params_distribution():
    """
    Test that ``log_params`` samples the named parameters uniformly in
    log10.
    """
    params = make_model_params((10000, 10000), 20000, flux=(1, 1e4),
                               min_separation=0, log_params=('flux',),
                               seed=0)
    flux = params['flux']
    assert np.all(flux >= 1)
    assert np.all(flux < 1e4)
    assert_allclose(np.mean(np.log10(flux)), 2.0, atol=0.03)


def test_log_params_unchanged_without():
    """
    Test that the output without ``log_params`` matches the docstring
    example exactly.
    """
    params = make_model_params((100, 100), 5, flux=(100, 500),
                               min_separation=3, border_size=10, seed=0)
    assert_allclose(params['x_0'], [60.956935, 31.582937, 13.277882,
                                    11.322211, 75.061619], rtol=1e-7)
    assert_allclose(params['y_0'], [72.967865, 29.149555, 80.118738,
                                    14.685443, 36.889365], rtol=1e-7)
    assert_allclose(params['flux'], [291.99517, 192.94917, 420.75223,
                                     469.41206, 206.45211], rtol=1e-7)


def test_log_params_errors():
    """
    Test that invalid ``log_params`` values raise an error.
    """
    match = 'log_params name "r_eff" is not a keyword parameter'
    with pytest.raises(ValueError, match=match):
        make_model_params((100, 100), 5, flux=(1, 10),
                          log_params=('r_eff',))

    match = 'bounds of log parameter "flux" must be positive'
    with pytest.raises(ValueError, match=match):
        make_model_params((100, 100), 5, flux=(0, 10),
                          log_params=('flux',))
    with pytest.raises(ValueError, match=match):
        make_model_params((100, 100), 5, flux=(-1, 10),
                          log_params=('flux',))


def test_kind_column():
    """
    Test that ``kind`` adds a ``kind`` column after the parameter
    columns, so tables of different populations can be stacked.
    """
    stars = make_model_params((100, 100), 3, flux=(1, 10), kind='star',
                              seed=0)
    assert stars.colnames == ['id', 'x_0', 'y_0', 'flux', 'kind']
    assert np.all(stars['kind'] == 'star')

    gals = make_model_params((100, 100), 2, flux=(1, 10), r_eff=(1, 3),
                             kind='galaxy', seed=1)
    combined = vstack([stars, gals])
    assert len(combined) == 5
    assert list(combined['kind']) == ['star'] * 3 + ['galaxy'] * 2

    params = make_model_params((100, 100), 3, flux=(1, 10), seed=0)
    assert 'kind' not in params.colnames


def test_make_random_models_table():
    """
    Test the basic functionality of ``make_random_models_table``.
    """
    param_ranges = {'x_0': (0, 300), 'y_0': (0, 500),
                    'gamma': (1, 3), 'alpha': (1.5, 3)}
    source_table = make_random_models_table(10, param_ranges)
    assert len(source_table) == 10
    assert 'id' in source_table.colnames
    cols = ('x_0', 'y_0', 'gamma', 'alpha')
    for col in cols:
        assert col in source_table.colnames
        assert np.min(source_table[col]) >= param_ranges[col][0]
        assert np.max(source_table[col]) <= param_ranges[col][1]


def test_make_random_models_table_inputs():
    """
    Test invalid inputs to ``make_random_models_table``.
    """
    match = 'amplitude must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_random_models_table(3, {'amplitude': (1, 2, 3)})

    with pytest.raises(ValueError, match=match):
        make_random_models_table(3, {'amplitude': 5})

    match = 'n_sources must be >= 0'
    with pytest.raises(ValueError, match=match):
        make_random_models_table(-1, {'amplitude': (1, 2)})

    match = "param_ranges cannot contain an 'id' key"
    with pytest.raises(ValueError, match=match):
        make_random_models_table(3, {'id': (1, 2)})


def test_params_table_to_models():
    """
    Test the basic functionality of ``params_table_to_models``.
    """
    tbl = QTable()
    tbl['x_0'] = [1, 2, 3]
    tbl['y_0'] = [4, 5, 6]
    tbl['flux'] = [100, 200, 300]
    tbl['name'] = ['a', 'b', 'c']
    model = CircularGaussianPSF()
    models = params_table_to_models(tbl, model)

    assert len(models) == 3
    for i, model in enumerate(models):
        assert model.x_0 == tbl['x_0'][i]
        assert model.y_0 == tbl['y_0'][i]
        assert model.flux == tbl['flux'][i]
        assert model.name == tbl['name'][i]

    tbl = QTable()
    tbl['invalid1'] = [1, 2, 3]
    tbl['invalid2'] = [4, 5, 6]
    match = 'No matching model parameter names found in params_table'
    with pytest.raises(ValueError, match=match):
        params_table_to_models(tbl, model)

    match = 'model must be a Model instance'
    with pytest.raises(TypeError, match=match):
        params_table_to_models(tbl, None)
