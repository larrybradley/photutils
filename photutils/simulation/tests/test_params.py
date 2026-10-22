# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the params module.
"""

import numpy as np
import pytest
from astropy.table import QTable, Table, vstack
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal

from photutils.psf import CircularGaussianPSF
from photutils.simulation import (make_model_params, make_pair_params,
                                  make_random_models_table,
                                  make_satellite_params,
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


def pair_members(params):
    """
    Return the first and second members of each pair as two tables.
    """
    return params[0::2], params[1::2]


def pair_separations(params):
    """
    Return the distance between the two members of each pair.
    """
    first, second = pair_members(params)
    return np.hypot(second['x_0'] - first['x_0'],
                    second['y_0'] - first['y_0'])


def test_pair_columns():
    """
    Test the column order and pair IDs of ``make_pair_params``.
    """
    params = make_pair_params((200, 200), 4, 3.0, r_eff=(1, 2),
                              kind='pair', seed=0)
    assert params.colnames == ['id', 'x_0', 'y_0', 'flux', 'pair_id',
                               'r_eff', 'kind']
    assert len(params) == 8
    assert_array_equal(params['id'], np.arange(1, 9))
    assert_array_equal(params['pair_id'], [1, 1, 2, 2, 3, 3, 4, 4])
    assert np.all(params['kind'] == 'pair')


def test_pair_separation_scalar():
    """
    Test that every pair has the requested scalar separation.
    """
    params = make_pair_params((200, 200), 20, 4.5, seed=0)
    assert_allclose(pair_separations(params), 4.5, rtol=0, atol=1e-12)


def test_pair_separation_range():
    """
    Test that a separation range is sampled per pair.
    """
    params = make_pair_params((1000, 1000), 200, (2, 6), seed=0)
    separations = pair_separations(params)
    assert np.all(separations >= 2)
    assert np.all(separations < 6)
    assert separations.min() < 2.5
    assert separations.max() > 5.5


def test_pair_flux_ratio():
    """
    Test that the second flux is the first flux times ``flux_ratio``.
    """
    params = make_pair_params((500, 500), 50, 3.0, flux=(10, 100),
                              flux_ratio=(0.1, 1), seed=0)
    first, second = pair_members(params)
    ratio = np.array(second['flux'] / first['flux'])
    assert np.all(first['flux'] >= 10)
    assert np.all(first['flux'] < 100)
    assert np.all(ratio >= 0.1)
    assert np.all(ratio < 1)

    params = make_pair_params((500, 500), 10, 3.0, flux=(5, 5),
                              flux_ratio=0.5, seed=0)
    first, second = pair_members(params)
    assert_allclose(first['flux'], 5)
    assert_allclose(second['flux'], 2.5)


def test_pair_orientation_random():
    """
    Test that pair orientations cover all directions.
    """
    params = make_pair_params((2000, 2000), 500, 3.0, seed=0)
    first, second = pair_members(params)
    angles = np.arctan2(second['y_0'] - first['y_0'],
                        second['x_0'] - first['x_0'])
    counts, _ = np.histogram(angles, bins=4, range=(-np.pi, np.pi))
    assert np.all(counts > 50)


def test_pairs_inside_image():
    """
    Test that both members of every pair lie inside the border margins.
    """
    shape = (60, 80)
    border = (5, 7)
    params = make_pair_params(shape, 4, (2, 8), border_size=border,
                              seed=0)
    assert np.all(params['x_0'] >= border[1])
    assert np.all(params['x_0'] < shape[1] - border[1])
    assert np.all(params['y_0'] >= border[0])
    assert np.all(params['y_0'] < shape[0] - border[0])


def test_pairs_do_not_interleave():
    """
    Test that pair midpoints are at least three times the largest
    separation apart.
    """
    params = make_pair_params((500, 500), 100, (2, 5), seed=0)
    first, second = pair_members(params)
    x_mid = 0.5 * (first['x_0'] + second['x_0'])
    y_mid = 0.5 * (first['y_0'] + second['y_0'])
    dist = np.hypot(x_mid[:, None] - x_mid[None, :],
                    y_mid[:, None] - y_mid[None, :])
    dist[np.diag_indices_from(dist)] = np.inf
    assert dist.min() >= 15


def test_pair_too_many_warns():
    """
    Test that an impossible request warns and returns whole pairs.
    """
    match = r'Unable to produce .* coordinates within the given shape'
    with pytest.warns(AstropyUserWarning, match=match):
        params = make_pair_params((50, 50), 100, 4.0, seed=0)
    assert 0 < len(params) < 200
    assert len(params) % 2 == 0


def test_pair_errors():
    """
    Test the input validation of ``make_pair_params``.
    """
    match = 'separation must be positive'
    with pytest.raises(ValueError, match=match):
        make_pair_params((100, 100), 5, 0.0)
    with pytest.raises(ValueError, match=match):
        make_pair_params((100, 100), 5, (-1, 2))

    match = 'border_size and separation are too large for the given shape'
    with pytest.raises(ValueError, match=match):
        make_pair_params((10, 10), 5, 3.0, border_size=4)

    for name in ('id', 'x_0', 'y_0', 'pair_id'):
        match = f'{name} cannot be used as a keyword argument'
        with pytest.raises(ValueError, match=match):
            make_pair_params((100, 100), 5, 3.0, **{name: (1, 2)})


def make_hosts(with_id=True):
    """
    Make a small table of host galaxies.
    """
    hosts = QTable()
    if with_id:
        hosts['id'] = [7, 9, 12]
    hosts['x_0'] = [100.0, 300.0, 500.0]
    hosts['y_0'] = [200.0, 250.0, 120.0]
    hosts['r_eff'] = [5.0, 10.0, 20.0]
    return hosts


def satellite_distances(satellites, hosts, host_index):
    """
    Return the distance of each satellite to its host.
    """
    x_host = np.asarray(hosts['x_0'])[host_index]
    y_host = np.asarray(hosts['y_0'])[host_index]
    return np.hypot(satellites['x_0'] - x_host, satellites['y_0'] - y_host)


def test_satellite_columns():
    """
    Test the columns and row count of ``make_satellite_params``.
    """
    hosts = make_hosts()
    satellites = make_satellite_params(hosts, 4, (2, 5), flux=(1, 10),
                                       kind='star_on_gal', seed=0)
    assert satellites.colnames == ['id', 'x_0', 'y_0', 'host_id', 'flux',
                                   'kind']
    assert len(satellites) == 12
    assert_array_equal(satellites['id'], np.arange(1, 13))
    assert np.all(satellites['kind'] == 'star_on_gal')


def test_satellite_radius_pixel():
    """
    Test that satellite distances in pixels are within the radius range.
    """
    hosts = make_hosts()
    satellites = make_satellite_params(hosts, 200, (2, 5), seed=0)
    distances = satellite_distances(satellites, hosts,
                                    np.repeat(np.arange(3), 200))
    assert np.all(distances >= 2 - 1e-12)
    assert np.all(distances < 5)
    assert distances.min() < 2.2
    assert distances.max() > 4.8


def test_satellite_radius_r_eff():
    """
    Test that the radius range is in units of each host's ``r_eff``.
    """
    hosts = make_hosts()
    satellites = make_satellite_params(hosts, 200, (0.5, 2.5),
                                       radius_unit='r_eff', seed=0)
    host_index = np.repeat(np.arange(3), 200)
    scaled = (satellite_distances(satellites, hosts, host_index)
              / np.asarray(hosts['r_eff'])[host_index])
    assert np.all(scaled >= 0.5 - 1e-12)
    assert np.all(scaled < 2.5)

    hosts.rename_column('r_eff', 'size')
    satellites = make_satellite_params(hosts, 5, 1.0, radius_unit='r_eff',
                                       r_eff_name='size', seed=0)
    distances = satellite_distances(satellites, hosts,
                                    np.repeat(np.arange(3), 5))
    assert_allclose(distances, np.repeat([5.0, 10.0, 20.0], 5))


def test_satellite_host_id():
    """
    Test that ``host_id`` is the host's ``id`` value, or its 1-based row
    number when the host table has no ``id`` column.
    """
    satellites = make_satellite_params(make_hosts(), 2, 3.0, seed=0)
    assert_array_equal(satellites['host_id'], [7, 7, 9, 9, 12, 12])

    satellites = make_satellite_params(make_hosts(with_id=False), 2, 3.0,
                                       seed=0)
    assert_array_equal(satellites['host_id'], [1, 1, 2, 2, 3, 3])


def test_satellite_errors():
    """
    Test the input validation of ``make_satellite_params``.
    """
    hosts = make_hosts()
    match = "radius_unit must be 'pixel' or 'r_eff'"
    with pytest.raises(ValueError, match=match):
        make_satellite_params(hosts, 2, 3.0, radius_unit='arcsec')

    match = 'host_params has no "size" column'
    with pytest.raises(ValueError, match=match):
        make_satellite_params(hosts, 2, 3.0, radius_unit='r_eff',
                              r_eff_name='size')

    for name in ('id', 'x_0', 'y_0', 'host_id'):
        match = f'{name} cannot be used as a keyword argument'
        with pytest.raises(ValueError, match=match):
            make_satellite_params(hosts, 2, 3.0, **{name: (1, 2)})
