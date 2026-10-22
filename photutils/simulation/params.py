# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making tables of model parameters or making models from a
table of model parameters.
"""

import numpy as np
from astropy.modeling import Model
from astropy.table import QTable

from photutils.utils._coords import make_random_xycoords
from photutils.utils._misc import _get_meta
from photutils.utils._parameters import as_pair

__all__ = ['make_model_params', 'make_pair_params',
           'make_random_models_table', 'make_satellite_params',
           'params_table_to_models']


def make_model_params(shape, n_sources, *, x_name='x_0', y_name='y_0',
                      min_separation=1, border_size=(0, 0), seed=0,
                      log_params=(), kind=None, **kwargs):
    """
    Make a table of randomly generated model positions and additional
    parameters for simulated sources.

    By default, the output table contains only the source ``id`` and
    the ``x_0`` and ``y_0`` positions. Additional parameters can be
    specified as keyword arguments with their lower and upper bounds
    as 2-tuples. They are added as columns in the order given. The
    parameter values will be uniformly sampled from the half-open
    interval ``[lower, upper)``.

    Parameters
    ----------
    shape : 2-tuple of int or int
        The (ny, nx) shape of the image in which the sources will be
        placed. If a single integer is provided, it is used for both
        dimensions.

    n_sources : int
        The number of sources to generate. If ``min_separation`` is too
        large, the requested number of sources may not fit within the
        given ``shape``. In that case, fewer than ``n_sources`` sources
        are generated and an
        `~astropy.utils.exceptions.AstropyUserWarning` is issued.

    x_name : str, optional
        The name of the model parameter that corresponds to the x
        position of the sources. This will be the column name in the
        output table.

    y_name : str, optional
        The name of the model parameter that corresponds to the y
        position of the sources. This will be the column name in the
        output table.

    min_separation : float, optional
        The minimum separation in pixels between the centers of two
        sources. If the minimum separation is too large, fewer than
        ``n_sources`` sources are generated and a warning is issued.

    border_size : tuple of 2 int or int, optional
        The (ny, nx) size of the border around the image where no
        sources will be generated (i.e., the source center will not be
        located within the border). If a single integer is provided, it
        will be used for both dimensions. The x positions are uniformly
        sampled from the half-open interval ``[nx_border, shape[1] -
        nx_border)`` and the y positions from ``[ny_border, shape[0] -
        ny_border)``.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    log_params : tuple of str, optional
        The names of the keyword parameters (e.g., ``('flux',
        'r_eff')``) whose values are sampled uniformly in log10 between
        their lower and upper bounds, instead of uniformly. The bounds of
        these parameters must be positive.

    kind : str or `None`, optional
        If not `None`, a ``'kind'`` column filled with this value is
        added after the parameter columns. It labels the population, so
        tables of different populations can be stacked with
        `~astropy.table.vstack` and told apart afterward.
        Populations that will be stacked need different ``seed``
        values, or their positions are correlated.

    **kwargs
        Keyword arguments are accepted for additional model parameters.
        The values should be 2-tuples of the lower and upper bounds for
        the parameter range. The bounds must be numbers without units.
        The parameter values will be uniformly sampled from the
        half-open interval ``[lower, upper)``, or uniformly in log10 for
        the parameters named in ``log_params``. The names ``id``,
        ``x_name``, and ``y_name`` cannot be used because they are the
        names of the ID and position columns.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table containing the model parameters of the generated
        sources. The table will also contain an ``'id'`` column with
        unique source IDs.

    Raises
    ------
    ValueError
        If ``n_sources`` is negative, if ``border_size`` leaves no
        room for sources within the given ``shape``, if a keyword
        argument value is not a 2-tuple, if a keyword argument is
        named ``id`` or has the same name as ``x_name`` or ``y_name``,
        if a ``log_params`` name is not a keyword parameter, or if the
        bounds of a ``log_params`` parameter are not positive.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If fewer than ``n_sources`` sources could be generated with
        the given ``shape`` and ``min_separation``.

    Examples
    --------
    >>> from photutils.simulation import make_model_params
    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id    x_0       y_0       flux
    --- --------- --------- ---------
      1 60.956935 72.967865 291.99517
      2 31.582937 29.149555 192.94917
      3 13.277882 80.118738 420.75223
      4 11.322211 14.685443 469.41206
      5 75.061619 36.889365 206.45211

    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            x_name='x_mean', y_name='y_mean',
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id   x_mean    y_mean     flux
    --- --------- --------- ---------
      1 60.956935 72.967865 291.99517
      2 31.582937 29.149555 192.94917
      3 13.277882 80.118738 420.75223
      4 11.322211 14.685443 469.41206
      5 75.061619 36.889365 206.45211

    >>> params = make_model_params((100, 100), 5, flux=(100, 500),
    ...                            sigma=(1, 2), alpha=(0, 1),
    ...                            min_separation=3, border_size=10, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.5g'  # for consistent table output
    >>> print(params)
     id  x_0    y_0    flux  sigma   alpha
    --- ------ ------ ------ ------ --------
      1 60.957 72.968    292 1.5389  0.61437
      2 31.583  29.15 192.95 1.4428 0.028365
      3 13.278 80.119 420.75  1.931  0.71922
      4 11.322 14.685 469.41 1.0405 0.015992
      5 75.062 36.889 206.45  1.732  0.75795
    """
    shape = as_pair('shape', shape, lower_bound=(0, 0))
    border_size = as_pair('border_size', border_size, lower_bound=(0, 1))

    if n_sources < 0:
        msg = 'n_sources must be >= 0'
        raise ValueError(msg)

    xrange = (border_size[1], shape[1] - border_size[1])
    yrange = (border_size[0], shape[0] - border_size[0])

    if xrange[0] >= xrange[1] or yrange[0] >= yrange[1]:
        msg = 'border_size is too large for the given shape'
        raise ValueError(msg)

    for name in log_params:
        if name not in kwargs:
            msg = f'log_params name "{name}" is not a keyword parameter'
            raise ValueError(msg)

    rng = np.random.default_rng(seed)
    xycoords = make_random_xycoords(n_sources, xrange, yrange,
                                    min_separation=min_separation,
                                    seed=rng)
    x, y = np.transpose(xycoords)

    model_params = QTable()
    model_params.meta.update(_get_meta())  # keep model_params.meta type
    model_params['id'] = np.arange(len(x)) + 1
    model_params[x_name] = x
    model_params[y_name] = y

    for param, prange in kwargs.items():
        if param in ('id', x_name, y_name):
            msg = (f'{param} cannot be used as a keyword argument because '
                   'it is the name of the ID or a position column')
            raise ValueError(msg)
        if np.shape(prange) != (2,):
            msg = f'{param} must be a 2-tuple'
            raise ValueError(msg)
        model_params[param] = _sample_param(
            rng, prange, len(model_params), log=param in log_params,
            name=param)

    _add_kind_column(model_params, kind)

    return model_params


def make_pair_params(shape, n_pairs, separation, *, flux=(1, 1),
                     flux_ratio=1.0, x_name='x_0', y_name='y_0',
                     border_size=(0, 0), seed=0, kind=None, **kwargs):
    """
    Make a table of parameters for close pairs of simulated sources.

    Each pair has two sources at a given separation and random
    orientation. The table has two rows per pair, and the two rows of
    a pair share a ``pair_id`` value. Pair midpoints are at least three
    times the largest separation apart, so pairs never interleave, and
    both sources of every pair lie inside the ``border_size`` margins.

    Parameters
    ----------
    shape : 2-tuple of int or int
        The (ny, nx) shape of the image in which the sources will be
        placed. If a single integer is provided, it is used for both
        dimensions.

    n_pairs : int
        The number of pairs to generate. If the pairs do not fit within
        the given ``shape``, fewer than ``n_pairs`` pairs are generated
        and an `~astropy.utils.exceptions.AstropyUserWarning` is
        issued.

    separation : float or 2-tuple of float
        The separation in pixels between the two sources of a pair. A
        2-tuple gives the lower and upper bounds of the half-open
        interval ``[lower, upper)`` from which the separation of each
        pair is sampled uniformly. Separations must be positive.

    flux : float or 2-tuple of float, optional
        The flux of the first source of each pair. A 2-tuple gives the
        lower and upper bounds from which the flux of each pair is
        sampled uniformly.

    flux_ratio : float or 2-tuple of float, optional
        The flux of the second source of each pair divided by the flux
        of the first. A 2-tuple gives the lower and upper bounds from
        which the ratio of each pair is sampled uniformly.

    x_name : str, optional
        The name of the model parameter that corresponds to the x
        position of the sources. This will be the column name in the
        output table.

    y_name : str, optional
        The name of the model parameter that corresponds to the y
        position of the sources. This will be the column name in the
        output table.

    border_size : tuple of 2 int or int, optional
        The (ny, nx) size of the border around the image where no
        sources will be generated. If a single integer is provided, it
        will be used for both dimensions.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    kind : str or `None`, optional
        If not `None`, a ``'kind'`` column filled with this value is
        added as the last column.
        Populations that will be stacked need different ``seed``
        values, or their positions are correlated.

    **kwargs
        Keyword arguments are accepted for additional model parameters.
        The values can be a scalar, used for every source, or a 2-tuple
        of the lower and upper bounds from which the value of each
        source is sampled uniformly. The names ``id``, ``x_name``,
        ``y_name``, and ``pair_id`` cannot be used.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table with the columns ``id``, ``x_name``, ``y_name``,
        ``flux``, and ``pair_id``, followed by one column for each
        keyword parameter and the optional ``kind`` column. Rows ``2k``
        and ``2k + 1`` are the two sources of pair ``k + 1``.

    Raises
    ------
    ValueError
        If a separation is not positive, if ``border_size`` and the
        largest separation leave no room for pairs within the given
        ``shape``, or if a keyword argument has a reserved name.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If fewer than ``n_pairs`` pairs could be generated.

    See Also
    --------
    make_model_params, make_satellite_params

    Examples
    --------
    >>> from photutils.simulation import make_pair_params
    >>> params = make_pair_params((100, 100), 3, (2, 5), flux=(100, 500),
    ...                           flux_ratio=(0.1, 1), seed=0)
    >>> print(params.colnames)
    ['id', 'x_0', 'y_0', 'flux', 'pair_id']
    >>> print(len(params))
    6
    """
    shape = as_pair('shape', shape, lower_bound=(0, 0))
    border_size = as_pair('border_size', border_size, lower_bound=(0, 1))

    if np.any(np.asarray(separation) <= 0):
        msg = 'separation must be positive'
        raise ValueError(msg)

    for param in kwargs:
        if param in ('id', x_name, y_name, 'pair_id'):
            msg = (f'{param} cannot be used as a keyword argument because '
                   'it is the name of an output column')
            raise ValueError(msg)

    max_separation = np.max(separation)
    margin = 0.5 * max_separation
    xrange = (border_size[1] + margin, shape[1] - border_size[1] - margin)
    yrange = (border_size[0] + margin, shape[0] - border_size[0] - margin)
    if xrange[0] >= xrange[1] or yrange[0] >= yrange[1]:
        msg = 'border_size and separation are too large for the given shape'
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    midpoints = make_random_xycoords(n_pairs, xrange, yrange,
                                     min_separation=3 * max_separation,
                                     seed=rng)
    n_made = len(midpoints)
    x_mid, y_mid = np.transpose(midpoints)

    separations = _sample_param(rng, separation, n_made, name='separation')
    angles = rng.uniform(0, 2 * np.pi, n_made)
    first_flux = _sample_param(rng, flux, n_made, name='flux')
    ratios = _sample_param(rng, flux_ratio, n_made, name='flux_ratio')

    dx = 0.5 * separations * np.cos(angles)
    dy = 0.5 * separations * np.sin(angles)

    def interleave(first, second):  # numpydoc ignore=GL08
        return np.column_stack((first, second)).ravel()

    params = QTable()
    params.meta.update(_get_meta())  # keep params.meta type
    params['id'] = np.arange(2 * n_made) + 1
    params[x_name] = interleave(x_mid - dx, x_mid + dx)
    params[y_name] = interleave(y_mid - dy, y_mid + dy)
    params['flux'] = interleave(first_flux, first_flux * ratios)
    params['pair_id'] = np.repeat(np.arange(n_made) + 1, 2)

    for param, prange in kwargs.items():
        params[param] = _sample_param(rng, prange, len(params), name=param)

    _add_kind_column(params, kind)

    return params


def make_satellite_params(host_params, n_per_host, radius, *,
                          radius_unit='pixel', r_eff_name='r_eff',
                          x_name='x_0', y_name='y_0', seed=0, kind=None,
                          **kwargs):
    """
    Make a table of parameters for simulated sources placed around host
    sources.

    For each row of ``host_params``, ``n_per_host`` sources are placed at
    random angles and at distances from the host center sampled
    uniformly from the ``radius`` range. A typical use is placing faint
    stars on bright galaxies to test detection on top of extended
    emission.

    The positions are not limited to any image, so sources around a host
    near an image edge can fall outside the image.

    Parameters
    ----------
    host_params : `~astropy.table.Table`
        A table of host sources, with ``x_name`` and ``y_name`` position
        columns and, if ``radius_unit='r_eff'``, an ``r_eff_name``
        column. These columns are in pixels and must not have units.

    n_per_host : int
        The number of sources placed around each host.

    radius : float or 2-tuple of float
        The distance from the host center. A 2-tuple gives the lower and
        upper bounds of the half-open interval ``[lower, upper)`` from
        which the distance of each source is sampled uniformly.

    radius_unit : {'pixel', 'r_eff'}, optional
        The unit of ``radius``. With ``'r_eff'``, the radius is in units
        of each host's ``r_eff_name`` column.

    r_eff_name : str, optional
        The name of the ``host_params`` column used when
        ``radius_unit='r_eff'``.

    x_name : str, optional
        The name of the x position column, in both ``host_params`` and
        the output table.

    y_name : str, optional
        The name of the y position column, in both ``host_params`` and
        the output table.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    kind : str or `None`, optional
        If not `None`, a ``'kind'`` column filled with this value is
        added as the last column.
        Populations that will be stacked need different ``seed``
        values, or their positions are correlated.

    **kwargs
        Keyword arguments are accepted for additional model parameters.
        The values can be a scalar, used for every source, or a 2-tuple
        of the lower and upper bounds from which the value of each
        source is sampled uniformly. The names ``id``, ``x_name``,
        ``y_name``, and ``host_id`` cannot be used.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table with the columns ``id``, ``x_name``, ``y_name``, and
        ``host_id``, followed by one column for each keyword parameter
        and the optional ``kind`` column. The ``host_id`` is the host's
        ``id`` value if ``host_params`` has an ``id`` column, and its
        1-based row number otherwise.

    Raises
    ------
    ValueError
        If ``radius_unit`` is invalid, if ``host_params`` has no
        ``r_eff_name`` column when ``radius_unit='r_eff'``, if a host
        position or ``r_eff_name`` column has units, or if a keyword
        argument has a reserved name.

    See Also
    --------
    make_model_params, make_pair_params

    Examples
    --------
    >>> from astropy.table import QTable
    >>> from photutils.simulation import make_satellite_params
    >>> hosts = QTable({'x_0': [50.0, 150.0], 'y_0': [60.0, 40.0],
    ...                 'r_eff': [5.0, 8.0]})
    >>> stars = make_satellite_params(hosts, 3, (0.5, 2.5),
    ...                               radius_unit='r_eff', flux=(5, 300),
    ...                               seed=0)
    >>> print(stars.colnames)
    ['id', 'x_0', 'y_0', 'host_id', 'flux']
    >>> print(stars['host_id'].tolist())
    [1, 1, 1, 2, 2, 2]
    """
    if radius_unit not in ('pixel', 'r_eff'):
        msg = "radius_unit must be 'pixel' or 'r_eff'"
        raise ValueError(msg)
    if radius_unit == 'r_eff' and r_eff_name not in host_params.colnames:
        msg = f'host_params has no "{r_eff_name}" column'
        raise ValueError(msg)

    unit_columns = [x_name, y_name]
    if radius_unit == 'r_eff':
        unit_columns.append(r_eff_name)
    for column in unit_columns:
        if getattr(host_params[column], 'unit', None) is not None:
            msg = f'host_params column "{column}" must not have units'
            raise ValueError(msg)

    for param in kwargs:
        if param in ('id', x_name, y_name, 'host_id'):
            msg = (f'{param} cannot be used as a keyword argument because '
                   'it is the name of an output column')
            raise ValueError(msg)

    n_hosts = len(host_params)
    host_index = np.repeat(np.arange(n_hosts), n_per_host)
    n_sources = len(host_index)

    rng = np.random.default_rng(seed)
    radii = _sample_param(rng, radius, n_sources, name='radius')
    if radius_unit == 'r_eff':
        radii *= np.asarray(host_params[r_eff_name], dtype=float)[host_index]
    angles = rng.uniform(0, 2 * np.pi, n_sources)

    if 'id' in host_params.colnames:
        host_ids = np.asarray(host_params['id'])
    else:
        host_ids = np.arange(n_hosts) + 1

    x_host = np.asarray(host_params[x_name], dtype=float)[host_index]
    y_host = np.asarray(host_params[y_name], dtype=float)[host_index]

    params = QTable()
    params.meta.update(_get_meta())  # keep params.meta type
    params['id'] = np.arange(n_sources) + 1
    params[x_name] = x_host + radii * np.cos(angles)
    params[y_name] = y_host + radii * np.sin(angles)
    params['host_id'] = host_ids[host_index]

    for param, prange in kwargs.items():
        params[param] = _sample_param(rng, prange, n_sources, name=param)

    _add_kind_column(params, kind)

    return params


def _sample_param(rng, bounds, size, *, log=False, name='parameter'):
    """
    Sample random values of a parameter.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        The random number generator.

    bounds : float or 2-tuple of float
        A scalar gives a constant value. A 2-tuple gives the lower and
        upper bounds of the half-open interval ``[lower, upper)`` from
        which the values are sampled.

    size : int
        The number of values.

    log : bool, optional
        Whether to sample uniformly in log10 instead of uniformly. Used
        only when ``bounds`` is a 2-tuple, whose values must then be
        positive.

    name : str, optional
        The parameter name, used in error messages.

    Returns
    -------
    values : `~numpy.ndarray`
        The sampled values.
    """
    if np.ndim(bounds) == 0:
        return np.full(size, bounds, dtype=float)

    low, high = bounds
    if not log:
        return rng.uniform(low, high, size)

    if low <= 0 or high <= 0:
        msg = f'bounds of log parameter "{name}" must be positive'
        raise ValueError(msg)
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size)


def _add_kind_column(table, kind):
    """
    Add a ``'kind'`` column to a table in place.

    Parameters
    ----------
    table : `~astropy.table.Table`
        The table.

    kind : str or `None`
        The value of every row in the ``'kind'`` column. If `None`, no
        column is added.
    """
    if kind is not None:
        table['kind'] = np.full(len(table), kind)


def make_random_models_table(n_sources, param_ranges, *, seed=None):
    """
    Make a `~astropy.table.QTable` containing randomly generated
    parameters for an Astropy model to simulate a set of sources.

    Each row of the table corresponds to a source whose parameters
    are defined by the column names. The parameters are uniformly
    sampled from the half-open intervals ``[lower, upper)`` given by the
    specified input ranges.

    The output table can be input into :func:`make_model_image` to
    create an image containing the model sources.

    Parameters
    ----------
    n_sources : int
        The number of random model sources to generate.

    param_ranges : dict
        The lower and upper boundaries for each of the model parameters
        as a dictionary mapping the parameter name to its ``(lower,
        upper)`` bounds. The bounds must be numbers without units. The
        parameter values will be uniformly sampled from the half-open
        interval ``[lower, upper)``. The key ``'id'`` cannot be used
        because it is the name of the source ID column.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of parameters for the randomly generated sources. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names. The column names will be the
        keys of the dictionary ``param_ranges``. The table will also
        contain an ``'id'`` column with unique source IDs.

    Raises
    ------
    ValueError
        If ``n_sources`` is negative, if a ``param_ranges`` value is
        not a 2-tuple, or if ``param_ranges`` contains an ``'id'`` key.

    Notes
    -----
    To generate identical parameter values from separate function
    calls, ``n_sources``, the ``seed`` (which must not be `None`),
    and the parameter ranges in ``param_ranges``, including the order
    of its keys, must all be the same.

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.simulation import make_random_models_table
    >>> n_sources = 5
    >>> param_ranges = {'amplitude': [500, 1000],
    ...                 'x_mean': [0, 500],
    ...                 'y_mean': [0, 300],
    ...                 'x_stddev': [1, 5],
    ...                 'y_stddev': [1, 5],
    ...                 'theta': [0, np.pi]}
    >>> params = make_random_models_table(n_sources, param_ranges, seed=0)
    >>> for col in params.colnames:
    ...     params[col].info.format = '%.8g'  # for consistent table output
    >>> print(params)
     id amplitude   x_mean    y_mean    x_stddev  y_stddev   theta
    --- --------- --------- ---------- --------- --------- ---------
      1 818.48084 456.37779  244.75607 1.7026225 1.1132787 1.2053586
      2 634.89336 303.31789 0.82155005 4.4527157 1.4971331 3.1328274
      3 520.48676 364.74828  257.22128 3.1658449 3.6824977 3.0813851
      4 508.26382  271.8125  10.075673 2.1988476  3.588758 2.1536937
      5 906.63512 467.53621  218.89663 2.6907489 3.4615404 2.0434781
    """
    if n_sources < 0:
        msg = 'n_sources must be >= 0'
        raise ValueError(msg)

    if 'id' in param_ranges:
        msg = ("param_ranges cannot contain an 'id' key because it is the "
               'name of the source ID column')
        raise ValueError(msg)

    rng = np.random.default_rng(seed)

    sources = QTable()
    sources.meta.update(_get_meta())  # keep sources.meta type
    sources['id'] = np.arange(n_sources) + 1
    for param_name, prange in param_ranges.items():
        if np.shape(prange) != (2,):
            msg = f'{param_name} must be a 2-tuple'
            raise ValueError(msg)
        # Generate a column for every item in param_ranges, even if it
        # is not in the model (e.g., flux).
        sources[param_name] = rng.uniform(prange[0], prange[1], n_sources)

    return sources


def params_table_to_models(params_table, model):
    """
    Create a list of models from a table of model parameters.

    Parameters
    ----------
    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source.
        Each row of the table corresponds to a different model whose
        parameters are defined by the column names. Model parameters
        not defined in the table keep the values of the input
        ``model``. Columns whose names do not match a model parameter
        name are ignored. At least one column name must match a
        parameter name of ``model``. To attach units to model
        parameters, ``params_table`` must be input as a
        `~astropy.table.QTable`. A column named ``'name'`` can also be
        included in the table to assign a name to each model.

    model : `astropy.modeling.Model` instance
        The model to copy for each row of ``params_table``. The
        parameters of each copy are set from the table. The input
        ``model`` is not modified.

    Returns
    -------
    models : list of `astropy.modeling.Model`
        A list of models created from the input table of model
        parameters.

    Raises
    ------
    TypeError
        If ``model`` is not an `astropy.modeling.Model` instance.

    ValueError
        If no column name in ``params_table`` matches a parameter name
        of ``model``.

    Examples
    --------
    >>> from astropy.table import QTable
    >>> from photutils.simulation import params_table_to_models
    >>> from photutils.psf import CircularGaussianPSF
    >>> tbl = QTable()
    >>> tbl['x_0'] = [1, 2, 3]
    >>> tbl['y_0'] = [4, 5, 6]
    >>> tbl['flux'] = [100, 200, 300]
    >>> model = CircularGaussianPSF()
    >>> models = params_table_to_models(tbl, model)
    >>> models
    [<CircularGaussianPSF(flux=100., x_0=1., y_0=4., fwhm=1.)>,
     <CircularGaussianPSF(flux=200., x_0=2., y_0=5., fwhm=1.)>,
     <CircularGaussianPSF(flux=300., x_0=3., y_0=6., fwhm=1.)>]
    """
    if not isinstance(model, Model):
        msg = 'model must be a Model instance'
        raise TypeError(msg)

    param_names = set(model.param_names)
    colnames = set(params_table.colnames)
    if param_names.isdisjoint(colnames):
        msg = 'No matching model parameter names found in params_table'
        raise ValueError(msg)

    param_names = [*sorted(param_names), 'name']
    models = []
    for row in params_table:
        new_model = model.copy()
        for param_name in param_names:
            if param_name not in colnames:
                continue
            setattr(new_model, param_name, row[param_name])
        models.append(new_model)

    return models
