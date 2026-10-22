# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making simulated images from astropy models.
"""

import pathlib

import astropy.units as u
import numpy as np
from astropy.modeling import Model
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NoOverlapError
from astropy.table import QTable, Table
from astropy.utils.data import get_pkg_data_path

from photutils.simulation.params import make_model_params
from photutils.utils._overlap import overlap_slices
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar

__all__ = [
    'make_4gaussians_image',
    'make_100gaussians_image',
    'make_model_image',
    'make_psf_model_image',
]

_DATA_DIR = pathlib.Path(get_pkg_data_path('simulation', 'data',
                                           package='photutils'))


def make_model_image(shape, model, params_table, *, model_shape=None,
                     bbox_factor=None, x_name='x_0', y_name='y_0',
                     params_map=None, progress_bar=False):
    """
    Make a 2D image containing sources generated from a user-specified
    astropy 2D model.

    The model parameters for each source are taken from the input
    ``params_table`` table. By default, the table is searched for column
    names that match model parameter names and the values specified by
    ``x_name`` and ``y_name``. However, the user can specify a different
    mapping between model parameter names and column names using the
    ``params_map`` keyword.

    Each source is rendered by evaluating the model at the center of
    each pixel. To simulate the pixelated image of a source, the value
    of the model at a position must be the flux in a pixel centered at
    that position, as for the analytic models whose names end in
    ``PRF`` and for an image-based model made from an ePSF (see
    :ref:`psf-terminology`).

    Parameters
    ----------
    shape : 2-tuple of int or int
        The shape of the output image. If a single integer is provided,
        it will be used for both dimensions.

    model : 2D `astropy.modeling.Model`
        The 2D model to be used to render the sources. The model must
        be two-dimensional, meaning that it accepts 2 inputs (i.e.,
        (x, y)) and has 1 output. The model must have parameters for
        the x and y positions of the sources. Typically, these
        parameters are named 'x_0' and 'y_0', but the parameter names
        can be specified using the ``x_name`` and ``y_name`` keywords.

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source. Each
        row of the table corresponds to a source whose model parameters
        are defined by the column names, which must match the model
        parameter names. The table must contain columns for the x
        and y positions of the sources. The column names for the x
        and y positions can be specified using the ``x_name`` and
        ``y_name`` keywords. Model parameters not defined in the table
        or ``params_map`` keep the values of the input ``model``. To
        attach units to model parameters, ``params_table`` must be input
        as a `~astropy.table.QTable`. Rows that contain any nonfinite
        model parameters or a nonfinite ``'local_bkg'`` value will be
        skipped.

        If the table contains a column named ``'model_shape'``, then
        the values in that column will be used to override the
        ``model_shape`` keyword and model ``bounding_box`` for each
        source. This can be used to render each source with a different
        shape.

        If the table contains a column named ``'local_bkg'``, then the
        per-pixel local background values in that column will be
        added to each model source over the region defined by its
        ``model_shape``. The ``'local_bkg'`` column must have the same
        flux units as the output image (e.g., if the input ``model``
        has 'amplitude' or 'flux' parameters with units). Including
        ``'local_bkg'`` should be used with care, especially in crowded
        fields where the ``model_shape`` of sources overlap (see Notes
        below).

        Except for the ``'model_shape'`` and ``'local_bkg'`` columns,
        columns whose names do not match a model parameter name are
        ignored unless they are mapped to a model parameter by
        ``params_map``.

    model_shape : 2-tuple of int, int, or `None`, optional
        The shape around the (x, y) center of each source that will be
        used to evaluate the ``model``. If ``model_shape`` is a scalar
        integer, then a square shape of size ``model_shape`` will be
        used. If `None`, then the bounding box of the model will be
        used (which can optionally be scaled using the ``bbox_factor``
        keyword if the model supports it). If the model does not have
        a ``bounding_box`` attribute, then either this keyword or a
        ``'model_shape'`` column in ``params_table`` must be
        specified. If specified, this keyword overrides the model
        ``bounding_box`` attribute. To use a different shape for
        each source, include a column named ``'model_shape'`` in the
        ``params_table``. For that case, this keyword is ignored.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. If the model
        ``bounding_box`` method does not accept a ``factor`` keyword,
        then this keyword is ignored. If `None`, the default model
        bounding box will be used. This keyword is ignored if
        ``model_shape`` is specified or if the ``params_table`` contains
        a ``'model_shape'`` column. Note that some Photutils PSF models
        have a ``bbox_factor`` keyword that is used to define the model
        bounding box. In that case, this keyword is ignored.

    x_name : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources. This value must also be a column name
        in ``params_table`` unless ``params_map`` maps this parameter
        to a different column.

    y_name : str, optional
        The name of the ``model`` parameter that corresponds to the y
        position of the sources. This value must also be a column name
        in ``params_table`` unless ``params_map`` maps this parameter
        to a different column.

    params_map : dict or None, optional
        A dictionary mapping the model parameter names to the column
        names in the input ``params_table``. The dictionary keys are
        the model parameter names and the values are the column names
        in the input ``params_table``. This can be used to map column
        names to model parameter names that are different. For example,
        if the input column name is 'flux_f200w' and the model parameter
        name is 'flux', then use ``params_map={'flux': 'flux_f200w'}``.
        This dictionary may also be used to map the model x and y
        parameters to columns with names other than ``x_name`` and
        ``y_name``. In that case, the dictionary keys for the positions
        must be the ``x_name`` and ``y_name`` parameter names. If
        those keys are not in the dictionary, the ``x_name`` and
        ``y_name`` columns are used.

    progress_bar : bool, optional
        Whether to display a progress bar while adding the sources
        to the image. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Returns
    -------
    array : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The rendered image containing the model sources. The image is
        a `~astropy.units.Quantity` if the model output has units
        (e.g., if ``params_table`` is a `~astropy.table.QTable` with a
        flux column that has units).

    Raises
    ------
    TypeError
        If ``model`` is not an `astropy.modeling.Model` instance or if
        ``params_table`` is not an `~astropy.table.Table`.

    ValueError
        If ``model`` does not have 2 inputs and 1 output, if a
        ``params_map`` key is not a model parameter name, if a required
        column is not in ``params_table``, if the model has no
        ``bounding_box`` attribute and no model shape is specified, if
        the ``'local_bkg'`` column does not have the same flux units as
        the output image, or if a ``params_table`` column has units
        and ``params_table`` is not a `~astropy.table.QTable`.

    Notes
    -----
    The local background value around each source is optionally included
    using the ``'local_bkg'`` column in the input ``params_table``. This
    local background is added to each source over its ``model_shape``
    region. In regions where the ``model_shape`` of sources overlap,
    the local background will be added multiple times. This is not an
    issue if the sources are well-separated, but for crowded fields,
    this option should be used with care.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.modeling.models import Moffat2D
        from photutils.simulation import (make_model_image,
                                          make_random_models_table)

        model = Moffat2D()
        n_sources = 25
        shape = (100, 100)
        param_ranges = {'amplitude': [100, 200],
                        'x_0': [0, shape[1]],
                        'y_0': [0, shape[0]],
                        'gamma': [1, 2],
                        'alpha': [1, 2]}
        params = make_random_models_table(n_sources, param_ranges, seed=0)

        model_shape = (15, 15)
        data = make_model_image(shape, model, params, model_shape=model_shape)

        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
        fig.tight_layout()

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.modeling.models import Gaussian2D
        from photutils.simulation import make_model_image, make_model_params

        model = Gaussian2D()
        shape = (500, 500)
        n_sources = 100
        params = make_model_params(shape, n_sources, x_name='x_mean',
                                   y_name='y_mean', min_separation=25,
                                   amplitude=(100, 500), x_stddev=(1, 3),
                                   y_stddev=(1, 3), theta=(0, np.pi))
        model_shape = (25, 25)
        data = make_model_image(shape, model, params, model_shape=model_shape,
                                x_name='x_mean', y_name='y_mean')

        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
        fig.tight_layout()
    """
    shape = as_pair('shape', shape, lower_bound=(0, 0))

    if not isinstance(model, Model):
        msg = 'model must be a Model instance'
        raise TypeError(msg)
    if model.n_inputs != 2 or model.n_outputs != 1:
        msg = 'model must be a 2D model'
        raise ValueError(msg)
    if not isinstance(params_table, Table):
        msg = 'params_table must be an astropy Table'
        raise TypeError(msg)

    xypos_map = {x_name: x_name, y_name: y_name}

    # By default, use the model parameter names as the column names
    # if they are in the table
    params_to_set = set(params_table.colnames) & set(model.param_names)
    xypos_map.update({param: param for param in params_to_set})

    if params_map is not None:
        # params_map takes precedence over x_name and y_name and
        # any matching column names in params_table
        xypos_map.update(params_map)
    params_map = xypos_map

    for key, value in params_map.items():
        if key not in model.param_names:
            msg = f'key "{key}" not in model parameter names'
            raise ValueError(msg)
        if value not in params_table.colnames:
            msg = f'value "{value}" not in params_table column names'
            raise ValueError(msg)

    if model_shape is not None:
        model_shape = as_pair('model_shape', model_shape, lower_bound=(0, 0))

    variable_shape = False
    if 'model_shape' in params_table.colnames:
        model_shape = np.array([as_pair('model_shape', mshape,
                                        lower_bound=(0, 0))
                                for mshape in params_table['model_shape']])
        variable_shape = True

    if model_shape is None:
        try:
            _ = model.bounding_box
        except NotImplementedError as exc:
            msg = ('model_shape must be specified if the model does not '
                   'have a bounding_box attribute')
            raise ValueError(msg) from exc

    if 'local_bkg' in params_table.colnames:
        local_bkg = params_table['local_bkg']
    else:
        local_bkg = np.zeros(len(params_table))

    # Copy the input model to leave it unchanged
    model = model.copy()

    image = np.zeros(shape, dtype=float)

    # Determine the output image units, if any, by evaluating the model
    # with unit scalars derived from the mapped table columns. Unmapped
    # model parameters keep their default values (and units). This is
    # done up front so that the output units do not depend on whether
    # any sources are actually rendered (e.g., a zero-row table or
    # sources that do not overlap the image).
    for key, param in params_map.items():
        unit = getattr(params_table[param], 'unit', None)
        setattr(model, key, 1 if unit is None else 1 * unit)
    result = model(0, 0)
    if isinstance(result, u.Quantity):
        image <<= result.unit

    if progress_bar:
        desc = 'Add model sources'
        params_table = add_progress_bar(params_table, desc=desc)

    for i, source in enumerate(params_table):
        params = {key: source[param] for key, param in params_map.items()}

        # Skip sources that have any nonfinite parameter or local
        # background value
        if (not all(np.isfinite(value) for value in params.values())
                or not np.isfinite(local_bkg[i])):
            continue

        try:
            for key, value in params.items():
                setattr(model, key, value)
        except u.UnitsError as exc:
            msg = ('params_table must be a QTable when its columns have '
                   'units')
            raise ValueError(msg) from exc

        # This assumes that if the user also uses params_table to
        # override the (x/y)_name mapping that the x_name and y_name
        # values are correct (i.e., the mapping keys include x_name and
        # y_name). There is no good way to check/enforce this.
        x0 = getattr(model, x_name).value
        y0 = getattr(model, y_name).value

        if variable_shape:
            mod_shape = model_shape[i]
        elif model_shape is None:
            # The bounding box size generally depends on model
            # parameters, so needs to be calculated for each source
            mod_shape = _model_shape_from_bbox(model, bbox_factor=bbox_factor)
        else:
            mod_shape = model_shape

        try:
            slc_lg, _ = overlap_slices(shape, mod_shape, (y0, x0), mode='trim')
        except NoOverlapError:
            continue

        yy, xx = np.mgrid[slc_lg]
        subimg = model(xx, yy)

        try:
            image[slc_lg] += subimg + local_bkg[i]
        except u.UnitConversionError as exc:
            msg = ('The local_bkg column must have the same flux units as '
                   'the output image')
            raise ValueError(msg) from exc

    return image


def make_psf_model_image(shape, psf_model, n_sources, *, model_shape=None,
                         min_separation=1, border_size=None, seed=0,
                         progress_bar=False, **kwargs):
    """
    Make an example image containing PSF model images.

    Source parameters are randomly generated using an optional ``seed``.

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output image.

    psf_model : 2D `astropy.modeling.Model`
        The PSF model. The model must have parameters named ``x_0``,
        ``y_0``, and ``flux``, corresponding to the center (x, y)
        position and flux, or it must have 'x_name', 'y_name',
        and 'flux_name' attributes that map to the x, y, and flux
        parameters. The model must be two-dimensional such that it
        accepts 2 inputs (e.g., x and y) and provides 1 output.

        The model is evaluated at the center of each pixel. To simulate
        the pixelated image of a source, the value of the model at a
        position must be the flux in a pixel centered at that position,
        as for the analytic models whose names end in ``PRF`` and for
        an image-based model made from an ePSF (see
        :ref:`psf-terminology`).

    n_sources : int
        The number of sources to generate. If ``min_separation`` is too
        large, the number of requested sources may not fit within the
        given ``shape`` and therefore the number of sources generated
        may be less than ``n_sources``.

    model_shape : `None`, int, or 2-tuple of int, optional
        The shape around the center (x, y) position that will be used
        to evaluate the ``psf_model``. If a single integer is provided,
        a square shape is used. If `None`, then the shape will be
        determined from the ``psf_model`` bounding box (an error will be
        raised if the model does not have a bounding box).

    min_separation : float, optional
        The minimum separation between the centers of two sources. Note
        that if the minimum separation is too large, the number of
        sources generated may be less than ``n_sources``.

    border_size : `None`, tuple of 2 int, or int, optional
        The (ny, nx) size of the exclusion border around the image edges
        where no sources will be generated that have centers within
        the border region. If a single integer is provided, it will be
        used for both dimensions. If `None`, then a border size equal
        to half the (y, x) size of the evaluated PSF model (taking any
        oversampling into account) will be used.

    seed : int, optional
        A seed to initialize the `numpy.random.BitGenerator`. If `None`,
        then fresh, unpredictable entropy will be pulled from the OS.

    progress_bar : bool, optional
        Whether to display a progress bar when creating the sources. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed.

    **kwargs : dict
        Keyword arguments are accepted for additional model parameters.
        The values should be 2-tuples of the lower and upper bounds for
        the parameter range. The parameter values will be uniformly
        sampled over the half-open interval defined by the lower and
        upper bounds (i.e., the lower bound is inclusive and the upper
        bound is exclusive). A ``flux`` keyword is mapped to the model's
        flux parameter name (e.g., for models with a ``flux_name``
        attribute). If the parameter is not in the input ``psf_model``
        parameter names, it will be ignored. Keywords matching the
        model's x and y position parameter names are also ignored
        because the source positions are randomly generated.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The simulated image.

    table : `~astropy.table.QTable`
        A table containing the parameters of the generated sources.
        It has an ``'id'`` column with unique source IDs, columns for
        the x and y positions named after the ``psf_model`` position
        parameters, and one column for each model parameter given in
        ``**kwargs``, named after the model parameter. A parameter
        that is not given, including the flux, has no column and takes
        the value set in ``psf_model`` for every source.

    Warns
    -----
    `~astropy.utils.exceptions.AstropyUserWarning`
        If fewer than ``n_sources`` sources could be generated with
        the given ``shape`` and ``min_separation``.

    Examples
    --------
    >>> from photutils.psf import CircularGaussianPRF
    >>> from photutils.simulation import make_psf_model_image
    >>> shape = (150, 200)
    >>> psf_model = CircularGaussianPRF(fwhm=3.5)
    >>> n_sources = 10
    >>> data, params = make_psf_model_image(shape, psf_model, n_sources,
    ...                                     flux=(100, 250),
    ...                                     min_separation=10,
    ...                                     seed=0)
    >>> params['x_0'].info.format = '.4f'  # optional format
    >>> params['y_0'].info.format = '.4f'
    >>> params['flux'].info.format = '.4f'
    >>> print(params)
     id   x_0      y_0      flux
    --- -------- -------- --------
      1 125.2010  72.3184 147.9522
      2  57.6408  39.1380 128.1262
      3  15.5391 115.4520 200.8790
      4  11.0411 131.7530 129.2661
      5 157.6417  43.6615 186.6532
      6 175.9470  80.2172 190.3359
      7 142.2274 132.7563 244.3635
      8 108.0270  13.4284 110.8398
      9 180.0533 106.0888 174.9959
     10 158.1171  90.3260 211.6146

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf import CircularGaussianPRF
        from photutils.simulation import make_psf_model_image

        shape = (150, 200)
        psf_model = CircularGaussianPRF(fwhm=3.5)
        n_sources = 10
        data, params = make_psf_model_image(shape, psf_model, n_sources,
                                            flux=(100, 250),
                                            min_separation=10,
                                            seed=0)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """
    # Import here because a module-level import is circular. The
    # photutils.psf subpackage imports photutils.simulation.
    from photutils.psf.utils import _get_psf_model_main_params

    main_params = _get_psf_model_main_params(psf_model)

    if model_shape is not None:
        model_shape = as_pair('model_shape', model_shape, lower_bound=(0, 0))
    else:
        try:
            model_shape = _model_shape_from_bbox(psf_model)
        except ValueError as exc:
            msg = ('model_shape must be specified if the model does not '
                   'have a bounding_box attribute')
            raise ValueError(msg) from exc

    if border_size is None:
        border_size = (np.array(model_shape) - 1) // 2

    other_params = {}
    if kwargs:
        # Include only kwargs that are model parameters, but not the
        # x and y position parameters
        for key, val in kwargs.items():
            param = key
            if key == 'flux' and 'flux' not in psf_model.param_names:
                # Map the flux kwarg to the model's flux parameter name
                param = main_params[2]
            if (param not in psf_model.param_names
                    or param in main_params[0:2]):
                continue  # skip the x, y parameters
            other_params[param] = val

    x_name, y_name = main_params[0:2]
    params = make_model_params(shape, n_sources, x_name=x_name, y_name=y_name,
                               min_separation=min_separation,
                               border_size=border_size, seed=seed,
                               **other_params)

    data = make_model_image(shape, psf_model, params, model_shape=model_shape,
                            x_name=x_name, y_name=y_name,
                            progress_bar=progress_bar)

    return data, params


def _make_gaussians_image(filename, shape, noise, noise_stddev,
                          **kwargs):
    """
    Make an image containing 2D Gaussians from a parameter file.

    Parameters
    ----------
    filename : str
        The name of the ECSV file containing the Gaussian parameters.

    shape : 2-tuple of int
        The shape of the output image.

    noise : bool
        Whether to include noise in the output image.

    noise_stddev : float
        The standard deviation of the Gaussian noise.

    **kwargs
        Additional keyword arguments passed to `make_model_image`.

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing 2D Gaussian sources.
    """
    model = Gaussian2D()
    params = QTable.read(_DATA_DIR / filename, format='ascii.ecsv')
    data = make_model_image(shape, model, params, x_name='x_mean',
                            y_name='y_mean', **kwargs)
    data += 5.0  # background

    if noise:
        rng = np.random.default_rng(seed=0)
        data += rng.normal(loc=0.0, scale=noise_stddev, size=shape)

    return data


def make_4gaussians_image(*, noise=True):
    """
    Make an example image containing four 2D Gaussians plus a constant
    background.

    The constant background has a value of 5.

    If ``noise`` is `True`, then Gaussian noise with a mean of 0 and a
    standard deviation of 5 is added to the output image. The noise is
    generated with a fixed random seed, so the output image is the same
    for every call.

    Parameters
    ----------
    noise : bool, optional
        Whether to include noise in the output image (default is
        `True`).

    Returns
    -------
    image : 2D `~numpy.ndarray`
        The image, with shape (100, 200), containing four 2D Gaussian
        sources.

    See Also
    --------
    make_100gaussians_image

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm
        from photutils.simulation import make_4gaussians_image

        image = make_4gaussians_image()
        fig, ax = plt.subplots()
        norm = simple_norm(image, 'sqrt', percent=99.5)
        ax.imshow(image, norm=norm, origin='lower')
    """
    return _make_gaussians_image('4gaussians_params.ecsv', (100, 200),
                                 noise, noise_stddev=5.0)


def make_100gaussians_image(*, noise=True):
    """
    Make an example image containing 100 2D Gaussians plus a constant
    background.

    The constant background has a value of 5.

    If ``noise`` is `True`, then Gaussian noise with a mean of 0 and a
    standard deviation of 2 is added to the output image. The noise is
    generated with a fixed random seed, so the output image is the same
    for every call.

    Parameters
    ----------
    noise : bool, optional
        Whether to include noise in the output image (default is
        `True`).

    Returns
    -------
    image : 2D `~numpy.ndarray`
        The image, with shape (300, 500), containing 100 2D Gaussian
        sources.

    See Also
    --------
    make_4gaussians_image

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from astropy.visualization import simple_norm
        from photutils.simulation import make_100gaussians_image

        image = make_100gaussians_image()
        fig, ax = plt.subplots()
        norm = simple_norm(image, 'sqrt', percent=99.5)
        ax.imshow(image, norm=norm, origin='lower')
    """
    return _make_gaussians_image('100gaussians_params.ecsv', (300, 500),
                                 noise, noise_stddev=2.0, bbox_factor=6.0)


def _model_shape_from_bbox(model, *, bbox_factor=None):
    """
    Calculate the model shape from the model bounding box.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model to be used to render the sources.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. If the model
        ``bounding_box`` method does not accept a ``factor`` keyword,
        then this keyword is ignored. If `None`, the default model
        bounding box will be used.

    Returns
    -------
    model_shape : 2-tuple of int
        The shape around the (x, y) center of the model that will be
        used to evaluate the model.

    Raises
    ------
    ValueError
        If the model does not have a bounding_box attribute.
    """
    try:
        _ = model.bounding_box
    except NotImplementedError as exc:
        msg = 'model does not have a bounding_box attribute'
        raise ValueError(msg) from exc

    if bbox_factor is not None:
        try:
            bbox = model.bounding_box(factor=bbox_factor)
        except NotImplementedError:
            bbox = model.bounding_box.bounding_box()
    else:
        bbox = model.bounding_box.bounding_box()

    return (int(np.ceil(bbox[0][1] - bbox[0][0])),
            int(np.ceil(bbox[1][1] - bbox[1][0])))
