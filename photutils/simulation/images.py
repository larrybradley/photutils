# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for making simulated images from astropy models.
"""

import pathlib
from math import ceil

import astropy.units as u
import numpy as np
from astropy.modeling import Model
from astropy.modeling.models import Gaussian2D
from astropy.nddata import NoOverlapError
from astropy.table import QTable, Table
from astropy.utils.data import get_pkg_data_path

from photutils.simulation._cutouts import (psf_half_size, render_cells,
                                           render_convolved, trim_slices,
                                           validate_n_nodes,
                                           validate_oversample,
                                           validate_psf_model)
from photutils.simulation.params import make_model_params
from photutils.utils._overlap import overlap_slices
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar

__all__ = [
    'make_4gaussians_image',
    'make_100gaussians_image',
    'make_convolved_model_image',
    'make_model_image',
    'make_psf_model_image',
]

_DATA_DIR = pathlib.Path(get_pkg_data_path('simulation', 'data',
                                           package='photutils'))


def make_model_image(shape, model, params_table, *, model_shape=None,
                     bbox_factor=None, x_name='x_0', y_name='y_0',
                     params_map=None, n_nodes=None, progress_bar=False):
    """
    Make a 2D image containing sources generated from a user-specified
    astropy 2D model.

    The model parameters for each source are taken from the input
    ``params_table`` table. By default, the table is searched for column
    names that match model parameter names and the values specified by
    ``x_name`` and ``y_name``. However, the user can specify a different
    mapping between model parameter names and column names using the
    ``params_map`` keyword.

    By default, each source is rendered by evaluating the model at
    the center of each pixel. To simulate the pixelated image of a
    source, the value of the model at a position must then be the
    flux in a pixel centered at that position, as for the analytic
    models whose names end in ``PRF`` and for an image-based model
    made from an ePSF (see :ref:`psf-terminology`). For models that
    are not integrated over pixels, use the ``n_nodes`` keyword to
    integrate each pixel.

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

    n_nodes : `None` or int, optional
        If `None`, the model is evaluated at the center of each pixel.
        This is correct for models that give the flux in a pixel, such
        as an ePSF or a PRF model. If an integer, each pixel is instead
        integrated with ``n_nodes`` x ``n_nodes`` Gauss-Legendre nodes.
        This is for models that are not integrated over pixels, such
        as a PSF model or a galaxy profile, and it takes the role of
        the ``discretize_method`` and ``discretize_oversample``
        keywords of the deprecated
        ``photutils.datasets.make_model_image``. The pixel that
        contains the source center and its eight neighbors are refined
        by splitting each pixel into pieces at most 1/8 pixel wide
        that are each integrated with the same number of nodes. The
        piece boundaries are aligned to the source center, so the
        accuracy does not depend on where the center falls within a
        pixel. A value of 4 gives about 0.1% accuracy relative to the
        peak for typical galaxies. A value of 8 or more is needed for
        Sersic indices of 6 or more and for effective radii below 0.5
        pixels. The number of model evaluations scales as ``n_nodes``
        squared.

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
        If ``model`` does not have 2 inputs and 1 output, if
        ``n_nodes`` is not `None` or a positive integer, if a
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
    n_nodes = validate_n_nodes(n_nodes)
    _validate_model_and_table(model, params_table)
    params_map = _make_params_map(model, params_table, x_name, y_name,
                                  params_map)
    model_shape, variable_shape = _validate_model_shape(model, params_table,
                                                        model_shape)

    if 'local_bkg' in params_table.colnames:
        local_bkg = params_table['local_bkg']
    else:
        local_bkg = np.zeros(len(params_table))

    # Copy the input model to leave it unchanged
    model = model.copy()

    image = np.zeros(shape, dtype=float)
    unit = _get_output_unit(model, params_table, params_map)
    if unit is not None:
        image <<= unit

    sources = _iter_sources(model, params_table, params_map, local_bkg,
                            model_shape=model_shape,
                            variable_shape=variable_shape,
                            bbox_factor=bbox_factor, x_name=x_name,
                            y_name=y_name, progress_bar=progress_bar)
    for i, x0, y0, mod_shape in sources:
        try:
            slc_lg, _ = overlap_slices(shape, mod_shape, (y0, x0), mode='trim')
        except NoOverlapError:
            continue

        yy, xx = np.mgrid[slc_lg]
        subimg = render_cells(model, xx, yy, x0, y0, n_nodes=n_nodes)

        try:
            image[slc_lg] += subimg + local_bkg[i]
        except u.UnitConversionError as exc:
            msg = ('The local_bkg column must have the same flux units as '
                   'the output image')
            raise ValueError(msg) from exc

    return image


def make_convolved_model_image(shape, model, params_table, psf_model, *,
                               oversample=4, n_nodes=4, model_shape=None,
                               bbox_factor=None, x_name='x_0', y_name='y_0',
                               params_map=None, progress_bar=False):
    """
    Make a 2D image containing sources convolved with a PSF model.

    Use this function for extended sources, such as galaxies made
    with `~photutils.simulation.SersicSource`, observed through a PSF.
    The model parameters for each source are taken from the input
    ``params_table`` table, as for `make_model_image`.

    Each source is rendered in three steps:

    1. The source model is integrated over the square cells of a grid
       with a spacing of ``1 / oversample`` pixels whose points include
       every pixel center of the source cutout. Each cell is integrated
       with ``n_nodes`` x ``n_nodes`` Gauss-Legendre nodes, and the
       cells near the source center are refined as described for the
       ``n_nodes`` keyword of `make_model_image`.
    2. The cell integrals are convolved with the PSF model evaluated at
       the grid offsets inside its bounding box. The PSF model must be
       integrated over the pixels, so it already includes the pixel
       response.
    3. The result is kept at the pixel centers.

    Each source is integrated over the cells before it is convolved,
    so the result is the source convolved with a square box of width
    ``1 / oversample`` pixels and the PSF model, and not the source
    convolved with the PSF model alone. The box broadens the source
    by a variance of ``1 / (12 * oversample**2)`` square pixels along
    each axis, which is negligible for most sources. A source that
    is much smaller than a cell falls in a single cell, so it renders
    as the PSF model at the center of that cell, displaced from the
    source center by up to half a cell along each axis. Averaged over
    source positions, this is the same broadening by the box. Such a
    source, centered on a grid point, renders exactly as the PSF model.
    Increase ``oversample`` to reduce the box, or render point sources
    with `make_model_image` and the PSF model. For an
    `~photutils.psf.ImagePSF`, the natural choice of ``oversample`` is
    its own oversampling factor, but any value works.

    Flux beyond the source cutout, which is defined by ``model_shape``
    or the model bounding box, is lost, as is flux beyond the bounding
    box of the PSF model. Sources are trimmed at the image edges, and
    a source outside the image whose light reaches the image through
    the PSF is included.

    Parameters
    ----------
    shape : 2-tuple of int or int
        The shape of the output image. If a single integer is provided,
        it will be used for both dimensions.

    model : 2D `astropy.modeling.Model`
        The 2D model of the sources before convolution, such as
        `~photutils.simulation.SersicSource`. It must not be integrated
        over the pixels. The model must have parameters for the x and
        y positions of the sources, named by the ``x_name`` and
        ``y_name`` keywords.

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source, as
        for `make_model_image`, including the optional ``'model_shape'``
        column. Rows that contain any nonfinite model parameters will be
        skipped. The columns must not have units, and the table must not
        have a ``'local_bkg'`` column.

    psf_model : 2D `astropy.modeling.Model`
        The PSF model. It must be integrated over the pixels, such as
        an ePSF (`~photutils.psf.ImagePSF` or
        `~photutils.psf.GriddedPSFModel`) or an analytic model whose
        name ends in ``PRF`` (see :ref:`psf-terminology`). A model
        whose name ends in ``PSF`` is rejected, except for an
        `~photutils.psf.ImagePSF`. The model must have a bounding box
        and parameters named ``x_0``, ``y_0``, and ``flux``, or
        ``x_name``, ``y_name``, and ``flux_name`` attributes that name
        them. Its other parameters are used for every source. A
        `~photutils.psf.GriddedPSFModel` is evaluated at the pixel
        nearest to each source center.

    oversample : int, optional
        The number of grid points per pixel along each axis.

    n_nodes : `None` or int, optional
        The number of Gauss-Legendre nodes per axis used to integrate
        each cell, or `None` to evaluate the source model at the cell
        centers. As for `make_model_image`, a value of 4 gives about
        0.1% accuracy relative to the peak for typical galaxies, and
        a value of 8 or more is needed for Sersic indices of 6 or more
        and for effective radii below 0.5 pixels. The number of model
        evaluations scales as ``n_nodes`` squared and as ``oversample``
        squared.

    model_shape : 2-tuple of int, int, or `None`, optional
        The shape around the (x, y) center of each source over which
        the source model is integrated, before convolution. If `None`,
        then the bounding box of the model will be used (which can
        optionally be scaled using the ``bbox_factor`` keyword if the
        model supports it). See `make_model_image`.

    bbox_factor : `None` or float, optional
        The multiplicative factor to pass to the model ``bounding_box``
        method to determine the model shape. See `make_model_image`.

    x_name : str, optional
        The name of the ``model`` parameter that corresponds to the x
        position of the sources.

    y_name : str, optional
        The name of the ``model`` parameter that corresponds to the y
        position of the sources.

    params_map : dict or None, optional
        A dictionary mapping the model parameter names to the column
        names in the input ``params_table``. See `make_model_image`.

    progress_bar : bool, optional
        Whether to display a progress bar while adding the sources
        to the image. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Returns
    -------
    array : 2D `~numpy.ndarray`
        The rendered image containing the convolved sources.

    Raises
    ------
    TypeError
        If ``model`` or ``psf_model`` is not an `astropy.modeling.Model`
        instance, if ``params_table`` is not an `~astropy.table.Table`,
        or if ``psf_model`` is not integrated over the pixels.

    ValueError
        If ``model`` does not have 2 inputs and 1 output, if
        ``oversample`` is not a positive integer, if ``n_nodes`` is
        not `None` or a positive integer, if ``psf_model`` does not
        have a bounding box or its x, y, and flux parameters cannot
        be found, if a ``params_map`` key is not a model parameter
        name, if a required column is not in ``params_table``, if
        the model has no ``bounding_box`` attribute and no model shape
        is specified, if a ``params_table`` column has units or the
        model output has units, or if ``params_table`` has a
        ``'local_bkg'`` column.

    See Also
    --------
    make_model_image

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from photutils.psf import CircularGaussianPRF
        from photutils.simulation import (SersicSource,
                                          make_convolved_model_image,
                                          make_model_params)

        shape = (200, 200)
        params = make_model_params(shape, 20, min_separation=20,
                                   border_size=10, flux=(500, 1000),
                                   r_eff=(1.5, 4), n=(0.5, 4),
                                   ellip=(0, 0.6), theta=(0, 3.14),
                                   seed=0)
        psf_model = CircularGaussianPRF(fwhm=3.0)
        data = make_convolved_model_image(shape, SersicSource(), params,
                                          psf_model)

        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """
    # Import here because a module-level import is circular. The
    # photutils.psf subpackage imports photutils.simulation.
    from photutils.psf.utils import _get_psf_model_main_params

    shape = as_pair('shape', shape, lower_bound=(0, 0))
    n_nodes = validate_n_nodes(n_nodes)
    validate_oversample(oversample)

    _validate_model_and_table(model, params_table)
    validate_psf_model(psf_model)
    _get_psf_model_main_params(psf_model)
    half_y, half_x = psf_half_size(psf_model, oversample)

    params_map = _make_params_map(model, params_table, x_name, y_name,
                                  params_map)
    model_shape, variable_shape = _validate_model_shape(model, params_table,
                                                        model_shape)

    # Copy the input model to leave it unchanged
    model = model.copy()
    _reject_units_and_local_bkg('make_convolved_model_image', model,
                                params_table, params_map)

    margins = _convolution_margins(half_y, half_x, oversample)

    image = np.zeros(shape, dtype=float)
    sources = _iter_sources(model, params_table, params_map, None,
                            model_shape=model_shape,
                            variable_shape=variable_shape,
                            bbox_factor=bbox_factor, x_name=x_name,
                            y_name=y_name, progress_bar=progress_bar)
    for _, x0, y0, mod_shape in sources:
        slices = _source_slices(mod_shape, x0, y0)
        if not _can_reach_image(slices, margins, shape):
            continue
        # Only the image and the PSF margin around it can contribute
        slices = tuple(slice(max(slc.start, -mar), min(slc.stop, size + mar))
                       for slc, mar, size in zip(slices, margins, shape,
                                                 strict=True))

        out_slices, cutout = render_convolved(model, psf_model, slices, x0,
                                              y0, oversample=oversample,
                                              n_nodes=n_nodes)
        overlap = trim_slices(out_slices, shape)
        if overlap is None:
            continue
        slices_image, slices_cutout = overlap
        image[slices_image] += cutout[slices_cutout]

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
        :ref:`psf-terminology`). A model whose name ends in ``PSF``
        is rejected with a `TypeError`, except for an
        `~photutils.psf.ImagePSF`.

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

    validate_psf_model(psf_model)
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
                            y_name='y_mean', n_nodes=None, **kwargs)
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


def _validate_model_and_table(model, params_table):
    """
    Validate the source model and the parameters table.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model used to render the sources.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source.

    Raises
    ------
    TypeError
        If ``model`` is not an `astropy.modeling.Model` instance or if
        ``params_table`` is not an `~astropy.table.Table`.

    ValueError
        If ``model`` does not have 2 inputs and 1 output.
    """
    if not isinstance(model, Model):
        msg = 'model must be a Model instance'
        raise TypeError(msg)
    if model.n_inputs != 2 or model.n_outputs != 1:
        msg = 'model must be a 2D model'
        raise ValueError(msg)
    if not isinstance(params_table, Table):
        msg = 'params_table must be an astropy Table'
        raise TypeError(msg)


def _make_params_map(model, params_table, x_name, y_name, params_map):
    """
    Make the mapping from model parameter names to table column names.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model used to render the sources.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source.

    x_name, y_name : str
        The names of the model parameters for the x and y positions.

    params_map : dict or None
        The user mapping from model parameter names to column names,
        which takes precedence over ``x_name``, ``y_name``, and the
        column names that match model parameter names.

    Returns
    -------
    params_map : dict
        The mapping from model parameter names to column names.

    Raises
    ------
    ValueError
        If a key is not a model parameter name or a value is not a
        column name.
    """
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

    return params_map


def _validate_model_shape(model, params_table, model_shape):
    """
    Validate the model shape of the sources.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model used to render the sources.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source. A
        ``'model_shape'`` column overrides ``model_shape``.

    model_shape : 2-tuple of int, int, or `None`
        The shape around the center of each source.

    Returns
    -------
    model_shape : `None`, 2-tuple of int, or 2D `~numpy.ndarray`
        The validated model shape, or an array of the (ny, nx) shapes
        of each source if ``variable_shape`` is `True`.

    variable_shape : bool
        Whether ``params_table`` has a ``'model_shape'`` column.

    Raises
    ------
    ValueError
        If no model shape is specified and the model does not have a
        bounding box.
    """
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

    return model_shape, variable_shape


def _get_output_unit(model, params_table, params_map):
    """
    Return the units of the model output, if any.

    The units are found by evaluating the model with unit scalars
    derived from the mapped table columns at (0, 0), so that they do
    not depend on whether any sources are rendered (e.g., a zero-row
    table or sources that do not overlap the image). Unmapped model
    parameters keep their values and units.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        A copy of the model used to render the sources. Its mapped
        parameters are changed.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source.

    params_map : dict
        The mapping from model parameter names to column names.

    Returns
    -------
    unit : `None` or `~astropy.units.UnitBase`
        The units of the model output, or `None` if it has no units.
    """
    for key, param in params_map.items():
        unit = getattr(params_table[param], 'unit', None)
        setattr(model, key, 1 if unit is None else 1 * unit)

    # Only the units are used, so ignore floating-point errors from
    # parameter values that are invalid for the model (e.g., an
    # ellipticity of 1)
    with np.errstate(all='ignore'):
        result = model(0, 0)
    if isinstance(result, u.Quantity):
        return result.unit
    return None


def _iter_sources(model, params_table, params_map, local_bkg, *,
                  model_shape, variable_shape, bbox_factor, x_name, y_name,
                  progress_bar):
    """
    Set the model parameters for each source in turn.

    Sources that have any nonfinite parameter or local background value
    are skipped.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        A copy of the model used to render the sources. Its parameters
        are set for each source before the source is yielded.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source.

    params_map : dict
        The mapping from model parameter names to column names.

    local_bkg : `None` or array_like
        The local background of each source, or `None` for none.

    model_shape : `None`, 2-tuple of int, or 2D `~numpy.ndarray`
        The model shape returned by `_validate_model_shape`.

    variable_shape : bool
        Whether ``model_shape`` holds the shape of each source.

    bbox_factor : `None` or float
        The factor passed to the model ``bounding_box`` method when
        ``model_shape`` is `None`.

    x_name, y_name : str
        The names of the model parameters for the x and y positions.

    progress_bar : bool
        Whether to display a progress bar.

    Yields
    ------
    index : int
        The row index of the source in ``params_table``.

    x0, y0 : float
        The source center.

    model_shape : 2-tuple of int
        The (ny, nx) shape around the source center.

    Raises
    ------
    ValueError
        If a ``params_table`` column has units and ``params_table`` is
        not a `~astropy.table.QTable`.
    """
    if progress_bar:
        desc = 'Add model sources'
        params_table = add_progress_bar(params_table, desc=desc)

    for i, source in enumerate(params_table):
        params = {key: source[param] for key, param in params_map.items()}

        # Skip sources that have any nonfinite parameter or local
        # background value
        if (not all(np.isfinite(value) for value in params.values())
                or (local_bkg is not None
                    and not np.isfinite(local_bkg[i]))):
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

        yield i, x0, y0, mod_shape


def _source_slices(model_shape, x0, y0):
    """
    Return the pixel slices of a source cutout, without trimming them
    at the image edges.

    The cutout is placed as `~astropy.nddata.overlap_slices` places it.

    Parameters
    ----------
    model_shape : 2-tuple of int
        The (ny, nx) shape of the cutout.

    x0, y0 : float
        The source center.

    Returns
    -------
    slices : tuple of 2 slice
        The (y, x) pixel slices, which may extend beyond the image,
        including to negative indices.
    """
    slices = []
    for size, center in zip(model_shape, (y0, x0), strict=True):
        start = int(np.ceil(center - size / 2.0))
        slices.append(slice(start, start + int(size)))
    return tuple(slices)


def _reject_units_and_local_bkg(caller, model, params_table, params_map):
    """
    Raise an error if the sources have units or a local background.

    Parameters
    ----------
    caller : str
        The name of the calling function, used in the error message.

    model : 2D `astropy.modeling.Model`
        A copy of the model used to render the sources. Its mapped
        parameters are changed.

    params_table : `~astropy.table.Table`
        The table of model parameters for each source.

    params_map : dict
        The mapping from model parameter names to column names.

    Raises
    ------
    ValueError
        If ``params_table`` has a ``'local_bkg'`` column, if a mapped
        column has units, or if the model output has units.
    """
    if 'local_bkg' in params_table.colnames:
        msg = f'{caller} does not support a local_bkg column'
        raise ValueError(msg)

    has_units = any(getattr(params_table[column], 'unit', None) is not None
                    for column in params_map.values())
    if has_units or _get_output_unit(model, params_table,
                                     params_map) is not None:
        msg = f'{caller} does not support units'
        raise ValueError(msg)


def _convolution_margins(half_y, half_x, oversample):
    """
    Return the number of pixels by which a convolution can extend a
    source cutout along each axis.

    Parameters
    ----------
    half_y, half_x : int
        The half-size of the PSF model bounding box, in units of the
        grid spacing ``1 / oversample``.

    oversample : int
        The number of grid points per pixel along each axis.

    Returns
    -------
    margins : 2-tuple of int
        The (y, x) margins, in pixels.
    """
    return (ceil((half_y + oversample // 2) / oversample),
            ceil((half_x + oversample // 2) / oversample))


def _can_reach_image(slices, margins, shape):
    """
    Return whether a convolved source can reach the image.

    Parameters
    ----------
    slices : tuple of 2 slice
        The (y, x) pixel slices of the source cutout before convolution.

    margins : 2-tuple of int
        The (y, x) margins returned by `_convolution_margins`.

    shape : 2-tuple of int
        The shape of the image.

    Returns
    -------
    result : bool
        `False` if the cutout is empty or if its margin does not
        overlap the image.
    """
    grown = tuple(slice(slc.start - margin, slc.stop + margin)
                  for slc, margin in zip(slices, margins, strict=True))
    return (all(slc.start < slc.stop for slc in slices)
            and trim_slices(grown, shape) is not None)
