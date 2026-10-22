# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating the signal-to-noise ratio of simulated sources.
"""

import numpy as np
from astropy.nddata import NoOverlapError

from photutils.simulation._cutouts import (psf_half_size, render_cells,
                                           render_convolved, trim_slices,
                                           validate_n_nodes,
                                           validate_oversample,
                                           validate_psf_model)
from photutils.simulation.images import (_can_reach_image,
                                         _convolution_margins, _iter_sources,
                                         _make_params_map,
                                         _reject_units_and_local_bkg,
                                         _source_slices,
                                         _validate_model_and_table,
                                         _validate_model_shape)
from photutils.utils._overlap import overlap_slices

__all__ = ['optimal_snr']


def optimal_snr(model, params_table, error, *, psf_model=None, n_nodes=None,
                oversample=4):
    r"""
    Calculate the optimal (matched-filter) signal-to-noise ratio of
    sources.

    For each source, the signal-to-noise ratio is

    .. math::

        \mathrm{SNR} = \sqrt{\sum_i \frac{s_i^2}{\sigma_i^2}}

    where :math:`s_i` is the flux of the source in pixel :math:`i`,
    :math:`\sigma_i` is the error in that pixel, and the sum is over the
    pixels of the source cutout that are in the image. This is the SNR
    of the flux measured with the filter matched to the source, so it
    is the highest SNR that any linear estimator of the flux can reach
    for independent pixel errors. The SNR of an aperture or of a fit
    is lower for sources that are not matched by the aperture or the
    fit model. The SNR is a property of the source and the noise, and
    does not depend on the detection algorithm.

    Each source is rendered exactly as it would be rendered in an
    image. If ``psf_model`` is `None`, it is rendered as by
    `~photutils.simulation.make_model_image` with the ``n_nodes``
    keyword. Otherwise, it is rendered as by
    `~photutils.simulation.make_convolved_model_image` with the
    ``psf_model``, ``oversample``, and ``n_nodes`` keywords. The
    sources are not added to an image, so each SNR is that of the
    source alone, even if the cutouts of sources overlap.

    The errors are assumed to be independent from pixel to pixel. For
    noise that is correlated, as in a resampled image, the SNR is
    overestimated (see `~photutils.simulation.noise_correlation_factor`).

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The 2D model of the sources, such as
        `~photutils.simulation.SersicSource` or an analytic PRF model.
        It must have parameters named ``x_0`` and ``y_0`` for the source
        position. Without ``psf_model``, it must give the flux in each
        pixel, or ``n_nodes`` must be given (see
        `~photutils.simulation.make_model_image`). With ``psf_model``,
        it must not be integrated over the pixels (see
        `~photutils.simulation.make_convolved_model_image`).

    params_table : `~astropy.table.Table`
        A table containing the model parameters for each source, whose
        column names match the model parameter names, as for
        `~photutils.simulation.make_model_image`. Model parameters
        that are not in the table keep the values of ``model``. If the
        table contains a ``'model_shape'`` column, its values set the
        shape of the cutout of each source. Otherwise, the cutout is
        the bounding box of the model. The columns must not have units,
        and the table must not have a ``'local_bkg'`` column.

    error : 2D `~numpy.ndarray`
        The 1-sigma error of each pixel of the image, in the same
        units as the model flux. It must have the shape of the image,
        which sets the extent of the image. Pixels whose error is not
        finite or not positive are excluded from the sum, as for a
        masked pixel.

    psf_model : `None` or 2D `astropy.modeling.Model`, optional
        The PSF model of the image, which must be integrated over the
        pixels, such as an ePSF or a model whose name ends in ``PRF``.
        If `None`, the sources are not convolved. See
        `~photutils.simulation.make_convolved_model_image` for the
        requirements of the model.

    n_nodes : `None` or int, optional
        The number of Gauss-Legendre nodes per axis used to integrate
        each pixel, or each cell if ``psf_model`` is given. If `None`,
        the source model is evaluated at the center of each pixel or
        cell. This is not the default of
        `~photutils.simulation.make_convolved_model_image`, so pass
        ``n_nodes=4`` to match its default. See
        `~photutils.simulation.make_model_image`.

    oversample : int, optional
        The number of grid points per pixel along each axis of
        the convolution grid. It is used only if ``psf_model`` is
        given. See `~photutils.simulation.make_convolved_model_image`.

    Returns
    -------
    snr : 1D `~numpy.ndarray`
        The signal-to-noise ratio of each source, in the order of the
        rows of ``params_table``. The SNR is NaN for a row that has a
        nonfinite model parameter and for a source that does not
        overlap the image. As for the image renderers, a source outside
        the image whose light reaches the image through the ``psf_model``
        has an SNR. The SNR is 0 for a source whose overlap with the
        image has no pixel with a positive, finite error.

    Raises
    ------
    TypeError
        If ``model`` or ``psf_model`` is not an `astropy.modeling.Model`
        instance, if ``params_table`` is not an `~astropy.table.Table`,
        or if ``psf_model`` is not integrated over the pixels.

    ValueError
        If ``error`` is not a 2D array or has units, if ``model`` does
        not have 2 inputs and 1 output or does not have ``x_0`` and
        ``y_0`` parameters, if ``n_nodes`` is not `None` or a positive
        integer, if ``oversample`` is not a positive integer, if a
        required column is not in ``params_table``, if the model has no
        ``bounding_box`` attribute and no ``'model_shape'`` column is
        given, if a ``params_table`` column has units or the model
        output has units, if ``params_table`` has a ``'local_bkg'``
        column, or, with ``psf_model``, if it does not have a bounding
        box or its x, y, and flux parameters cannot be found.

    See Also
    --------
    make_model_image, make_convolved_model_image

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.table import QTable
    >>> from photutils.psf import CircularGaussianPRF
    >>> from photutils.simulation import optimal_snr
    >>> params = QTable()
    >>> params['x_0'] = [20.3, 40.7]
    >>> params['y_0'] = [25.8, 12.2]
    >>> params['flux'] = [100.0, 400.0]
    >>> error = np.full((50, 60), 2.0)
    >>> snr = optimal_snr(CircularGaussianPRF(fwhm=3.0), params, error)
    >>> print(np.round(snr, 2))
    [10.8  43.18]
    """
    # Import here because a module-level import is circular. The
    # photutils.psf subpackage imports photutils.simulation.
    from photutils.psf.utils import _get_psf_model_main_params

    if getattr(error, 'unit', None) is not None:
        msg = 'error must not have units'
        raise ValueError(msg)
    error = np.asarray(error, dtype=float)
    if error.ndim != 2:
        msg = 'error must be a 2D array'
        raise ValueError(msg)
    shape = error.shape

    n_nodes = validate_n_nodes(n_nodes)
    _validate_model_and_table(model, params_table)
    if psf_model is not None:
        validate_oversample(oversample)
        validate_psf_model(psf_model)
        _get_psf_model_main_params(psf_model)
        half_y, half_x = psf_half_size(psf_model, oversample)
        margins = _convolution_margins(half_y, half_x, oversample)

    x_name = 'x_0'
    y_name = 'y_0'
    if x_name not in model.param_names or y_name not in model.param_names:
        msg = 'model must have parameters named x_0 and y_0'
        raise ValueError(msg)

    params_map = _make_params_map(model, params_table, x_name, y_name, None)
    model_shape, variable_shape = _validate_model_shape(model, params_table,
                                                        None)

    # Copy the input model to leave it unchanged
    model = model.copy()
    _reject_units_and_local_bkg('optimal_snr', model, params_table,
                                params_map)

    # Pixels with an error that is not finite and positive are excluded
    with np.errstate(invalid='ignore'):
        good = np.isfinite(error) & (error > 0)

    snr = np.full(len(params_table), np.nan)
    sources = _iter_sources(model, params_table, params_map, None,
                            model_shape=model_shape,
                            variable_shape=variable_shape, bbox_factor=None,
                            x_name=x_name, y_name=y_name, progress_bar=False)
    for i, x0, y0, mod_shape in sources:
        if psf_model is None:
            try:
                slices, _ = overlap_slices(shape, mod_shape, (y0, x0),
                                           mode='trim')
            except NoOverlapError:
                continue
            yy, xx = np.mgrid[slices]
            cutout = render_cells(model, xx, yy, x0, y0, n_nodes=n_nodes)
            slices_cutout = (slice(None), slice(None))
        else:
            slices = _source_slices(mod_shape, x0, y0)
            if not _can_reach_image(slices, margins, shape):
                continue
            out_slices, cutout = render_convolved(model, psf_model, slices,
                                                  x0, y0,
                                                  oversample=oversample,
                                                  n_nodes=n_nodes)
            overlap = trim_slices(out_slices, shape)
            if overlap is None:
                continue
            slices, slices_cutout = overlap

        use = good[slices]
        signal = cutout[slices_cutout][use]
        snr[i] = np.sqrt(np.sum(signal**2 / error[slices][use]**2))

    return snr
