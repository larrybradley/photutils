# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define helper utilities for making PSF models.
"""

import re

import numpy as np
from astropy.modeling import CompoundModel
from astropy.modeling.models import Const2D, Identity, Shift
from astropy.nddata import NDData
from astropy.units import Quantity
from scipy.integrate import dblquad, trapezoid

__all__ = ['grid_from_epsfs', 'make_psf_model']


def make_psf_model(model, *, x_name=None, y_name=None, flux_name=None,
                   normalize=True, dx=50, dy=50, subsample=100,
                   use_dblquad=False):
    """
    Make a PSF model that can be used with the PSF photometry classes
    (`PSFPhotometry` or `IterativePSFPhotometry`) from an Astropy
    fittable 2D model.

    If the ``x_name``, ``y_name``, or ``flux_name`` keywords are input,
    this function will map those ``model`` parameter names to ``x_0``,
    ``y_0``, or ``flux``, respectively.

    If any of the ``x_name``, ``y_name``, or ``flux_name`` keywords
    are `None`, then a new parameter will be added to the model
    corresponding to the missing parameter. Any new position parameters
    will be set to a default value of 0, and any new flux parameter will
    be set to a default value of 1.

    The output PSF model will have ``x_name``, ``y_name``, and
    ``flux_name`` attributes that contain the name of the corresponding
    model parameter.

    .. note::

        This function is needed only in cases where the 2D PSF model
        does not have ``x_0``, ``y_0``, and ``flux`` parameters.

        It is *not* needed for any of the PSF models provided by
        Photutils.

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        An Astropy fittable 2D model to use as a PSF.

    x_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the x
        center of the PSF. If `None`, the model will be assumed to be
        centered at x=0, and a new model parameter called ``xpos_0``
        will be added for the x position.

    y_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y center of the PSF. If `None`, the model will be assumed
        to be centered at y=0, and a new parameter called ``ypos_1``
        will be added for the y position.

    flux_name : `str` or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        total flux of a source. If `None`, a new model parameter called
        ``flux_3`` will be added for model flux.

    normalize : bool, optional
        If `True`, the input ``model`` will be integrated and rescaled
        so that its sum integrates to 1. This normalization occurs only
        once for the input ``model``. If the total flux of ``model``
        somehow depends on (x, y) position, then one will need to
        correct the fitted model fluxes for this effect.

    dx, dy : odd int, optional
        The size of the integration grid in x and y for normalization.
        Must be odd. These keywords are ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis
        for normalization. Each pixel will be sampled ``subsample`` x
        ``subsample`` times. This keyword is ignored if ``normalize`` is
        `False` or ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model for normalization. This is *much* slower than the default
        integration of the evaluated model, but it is more accurate.
        This keyword is ignored if ``normalize`` is `False`.

    Returns
    -------
    result : `~astropy.modeling.CompoundModel`
        A PSF model that can be used with the PSF photometry classes.
        The returned model will always be an Astropy compound model.

    Notes
    -----
    To normalize the model, by default it is discretized on a grid of
    size ``dx`` x ``dy`` from the model center with a subsampling factor
    of ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.

    Examples
    --------
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.psf import make_psf_model
    >>> model = Gaussian2D(x_stddev=2, y_stddev=2)
    >>> psf_model = make_psf_model(model, x_name='x_mean', y_name='y_mean')
    >>> print(psf_model.param_names)  # doctest: +SKIP
    ('amplitude_2', 'x_mean_2', 'y_mean_2', 'x_stddev_2', 'y_stddev_2',
     'theta_2', 'amplitude_3', 'amplitude_4')
    """
    input_model = model.copy()

    if x_name is None:
        x_model = _InverseShift(0, name='x_position')
        # "offset" is the _InverseShift parameter name;
        # the x inverse shift model is always the first submodel
        x_name = 'offset_0'
    else:
        if x_name not in input_model.param_names:
            msg = f'{x_name!r} parameter name not found in the input model'
            raise ValueError(msg)

        x_model = Identity(1)
        x_name = _shift_model_param(input_model, x_name, shift=2)

    if y_name is None:
        y_model = _InverseShift(0, name='y_position')
        # "offset" is the _InverseShift parameter name;
        # the y inverse shift model is always the second submodel
        y_name = 'offset_1'
    else:
        if y_name not in input_model.param_names:
            msg = f'{y_name!r} parameter name not found in the input model'
            raise ValueError(msg)

        y_model = Identity(1)
        y_name = _shift_model_param(input_model, y_name, shift=2)

    x_model.fittable = True
    y_model.fittable = True
    psf_model = (x_model & y_model) | input_model

    if flux_name is None:
        psf_model *= Const2D(1.0, name='flux')
        # "amplitude" is the Const2D parameter name;
        # the flux scaling is always the last component (prior to
        # normalization)
        flux_name = psf_model.param_names[-1]
    else:
        flux_name = _shift_model_param(input_model, flux_name, shift=2)

    if normalize:
        integral = _integrate_model(psf_model, x_name=x_name, y_name=y_name,
                                    dx=dx, dy=dy, subsample=subsample,
                                    use_dblquad=use_dblquad)

        if integral == 0:
            msg = ('Cannot normalize the model because the integrated flux '
                   'is zero')
            raise ValueError(msg)

        psf_model *= Const2D(1.0 / integral, name='normalization_scaling')

    # fix all the output model parameters that are not x, y, or flux
    for name in psf_model.param_names:
        psf_model.fixed[name] = name not in (x_name, y_name, flux_name)

    # final check that the x, y, and flux parameter names are in the
    # output model
    names = (x_name, y_name, flux_name)
    for name in names:
        if name not in psf_model.param_names:
            msg = f'{name!r} parameter name not found in the output model'
            raise ValueError(msg)

    # set the parameter names for the PSF photometry classes
    psf_model.x_name = x_name
    psf_model.y_name = y_name
    psf_model.flux_name = flux_name

    # set aliases
    psf_model.x_0 = getattr(psf_model, x_name)
    psf_model.y_0 = getattr(psf_model, y_name)
    psf_model.flux = getattr(psf_model, flux_name)

    return psf_model


class _InverseShift(Shift):
    """
    A model that is the inverse of the normal
    `astropy.modeling.functional_models.Shift` model.
    """

    @staticmethod
    def evaluate(x, offset):
        return x - offset

    @staticmethod
    def fit_deriv(x, offset):
        """
        One dimensional Shift model derivative with respect to
        parameter.
        """
        d_offset = -np.ones_like(x) + offset * 0.0
        return [d_offset]


def _integrate_model(model, x_name=None, y_name=None, dx=50, dy=50,
                     subsample=100, use_dblquad=False):
    """
    Integrate a model over a 2D grid.

    By default, the model is discretized on a grid of size ``dx``
    x ``dy`` from the model center with a subsampling factor of
    ``subsample``. The model is then integrated over the grid using
    trapezoidal integration.

    If the ``use_dblquad`` keyword is set to `True`, then the model is
    integrated using `scipy.integrate.dblquad`. This is *much* slower
    than the default integration of the evaluated model, but it is more
    accurate. Also, note that the ``dblquad`` integration can sometimes
    fail, e.g., return zero for a non-zero model. This can happen when
    the model function is sharply localized relative to the size of the
    integration interval.

    Parameters
    ----------
    model : `~astropy.modeling.Fittable2DModel`
        The Astropy 2D model.

    x_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        x-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    y_name : str or `None`, optional
        The name of the ``model`` parameter that corresponds to the
        y-axis center of the PSF. This parameter is required if
        ``use_dblquad`` is `False` and ignored if ``use_dblquad`` is
        `True`.

    dx, dy : odd int, optional
        The size of the integration grid in x and y. Must be odd.
        These keywords are ignored if ``use_dblquad`` is `True`.

    subsample : int, optional
        The subsampling factor for the integration grid along each axis.
        Each pixel will be sampled ``subsample`` x ``subsample`` times.
        This keyword is ignored if ``use_dblquad`` is `True`.

    use_dblquad : bool, optional
        If `True`, then use `scipy.integrate.dblquad` to integrate the
        model. This is *much* slower than the default integration of
        the evaluated model, but it is more accurate.

    Returns
    -------
    integral : float
        The integral of the model over the 2D grid.
    """
    if use_dblquad:
        return dblquad(model, -np.inf, np.inf, -np.inf, np.inf)[0]

    if dx <= 0 or dy <= 0:
        msg = 'dx and dy must be > 0'
        raise ValueError(msg)
    if subsample < 1:
        msg = 'subsample must be >= 1'
        raise ValueError(msg)

    xc = getattr(model, x_name)
    yc = getattr(model, y_name)

    if np.any(~np.isfinite((xc.value, yc.value))):
        msg = 'model x and y positions must be finite'
        raise ValueError(msg)

    hx = (dx - 1) / 2
    hy = (dy - 1) / 2
    nxpts = int(dx * subsample)
    nypts = int(dy * subsample)
    xvals = np.linspace(xc - hx, xc + hx, nxpts)
    yvals = np.linspace(yc - hy, yc + hy, nypts)

    # evaluate the model on the subsampled grid
    data = model(xvals.reshape(-1, 1), yvals.reshape(1, -1))
    if isinstance(data, Quantity):
        data = data.value

    # now integrate over the subsampled grid (first over x, then over y)
    int_func = trapezoid

    return int_func([int_func(row, xvals) for row in data], yvals)


def _shift_model_param(model, param_name, shift=2):
    if isinstance(model, CompoundModel):
        # for CompoundModel, add "shift" to the parameter suffix
        out = re.search(r'(.*)_([\d]*)$', param_name)
        new_name = out.groups()[0] + '_' + str(int(out.groups()[1]) + 2)
    else:
        # simply add the shift to the parameter name
        new_name = param_name + '_' + str(shift)

    return new_name


def _validate_epsf_consistency(epsfs, grid_xypos, reference_values):
    """
    Validate that all EPSFs have consistent properties.

    Parameters
    ----------
    epsfs : list of ImagePSF
        List of ImagePSF models to validate.
    grid_xypos : list or None
        Grid positions, if provided.
    reference_values : dict
        Dictionary containing reference values from the first EPSF.

    Raises
    ------
    ValueError
        If EPSFs have inconsistent properties.
    """
    for epsf in epsfs[1:]:  # Start from second EPSF
        if not np.array_equal(epsf.oversampling,
                              reference_values['oversampling']):
            msg = ('All input ImagePSF models must have the same value '
                   'for oversampling')
            raise ValueError(msg)

        if epsf.fill_value != reference_values['fill_value']:
            msg = ('All input ImagePSF models must have the same value '
                   'for fill_value')
            raise ValueError(msg)

        if epsf.data.ndim != reference_values['data_ndim']:
            msg = ('All input ImagePSF models must have data with the '
                   'same dimensions')
            raise ValueError(msg)

        # Check data units
        current_unit = getattr(epsf.data, 'unit', None)
        if current_unit != reference_values['data_unit']:
            msg = 'All input data must have the same unit'
            raise ValueError(msg)

        if epsf.flux != reference_values['flux']:
            msg = ('All input ImagePSF models must have the same value '
                   'for flux')
            raise ValueError(msg)

        # Check origin consistency only if using x_0, y_0 from EPSFs
        if (grid_xypos is None
            and not np.array_equal(epsf.origin,
                                   reference_values['origin'])):
            msg = ('If using (x_0, y_0) as fiducial point, origin must '
                   'match for each input EPSF')
            raise ValueError(msg)


def grid_from_epsfs(epsfs, grid_xypos=None, meta=None):
    """
    Create a GriddedPSFModel from a list of ImagePSF models.

    Given a list of `~photutils.psf.ImagePSF` models, this function will
    return a `~photutils.psf.GriddedPSFModel`. The fiducial points for
    each input ImagePSF can either be set on each individual model by
    setting the 'x_0' and 'y_0' attributes, or provided as a list of
    tuples (``grid_xypos``). If a ``grid_xypos`` list is provided, it
    must match the length of input EPSFs. In either case, the fiducial
    points must be on a grid.

    Optionally, a ``meta`` dictionary may be provided for the output
    GriddedPSFModel. If this dictionary contains the keys 'grid_xypos',
    'oversampling', or 'fill_value', they will be overridden.

    Note: If set on the input ImagePSF (x_0, y_0), then ``origin``
    must be the same for each input EPSF. Additionally data units and
    dimensions must be for each input EPSF, and values for ``flux`` and
    ``oversampling``, and ``fill_value`` must match as well.

    Parameters
    ----------
    epsfs : list of `photutils.psf.ImagePSF`
        A list of ImagePSF models representing the individual PSFs.
        Must contain at least one EPSF.
    grid_xypos : list of tuples, optional
        A list of fiducial points (x_0, y_0) for each PSF. Each element
        should be a tuple of two numeric values. If not provided, the
        x_0 and y_0 of each input EPSF will be considered the fiducial
        point for that PSF. Default is None.
    meta : dict, optional
        Additional metadata for the GriddedPSFModel. Note that, if
        they exist in the supplied ``meta``, any values under the keys
        ``grid_xypos``, ``oversampling``, or ``fill_value`` will be
        overridden. Default is None.

    Returns
    -------
    GriddedPSFModel: `photutils.psf.GriddedPSFModel`
        The gridded PSF model created from the input EPSFs.

    Raises
    ------
    TypeError
        If any input EPSF is not of type ImagePSF.
    ValueError
        If EPSFs list is empty, grid_xypos length doesn't match EPSFs length,
        or if EPSFs have inconsistent properties.
    """
    # prevent circular imports
    from photutils.psf import GriddedPSFModel, ImagePSF

    # Input validation
    if not epsfs:
        msg = 'epsfs list cannot be empty'
        raise ValueError(msg)

    if grid_xypos is not None and len(grid_xypos) != len(epsfs):
        msg = 'grid_xypos must be the same length as epsfs'
        raise ValueError(msg)

    # Validate input types
    for i, epsf in enumerate(epsfs):
        if not isinstance(epsf, ImagePSF):
            msg = (f'All input epsfs must be of type ImagePSF, got '
                   f'{type(epsf).__name__} at index {i}')
            raise TypeError(msg)

    # Extract reference values from the first EPSF
    first_epsf = epsfs[0]
    reference_values = {
        'oversampling': first_epsf.oversampling,
        'fill_value': first_epsf.fill_value,
        'data_ndim': first_epsf.data.ndim,
        'data_unit': getattr(first_epsf.data, 'unit', None),
        'flux': first_epsf.flux,
        'origin': first_epsf.origin if grid_xypos is None else None,
    }

    # Validate consistency across all EPSFs
    _validate_epsf_consistency(epsfs, grid_xypos, reference_values)

    # Extract data arrays and positions
    data_arrs = [epsf.data for epsf in epsfs]

    if grid_xypos is None:
        # Extract positions from EPSFs' x_0, y_0 attributes
        grid_xypos = [(float(epsf.x_0.value), float(epsf.y_0.value))
                      for epsf in epsfs]

    # Create the data cube
    data_cube = np.stack(data_arrs, axis=0)

    # Prepare metadata
    meta = {} if meta is None else meta.copy()  # Avoid modifying input dict

    # Override required metadata keys
    meta['grid_xypos'] = grid_xypos
    meta['oversampling'] = reference_values['oversampling']
    meta['fill_value'] = reference_values['fill_value']

    data = NDData(data_cube, meta=meta)

    return GriddedPSFModel(data, fill_value=reference_values['fill_value'])
