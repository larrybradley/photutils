# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the GriddedPSFModel and related tools.
"""

import copy
import itertools
from functools import lru_cache

import astropy
import numpy as np
from astropy.io import registry
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData
from astropy.utils import minversion
from astropy.utils.decorators import lazyproperty
from astropy.visualization import simple_norm
from scipy.interpolate import RectBivariateSpline

from photutils.psf.model_io import (GriddedPSFModelRead, _get_metadata,
                                    _read_stdpsf, is_stdpsf, is_webbpsf,
                                    stdpsf_reader, webbpsf_reader)
from photutils.utils._parameters import as_pair

__all__ = ['GriddedPSFModel', 'ModelGridPlotMixin', 'STDPSFGrid']
__doctest_skip__ = ['STDPSFGrid']


class ModelGridPlotMixin:
    """
    Mixin class to plot a grid of ePSF models.
    """

    def _reshape_grid(self, data):
        """
        Reshape the 3D ePSF grid as a 2D array of horizontally and
        vertically stacked ePSFs.
        """
        nypsfs = self._ygrid.shape[0]
        nxpsfs = self._xgrid.shape[0]
        ny, nx = self.data.shape[1:]
        data.shape = (nypsfs, nxpsfs, ny, nx)

        return data.transpose([0, 2, 1, 3]).reshape(nypsfs * ny, nxpsfs * nx)

    def plot_grid(self, *, ax=None, vmax_scale=None, peak_norm=False,
                  deltas=False, cmap=None, dividers=True,
                  divider_color='darkgray', divider_ls='-', figsize=None):
        """
        Plot the grid of ePSF models.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        vmax_scale : float, optional
            Scale factor to apply to the image stretch limits. This
            value is multiplied by the peak ePSF value to determine the
            plotting ``vmax``. The defaults are 1.0 for plotting the
            ePSF data and 0.03 for plotting the ePSF difference data
            (``deltas=True``). If ``deltas=True``, the ``vmin`` is set
            to ``-vmax``. If ``deltas=False`` the ``vmin`` is set to
            ``vmax`` / 1e4.

        peak_norm : bool, optional
            Whether to normalize the ePSF data by the peak value. The
            default shows the ePSF flux per pixel.

        deltas : bool, optional
            Set to `True` to show the differences between each ePSF
            and the average ePSF.

        cmap : str or `matplotlib.colors.Colormap`, optional
            The colormap to use. The default is `None`, which uses
            the 'viridis' colormap for plotting ePSF data and the
            'gray_r' colormap for plotting the ePSF difference data
            (``deltas=True``).

        dividers : bool, optional
            Whether to show divider lines between the ePSFs.

        divider_color, divider_ls : str, optional
            Matplotlib color and linestyle options for the divider
            lines between ePSFs. These keywords have no effect unless
            ``show_dividers=True``.

        figsize : (float, float), optional
            The figure (width, height) in inches.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The matplotlib figure object. This will be the current
            figure if ``ax=None``. Use ``fig.savefig()`` to save the
            figure to a file.

            Note that when calling this method in a notebook, if you do
            not store the return value of this function, the figure will
            be displayed twice due to the REPL automatically displaying
            the return value of the last function call. Alternatively,
            you can append a semicolon to the end of the function call
            to suppress the display of the return value.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm

        data = self.data.copy()
        if deltas:
            # Compute mean ignoring any blank (all zeros) ePSFs.
            # This is the case for MIRI with its non-square FOV.
            mask = np.zeros(data.shape[0], dtype=bool)
            for i, arr in enumerate(data):
                if np.count_nonzero(arr) == 0:
                    mask[i] = True
            data -= np.mean(data[~mask], axis=0)
            data[mask] = 0.0

        data = self._reshape_grid(data)

        if ax is None:
            if figsize is None and self.meta.get('detector', '') == 'NRCSW':
                figsize = (20, 8)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = plt.gcf()

        if peak_norm and data.max() != 0:
            # normalize relative to peak
            data /= data.max()

        if deltas:
            if cmap is None:
                cmap = cm.gray_r.copy()

            if vmax_scale is None:
                vmax_scale = 0.03
            vmax = data.max() * vmax_scale
            vmin = -vmax
            if minversion(astropy, '6.1.dev'):
                norm = simple_norm(data, 'linear', vmin=vmin, vmax=vmax)
            else:
                norm = simple_norm(data, 'linear', min_cut=vmin, max_cut=vmax)
        else:
            if cmap is None:
                cmap = cm.viridis.copy()

            if vmax_scale is None:
                vmax_scale = 1.0
            vmax = data.max() * vmax_scale
            vmin = vmax / 1.0e4
            if minversion(astropy, '6.1.dev'):
                norm = simple_norm(data, 'log', vmin=vmin, vmax=vmax,
                                   log_a=1.0e4)
            else:
                norm = simple_norm(data, 'log', min_cut=vmin, max_cut=vmax,
                                   log_a=1.0e4)

        # Set up the coordinate axes to later set tick labels based on
        # detector ePSF coordinates. This sets up axes to have, behind the
        # scenes, the ePSFs centered at integer coords 0, 1, 2, 3 etc.
        # extent order: left, right, bottom, top
        nypsfs = self._ygrid.shape[0]
        nxpsfs = self._xgrid.shape[0]
        extent = [-0.5, nxpsfs - 0.5, -0.5, nypsfs - 0.5]

        ax.imshow(data, extent=extent, norm=norm, cmap=cmap, origin='lower')

        # Use the axes set up above to set appropriate tick labels
        xticklabels = self._xgrid.astype(int)
        yticklabels = self._ygrid.astype(int)
        if self.meta.get('detector', '') == 'NRCSW':
            xticklabels = list(xticklabels[0:5]) * 4
            yticklabels = list(yticklabels[0:5]) * 2
        ax.set_xticks(np.arange(nxpsfs))
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('ePSF location in detector X pixels')
        ax.set_yticks(np.arange(nypsfs))
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel('ePSF location in detector Y pixels')

        if dividers:
            for ix in range(nxpsfs):
                ax.axvline(ix + 0.5, color=divider_color, ls=divider_ls)
            for iy in range(nypsfs):
                ax.axhline(iy + 0.5, color=divider_color, ls=divider_ls)

        instrument = self.meta.get('instrument', '')
        if not instrument:
            # WebbPSF output
            instrument = self.meta.get('instrume', '')
        detector = self.meta.get('detector', '')
        filtername = self.meta.get('filter', '')

        # WebbPSF outputs a tuple with the comment in the second element
        if isinstance(instrument, (tuple, list, np.ndarray)):
            instrument = instrument[0]
        if isinstance(detector, (tuple, list, np.ndarray)):
            detector = detector[0]
        if isinstance(filtername, (tuple, list, np.ndarray)):
            filtername = filtername[0]

        title = f'{instrument} {detector} {filtername}'
        if title != '':
            # add extra space at end
            title += ' '

        if deltas:
            minus = '\u2212'
            ax.set_title(f'{title}(ePSFs {minus} <ePSF>)')
            if peak_norm:
                label = 'Difference relative to average ePSF peak'
            else:
                label = 'Difference relative to average ePSF values'
        else:
            ax.set_title(f'{title}ePSFs')
            if peak_norm:
                label = 'Scale relative to ePSF peak pixel'
            else:
                label = 'ePSF flux per pixel'

        cbar = plt.colorbar(label=label, mappable=ax.images[0])
        if not deltas:
            cbar.ax.set_yscale('log')

        if self.meta.get('detector', '') == 'NRCSW':
            # NIRCam NRCSW STDPSF files contain all detectors.
            # The plot gets extra divider lines and SCA name labels.
            nxpsfs = len(self._xgrid)
            nypsfs = len(self._ygrid)
            plt.axhline(nypsfs / 2 - 0.5, color='orange')
            for i in range(1, 4):
                ax.axvline(nxpsfs / 4 * i - 0.5, color='orange')

            det_labels = [['A1', 'A3', 'B4', 'B2'], ['A2', 'A4', 'B3', 'B1']]
            for i in range(2):
                for j in range(4):
                    ax.text(j * nxpsfs / 4 - 0.45,
                            (i + 1) * nypsfs / 2 - 0.55,
                            det_labels[i][j], color='orange',
                            verticalalignment='top', fontsize=12)

        fig.tight_layout()

        return fig


class GriddedPSFModel(ModelGridPlotMixin, Fittable2DModel):
    """
    A fittable 2D model containing a grid ePSF models.

    The ePSF models are defined at fiducial detector locations and are
    bilinearly interpolated to calculate an ePSF model at an arbitrary
    (x, y) detector position.

    When evaluating this model, it cannot be called with x and y arrays
    that have greater than 2 dimensions.

    Parameters
    ----------
    nddata : `~astropy.nddata.NDData`
        A `~astropy.nddata.NDData` object containing the grid of
        reference ePSF arrays. The data attribute must contain a 3D
        `~numpy.ndarray` containing a stack of the 2D ePSFs with a shape
        of ``(N_psf, ePSF_ny, ePSF_nx)``.

        The meta attribute must be dictionary containing the following:

            * ``'grid_xypos'``: A list of the (x, y) grid positions of
              each reference ePSF. The order of positions should match the
              first axis of the 3D `~numpy.ndarray` of ePSFs. In other
              words, ``grid_xypos[i]`` should be the (x, y) position of
              the reference ePSF defined in ``data[i]``.

            * ``'oversampling'``: The integer oversampling factor(s) of
              the ePSF. If ``oversampling`` is a scalar then it will be
              used for both axes. If ``oversampling`` has two elements,
              they must be in ``(y, x)`` order.

        The meta attribute may contain other properties such as the
        telescope, instrument, detector, and filter of the ePSF.

    flux : float, optional
        The flux scaling factor for the model. The default is 1.0.

    x_0, y_0 : float, optional
        The (x, y) position in the output coordinate grid where the model
        is evaluated. The default is (0, 0).

    fill_value : float, optional
        The value to use for points outside of the input pixel grid.
        The default is 0.0.

    Methods
    -------
    read(*args, **kwargs)
        Class method to create a `GriddedPSFModel` instance from a
        STDPSF FITS file. This method uses :func:`stdpsf_reader` with
        the provided parameters.

    Notes
    -----
    Internally, the grid of ePSFs will be arranged and stored such that
    it is sorted first by y and then by x.
    """

    flux = Parameter(description='Intensity scaling factor for the ePSF '
                     'model.', default=1.0)
    x_0 = Parameter(description='x position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    y_0 = Parameter(description='y position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)

    read = registry.UnifiedReadWriteMethod(GriddedPSFModelRead)

    def __init__(self, nddata, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, fill_value=0.0):

        self._validate_data(nddata)
        self.data, self.grid_xypos = self._define_grid(nddata)
        # use _meta to avoid the meta descriptor
        self._meta = nddata.meta.copy()
        self.oversampling = as_pair('oversampling',
                                    nddata.meta['oversampling'],
                                    lower_bound=(0, 1))

        self.fill_value = fill_value

        self._grid_xpos, self._grid_ypos = np.transpose(self.grid_xypos)
        self._xgrid = np.unique(self._grid_xpos)  # also sorts values
        self._ygrid = np.unique(self._grid_ypos)  # also sorts values
        self.meta['grid_shape'] = (len(self._ygrid), len(self._xgrid))
        if (len(list(itertools.product(self._xgrid, self._ygrid)))
                != len(self.grid_xypos)):
            raise ValueError('"grid_xypos" must form a regular grid.')

        self._xidx = np.arange(self.data.shape[2], dtype=float)
        self._yidx = np.arange(self.data.shape[1], dtype=float)

        # Here we avoid decorating the instance method with @lru_cache
        # to prevent memory leaks; we set maxsize=128 to prevent the
        # cache from growing too large.
        self._calc_interpolator = lru_cache(maxsize=128)(
            self._calc_interpolator_uncached)

        super().__init__(flux, x_0, y_0)

    @staticmethod
    def _validate_data(data):
        if not isinstance(data, NDData):
            raise TypeError('data must be an NDData instance.')

        if data.data.ndim != 3:
            raise ValueError('The NDData data attribute must be a 3D numpy '
                             'ndarray')

        if not np.all(np.isfinite(data.data)):
            raise ValueError('All elements of input data must be finite.')

        # this is required by RectBivariateSpline for kx=3, ky=3
        if np.any(np.array(data.data.shape[1:]) < 4):
            raise ValueError('The length of the PSF x and y axes must both '
                             'be at least 4.')

        if 'grid_xypos' not in data.meta:
            raise ValueError('"grid_xypos" must be in the nddata meta '
                             'dictionary.')
        if len(data.meta['grid_xypos']) != data.data.shape[0]:
            raise ValueError('The length of grid_xypos must match the number '
                             'of input ePSFs.')

        if 'oversampling' not in data.meta:
            raise ValueError('"oversampling" must be in the nddata meta '
                             'dictionary.')

    def _define_grid(self, nddata):
        """
        Sort the input ePSF data into a regular grid where the ePSFs are
        sorted first by y and then by x.

        Parameters
        ----------
        nddata : `~astropy.nddata.NDData`
            The input NDData object containing the ePSF data.

        Returns
        -------
        data : 3D `~numpy.ndarray`
            The 3D array of ePSFs.
        grid_xypos : array of (x, y) pairs
            The (x, y) positions of the ePSFs, sorted first by y and
            then by x.
        """
        grid_xypos = np.array(nddata.meta['grid_xypos'])
        # sort by y and then by x
        idx = np.lexsort((grid_xypos[:, 0], grid_xypos[:, 1]))
        grid_xypos = grid_xypos[idx]
        data = nddata.data[idx]

        return data, grid_xypos

    def _cls_info(self):
        cls_info = []

        keys = ('STDPSF', 'instrument', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', tuple(self.oversampling))])
        return cls_info

    def __str__(self):
        return self._format_str(keywords=self._cls_info())

    def copy(self):
        """
        Return a copy of this model where only the model parameters are
        copied.

        All other copied model attributes are references to the
        original model. This prevents copying the ePSF grid data, which
        may contain a large array.

        This method is useful if one is interested in only changing
        the model parameters in a model copy. It is used in the PSF
        photometry classes during model fitting.

        Use the `deepcopy` method if you want to copy all of the model
        attributes, including the ePSF grid data.

        Returns
        -------
        result : `GriddedPSFModel`
            A copy of this model with only the model parameters copied.
        """
        newcls = object.__new__(self.__class__)

        for key, val in self.__dict__.items():
            if key in self.param_names:  # copy only the parameter values
                newcls.__dict__[key] = copy.copy(val)
            else:
                newcls.__dict__[key] = val

        return newcls

    def deepcopy(self):
        """
        Return a deep copy of this model.

        Returns
        -------
        result : `GriddedPSFModel`
            A deep copy of this model.
        """
        return copy.deepcopy(self)

    def clear_cache(self):
        """
        Clear the internal cache.
        """
        self._calc_interpolator.cache_clear()

    def _cache_info(self):
        """
        Return information about the internal cache.
        """
        return self._calc_interpolator.cache_info()

    def bounding_box(self):
        """
        Return a bounding box defining the limits of the model.

        Returns
        -------
        bounding_box : `astropy.modeling.bounding_box.ModelBoundingBox`
            A bounding box defining the limits of the model.

        Examples
        --------
        >>> from itertools import product
        >>> import numpy as np
        >>> from astropy.nddata import NDData
        >>> from photutils.psf import GaussianPSF, GriddedPSFModel
        >>> psfs = []
        >>> yy, xx = np.mgrid[0:101, 0:101]
        >>> for i in range(16):
        ...     theta = np.deg2rad(i * 10.0)
        ...     gmodel = GaussianPSF(flux=1, x_0=50, y_0=50, x_fwhm=10,
        ...                          y_fwhm=5, theta=theta)
        ...     psfs.append(gmodel(xx, yy))
        >>> xgrid = [0, 40, 160, 200]
        >>> ygrid = [0, 60, 140, 200]
        >>> meta = {}
        >>> meta['grid_xypos'] = list(product(xgrid, ygrid))
        >>> meta['oversampling'] = 4
        >>> nddata = NDData(psfs, meta=meta)
        >>> model = GriddedPSFModel(nddata, flux=1, x_0=0, y_0=0)
        >>> model.bounding_box  # doctest: +FLOAT_CMP
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-12.625, upper=12.625)
                y: Interval(lower=-12.625, upper=12.625)
            }
            model=GriddedPSFModel(inputs=('x', 'y'))
            order='C'
        )
        """
        dy, dx = np.array(self.data.shape[1:]) / 2 / self.oversampling
        return ((self.y_0 - dy, self.y_0 + dy),
                (self.x_0 - dx, self.x_0 + dx))

    @staticmethod
    def _find_start_idx(data, x):
        """
        Find the index of the lower bound where ``x`` should be inserted
        into ``a`` to maintain order.

        The index of the upper bound is the index of the lower bound
        plus 2. Both bound indices must be within the array.

        Parameters
        ----------
        data : 1D `~numpy.ndarray`
            The 1D array to search.

        x : float
            The value to insert.

        Returns
        -------
        index : int
            The index of the lower bound.
        """
        idx = np.searchsorted(data, x)
        if idx == 0:
            idx0 = 0
        elif idx == len(data):  # pragma: no cover
            idx0 = idx - 2
        else:
            idx0 = idx - 1
        return idx0

    def _find_bounding_points(self, x, y):
        """
        Find the indices of the grid points that bound the input ``(x,
        y)`` position.

        Parameters
        ----------
        x, y : float
            The ``(x, y)`` position where the ePSF is to be evaluated.
            The position must be inside the region defined by the grid
            of ePSF positions.

        Returns
        -------
        indices : list of int
            A list of indices of the bounding grid points.
        """
        x0 = self._find_start_idx(self._xgrid, x)
        y0 = self._find_start_idx(self._ygrid, y)
        xypoints = list(itertools.product(self._xgrid[x0:x0 + 2],
                                          self._ygrid[y0:y0 + 2]))

        # find the grid_xypos indices of the reference xypoints
        indices = []
        for xx, yy in xypoints:
            indices.append(np.argsort(np.hypot(self._grid_xpos - xx,
                                               self._grid_ypos - yy))[0])

        return indices

    @staticmethod
    def _bilinear_interp(xyref, zref, xi, yi):
        """
        Perform bilinear interpolation of four 2D arrays located at
        points on a regular grid.

        Parameters
        ----------
        xyref : list of 4 (x, y) pairs
            A list of 4 ``(x, y)`` pairs that form a rectangle.

        zref : 3D `~numpy.ndarray`
            A 3D `~numpy.ndarray` of shape ``(4, nx, ny)``. The first
            axis corresponds to ``xyref``, i.e., ``refdata[0, :, :]`` is
            the 2D array located at ``xyref[0]``.

        xi, yi : float
            The ``(xi, yi)`` point at which to perform the
            interpolation. The ``(xi, yi)`` point must lie within the
            rectangle defined by ``xyref``.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The 2D interpolated array.
        """
        xyref = [tuple(i) for i in xyref]
        idx = sorted(range(len(xyref)), key=xyref.__getitem__)
        xyref = sorted(xyref)  # sort by x, then y
        (x0, y0), (_x0, y1), (x1, _y0), (_x1, _y1) = xyref

        if x0 != _x0 or x1 != _x1 or y0 != _y0 or y1 != _y1:
            raise ValueError('The refxy points do not form a rectangle.')

        if not np.isscalar(xi):
            xi = xi[0]
        if not np.isscalar(yi):
            yi = yi[0]

        if not x0 <= xi <= x1 or not y0 <= yi <= y1:
            raise ValueError('The (x, y) input is not within the rectangle '
                             'defined by xyref.')

        data = np.asarray(zref)[idx]
        weights = np.array([(x1 - xi) * (y1 - yi), (x1 - xi) * (yi - y0),
                            (xi - x0) * (y1 - yi), (xi - x0) * (yi - y0)])
        norm = (x1 - x0) * (y1 - y0)

        return np.sum(data * weights[:, None, None], axis=0) / norm

    def _calc_interpolator_uncached(self, x_0, y_0):
        """
        Return the local interpolation function for the ePSF model at
        (x_0, y_0).

        Note that the interpolator will be cached by _calc_interpolator.
        It can be cleared by calling the clear_cache method.
        """
        if (x_0 < self._xgrid[0] or x_0 > self._xgrid[-1]
                or y_0 < self._ygrid[0] or y_0 > self._ygrid[-1]):
            # position is outside of the grid, so simply use the
            # closest reference ePSF
            ref_index = np.argsort(np.hypot(self._grid_xpos - x_0,
                                            self._grid_ypos - y_0))[0]
            psf_image = self.data[ref_index, :, :]
        else:
            # find the four bounding reference ePSFs and interpolate
            ref_indices = self._find_bounding_points(x_0, y_0)
            xyref = self.grid_xypos[ref_indices]
            psfs = self.data[ref_indices, :, :]

            psf_image = self._bilinear_interp(xyref, psfs, x_0, y_0)

        return RectBivariateSpline(self._xidx, self._yidx, psf_image.T,
                                   kx=3, ky=3, s=0)

    @lazyproperty
    def origin(self):
        """
        A 1D `~numpy.ndarray` (x, y) pixel coordinates within the
        model's 2D image of the origin of the coordinate system.
        """
        xyorigin = (np.array(self.data.shape) - 1) / 2
        return xyorigin[::-1]

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the `GriddedPSFModel` for the input parameters.

        Parameters
        ----------
        x, y : float or `~numpy.ndarray`
            The x and y positions at which to evaluate the model.

        flux : float
            The flux scaling factor for the model.

        x_0, y_0 : float
            The (x, y) position of the model.

        Returns
        -------
        evaluated_model : `~numpy.ndarray`
            The evaluated model.
        """
        if x.ndim > 2:
            raise ValueError('x and y must be 1D or 2D.')

        # NOTE: the astropy base Model.__call__() method converts scalar
        # inputs to size-1 arrays before calling evaluate().
        if not np.isscalar(flux):
            flux = flux[0]
        if not np.isscalar(x_0):
            x_0 = x_0[0]
        if not np.isscalar(y_0):
            y_0 = y_0[0]

        # Calculate the local interpolation function for the ePSF at
        # (x_0, y_0). Only the integer part of the position is input in
        # order to have effective caching.
        interpolator = self._calc_interpolator(int(x_0), int(y_0))

        # now evaluate the ePSF at the (x_0, y_0) subpixel position on
        # the input (x, y) values
        xi = self.oversampling[1] * (np.asarray(x, dtype=float) - x_0)
        yi = self.oversampling[0] * (np.asarray(y, dtype=float) - y_0)
        xi += self.origin[0]
        yi += self.origin[1]

        evaluated_model = flux * interpolator(xi, yi, grid=False)

        if self.fill_value is not None:
            # set pixels that are outside the input pixel grid to the
            # fill_value to avoid extrapolation; these bounds match the
            # RegularGridInterpolator bounds
            ny, nx = self.data.shape[1:]
            invalid = (xi < 0) | (xi > nx - 1) | (yi < 0) | (yi > ny - 1)
            evaluated_model[invalid] = self.fill_value

        return evaluated_model


class STDPSFGrid(ModelGridPlotMixin):
    """
    Class to read and plot "STDPSF" format ePSF model grids.

    STDPSF files are FITS files that contain a 3D array of ePSFs with
    the header detailing where the fiducial ePSFs are located in the
    detector coordinate frame.

    The oversampling factor for STDPSF FITS files is assumed to be 4.

    Parameters
    ----------
    filename : str
        The name of the STDPDF FITS file. A URL can also be used.

    Examples
    --------
    >>> psfgrid = STDPSFGrid.read('STDPSF_ACSWFC_F814W.fits')
    >>> psfgrid.plot_grid()
    """

    def __init__(self, filename):
        grid_data = _read_stdpsf(filename)
        self.data = grid_data['data']
        self._xgrid = grid_data['xgrid']
        self._ygrid = grid_data['ygrid']
        xy_grid = [yx[::-1] for yx in itertools.product(self._ygrid,
                                                        self._xgrid)]
        oversampling = 4  # assumption for STDPSF files
        self.grid_xypos = xy_grid
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        meta = {'grid_shape': (len(self._ygrid), len(self._xgrid)),
                'grid_xypos': xy_grid,
                'oversampling': oversampling}

        # try to get additional metadata from the filename because this
        # information is not currently available in the FITS headers
        file_meta = _get_metadata(filename, None)
        if file_meta is not None:
            meta.update(file_meta)

        self.meta = meta

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        cls_info = []

        keys = ('STDPSF', 'detector', 'filter', 'grid_shape')
        for key in keys:
            if key in self.meta:
                name = key.capitalize() if key != 'STDPSF' else key
                cls_info.append((name, self.meta[key]))

        cls_info.extend([('Number of PSFs', len(self.grid_xypos)),
                         ('PSF shape (oversampled pixels)',
                          self.data.shape[1:]),
                         ('Oversampling', self.oversampling)])

        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'{key}: {val}' for key, val in cls_info]

        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()


with registry.delay_doc_updates(GriddedPSFModel):
    registry.register_reader('stdpsf', GriddedPSFModel, stdpsf_reader)
    registry.register_identifier('stdpsf', GriddedPSFModel, is_stdpsf)
    registry.register_reader('webbpsf', GriddedPSFModel, webbpsf_reader)
    registry.register_identifier('webbpsf', GriddedPSFModel, is_webbpsf)
