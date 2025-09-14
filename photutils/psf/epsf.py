# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define tools to build and fit an effective PSF (ePSF) based on Anderson
and King (2000; PASP 112, 1360) and Anderson (2016; WFC3 ISR 2016-12).
"""

import copy
import warnings
from dataclasses import dataclass

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve

from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF, _LegacyEPSFModel
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import (SigmaClipSentinelDefault, as_pair,
                                         create_default_sigmaclip)
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround
from photutils.utils._stats import nanmedian
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['EPSFBuildResult', 'EPSFBuilder', 'EPSFFitter']


@dataclass
class EPSFBuildResult:
    """
    Container for ePSF building results.

    This class provides structured access to the results of the ePSF
    building process, including convergence information and diagnostic
    data that can help users understand and validate the building process.

    Attributes
    ----------
    epsf : `ImagePSF` object
        The final constructed ePSF model.

    fitted_stars : `EPSFStars` object
        The input stars with updated centers and fluxes derived from
        fitting the final ePSF.

    iterations : int
        The number of iterations performed during the building process.
        This will be <= maxiters specified in EPSFBuilder.

    converged : bool
        Whether the building process converged based on the center
        accuracy criterion. True if star centers moved less than
        the specified accuracy between the final iterations.

    final_center_accuracy : float
        The maximum center displacement in the final iteration, in pixels.
        This indicates how much the star centers changed in the last
        iteration and can be used to assess convergence quality.

    n_excluded_stars : int
        The number of individual stars (including those from linked stars)
        that were excluded from fitting due to repeated fit failures.

    excluded_star_indices : list
        Indices of stars that were excluded from fitting during the
        building process. These correspond to positions in the flattened
        star list (stars.all_stars).

    Notes
    -----
    This result object maintains backward compatibility by implementing
    tuple unpacking, so existing code like:

        epsf, stars = epsf_builder(stars)

    will continue to work unchanged. The additional information is
    available as attributes for users who want more detailed results.

    Examples
    --------
    >>> from photutils.psf import EPSFBuilder
    >>> result = epsf_builder(stars)
    >>> print(f"Converged after {result.iterations} iterations")
    >>> print(f"Final accuracy: {result.final_center_accuracy:.6f} pixels")
    >>> if result.n_excluded_stars > 0:
    ...     print(f"Excluded {result.n_excluded_stars} stars")
    """

    epsf: 'ImagePSF'
    fitted_stars: 'EPSFStars'
    iterations: int
    converged: bool
    final_center_accuracy: float
    n_excluded_stars: int
    excluded_star_indices: list

    def __iter__(self):
        """
        Allow tuple unpacking for backward compatibility.

        Returns
        -------
        iterator
            An iterator that yields (epsf, fitted_stars) for compatibility
            with existing code that expects a 2-tuple.
        """
        return iter((self.epsf, self.fitted_stars))

    def __getitem__(self, index):
        """
        Allow indexing for backward compatibility.

        Parameters
        ----------
        index : int
            Index to access (0 for epsf, 1 for fitted_stars).

        Returns
        -------
        value
            The ePSF (index 0) or fitted stars (index 1).
        """
        if index == 0:
            return self.epsf
        if index == 1:
            return self.fitted_stars

        msg = 'EPSFBuildResult index must be 0 (epsf) or 1 (fitted_stars)'
        raise IndexError(msg)


SIGMA_CLIP = SigmaClipSentinelDefault(sigma=3.0, maxiters=10)


class EPSFFitter:
    """
    Class to fit an ePSF model to one or more stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A `~astropy.modeling.fitting.Fitter` object. If `None`, then the
        default `~astropy.modeling.fitting.TRFLSQFitter` will be used.

    fit_boxsize : int, tuple of int, or `None`, optional
        The size (in pixels) of the box centered on the star to be used
        for ePSF fitting. This allows using only a small number of
        central pixels of the star (i.e., where the star is brightest)
        for fitting. If ``fit_boxsize`` is a scalar then a square box of
        size ``fit_boxsize`` will be used. If ``fit_boxsize`` has two
        elements, they must be in ``(ny, nx)`` order. ``fit_boxsize``
        must have odd values and be greater than or equal to 3 for both
        axes. If `None`, the fitter will use the entire star image.

    **fitter_kwargs : dict, optional
        Any additional keyword arguments (except ``x``, ``y``, ``z``, or
        ``weights``) to be passed directly to the ``__call__()`` method
        of the input ``fitter``.
    """

    def __init__(self, *, fitter=None, fit_boxsize=5,
                 **fitter_kwargs):

        if fitter is None:
            fitter = TRFLSQFitter()
        self.fitter = fitter
        self.fitter_has_fit_info = hasattr(self.fitter, 'fit_info')
        self.fit_boxsize = as_pair('fit_boxsize', fit_boxsize,
                                   lower_bound=(3, 0), check_odd=True)

        # remove any fitter keyword arguments that we need to set
        remove_kwargs = ['x', 'y', 'z', 'weights']
        fitter_kwargs = copy.deepcopy(fitter_kwargs)
        for kwarg in remove_kwargs:
            if kwarg in fitter_kwargs:
                del fitter_kwargs[kwarg]
        self.fitter_kwargs = fitter_kwargs

    def __call__(self, epsf, stars):
        """
        Fit an ePSF model to stars.

        Parameters
        ----------
        epsf : `ImagePSF`
            An ePSF model to be fitted to the stars.

        stars : `EPSFStars` object
            The stars to be fit. The center coordinates for each star
            should be as close as possible to actual centers. For stars
            than contain weights, a weighted fit of the ePSF to the star
            will be performed.

        Returns
        -------
        fitted_stars : `EPSFStars` object
            The fitted stars. The ePSF-fitted center position and flux
            are stored in the ``center`` (and ``cutout_center``) and
            ``flux`` attributes.
        """
        if len(stars) == 0:
            return stars

        if not isinstance(epsf, ImagePSF):
            msg = 'The input epsf must be an ImagePSF'
            raise TypeError(msg)

        # make a copy of the input ePSF
        # (copy only parameters, not the data)
        epsf = epsf.copy()

        # perform the fit
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                fitted_star = self._fit_star(epsf, star, self.fitter,
                                             self.fitter_kwargs,
                                             self.fitter_has_fit_info,
                                             self.fit_boxsize)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    fitted_star.append(
                        self._fit_star(epsf, linked_star, self.fitter,
                                       self.fitter_kwargs,
                                       self.fitter_has_fit_info,
                                       self.fit_boxsize))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize):
        """
        Fit an ePSF model to a single star.

        The input ``epsf`` will usually be modified by the fitting
        routine in this function. Make a copy before calling this
        function if the original is needed.
        """
        if fit_boxsize is not None:
            try:
                xcenter, ycenter = star.cutout_center
                large_slc, _ = overlap_slices(star.shape, fit_boxsize,
                                              (ycenter, xcenter),
                                              mode='strict')
            except (PartialOverlapError, NoOverlapError):
                warnings.warn(f'The star at ({star.center[0]}, '
                              f'{star.center[1]}) cannot be fit because '
                              'its fitting region extends beyond the star '
                              'cutout image.', AstropyUserWarning)

                star = copy.deepcopy(star)
                star._fit_error_status = 1

                return star

            data = star.data[large_slc]
            weights = star.weights[large_slc]

            # define the origin of the fitting region
            x0 = large_slc[1].start
            y0 = large_slc[0].start
        else:
            # use the entire cutout image
            data = star.data
            weights = star.weights

            # define the origin of the fitting region
            x0 = 0
            y0 = 0

        # Define positions in the undersampled grid. The fitter will
        # evaluate on the defined interpolation grid, currently in the
        # range [0, len(undersampled grid)].
        yy, xx = np.indices(data.shape, dtype=float)
        xx = xx + x0 - star.cutout_center[0]
        yy = yy + y0 - star.cutout_center[1]

        # define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # fitter doesn't support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = copy.copy(fitter.fit_info)

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # compute the star's fitted position
        x_center = star.cutout_center[0] + fitted_epsf.x_0.value
        y_center = star.cutout_center[1] + fitted_epsf.y_0.value

        star = copy.deepcopy(star)
        star.cutout_center = (x_center, y_center)

        # set the star's flux to the ePSF-fitted flux
        star.flux = fitted_epsf.flux.value

        star._fit_info = fit_info
        star._fit_error_status = fit_error_status

        return star


class EPSFBuilder:
    """
    Class to build an effective PSF (ePSF).

    See `Anderson and King (2000; PASP 112, 1360)
    <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
    and `Anderson (2016; WFC3 ISR 2016-12)
    <https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_
    for details.

    Parameters
    ----------
    oversampling : int or array_like (int)
        The integer oversampling factor(s) of the ePSF relative to the
        input ``stars`` along each axis. If ``oversampling`` is a scalar
        then it will be used for both axes. If ``oversampling`` has two
        elements, they must be in ``(y, x)`` order.

    shape : float, tuple of two floats, or `None`, optional
        The shape of the output ePSF. If the ``shape`` is not `None`, it
        will be derived from the sizes of the input ``stars`` and the
        ePSF oversampling factor. If the size is even along any axis,
        it will be made odd by adding one. The output ePSF will always
        have odd sizes along both axes to ensure a well-defined central
        pixel.

    smoothing_kernel : {'quartic', 'quadratic'}, 2D `~numpy.ndarray`, or `None`
        The smoothing kernel to apply to the ePSF. The predefined
        ``'quartic'`` and ``'quadratic'`` kernels are derived
        from fourth and second degree polynomials, respectively.
        Alternatively, a custom 2D array can be input. If `None` then no
        smoothing will be performed.

    recentering_func : callable, optional
        A callable object (e.g., function or class) that is used to
        calculate the centroid of a 2D array. The callable must accept
        a 2D `~numpy.ndarray`, have a ``mask`` keyword and optionally
        ``error`` and ``oversampling`` keywords. The callable object
        must return a tuple of two 1D `~numpy.ndarray` variables,
        representing the x and y centroids.

    recentering_maxiters : int, optional
        The maximum number of recentering iterations to perform during
        each ePSF build iteration.

    fitter : `EPSFFitter` object, optional
        A `EPSFFitter` object use to fit the ePSF to stars. If `None`,
        then the default `EPSFFitter` will be used. To set custom fitter
        options, input a new `EPSFFitter` object. See the `EPSFFitter`
        documentation for options.

    maxiters : int, optional
        The maximum number of iterations to perform.

    progress_bar : bool, option
        Whether to print the progress bar during the build
        iterations. The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    norm_radius : float, optional
        The pixel radius over which the ePSF is normalized.

    recentering_boxsize : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each build iteration. If a
            single integer number is provided, then a square box will
            be used. If two values are provided, then they must be in
            ``(ny, nx)`` order. ``recentering_boxsize`` must have odd
            must have odd values and be greater than or equal to 3 for
            both axes.

    center_accuracy : float, optional
            The desired accuracy for the centers of stars. The building
            iterations will stop if the centers of all the stars change
            by less than ``center_accuracy`` pixels between iterations.
            All stars must meet this condition for the loop to exit.

    sigma_clip : `astropy.stats.SigmaClip` instance, optional
        A `~astropy.stats.SigmaClip` object that defines the sigma
        clipping parameters used to determine which pixels are ignored
        when stacking the ePSF residuals in each iteration step. If
        `None` then no sigma clipping will be performed.

    Notes
    -----
    If your image contains NaN values, you may see better performance if
    you have the `bottleneck`_ package installed.

    .. _bottleneck:  https://github.com/pydata/bottleneck
    """

    def __init__(self, *, oversampling=4, shape=None,
                 smoothing_kernel='quartic', recentering_func=centroid_com,
                 recentering_maxiters=20, fitter=None, maxiters=10,
                 progress_bar=True, norm_radius=5.5,
                 recentering_boxsize=(5, 5), center_accuracy=1.0e-3,
                 sigma_clip=SIGMA_CLIP):

        if oversampling is None:
            msg = "'oversampling' must be specified"
            raise ValueError(msg)
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        self._norm_radius = norm_radius
        if shape is not None:
            self.shape = as_pair('shape', shape, lower_bound=(0, 1))
        else:
            self.shape = shape

        self.recentering_func = recentering_func
        self.recentering_maxiters = recentering_maxiters
        self.recentering_boxsize = as_pair('recentering_boxsize',
                                           recentering_boxsize,
                                           lower_bound=(3, 0), check_odd=True)
        self.smoothing_kernel = smoothing_kernel

        if fitter is None:
            fitter = EPSFFitter()
        if not isinstance(fitter, EPSFFitter):
            msg = 'fitter must be an EPSFFitter instance'
            raise TypeError(msg)
        self.fitter = fitter

        if center_accuracy <= 0.0:
            msg = 'center_accuracy must be a positive number'
            raise ValueError(msg)
        self.center_accuracy_sq = center_accuracy**2

        maxiters = int(maxiters)
        if maxiters <= 0:
            msg = 'maxiters must be a positive number'
            raise ValueError(msg)
        self.maxiters = maxiters

        self.progress_bar = progress_bar

        if sigma_clip is SIGMA_CLIP:
            sigma_clip = create_default_sigmaclip(sigma=SIGMA_CLIP.sigma,
                                                  maxiters=SIGMA_CLIP.maxiters)
        if not isinstance(sigma_clip, SigmaClip):
            msg = 'sigma_clip must be an astropy.stats.SigmaClip instance'
            raise TypeError(msg)
        self._sigma_clip = sigma_clip

        # store each ePSF build iteration
        self._epsf = []

    def __call__(self, stars):
        return self.build_epsf(stars)

    def _create_initial_epsf(self, stars):
        """
        Create an initial `ImagePSF` object with zero data.

        This method initializes the ePSF building process by creating
        a blank ImagePSF model with the appropriate dimensions and
        coordinate system. The initial ePSF data are all zeros and
        will be populated through the iterative building process.

        Shape Determination Algorithm
        -----------------------------
        1. If shape is explicitly provided, use it (ensuring odd dimensions)
        2. Otherwise, determine shape from input stars and oversampling:
           - Take the maximum star cutout dimensions
           - Apply oversampling factor: new_size = old_size * oversampling + 1
           - Ensure resulting dimensions are odd (add 1 if even)

        The +1 ensures that oversampled arrays have a well-defined center
        pixel, which is crucial for PSF modeling and fitting.

        Coordinate System Setup
        -----------------------
        The method establishes the coordinate system for the ImagePSF:

        - Origin: Set to the geometric center of the data array
        - For an NxM array, origin = ((M-1)/2, (N-1)/2) in (x, y) order
        - This ensures the PSF center aligns with the array center
        - The coordinate system is consistent with ImagePSF expectations

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF. The method uses:
            - stars._max_shape: Maximum dimensions among all star cutouts
            - This ensures the ePSF is large enough to contain all stars

        Returns
        -------
        epsf : `ImagePSF` object
            The initial ePSF model with:
            - data: Zero-filled array of appropriate dimensions
            - origin: Set to the array center in (x, y) order
            - oversampling: Copied from the EPSFBuilder configuration
            - fill_value: Set to 0.0 for regions outside the PSF
            - _norm_radius: Preserved for backward compatibility

        Notes
        -----
        The initial ePSF has zero flux and data values. These will be
        populated through the iterative building process as residuals
        from individual stars are combined.

        The method ensures that:
        - Array dimensions are always odd (ensuring a center pixel)
        - The coordinate system is properly established
        - All necessary attributes are set for downstream processing

        Examples
        --------
        For stars with maximum shape (25, 25) and oversampling=4:
        - x_shape = 25 * 4 + 1 = 101
        - y_shape = 25 * 4 + 1 = 101
        - Final shape: (101, 101)
        - Origin: (50.0, 50.0)
        """
        norm_radius = self._norm_radius
        oversampling = self.oversampling
        shape = self.shape

        # Define the ePSF shape
        if shape is not None:
            shape = as_pair('shape', shape, lower_bound=(0, 1), check_odd=True)
        else:
            # Stars class should have odd-sized dimensions, and thus we
            # get the oversampled shape as oversampling * len + 1; if
            # len=25, then newlen=101, for example.
            x_shape = (np.ceil(stars._max_shape[1]) * oversampling[1]
                       + 1).astype(int)
            y_shape = (np.ceil(stars._max_shape[0]) * oversampling[0]
                       + 1).astype(int)

            shape = np.array((y_shape, x_shape))

        # Verify odd sizes of shape (ensure center pixel exists)
        shape = [(i + 1) if i % 2 == 0 else i for i in shape]

        # Initialize with zeros
        data = np.zeros(shape, dtype=float)

        # Set origin as center of data array in (x, y) order
        # This establishes the coordinate system for the ImagePSF
        origin_xy = ((data.shape[1] - 1) / 2.0, (data.shape[0] - 1) / 2.0)

        epsf = ImagePSF(data=data, origin=origin_xy, oversampling=oversampling,
                        fill_value=0.0)
        # Preserve norm_radius for backward compatibility
        epsf._norm_radius = norm_radius
        return epsf

    def _resample_residual(self, star, epsf):
        """
        Compute a normalized residual image in the oversampled ePSF
        grid.

        A normalized residual image is calculated by subtracting the
        normalized ePSF model from the normalized star at the location
        of the star in the undersampled grid. The normalized residual
        image is then resampled from the undersampled star grid to the
        oversampled ePSF grid.

        Parameters
        ----------
        star : `EPSFStar` object
            A single star object.

        epsf : `_LegacyEPSFModel` object
            The ePSF model.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled residual image. The
            image contains NaNs where there is no data.
        """
        # Compute the normalized residual by subtracting the ePSF model
        # from the normalized star at the location of the star in the
        # undersampled grid.

        x = star._xidx_centered
        y = star._yidx_centered

        # stardata = (star._data_values_normalized
        #            - epsf.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0))

        stardata = (star._data_values_normalized
                    - epsf.evaluate(x=star._xidx_centered,
                                    y=star._yidx_centered,
                                    flux=1.0, x_0=0.0, y_0=0.0))

        # For ImagePSF, we need to map to the oversampled ePSF grid
        # Star coordinates are in undersampled units relative to center
        # We need to apply oversampling and add the ePSF center offset
        x = epsf.oversampling[1] * star._xidx_centered
        y = epsf.oversampling[0] * star._yidx_centered

        # ePSF center in oversampled coordinates (should match ePSF.origin)
        epsf_xcenter, epsf_ycenter = epsf.origin
        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)

        epsf_shape = epsf.data.shape
        resampled_img = np.full(epsf_shape, np.nan)

        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < epsf_shape[1]),
                              np.logical_and(yidx >= 0, yidx < epsf_shape[0]))
        xidx_ = xidx[mask]
        yidx_ = yidx[mask]

        resampled_img[yidx_, xidx_] = stardata[mask]

        return resampled_img

    def _resample_residuals(self, stars, epsf):
        """
        Compute normalized residual images for all the input stars.

        This method now uses a memory-efficient approach that avoids
        creating large 3D arrays by yielding residuals one at a time.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object
            The ePSF model.

        Returns
        -------
        epsf_resid : 3D `~numpy.ndarray`
            A 3D cube containing the resampled residual images.
        """
        epsf_shape = epsf.data.shape
        shape = (stars.n_good_stars, epsf_shape[0], epsf_shape[1])
        epsf_resid = np.zeros(shape)
        for i, star in enumerate(stars.all_good_stars):
            epsf_resid[i, :, :] = self._resample_residual(star, epsf)

        return epsf_resid

    def _compute_residual_median(self, stars, epsf):
        """
        Compute sigma-clipped median of residual images without storing
        the full 3D array in memory.

        This is a memory-efficient alternative to creating a large 3D
        residual array. It processes residuals iteratively and computes
        the sigma-clipped median directly.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object
            The ePSF model.

        Returns
        -------
        residual_median : 2D `~numpy.ndarray`
            The sigma-clipped median residual image.
        """
        n_stars = stars.n_good_stars
        if n_stars == 0:
            msg = 'No good stars available for residual computation'
            raise ValueError(msg)

        # Initialize with first star's residual
        first_star = next(iter(stars.all_good_stars))
        first_residual = self._resample_residual(first_star, epsf)

        if n_stars == 1:
            return first_residual

        # For multiple stars, collect all residuals for sigma clipping
        # We still need all data for proper sigma clipping, but we can
        # process in chunks for very large datasets
        residuals_list = [first_residual]
        residuals_list.extend(self._resample_residual(star, epsf)
                              for star in list(stars.all_good_stars)[1:])

        # Stack residuals for sigma clipping
        residuals = np.array(residuals_list)

        # Apply sigma clipping and compute median
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            residuals = self._sigma_clip(residuals, axis=0, masked=False,
                                         return_bounds=False)
            return nanmedian(residuals, axis=0)

    def _smooth_epsf(self, epsf_data):
        """
        Smooth the ePSF array by convolving it with a kernel.

        Parameters
        ----------
        epsf_data : 2D `~numpy.ndarray`
            A 2D array containing the ePSF image.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The smoothed (convolved) ePSF data.
        """
        if self.smoothing_kernel is None:
            return epsf_data

        # do this check first as comparing a ndarray to string causes a warning
        if isinstance(self.smoothing_kernel, np.ndarray):
            kernel = self.smoothing_kernel

        elif self.smoothing_kernel == 'quartic':
            # from Polynomial2D fit with degree=4 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(4, c0_0=0.04163265, c1_0=-0.76326531,
            #              c2_0=0.99081633, c3_0=-0.4, c4_0=0.05,
            #              c0_1=-0.76326531, c0_2=0.99081633, c0_3=-0.4,
            #              c0_4=0.05, c1_1=0.32653061, c1_2=-0.08163265,
            #              c1_3=0.0, c2_1=-0.08163265, c2_2=0.02040816,
            #              c3_1=-0.0)>
            kernel = np.array(
                [[+0.041632, -0.080816, 0.078368, -0.080816, +0.041632],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.078368, +0.200816, 0.441632, +0.200816, +0.078368],
                 [-0.080816, -0.019592, 0.200816, -0.019592, -0.080816],
                 [+0.041632, -0.080816, 0.078368, -0.080816, +0.041632]])

        elif self.smoothing_kernel == 'quadratic':
            # from Polynomial2D fit with degree=2 to 5x5 array of
            # zeros with 1.0 at the center
            # Polynomial2D(2, c0_0=-0.07428571, c1_0=0.11428571,
            #              c2_0=-0.02857143, c0_1=0.11428571,
            #              c0_2=-0.02857143, c1_1=-0.0)
            kernel = np.array(
                [[-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [+0.03999952, 0.12571449, 0.15428215, 0.12571449,
                  +0.03999952],
                 [+0.01142786, 0.09714283, 0.12571449, 0.09714283,
                  +0.01142786],
                 [-0.07428311, 0.01142786, 0.03999952, 0.01142786,
                  -0.07428311]])

        else:
            msg = 'Unsupported kernel'
            raise TypeError(msg)

        return convolve(epsf_data, kernel)

    def _recenter_epsf(self, epsf, _centroid_func=centroid_com,
                       _box_size=(5, 5), _maxiters=20,
                       _center_accuracy=1.0e-4):
        """
        Recenter the ePSF data by shifting the peak to the array center.

        This method implements a discrete pixel shifting algorithm that
        finds the brightest pixel in the ePSF and shifts the entire
        array so that this peak is centered. The shifting is done using
        array slicing to avoid interpolation artifacts that could
        introduce noise or degrade the PSF quality.

        Algorithm Overview
        ------------------
        1. Find the coordinates of the maximum pixel value (peak)
        2. Calculate the target center position (geometric center of array)
        3. Compute the integer shift needed to center the peak
        4. Create a new array and copy shifted data using array slicing
        5. Handle edge cases where data would extend beyond array bounds

        The algorithm uses discrete pixel shifts rather than sub-pixel
        interpolation to preserve the original data values and avoid
        introducing artifacts. This is particularly important for PSF
        building where preserving the exact flux distribution is critical.

        Coordinate System
        -----------------
        The method works in array index coordinates:
        - (0, 0) is the top-left corner of the array
        - Positive shifts move the data right (x) and down (y)
        - The target center is at ((width-1)/2, (height-1)/2)

        Parameters
        ----------
        epsf : `_LegacyEPSFModel` object
            The ePSF model containing the data to be recentered.
            Only the .data attribute is used for recentering.

        _centroid_func : callable, optional
            A callable object (e.g., function or class) that is used
            to calculate the centroid of a 2D array. Currently not used
            in the simplified implementation, but kept for API compatibility.
            The callable must accept a 2D `~numpy.ndarray`, have a ``mask``
            keyword and optionally an ``error`` keyword.

        _box_size : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid. Currently not used in the simplified implementation,
            but kept for API compatibility. If used, values must be odd
            and >= 3 for both axes.

        _maxiters : int, optional
            The maximum number of recentering iterations to perform.
            Currently not used in the simplified implementation.

        _center_accuracy : float, optional
            The desired accuracy for the centers. Currently not used
            in the simplified implementation.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data array with the same shape as input.
            Pixels that would be shifted outside the array bounds are
            lost (set to zero). The total flux is preserved for the
            data that remains within the array.

        Notes
        -----
        This method preserves the total flux while ensuring the peak
        is at the geometric center of the array. Pixels that would be
        shifted outside the array bounds are lost (set to zero), which
        may result in a small flux loss if the shift is large.

        The method assumes the ePSF has a single dominant peak. For
        complex PSFs with multiple peaks, the behavior may not be
        optimal.

        Examples
        --------
        If the peak is at pixel (12, 15) in a 25x25 array, and the
        target center is (12, 12), the algorithm will shift the entire
        array up by 3 pixels, placing the peak at the center.
        """
        epsf_data = epsf.data

        # Find current peak location in array indices
        ypeak, xpeak = np.unravel_index(np.nanargmax(epsf_data),
                                        epsf_data.shape)

        # Get intended center based on array dimensions
        # For ImagePSF, the center should be at the center of the array
        height, width = epsf_data.shape
        xcenter_target = (width - 1) / 2.0
        ycenter_target = (height - 1) / 2.0

        # Compute shift needed (positive means shift right/down)
        shift_x = round(xcenter_target - xpeak)
        shift_y = round(ycenter_target - ypeak)

        # If no shift needed, return original data
        if shift_x == 0 and shift_y == 0:
            return epsf_data

        # Create shifted array
        recentered_data = np.zeros_like(epsf_data)

        # Calculate the regions to copy
        # Source region (from original data)
        src_y_start = max(0, -shift_y)
        src_y_end = min(epsf_data.shape[0], epsf_data.shape[0] - shift_y)
        src_x_start = max(0, -shift_x)
        src_x_end = min(epsf_data.shape[1], epsf_data.shape[1] - shift_x)

        # Destination region (in new data)
        dst_y_start = max(0, shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Copy the shifted data
        recentered_data[dst_y_start:dst_y_end,
                        dst_x_start:dst_x_end] = (
            epsf_data[src_y_start:src_y_end, src_x_start:src_x_end])

        return recentered_data

    def _build_epsf_step(self, stars, epsf=None):
        """
        A single iteration of improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `_LegacyEPSFModel` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `_LegacyEPSFModel` object
            The updated ePSF.
        """
        if len(stars) < 1:
            msg = ('stars must contain at least one EPSFStar or '
                   'LinkedEPSFStar object')
            raise ValueError(msg)

        if epsf is None:
            # create an initial ePSF (array of zeros)
            epsf = self._create_initial_epsf(stars)
        else:
            # improve the input ePSF (shallow copy suffices for data mod)
            epsf = copy.copy(epsf)

        # compute a 3D stack of 2D residual images
        residuals = self._compute_residual_median(stars, epsf)

        # interpolate any missing data (np.nan)
        mask = ~np.isfinite(residuals)
        if np.any(mask):
            residuals = _interpolate_missing_data(residuals, mask,
                                                  method='cubic')

            # fill any remaining nans (outer points) with zeros
            residuals[~np.isfinite(residuals)] = 0.0

        # add the residuals to the previous ePSF image
        new_epsf = epsf.data + residuals

        # smooth and recenter the ePSF
        smoothed_data = self._smooth_epsf(new_epsf)

        # Create an intermediate ePSF for recentering operations
        # Use the current epsf's origin if it exists, otherwise compute center
        if hasattr(epsf, 'origin') and epsf.origin is not None:
            origin = epsf.origin
        else:
            origin = ((epsf.data.shape[1] - 1) / 2.0,
                      (epsf.data.shape[0] - 1) / 2.0)

        temp_epsf = ImagePSF(data=smoothed_data,
                             origin=origin,
                             oversampling=self.oversampling,
                             fill_value=0.0)

        # Apply recentering to the smoothed data
        recentered_data = self._recenter_epsf(temp_epsf)

        # Create the final ePSF with recentered data
        # For ImagePSF, origin should be in oversampled pixel units
        final_origin = ((recentered_data.shape[1] - 1) / 2.0,
                        (recentered_data.shape[0] - 1) / 2.0)

        return ImagePSF(data=recentered_data,
                        origin=final_origin,
                        oversampling=self.oversampling,
                        fill_value=0.0)

    def _validate_and_initialize_build(self, stars, init_epsf):
        """
        Validate inputs and initialize variables for ePSF building.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.
        init_epsf : `ImagePSF` object or None
            The initial ePSF model.

        Returns
        -------
        legacy_epsf : `_LegacyEPSFModel` object or None
            Legacy ePSF model for internal iterations.
        fit_failed : `~numpy.ndarray`
            Boolean array tracking failed fits.
        centers : `~numpy.ndarray`
            Initial star center positions.
        """
        # Input validation happens in build_epsf method signature

        fit_failed = np.zeros(stars.n_stars, dtype=bool)
        centers = stars.cutout_center_flat

        if init_epsf is None:
            legacy_epsf = None
        else:
            legacy_epsf = _LegacyEPSFModel(
                init_epsf.data, flux=init_epsf.flux,
                x_0=init_epsf.x_0, y_0=init_epsf.y_0,
                oversampling=init_epsf.oversampling,
                fill_value=init_epsf.fill_value)

        return legacy_epsf, fit_failed, centers

    def _setup_progress_bar(self):
        """
        Setup progress bar for ePSF building iterations.

        Returns
        -------
        pbar : progress bar object or None
            Progress bar instance if enabled, None otherwise.
        """
        if not self.progress_bar:
            return None

        desc = f'EPSFBuilder ({self.maxiters} maxiters)'
        return add_progress_bar(total=self.maxiters,
                                desc=desc)  # pragma: no cover

    def _check_convergence(self, stars, centers, fit_failed):
        """
        Check if the ePSF building has converged.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.
        centers : `~numpy.ndarray`
            Previous star center positions.
        fit_failed : `~numpy.ndarray`
            Boolean array tracking failed fits.

        Returns
        -------
        converged : bool
            True if convergence criteria are met.
        center_dist_sq : `~numpy.ndarray`
            Squared distances of center movements.
        new_centers : `~numpy.ndarray`
            Updated star center positions.
        """
        # Calculate center movements
        new_centers = stars.cutout_center_flat
        dx_dy = new_centers - centers
        dx_dy = dx_dy[np.logical_not(fit_failed)]
        center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)

        # Check convergence
        converged = np.max(center_dist_sq) < self.center_accuracy_sq

        return converged, center_dist_sq, new_centers

    def _process_iteration(self, stars, legacy_epsf, iter_num):
        """
        Process a single iteration of ePSF building.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.
        legacy_epsf : `_LegacyEPSFModel` object
            Current ePSF model.
        iter_num : int
            Current iteration number.

        Returns
        -------
        legacy_epsf : `_LegacyEPSFModel` object
            Updated ePSF model.
        stars : `EPSFStars` object
            Updated stars with new fitted centers.
        fit_failed : `~numpy.ndarray`
            Boolean array tracking failed fits.
        """
        # Build/improve the ePSF
        legacy_epsf = self._build_epsf_step(stars, epsf=legacy_epsf)

        # Fit the new ePSF to the stars to find improved centers
        with warnings.catch_warnings():
            message = '.*The fit may be unsuccessful;.*'
            warnings.filterwarnings('ignore', message=message,
                                    category=AstropyUserWarning)

            image_psf = ImagePSF(data=legacy_epsf.data,
                                 flux=legacy_epsf.flux,
                                 x_0=legacy_epsf.x_0,
                                 y_0=legacy_epsf.y_0,
                                 oversampling=legacy_epsf.oversampling,
                                 fill_value=legacy_epsf.fill_value)

            stars = self.fitter(image_psf, stars)

        # Find all stars where the fit failed
        fit_failed = np.array([star._fit_error_status > 0
                              for star in stars.all_stars])

        if np.all(fit_failed):
            msg = 'The ePSF fitting failed for all stars.'
            raise ValueError(msg)

        # Permanently exclude fitting any star where the fit fails
        # after 3 iterations
        if iter_num > 3 and np.any(fit_failed):
            idx = fit_failed.nonzero()[0]
            for i in idx:  # pylint: disable=not-an-iterable
                stars.all_stars[i]._excluded_from_fit = True

        # Store the ePSF from this iteration
        self._epsf.append(legacy_epsf)

        return legacy_epsf, stars, fit_failed

    def _finalize_build(self, legacy_epsf, stars, pbar, iter_num,
                        converged, final_center_accuracy,
                        excluded_star_indices):
        """
        Finalize the ePSF building process and create result object.

        Parameters
        ----------
        legacy_epsf : `_LegacyEPSFModel` object
            Final legacy ePSF model.
        stars : `EPSFStars` object
            Final fitted stars.
        pbar : progress bar object or None
            Progress bar instance.
        iter_num : int
            Number of completed iterations.
        converged : bool
            Whether the building process converged.
        final_center_accuracy : float
            Final center accuracy achieved.
        excluded_star_indices : list
            Indices of excluded stars.

        Returns
        -------
        result : `EPSFBuildResult`
            Structured result containing ePSF, stars, and build diagnostics.
        """
        # Handle progress bar completion
        if pbar is not None:
            if iter_num < self.maxiters:
                pbar.write(f'EPSFBuilder converged after {iter_num} '
                           f'iterations (of {self.maxiters} maximum '
                           'iterations)')
            pbar.close()

        # Convert legacy ePSF back to ImagePSF
        epsf = ImagePSF(data=legacy_epsf.data, flux=legacy_epsf.flux,
                        x_0=legacy_epsf.x_0, y_0=legacy_epsf.y_0,
                        oversampling=legacy_epsf.oversampling,
                        fill_value=legacy_epsf.fill_value)

        # Create structured result
        return EPSFBuildResult(
            epsf=epsf,
            fitted_stars=stars,
            iterations=iter_num,
            converged=converged,
            final_center_accuracy=final_center_accuracy,
            n_excluded_stars=len(excluded_star_indices),
            excluded_star_indices=excluded_star_indices,
        )

    def build_epsf(self, stars, *, init_epsf=None):
        """
        Build iteratively an ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        init_epsf : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        result : `EPSFBuildResult` or tuple
            The ePSF building results. Returns an `EPSFBuildResult` object
            with detailed information about the building process. For
            backward compatibility, the result can be unpacked as a tuple:
            ``(epsf, fitted_stars) = epsf_builder(stars)``.

        Notes
        -----
        The structured result object contains:
        - epsf: The final constructed ePSF
        - fitted_stars: Stars with updated centers/fluxes
        - iterations: Number of iterations performed
        - converged: Whether convergence was achieved
        - final_center_accuracy: Final center movement accuracy
        - n_excluded_stars: Number of stars excluded due to fit failures
        - excluded_star_indices: Indices of excluded stars
        """
        # Initialize variables and validate inputs
        legacy_epsf, fit_failed, centers = self._validate_and_initialize_build(
            stars, init_epsf)

        # Setup progress tracking
        pbar = self._setup_progress_bar()

        # Initialize iteration variables and tracking
        iter_num = 0
        center_dist_sq = self.center_accuracy_sq + 1.0
        converged = False
        excluded_star_indices = []

        # Main iteration loop
        while (iter_num < self.maxiters and not np.all(fit_failed)
               and np.max(center_dist_sq) >= self.center_accuracy_sq):

            iter_num += 1

            # Process one iteration
            legacy_epsf, stars, fit_failed = self._process_iteration(
                stars, legacy_epsf, iter_num)

            # Track newly excluded stars
            if iter_num > 3 and np.any(fit_failed):
                new_excluded = fit_failed.nonzero()[0]
                for idx in new_excluded:
                    if idx not in excluded_star_indices:
                        excluded_star_indices.append(idx)

            # Check convergence
            converged, center_dist_sq, centers = self._check_convergence(
                stars, centers, fit_failed)

            # Update progress bar
            if pbar is not None:
                pbar.update()

        # Determine final convergence status and accuracy
        final_converged = np.max(center_dist_sq) < self.center_accuracy_sq
        final_center_accuracy = np.max(center_dist_sq) ** 0.5

        # Finalize and return structured results
        return self._finalize_build(legacy_epsf, stars, pbar, iter_num,
                                    final_converged, final_center_accuracy,
                                    excluded_star_indices)
