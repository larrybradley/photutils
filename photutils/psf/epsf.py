# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools to build and fit an effective PSF (ePSF)
based on Anderson and King (2000; PASP 112, 1360) and Anderson (2016;
WFC3 ISR 2016-12).
"""

import copy
import warnings

import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata.utils import NoOverlapError, PartialOverlapError
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import convolve

from photutils.aperture import CircularAperture
from photutils.centroids import centroid_com
from photutils.psf.epsf_stars import EPSFStar, EPSFStars, LinkedEPSFStar
from photutils.psf.image_models import ImagePSF
from photutils.psf.utils import _interpolate_missing_data
from photutils.utils._parameters import as_pair
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._round import py2intround
from photutils.utils._stats import nanmedian
from photutils.utils.cutouts import _overlap_slices as overlap_slices

__all__ = ['EPSFBuilder', 'EPSFFitter']

FITTER_DEFAULT = TRFLSQFitter()
SIGMA_CLIP_DEFAULT = SigmaClip(sigma=3, cenfunc='median', maxiters=10)


class EPSFFitter:
    """
    Class to fit an ePSF model to one or more stars.

    Parameters
    ----------
    fitter : `astropy.modeling.fitting.Fitter`, optional
        A `~astropy.modeling.fitting.Fitter` object.

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

    def __init__(self, *, fitter=FITTER_DEFAULT, fit_boxsize=5,
                 **fitter_kwargs):

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
        epsf = copy.deepcopy(epsf)

        # perform the fit
        fitted_stars = []
        for star in stars:
            if isinstance(star, EPSFStar):
                fitted_star = self._fit_star(epsf, star, self.fitter,
                                             self.fitter_kwargs,
                                             self.fitter_has_fit_info,
                                             self.fit_boxsize,
                                             oversampling=epsf.oversampling)

            elif isinstance(star, LinkedEPSFStar):
                fitted_star = []
                for linked_star in star:
                    fitted_star.append(
                        self._fit_star(epsf, linked_star, self.fitter,
                                       self.fitter_kwargs,
                                       self.fitter_has_fit_info,
                                       self.fit_boxsize,
                                       oversampling=epsf.oversampling))

                fitted_star = LinkedEPSFStar(fitted_star)
                fitted_star.constrain_centers()

            else:
                msg = ('stars must contain only EPSFStar and/or '
                       'LinkedEPSFStar objects')
                raise TypeError(msg)

            fitted_stars.append(fitted_star)

        return EPSFStars(fitted_stars)

    def _fit_star(self, epsf, star, fitter, fitter_kwargs,
                  fitter_has_fit_info, fit_boxsize, oversampling):
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

        # define positions in the ePSF oversampled grid
        yy, xx = np.indices(data.shape, dtype=float)
        xx = (xx - (star.cutout_center[0] - x0)) * oversampling[1]
        yy = (yy - (star.cutout_center[1] - y0)) * oversampling[0]
        scaled_data = data / np.prod(oversampling)

        # define the initial guesses for fitted flux and shifts
        epsf.flux = star.flux
        epsf.x_0 = 0.0
        epsf.y_0 = 0.0

        # The oversampling factor is used in the ImagePSF
        # evaluate method (which is use when fitting).  We do not want
        # to use oversampling here because it has been set by the ratio
        # of the ePSF and EPSFStar pixel scales.  This allows for
        # oversampling factors that differ between stars and also for
        # the factor to be different along the x and y axes.
        # FIXME
        # epsf._oversampling = 1.

        try:
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=scaled_data,
                                 weights=weights, **fitter_kwargs)
        except TypeError:
            # fitter doesn't support weights
            fitted_epsf = fitter(model=epsf, x=xx, y=yy, z=scaled_data,
                                 **fitter_kwargs)

        fit_error_status = 0
        if fitter_has_fit_info:
            fit_info = copy.copy(fitter.fit_info)

            if 'ierr' in fit_info and fit_info['ierr'] not in [1, 2, 3, 4]:
                fit_error_status = 2  # fit solution was not found
        else:
            fit_info = None

        # compute the star's fitted position
        x_center = (star.cutout_center[0] +
                    (fitted_epsf.x_0.value / oversampling[1]))
        y_center = (star.cutout_center[1] +
                    (fitted_epsf.y_0.value / oversampling[0]))

        star = copy.deepcopy(star)
        star.cutout_center = (x_center, y_center)

        # set the star's flux to the ePSF-fitted flux
        star.flux = fitted_epsf.flux.value

        star._fit_info = fit_info
        star._fit_error_status = fit_error_status

        return star


EPSF_FITTER = EPSFFitter()


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
        The smoothing kernel to apply to the ePSF during build
        iterations. The predefined ``'quartic'`` and ``'quadratic'``
        kernels are derived from fourth and second degree polynomials,
        respectively. Alternatively, a custom 2D array can be input. If
        `None` then no smoothing will be performed.

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
        A `EPSFFitter` object use to fit the ePSF to stars. To set
        custom fitter options, input a new `EPSFFitter` object. See the
        `EPSFFitter` documentation for options.

    maxiters : int, optional
        The maximum number of ePSF build iterations to perform.

    progress_bar : bool, option
        Whether to print the progress bar during the ePSF build
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
        The desired accuracy for the centers of stars. The ePSF building
        iterations will stop if the centers of all the stars change by
        less than ``center_accuracy`` pixels between iterations. All
        stars must meet this condition for the loop to exit.

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
                 sigma_clip=SIGMA_CLIP_DEFAULT,
                 smoothing_kernel='quartic', recentering_func=centroid_com,
                 recentering_boxsize=(5, 5), recentering_maxiters=20,
                 fitter=EPSF_FITTER, center_accuracy=1.0e-3, maxiters=10,
                 progress_bar=True, norm_radius=5.5):

        if oversampling is None:
            msg = "'oversampling' must be specified"
            raise ValueError(msg)
        self.oversampling = as_pair('oversampling', oversampling,
                                    lower_bound=(0, 1))
        if shape is not None:
            self.shape = as_pair('shape', shape, lower_bound=(0, 1))
        else:
            self.shape = shape

        if not isinstance(sigma_clip, SigmaClip):
            raise TypeError('sigma_clip must be an astropy.stats.SigmaClip '
                            'instance.')
        self._sigma_clip = sigma_clip

        self.smoothing_kernel = smoothing_kernel
        self.recentering_func = recentering_func
        self.recentering_boxsize = as_pair('recentering_boxsize',
                                           recentering_boxsize,
                                           lower_bound=(3, 0), check_odd=True)
        self.recentering_maxiters = recentering_maxiters

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
        self._norm_radius = norm_radius

        if not isinstance(sigma_clip, SigmaClip):
            msg = 'sigma_clip must be an astropy.stats.SigmaClip instance'
            raise TypeError(msg)
        self._sigma_clip = sigma_clip

        # store each ePSF build iteration
        # TODO: remove
        # store some data during each ePSF build iteration for debugging
        self._nfit_failed = []
        self._center_dist_sq = []
        self._max_center_dist_sq = []
        self._epsf = []
        self._residuals = []
        self._residuals_sigclip = []
        self._residuals_interp = []

    def __call__(self, stars):
        return self.build_epsf(stars)

    def _define_epsf_shape(self, stars):
        """
        Define the shape of the ePSF data array.

        If ``shape`` is not specified, the shape of the ePSF data array
        is determined from the shape of the input ``stars`` and the
        oversampling factor. If the size is even along any axis, it will
        be made odd by adding one. The output ePSF will always have odd
        sizes along both axes to ensure a central pixel.

        stars   oversampling   shape
        -----   ------------   -----
        even    even           even -> +1
        odd     even           even -> +1
        even    odd            even -> +1
        odd     odd            odd -> +0

        TODO: is an odd shape always necessary?
        """
        if self.shape is not None:
            shape = self.shape
        else:
            x_shape = (np.ceil(stars._max_shape[0])
                       * self.oversampling[1]).astype(int)
            y_shape = (np.ceil(stars._max_shape[1])
                       * self.oversampling[0]).astype(int)
            shape = np.array((y_shape, x_shape))

        # ensure odd sizes
        return [(i + 1) if i % 2 == 0 else i for i in shape]

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

        epsf : `ImagePSF` object
            The ePSF model.

        Returns
        -------
        image : 2D `~numpy.ndarray`
            A 2D image containing the resampled residual image. The
            image contains NaNs where there is no data.
        """
        # find the integer index of EPSFStar pixels in the oversampled
        # ePSF grid
        x = self.oversampling[1] * star._xidx_centered
        y = self.oversampling[0] * star._yidx_centered
        epsf_xcenter, epsf_ycenter = epsf.origin
        xidx = py2intround(x + epsf_xcenter)
        yidx = py2intround(y + epsf_ycenter)

        shape = epsf.data.shape
        mask = np.logical_and(np.logical_and(xidx >= 0, xidx < shape[1]),
                              np.logical_and(yidx >= 0, yidx < shape[0]))
        xidx = xidx[mask]
        yidx = yidx[mask]

        # Compute the normalized residual image by subtracting the
        # normalized ePSF model from the normalized star at the location
        # of the star in the undersampled grid.  Then, resample the
        # normalized residual image in the oversampled ePSF grid.
        # [(star - (epsf * xov * yov)) / (xov * yov)]
        # --> [(star / (xov * yov)) - epsf]
        epsf_tmp = copy.deepcopy(epsf)
        epsf_tmp.oversampling = (1, 1)
        stardata = ((star._data_values_normalized / np.prod(self.oversampling))
                    - epsf_tmp.evaluate(x=x, y=y, flux=1.0, x_0=0.0, y_0=0.0))

        resampled_img = np.full(epsf.data.shape, np.nan)
        resampled_img[yidx, xidx] = stardata[mask]

        return resampled_img

    def _resample_residuals(self, stars, epsf):
        """
        Compute normalized residual images in the ePSF grid for all the
        good input stars.

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
        resid = []
        for i, star in enumerate(stars.all_good_stars):
            resid.append(self._resample_residual(star, epsf))
        return np.array(resid)  # 3D cube

        # shape = (stars.n_good_stars, epsf.data.shape[0], epsf.data.shape[1])
        # epsf_resid = np.zeros(shape)
        # for i, star in enumerate(stars.all_good_stars):
        #     epsf_resid[i, :, :] = self._resample_residual(star, epsf)
        # return epsf_resid

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

        if not isinstance(self.smoothing_kernel, (np.ndarray, str)):
            raise TypeError('smoothing_kernel must be a 2D numpy.ndarray '
                            'or a string.')

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
            msg = 'Unsupported smoothing kernel'
            raise TypeError(msg)

        return convolve(epsf_data, kernel)

    #def _recenter_epsf(self, epsf, centroid_func=centroid_com,
    #                   box_size=(5, 5), maxiters=20, center_accuracy=1.0e-4):
    def _recenter_epsf(self, epsf):
        """
        Calculate the center of the ePSF data and shift the data so the
        ePSF center is at the center of the ePSF data array.

        Parameters
        ----------
        epsf : `ImagePSF` object
            The ePSF model.

        # TODO: remove kwargs
        centroid_func : callable, optional
            A callable object (e.g., function or class) that is used
            to calculate the centroid of a 2D array. The callable must
            accept a 2D `~numpy.ndarray`, have a ``mask`` keyword
            and optionally an ``error`` keyword. The callable object
            must return a tuple of two 1D `~numpy.ndarray` arrays,
            representing the x and y centroids.

        box_size : float or tuple of two floats, optional
            The size (in pixels) of the box used to calculate the
            centroid of the ePSF during each build iteration. If a
            single integer number is provided, then a square box will
            be used. If two values are provided, then they must be in
            ``(ny, nx)`` order. ``box_size`` must have odd values and be
            greater than or equal to 3 for both axes.

        maxiters : int, optional
            The maximum number of recentering iterations to perform.

        recenter_accuracy : float, optional
            The desired accuracy for the centers of stars. The
            recentering iterations will stop if the center of the ePSF
            changes by less than ``recenter_accuracy`` pixels between
            iterations.

        Returns
        -------
        result : 2D `~numpy.ndarray`
            The recentered ePSF data.
        """
        # TODO: remove this after testing
        centroid_func = self.recentering_func
        box_size = self.recentering_boxsize
        maxiters = self.recentering_maxiters

        # this is not the same as self.center_accuracy
        recenter_accuracy = 1.0e-4

        epsf_data = epsf.data

        # LDB
        epsf = ImagePSF(data=epsf.data, oversampling=epsf.oversampling)
        epsf.fill_value = 0.0
        xcenter, ycenter = epsf.origin

        dx_total = 0
        dy_total = 0
        y, x = np.indices(epsf.data.shape, dtype=float)

        iter_num = 0
        recenter_accuracy_sq = recenter_accuracy ** 2
        center_dist_sq = recenter_accuracy_sq + 1.0e6
        center_dist_sq_prev = center_dist_sq + 1
        epsf_tmp = copy.deepcopy(epsf)
        epsf_tmp.oversampling = (1, 1)
        while (iter_num < maxiters and center_dist_sq >= recenter_accuracy_sq):
            iter_num += 1

            # extract a cutout from the ePSF
            slices_large, _ = overlap_slices(epsf_data.shape, box_size,
                                             (ycenter, xcenter))
            epsf_cutout = epsf_data[slices_large]
            mask = ~np.isfinite(epsf_cutout)

            # find a new center position
            xcenter_new, ycenter_new = centroid_func(epsf_cutout, mask=mask)
            xcenter_new += slices_large[1].start
            ycenter_new += slices_large[0].start

            # calculate the shift
            dx = xcenter - xcenter_new
            dy = ycenter - ycenter_new
            center_dist_sq = dx**2 + dy**2
            if center_dist_sq >= center_dist_sq_prev:  # don't shift
                # diverging; should converge quickly without big jumps
                break
            center_dist_sq_prev = center_dist_sq

            # Resample the ePSF data to a shifted grid to place the
            # centroid in the center of the central pixel. The shift is
            # always performed on the input epsf_data.
            dx_total += dx    # accumulated shifts for the input epsf_data
            dy_total += dy
            epsf_data = epsf_tmp.evaluate(x=x, y=y, flux=1.0,
                                          x_0=xcenter + dx_total,
                                          y_0=ycenter + dy_total)

        return epsf_data

    def _build_epsf_step(self, stars, epsf=None):
        """
        A single iteration of building or improving an ePSF.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        epsf : `ImagePSF` object, optional
            The ePSF model to build or improve. If not input, then the
            ePSF will be built from scratch.

        Returns
        -------
        epsf : `ImagePSF` object
            The updated ePSF model.
        """
        if len(stars) < 1:
            msg = ('stars must contain at least one EPSFStar or '
                   'LinkedEPSFStar object')
            raise ValueError(msg)

        if epsf is None:
            # create an initial ePSF (array of zeros)
            data = np.zeros(self.shape, dtype=float)
            epsf = ImagePSF(data=data, oversampling=self.oversampling)
        else:
            # improve the input ePSF
            # TODO: is a copy needed here?
            epsf = copy.deepcopy(epsf)

        # compute a 3D stack of 2D residual images
        residuals = self._resample_residuals(stars, epsf)

        # compute the sigma-clipped average along the 3D stack
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            residuals = self._sigma_clip(residuals, axis=0, masked=False,
                                         return_bounds=False)
            residuals = nanmedian(residuals, axis=0)

        # TEMP
        self._residuals_sigclip.append(residuals)

        # interpolate any missing data (np.nan)
        # TODO: every pixel is nan in the residuals 3D stack
        mask = ~np.isfinite(residuals)
        if np.any(mask):
            residuals = _interpolate_missing_data(residuals, mask,
                                                  method='cubic')

            # TODO: improve this?
            # fill any remaining nans (outer points) with zeros
            residuals[~np.isfinite(residuals)] = 0.0

        # TEMP
        self._residuals_interp.append(residuals)

        # add the residuals to the previous normalized ePSF image
        if np.sum(epsf.data) != 0:
            epsf_data_norm = epsf.data / np.sum(epsf.data)
        else:
            epsf_data_norm = epsf.data
        new_epsf = epsf_data_norm + residuals

        # smooth the ePSF
        new_epsf = self._smooth_epsf(new_epsf)

        new_epsf = ImagePSF(data=new_epsf, oversampling=epsf.oversampling)

        # recenter the ePSF
        # TODO: add check to ensure centering?
        new_epsf_data = self._recenter_epsf(new_epsf)

        # normalize the ePSF
        new_epsf_data /= new_epsf_data.sum()

        new_epsf.data = new_epsf_data

        return new_epsf

    def build_epsf(self, stars, *, init_epsf=None):
        """
        Iteratively build or improve an ePSF from star cutouts.

        Parameters
        ----------
        stars : `EPSFStars` object
            The stars used to build the ePSF.

        init_epsf : `ImagePSF` object, optional
            The initial ePSF model. If not input, then the ePSF will be
            built from scratch.

        Returns
        -------
        epsf : `ImagePSF` object
            The constructed ePSF.

        fitted_stars : `EPSFStars` object
            The input stars with updated centers and fluxes derived
            from fitting the output ``epsf``.
        """
        self.shape = self._define_epsf_shape(stars)

        iter_num = 0
        fit_failed = np.zeros(stars.n_stars, dtype=bool)
        epsf = init_epsf
        center_dist_sq = self.center_accuracy_sq + 1.0
        centers = stars.cutout_center_flat

        pbar = None
        if self.progress_bar:
            desc = f'EPSFBuilder ({self.maxiters} maxiters)'
            pbar = add_progress_bar(total=self.maxiters,
                                    desc=desc)  # pragma: no cover

        while (iter_num < self.maxiters and not np.all(fit_failed)
               and np.max(center_dist_sq) >= self.center_accuracy_sq):

            iter_num += 1

            # build/improve the ePSF
            epsf = self._build_epsf_step(stars, epsf=epsf)

            # fit the new ePSF to the stars to find improved centers
            # we catch fit warnings here -- stars with unsuccessful fits
            # are excluded from the ePSF build process
            with warnings.catch_warnings():
                message = '.*The fit may be unsuccessful;.*'
                warnings.filterwarnings('ignore', message=message,
                                        category=AstropyUserWarning)
                # Note: stars get new centers
                stars = self.fitter(epsf, stars)

            # find all stars where the fit failed
            fit_failed = np.array([star._fit_error_status > 0
                                   for star in stars.all_stars])
            if np.all(fit_failed):
                msg = 'The ePSF fitting failed for all stars.'
                raise ValueError(msg)

            # permanently exclude fitting any star where the fit fails
            # after 3 iterations
            if iter_num > 3 and np.any(fit_failed):
                idx = fit_failed.nonzero()[0]
                for i in idx:  # pylint: disable=not-an-iterable
                    stars.all_stars[i]._excluded_from_fit = True

            # if no star centers have moved by more than pixel accuracy,
            # stop the iteration loop early
            dx_dy = stars.cutout_center_flat - centers
            dx_dy = dx_dy[np.logical_not(fit_failed)]  # exclude bad fits
            center_dist_sq = np.sum(dx_dy * dx_dy, axis=1, dtype=np.float64)
            centers = stars.cutout_center_flat

            # TEMP
            self._nfit_failed.append(np.count_nonzero(fit_failed))
            self._center_dist_sq.append(center_dist_sq)
            self._max_center_dist_sq.append(np.max(center_dist_sq))
            self._epsf.append(epsf)

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            if iter_num < self.maxiters:
                pbar.write(f'EPSFBuilder converged after {iter_num} '
                           f'iterations (of {self.maxiters} maximum '
                           'iterations)')
            pbar.close()

        # TODO: remove
        epsf_data = _normalize_epsf(epsf.data, self._norm_radius,
                                    epsf.oversampling)
        epsf.data = epsf_data

        return epsf, stars


def _normalize_epsf(psf_data, norm_radius, oversampling):
    if np.sum(psf_data) == 0:
        return psf_data

    # NOTE: possibly remove this when EPSFModel is removed
    xypos = np.array(psf_data.shape) / 2.0
    xypos = xypos[::-1]
    # TODO: generalize "radius" (ellipse?) is oversampling is
    # different along x/y axes
    radius = norm_radius * oversampling[0]
    aper = CircularAperture(xypos, r=radius)
    flux, _ = aper.do_photometry(psf_data, method='exact')
    if flux[0] == 0:
        return psf_data
    return psf_data / (flux[0] / np.prod(oversampling))
