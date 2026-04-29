# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Photometry helpers for `~photutils.segmentation.SourceCatalog`.

`_Photometry` is held by the `SourceCatalog` instance via composition
(a `@lazyproperty` accessor) and provides the implementations of the
circular, Kron, and flux-radius photometry methods.  Each public
`SourceCatalog` photometry method/lazyproperty is a thin wrapper that
delegates to a method on this class.
"""

import math
import warnings

import astropy.units as u
import numpy as np
from scipy.optimize import root_scalar

from photutils.aperture import (BoundingBox, CircularAperture,
                                EllipticalAperture)
from photutils.geometry import (circular_overlap_grid,
                                circular_overlap_weighted_sum,
                                elliptical_overlap_grid)
from photutils.segmentation.utils import _mask_to_mirrored_value
from photutils.utils._progress_bars import add_progress_bar

__all__ = []


class _Photometry:
    """
    Compute aperture photometry, Kron radii/photometry, and flux
    radii for sources in a `~photutils.segmentation.SourceCatalog`.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        The host source catalog.
    """

    def __init__(self, catalog):
        self._catalog = catalog

    def _aperture_to_mask(self, aperture, **kwargs):
        """
        Call ``aperture.to_mask()`` after guarding against pathologically
        large aperture bounding boxes that could trigger out-of-memory
        errors.
        """
        catalog = self._catalog
        bbox = aperture.bbox
        max_size = max(catalog._data.size, 1_000_000)
        if bbox.shape[0] * bbox.shape[1] > max_size:
            return None
        return aperture.to_mask(**kwargs)

    def _make_aperture_data(self, label, x_centroid, y_centroid,
                            aperture_bbox, local_background, *,
                            make_error=True):
        """
        Make cutouts of data, error, and mask arrays for aperture
        photometry, applying the catalog's ``aperture_mask_method`` for
        neighboring sources.
        """
        catalog = self._catalog
        slc_lg, slc_sm = aperture_bbox.get_overlap_slices(catalog._data.shape)
        if slc_lg is None:
            return (None,) * 5

        data = catalog._data[slc_lg].astype(float) - local_background

        mask_cutout = (None if catalog._mask is None
                       else catalog._mask[slc_lg])
        data_mask = catalog._make_cutout_data_mask(data, mask_cutout)

        if make_error and catalog._error is not None:
            error = catalog._error[slc_lg]
        else:
            error = None

        cutout_xycen = (x_centroid - max(0, aperture_bbox.ixmin),
                        y_centroid - max(0, aperture_bbox.iymin))

        aperture_mask_method = catalog.aperture_mask_method
        if aperture_mask_method == 'none':
            mask = data_mask
        else:
            segment_img = catalog._segmentation_image.data[slc_lg]
            segm_mask = np.logical_and(segment_img != label, segment_img != 0)
            if aperture_mask_method == 'mask':
                mask = data_mask | segm_mask
            else:
                mask = data_mask

        if aperture_mask_method == 'correct':
            data = _mask_to_mirrored_value(data, segm_mask, cutout_xycen,
                                           mask=mask)
            if error is not None:
                error = _mask_to_mirrored_value(error, segm_mask, cutout_xycen,
                                                mask=mask)

        return data, error, mask, cutout_xycen, slc_sm

    # ------------------------------------------------------------------ #
    # Circular apertures
    # ------------------------------------------------------------------ #

    def _make_circular_apertures(self, radius):
        """
        Make a list of `CircularAperture` instances for each source.

        Returns ``None`` for sources where the centroid is non-finite or
        where the source is fully masked.
        """
        catalog = self._catalog
        radius = np.broadcast_to(radius, len(catalog._x_centroid))
        if np.any(radius <= 0):
            msg = 'radius must be > 0'
            raise ValueError(msg)

        apertures = []
        for (xcen, ycen, radius_, all_masked) in zip(catalog._x_centroid,
                                                     catalog._y_centroid,
                                                     radius,
                                                     catalog._all_masked,
                                                     strict=True):
            if all_masked or np.any(~np.isfinite((xcen, ycen, radius_))):
                apertures.append(None)
                continue

            apertures.append(CircularAperture((xcen, ycen), r=radius_))

        return apertures

    def _plot_circular_apertures(self, radius, ax, origin, **kwargs):
        """
        Plot circular apertures and return the list of patches.
        """
        apertures = self._make_circular_apertures(radius)
        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(ax=ax, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    def _circular_photometry(self, radius, name, overwrite):
        """
        Compute circular aperture photometry for each source.
        """
        catalog = self._catalog
        if radius <= 0:
            msg = 'radius must be > 0'
            raise ValueError(msg)

        apertures = self._make_circular_apertures(radius)
        kwargs = catalog._aperture_mask_kwargs['circ']
        flux, flux_err = self._aperture_photometry(
            apertures, desc='circular_photometry', **kwargs)

        if catalog._data_unit is not None:
            flux <<= catalog._data_unit
            flux_err <<= catalog._data_unit

        if catalog.isscalar:
            flux = flux[0]
            flux_err = flux_err[0]

        if name is not None:
            catalog.add_property(f'{name}_flux', flux, overwrite=overwrite)
            catalog.add_property(f'{name}_flux_err', flux_err,
                                 overwrite=overwrite)

        return flux, flux_err

    # ------------------------------------------------------------------ #
    # Elliptical / Kron apertures
    # ------------------------------------------------------------------ #

    def _make_elliptical_apertures(self, *, scale=6.0):
        """
        Return a list of elliptical (or fallback circular) apertures for
        each source, based on the source's shape parameters and ``scale``.
        """
        catalog = self._catalog
        xcen = catalog._x_centroid
        ycen = catalog._y_centroid
        major_size = catalog.semimajor_axis.value * scale
        minor_size = catalog.semiminor_axis.value * scale
        theta = catalog.orientation.to(u.radian).value
        if catalog.isscalar:
            major_size = (major_size,)
            minor_size = (minor_size,)
            theta = (theta,)

        aperture = []
        for values in zip(xcen, ycen, major_size, minor_size, theta,
                          catalog._all_masked, strict=True):
            if values[-1] or np.any(~np.isfinite(values[:-1])):
                aperture.append(None)
                continue

            # kron_radius = 0 -> scale = 0 -> major/minor_size = 0
            if values[2] == 0 and values[3] == 0:
                aperture.append(CircularAperture((values[0], values[1]),
                                                 r=catalog.kron_params[2]))
                continue

            (xcen_, ycen_, major_, minor_, theta_) = values[:-1]
            aperture.append(EllipticalAperture((xcen_, ycen_), major_, minor_,
                                               theta=theta_))

        return aperture

    def _measured_kron_radius(self):
        """
        Compute the *unscaled* first-moment Kron radius for each source as a
        plain `~numpy.ndarray` (no units).
        """
        catalog = self._catalog
        scale = 6.0

        xcen_arr = catalog._x_centroid
        ycen_arr = catalog._y_centroid
        a_arr = catalog.semimajor_axis.value * scale
        b_arr = catalog.semiminor_axis.value * scale
        theta_arr = catalog.orientation.to(u.radian).value
        cxx_arr = catalog.ellipse_cxx.value
        cxy_arr = catalog.ellipse_cxy.value
        cyy_arr = catalog.ellipse_cyy.value
        all_masked = catalog._all_masked

        if catalog.isscalar:
            a_arr = (a_arr,)
            b_arr = (b_arr,)
            theta_arr = (theta_arr,)
            cxx_arr = (cxx_arr,)
            cxy_arr = (cxy_arr,)
            cyy_arr = (cyy_arr,)

        data_full = catalog._data
        data_shape = data_full.shape
        mask_full = catalog._mask
        segm_data = catalog._segmentation_image.data
        max_size = max(data_full.size, 1_000_000)
        kron_min = catalog.kron_params[1]
        min_circ_radius = (catalog.kron_params[2]
                           if len(catalog.kron_params) == 3 else 0.0)
        aperture_mask_method = catalog.aperture_mask_method

        labels = catalog.labels
        if catalog.progress_bar:
            labels = add_progress_bar(labels, desc='kron_radius')

        kron_radius = []
        for (label, xc, yc, a, b, theta, cxx_, cxy_, cyy_,
             masked) in zip(labels, xcen_arr, ycen_arr, a_arr, b_arr,
                            theta_arr, cxx_arr, cxy_arr, cyy_arr,
                            all_masked, strict=True):
            if masked or not (math.isfinite(xc) and math.isfinite(yc)
                              and math.isfinite(a) and math.isfinite(b)
                              and math.isfinite(theta)):
                kron_radius.append(np.nan)
                continue

            # Circular aperture fallback when semimajor/semiminor are zero
            # (matching _make_elliptical_apertures behavior)
            use_circular = (a == 0 and b == 0)
            if use_circular:
                if min_circ_radius <= 0:
                    kron_radius.append(np.nan)
                    continue
                half_w = min_circ_radius
                half_h = min_circ_radius
            else:
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                half_w = math.sqrt(a * a * cos_theta * cos_theta
                                   + b * b * sin_theta * sin_theta)
                half_h = math.sqrt(a * a * sin_theta * sin_theta
                                   + b * b * cos_theta * cos_theta)

            # Compute bounding box from ellipse/circle parameters
            ixmin = math.floor(xc - half_w + 0.5)
            ixmax = math.floor(xc + half_w + 0.5) + 1
            iymin = math.floor(yc - half_h + 0.5)
            iymax = math.floor(yc + half_h + 0.5) + 1

            # OOM guard
            if (ixmax - ixmin) * (iymax - iymin) > max_size:
                kron_radius.append(np.nan)
                continue

            # Compute overlap slices with data boundaries
            dx_min = max(0, -ixmin)
            dy_min = max(0, -iymin)
            dx_max = max(0, ixmax - data_shape[1])
            dy_max = max(0, iymax - data_shape[0])
            lg_xmin = ixmin + dx_min
            lg_xmax = ixmax - dx_max
            lg_ymin = iymin + dy_min
            lg_ymax = iymax - dy_max
            if lg_xmin >= lg_xmax or lg_ymin >= lg_ymax:
                kron_radius.append(np.nan)
                continue

            slc_lg = (slice(lg_ymin, lg_ymax), slice(lg_xmin, lg_xmax))

            # Cutout data (local background explicitly zero for SE
            # agreement)
            data = data_full[slc_lg].astype(float)

            # Build data mask (non-finite + input mask)
            data_mask = ~np.isfinite(data)
            if mask_full is not None:
                data_mask |= mask_full[slc_lg]

            # Mask or correct neighboring sources
            if aperture_mask_method != 'none':
                seg_cut = segm_data[slc_lg]
                segm_mask = (seg_cut != label) & (seg_cut != 0)
                if aperture_mask_method == 'mask':
                    mask = data_mask | segm_mask
                else:
                    mask = data_mask
                if aperture_mask_method == 'correct':
                    cutout_xycen = (xc - max(0, ixmin), yc - max(0, iymin))
                    data = _mask_to_mirrored_value(data, segm_mask,
                                                   cutout_xycen,
                                                   mask=mask)
            else:
                mask = data_mask

            # Coordinate arrays (ogrid-style broadcasting avoids allocating
            # full 2D meshgrid arrays)
            ny, nx = data.shape
            xval = np.arange(nx) - (xc - lg_xmin)
            yval = np.arange(ny) - (yc - lg_ymin)
            yy = yval[:, np.newaxis]
            xx = xval[np.newaxis, :]

            # Elliptical radius
            rr_sq = cxx_ * xx * xx + cxy_ * xx * yy + cyy_ * yy * yy
            rr = np.sqrt(np.maximum(rr_sq, 0.0))

            # Aperture mask: for method='center', pixels whose center falls
            # inside the ellipse (rr <= scale) or circle
            if use_circular:
                dx = xx
                dy = yy
                pixel_mask = ((dx * dx + dy * dy)
                              <= min_circ_radius * min_circ_radius) & ~mask
            else:
                pixel_mask = (rr <= scale) & ~mask

            # Ignore RuntimeWarning for invalid data values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                flux_numer = np.sum(data[pixel_mask] * rr[pixel_mask])
                flux_denom = np.sum(data[pixel_mask])

            # Set Kron radius to the minimum Kron radius if numerator or
            # denominator is negative
            if flux_numer <= 0 or flux_denom <= 0:
                kron_radius.append(kron_min)
                continue

            kron_radius.append(flux_numer / flux_denom)

        return np.array(kron_radius)

    def _calc_kron_radius(self, kron_params):
        """
        Compute the unscaled Kron radius (with units), applying any
        minimum Kron or minimum circular radius from ``kron_params``.
        """
        catalog = self._catalog
        kron_radius = catalog._measured_kron_radius.copy()

        # Set values exceeding the measurement aperture scale (6.0) to NaN.
        # Such values are unphysical (the Kron radius cannot meaningfully
        # exceed the aperture used to measure it) and are caused by
        # near-cancellation in the denominator of the Kron formula due to
        # outlier pixels or noise.
        max_kron_radius = 6.0
        kron_radius[kron_radius > max_kron_radius] = np.nan

        # Set minimum (unscaled) kron radius
        kron_radius[kron_radius < kron_params[1]] = kron_params[1]

        # Check for minimum circular radius
        if len(kron_params) == 3:
            semimajor_axis = catalog.semimajor_axis.value
            semiminor_axis = catalog.semiminor_axis.value
            circ_radius = (kron_params[0] * kron_radius
                           * np.sqrt(semimajor_axis * semiminor_axis))
            kron_radius[circ_radius <= kron_params[2]] = 0.0

        return kron_radius << u.pix

    def _make_kron_apertures(self, kron_params):
        """
        Make Kron apertures for each source (always returned as a list).
        """
        catalog = self._catalog
        # NOTE: if kron_radius = NaN, scale = NaN and kron_aperture = None
        # Use the catalog's @as_scalar-decorated method so ``scale`` is a
        # plain scalar when ``catalog.isscalar`` is True.
        kron_radius = catalog._calc_kron_radius(kron_params)
        scale = kron_radius.value * kron_params[0]
        return self._make_elliptical_apertures(scale=scale)

    def _plot_kron_apertures(self, kron_params, ax, origin, **kwargs):
        """
        Plot Kron apertures and return the list of patches.
        """
        catalog = self._catalog
        if kron_params is None:
            apertures = catalog.kron_aperture
            if catalog.isscalar:
                apertures = (apertures,)
        else:
            apertures = self._make_kron_apertures(kron_params)

        patches = []
        for aperture in apertures:
            if aperture is not None:
                aperture.plot(ax=ax, origin=origin, **kwargs)
                patches.append(aperture._to_patch(origin=origin, **kwargs))
        return patches

    # ------------------------------------------------------------------ #
    # Aperture photometry (used by both circular and Kron paths)
    # ------------------------------------------------------------------ #

    def _aperture_photometry(self, apertures, *, desc='', **kwargs):
        """
        Perform aperture photometry on the given list of apertures, applying
        the catalog's ``aperture_mask_method`` to handle neighbors.
        """
        catalog = self._catalog
        labels = catalog.labels
        if catalog.progress_bar:
            labels = add_progress_bar(labels, desc=desc)

        flux = []
        flux_err = []
        for label, aperture, bkg in zip(labels, apertures,
                                        catalog._local_background,
                                        strict=True):
            # Return NaN for completely masked sources or sources where
            # the centroid is not finite
            if aperture is None:
                flux.append(np.nan)
                flux_err.append(np.nan)
                continue

            xcen, ycen = aperture.positions
            aperture_mask = catalog._aperture_to_mask(aperture, **kwargs)
            if aperture_mask is None:
                flux.append(np.nan)
                flux_err.append(np.nan)
                continue

            # Prepare cutouts of the data based on the aperture size
            data, error, mask, _, slc_sm = catalog._make_aperture_data(
                label, xcen, ycen, aperture_mask.bbox, bkg)

            aperture_weights = aperture_mask.data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask  # good pixels
            # Ignore RuntimeWarning for invalid data or error values
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                values = (aperture_weights * data)[pixel_mask]
                flux_ = np.nan if values.shape == (0,) else np.sum(values)
                flux.append(flux_)

                if error is None:
                    flux_err_ = np.nan
                else:
                    values = (aperture_weights * error**2)[pixel_mask]
                    if values.shape == (0,):
                        flux_err_ = np.nan
                    else:
                        flux_err_ = np.sqrt(np.sum(values))
                flux_err.append(flux_err_)

        return np.array(flux), np.array(flux_err)

    def _calc_kron_photometry(self, *, kron_params=None):
        """
        Compute the Kron flux and flux-error arrays (without units).

        Uses the cached ``kron_aperture`` when ``kron_params`` is None;
        otherwise rebuilds Kron apertures using the input ``kron_params``.
        """
        catalog = self._catalog
        if kron_params is None:
            kron_aperture = catalog.kron_aperture
            if catalog.isscalar:
                kron_aperture = (kron_aperture,)
        else:
            kron_params = catalog._validate_kron_params(kron_params)
            kron_aperture = self._make_kron_apertures(kron_params)

        labels = catalog.labels
        if catalog.progress_bar:
            labels = add_progress_bar(labels, desc='kron_photometry')

        _floor = math.floor
        max_size = max(catalog._data.size, 1_000_000)

        flux = []
        flux_err = []
        for label, aperture, bkg in zip(labels, kron_aperture,
                                        catalog._local_background,
                                        strict=True):
            if aperture is None:
                flux.append(np.nan)
                flux_err.append(np.nan)
                continue

            xcen, ycen = aperture.positions

            # Compute the aperture mask directly, bypassing the aperture's
            # to_mask() method and ApertureMask/BoundingBox property
            # overhead.
            if isinstance(aperture, CircularAperture):
                r = aperture.r
                ixmin = _floor(xcen - r + 0.5)
                ixmax = _floor(xcen + r + 1.5)
                iymin = _floor(ycen - r + 0.5)
                iymax = _floor(ycen + r + 1.5)
                nx = ixmax - ixmin
                ny = iymax - iymin
                if nx * ny > max_size:
                    flux.append(np.nan)
                    flux_err.append(np.nan)
                    continue
                edges = (ixmin - 0.5 - xcen, ixmax - 0.5 - xcen,
                         iymin - 0.5 - ycen, iymax - 0.5 - ycen)
                mask_data = circular_overlap_grid(
                    edges[0], edges[1], edges[2], edges[3],
                    nx, ny, r, 1, 1)
            else:
                a = aperture.a
                b = aperture.b
                theta_val = aperture.theta
                theta_rad = (theta_val.to(u.radian).value
                             if hasattr(theta_val, 'to')
                             else float(theta_val))
                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)
                x_ext = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
                y_ext = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
                ixmin = _floor(xcen - x_ext + 0.5)
                ixmax = _floor(xcen + x_ext + 1.5)
                iymin = _floor(ycen - y_ext + 0.5)
                iymax = _floor(ycen + y_ext + 1.5)
                nx = ixmax - ixmin
                ny = iymax - iymin
                if nx * ny > max_size:
                    flux.append(np.nan)
                    flux_err.append(np.nan)
                    continue
                edges = (ixmin - 0.5 - xcen, ixmax - 0.5 - xcen,
                         iymin - 0.5 - ycen, iymax - 0.5 - ycen)
                mask_data = elliptical_overlap_grid(
                    edges[0], edges[1], edges[2], edges[3],
                    nx, ny, a, b, theta_rad, 1, 1)

            bbox = BoundingBox(ixmin, ixmax, iymin, iymax)
            data, error, mask, _, slc_sm = catalog._make_aperture_data(
                label, xcen, ycen, bbox, bkg)
            if data is None:
                flux.append(np.nan)
                flux_err.append(np.nan)
                continue

            aperture_weights = mask_data[slc_sm]
            pixel_mask = (aperture_weights > 0) & ~mask
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                values = (aperture_weights * data)[pixel_mask]
                flux_ = np.nan if values.shape == (0,) else np.sum(values)
                flux.append(flux_)

                if error is None:
                    flux_err_ = np.nan
                else:
                    values = (aperture_weights * error ** 2)[pixel_mask]
                    if values.shape == (0,):
                        flux_err_ = np.nan
                    else:
                        flux_err_ = np.sqrt(np.sum(values))
                flux_err.append(flux_err_)

        return np.array(flux), np.array(flux_err)

    def _kron_photometry_table(self, kron_params, name, overwrite):
        """
        Public Kron photometry path: compute Kron flux/flux_err with units
        applied and optionally store them as named custom properties.
        """
        catalog = self._catalog
        kron_flux, kron_flux_err = self._calc_kron_photometry(
            kron_params=kron_params)
        if catalog._data_unit is not None:
            kron_flux <<= catalog._data_unit
            kron_flux_err <<= catalog._data_unit

        if catalog.isscalar:
            kron_flux = kron_flux[0]
            kron_flux_err = kron_flux_err[0]

        if name is not None:
            catalog.add_property(f'{name}_flux', kron_flux,
                                 overwrite=overwrite)
            catalog.add_property(f'{name}_flux_err', kron_flux_err,
                                 overwrite=overwrite)

        return kron_flux, kron_flux_err

    # ------------------------------------------------------------------ #
    # Flux radius
    # ------------------------------------------------------------------ #

    def _max_circular_kron_radius(self):
        """
        Return the maximum circular Kron radius used as the upper limit of
        ``flux_radius``.
        """
        catalog = self._catalog
        semimajor_sig = catalog.semimajor_axis.value
        kron_radius = catalog.kron_radius.value
        radius = semimajor_sig * kron_radius * catalog.kron_params[0]
        mask = radius == 0
        if np.any(mask):
            radius[mask] = catalog.kron_params[2]
        if catalog.isscalar:
            radius = np.array([radius])
        return radius

    @staticmethod
    def _flux_radius_fcn(radius, clean_data, grid_params, normflux):
        """
        Function whose root is found to compute the ``flux_radius``.

        Uses ``circular_overlap_weighted_sum`` directly on pre-computed
        cutout data (with masked pixels zeroed) to avoid allocating a
        per-call ``(ny, nx)`` weight array.
        """
        xmin_e, xmax_e, ymin_e, ymax_e, _nx, _ny, exact, subpx = grid_params
        flux = circular_overlap_weighted_sum(
            clean_data, xmin_e, xmax_e, ymin_e, ymax_e, radius, exact, subpx)
        return 1.0 - (flux / normflux)

    def _flux_radius_optimizer_args(self):
        """
        Pre-compute per-source argument tuples for the ``flux_radius``
        root-finding loop. Returns ``None`` for sources that cannot be
        measured.
        """
        catalog = self._catalog
        kron_flux = catalog._kron_photometry[:, 0]  # unitless
        max_radius = catalog._max_circular_kron_radius
        kwargs = catalog._aperture_mask_kwargs['flux_radius']

        # Translate mask method keywords to circular_overlap_grid parameters
        method = kwargs.get('method', 'exact')
        if method == 'exact':
            use_exact = 1
            subpixels = 1
        elif method == 'center':
            use_exact = 0
            subpixels = 1
        else:  # 'subpixel'
            use_exact = 0
            subpixels = kwargs.get('subpixels', 5)

        # Pre-fetch arrays used inside the loop
        data_arr = catalog._data
        mask_arr = catalog._mask
        segm_data = catalog._segmentation_image.data
        data_shape = data_arr.shape
        aperture_mask_method = catalog.aperture_mask_method
        max_aper_size = max(data_arr.size, 1_000_000)

        labels = catalog.labels
        if catalog.progress_bar:
            labels = add_progress_bar(labels, desc='flux_radius prep')

        args = []
        for label, xcen, ycen, kronflux, bkg, max_radius_ in zip(
                labels, catalog._x_centroid, catalog._y_centroid,
                kron_flux, catalog._local_background, max_radius,
                strict=True):

            if (np.any(~np.isfinite((xcen, ycen, kronflux, max_radius_)))
                    or kronflux == 0):
                args.append(None)
                continue

            # Compute the bounding box for the max-radius aperture inline,
            # replacing CircularAperture + _aperture_to_mask +
            # _make_aperture_data
            ixmin = math.floor(xcen - max_radius_ + 0.5)
            ixmax = math.ceil(xcen + max_radius_ + 0.5)
            iymin = math.floor(ycen - max_radius_ + 0.5)
            iymax = math.ceil(ycen + max_radius_ + 0.5)

            # OOM guard (same logic as _aperture_to_mask)
            bbox_ny = iymax - iymin
            bbox_nx = ixmax - ixmin
            if bbox_ny * bbox_nx > max_aper_size:
                args.append(None)
                continue

            # Clip to data boundaries
            data_ymin = max(0, iymin)
            data_ymax = min(data_shape[0], iymax)
            data_xmin = max(0, ixmin)
            data_xmax = min(data_shape[1], ixmax)
            if data_ymin >= data_ymax or data_xmin >= data_xmax:
                args.append(None)
                continue

            slc_lg = (slice(data_ymin, data_ymax),
                      slice(data_xmin, data_xmax))
            cutout_data = data_arr[slc_lg].astype(float) - bkg

            # Build data mask (non-finite + user mask)
            data_mask = ~np.isfinite(cutout_data)
            if mask_arr is not None:
                data_mask |= mask_arr[slc_lg]

            # Cutout centroid position
            cutout_xcen = xcen - data_xmin
            cutout_ycen = ycen - data_ymin

            # Handle neighboring sources
            if aperture_mask_method != 'none':
                seg_cut = segm_data[slc_lg]
                segm_mask = (seg_cut != label) & (seg_cut != 0)
                if aperture_mask_method == 'mask':
                    data_mask = data_mask | segm_mask
                elif aperture_mask_method == 'correct':
                    cutout_data = _mask_to_mirrored_value(
                        cutout_data, segm_mask,
                        (cutout_xcen, cutout_ycen), mask=data_mask)

            # Pre-zero masked pixels so the root-finding function can use a
            # simple sum without masking
            clean_data = cutout_data.copy()
            clean_data[data_mask] = 0.0

            # Pre-compute grid parameters for circular_overlap_grid
            ny, nx = clean_data.shape
            xmin_edge = -0.5 - cutout_xcen
            xmax_edge = nx - 0.5 - cutout_xcen
            ymin_edge = -0.5 - cutout_ycen
            ymax_edge = ny - 0.5 - cutout_ycen
            grid_params = (xmin_edge, xmax_edge, ymin_edge, ymax_edge,
                           nx, ny, use_exact, subpixels)

            args.append([clean_data, grid_params, kronflux, max_radius_])

        return args

    def _flux_radius(self, fraction, name, overwrite):
        """
        Compute the circular radius enclosing the specified fraction of
        the Kron flux for each source.
        """
        catalog = self._catalog
        if fraction <= 0 or fraction > 1:
            msg = 'fraction must be > 0 and <= 1'
            raise ValueError(msg)

        # Return cached result if available
        if fraction in catalog._flux_radius_cache:
            result = catalog._flux_radius_cache[fraction]
            if name is not None:
                catalog.add_property(name, result, overwrite=overwrite)
            return result

        args = catalog._flux_radius_optimizer_args
        if catalog.progress_bar:
            args = add_progress_bar(args, desc='flux_radius')

        radius = []
        for flux_radius_args in args:
            if flux_radius_args is None:
                radius.append(np.nan)
                continue

            clean_data, grid_params, kronflux, max_radius = flux_radius_args
            normflux = kronflux * fraction
            fcn_args = (clean_data, grid_params, normflux)

            # Try to find the root of self._flux_radius_fcn, which is bracketed
            # by a min and max radius. A ValueError is raised if the
            # bracket points do not have different signs, indicating no
            # solution or multiple solutions (e.g., a multi-valued
            # function). This can happen when at some radius, flux starts
            # decreasing with increasing radius (due to negative data
            # values), resulting in multiple possible solutions. If no
            # solution is found, we iteratively decrease the max radius to
            # narrow the bracket range until the root is found. If max
            # radius drops below the min radius (0.1), then no solution is
            # possible and NaN will be returned as the result.
            found = False
            min_radius = 0.1
            max_radius_delta = 0.1 * max_radius
            while max_radius > min_radius and found is False:
                try:
                    bracket = [min_radius, max_radius]
                    root_result = root_scalar(
                        self._flux_radius_fcn, args=fcn_args,
                        bracket=bracket, method='brentq')
                    result = root_result.root
                    found = True
                except ValueError:
                    # ValueError is raised if the bracket points do not have
                    # different signs
                    max_radius -= max_radius_delta

            # No solution found between min_radius and max_radius
            if found is False:
                result = np.nan

            radius.append(result)

        result = np.array(radius) << u.pix
        catalog._flux_radius_cache[fraction] = result

        if name is not None:
            catalog.add_property(name, result, overwrite=overwrite)

        return result
