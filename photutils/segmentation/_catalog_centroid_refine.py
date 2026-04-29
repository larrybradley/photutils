# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Refined-centroid algorithms for `~photutils.segmentation.SourceCatalog`.

`_CentroidRefiner` is held by the `SourceCatalog` instance via
composition (a `@lazyproperty` accessor) and computes the windowed and
quadratic-fit centroids on demand.
"""

import math
import warnings

import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.utils import lazyproperty

from photutils.segmentation._moments import centroid_win_step
from photutils.segmentation.utils import _mask_to_mirrored_value
from photutils.utils._progress_bars import add_progress_bar

__all__ = []


class _CentroidRefiner:
    """
    Compute refined centroids for sources in a
    `~photutils.segmentation.SourceCatalog`.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        The host source catalog.
    """

    def __init__(self, catalog):
        self._catalog = catalog

    @lazyproperty
    def centroid_win(self):
        """
        Compute the windowed centroid for each source in a `SourceCatalog`.

        The window centroid is computed using an iterative algorithm
        (equivalent to `SourceExtractor`_'s XWIN/YWIN parameters).

        Returns
        -------
        centroid_win : 2D `~numpy.ndarray`
            The windowed ``(x, y)`` centroid of each source, with shape
            ``(n_labels, 2)``. Sources for which the iteration falls
            outside the 1-sigma ellipse, where the half-light radius is
            non-finite, or where the aperture is fully outside the data
            will use the isophotal centroid (or NaN if the half-light
            radius is non-finite).

        .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
        """
        catalog = self._catalog
        # Use .copy() to avoid mutating the cached flux_radius value
        radius_hl = catalog.flux_radius(0.5).value.copy()
        if catalog.isscalar:
            radius_hl = np.array([radius_hl])

        # Track which sources have non-finite half-light radii (e.g.,
        # due to NaN kron_radius). These sources cannot have a meaningful
        # windowed centroid.
        nan_hl = ~np.isfinite(radius_hl)

        # Apply a minimum half-light radius of 0.5 pixels (matching
        # SourceExtractor) for valid but very small values
        min_radius = 0.5
        small_mask = np.isfinite(radius_hl) & (radius_hl < min_radius)
        radius_hl[small_mask] = min_radius

        labels = catalog.labels
        if catalog.progress_bar:
            labels = add_progress_bar(labels, desc='centroid_win')

        # Pre-fetch arrays used in the inner loop
        data_arr = catalog._data
        mask_arr = catalog._mask
        segm_data = catalog._segmentation_image.data
        data_shape = data_arr.shape
        aperture_mask_method = catalog.aperture_mask_method
        do_correct = aperture_mask_method == 'correct'
        do_segm_mask = aperture_mask_method != 'none'
        max_aper_size = max(data_arr.size, 1_000_000)

        max_iters = 16
        centroid_threshold = 0.0001

        xcen_win = []
        ycen_win = []
        for label, xcen, ycen, rad_hl, nan_hl_ in zip(
                labels, catalog._x_centroid, catalog._y_centroid, radius_hl,
                nan_hl, strict=True):

            if nan_hl_ or math.isnan(xcen) or math.isnan(ycen):
                xcen_win.append(np.nan)
                ycen_win.append(np.nan)
                continue

            sigma = 2.0 * rad_hl * gaussian_fwhm_to_sigma
            inv_2sigma2 = -1.0 / (2.0 * sigma * sigma)
            radius = 4.0 * sigma
            radius_sq = radius * radius

            # Compute the full (unclipped) bounding box for the aperture
            # using the initial centroid. The radius is fixed, so the bbox
            # size stays the same across iterations even if the center
            # shifts slightly.
            bbox_halfsize = int(radius + 1.5)
            full_ny = full_nx = 2 * bbox_halfsize + 1

            # OOM guard
            if full_ny * full_nx > max_aper_size:
                xcen_win.append(np.nan)
                ycen_win.append(np.nan)
                continue

            # Cache for cutout data when the integer bbox doesn't change
            prev_ixcen = prev_iycen = None
            cached_data = None
            cached_mask_u8 = None

            iter_ = 0
            dcen = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                while iter_ < max_iters and dcen > centroid_threshold:
                    # Compute integer bounding box
                    ixmin = int(xcen + 0.5) - bbox_halfsize
                    ixmax = ixmin + full_nx
                    iymin = int(ycen + 0.5) - bbox_halfsize
                    iymax = iymin + full_ny

                    # Clip to data boundaries
                    slc_y = slice(max(0, iymin), min(data_shape[0], iymax))
                    slc_x = slice(max(0, ixmin), min(data_shape[1], ixmax))
                    if (slc_y.start >= slc_y.stop
                            or slc_x.start >= slc_x.stop):
                        xcen = np.nan
                        ycen = np.nan
                        break

                    cur_ixcen = int(xcen + 0.5)
                    cur_iycen = int(ycen + 0.5)

                    # Recompute cutout data only when the integer center
                    # changes to avoid redundant _mask_to_mirrored_value
                    # calls
                    if cur_ixcen != prev_ixcen or cur_iycen != prev_iycen:
                        prev_ixcen = cur_ixcen
                        prev_iycen = cur_iycen

                        data = data_arr[slc_y, slc_x].astype(float)
                        data_mask = ~np.isfinite(data)
                        if mask_arr is not None:
                            data_mask |= mask_arr[slc_y, slc_x]

                        cutout_xycen = (xcen - max(0, ixmin),
                                        ycen - max(0, iymin))

                        if do_segm_mask:
                            seg_cut = segm_data[slc_y, slc_x]
                            segm_mask = ((seg_cut != label)
                                         & (seg_cut != 0))
                            if aperture_mask_method == 'mask':
                                data_mask = data_mask | segm_mask

                        if do_correct:
                            data = _mask_to_mirrored_value(
                                data, segm_mask, cutout_xycen,
                                mask=data_mask)

                        cached_data = data
                        cached_mask_u8 = data_mask.view(np.uint8)

                    # Centroid position in cutout coordinates
                    cx = xcen - max(0, ixmin)
                    cy = ycen - max(0, iymin)

                    # Gaussian-weighted moments inside the binary disk,
                    # computed in C to avoid allocating per-iteration
                    # coord/weight arrays and a per-pixel ``np.exp`` call.
                    total, mx, my = centroid_win_step(
                        cached_data, cached_mask_u8, cx, cy,
                        radius_sq, inv_2sigma2)
                    if total == 0.0:
                        xcen = np.nan
                        ycen = np.nan
                        break
                    dx = mx / total
                    dy = my / total

                    dcen = math.sqrt(dx * dx + dy * dy)
                    xcen += dx * 2.0
                    ycen += dy * 2.0
                    iter_ += 1

            xcen_win.append(xcen)
            ycen_win.append(ycen)

        xcen_win = np.array(xcen_win)
        ycen_win = np.array(ycen_win)

        # Reset to the isophotal centroid if the windowed centroid is
        # outside the 1-sigma ellipse or if the iteration failed (NaN
        # from aperture off-image). Sources with NaN half-light radius
        # keep NaN (no valid window size).
        dx = catalog._x_centroid - xcen_win
        dy = catalog._y_centroid - ycen_win
        cxx = catalog.ellipse_cxx.value
        cxy = catalog.ellipse_cxy.value
        cyy = catalog.ellipse_cyy.value
        if catalog.isscalar:
            cxx = (cxx,)
            cxy = (cxy,)
            cyy = (cyy,)

        reset = ((cxx * dx**2 + cxy * dx * dy + cyy * dy**2) > 1)
        nan_cen = np.isnan(xcen_win) | np.isnan(ycen_win)
        reset |= nan_cen & ~nan_hl
        if np.any(reset):
            xcen_win[reset] = catalog._x_centroid[reset]
            ycen_win[reset] = catalog._y_centroid[reset]

        return np.transpose((xcen_win, ycen_win))

    @lazyproperty
    def cutout_centroid_quad(self):
        """
        Compute the quadratic-fit centroid (in cutout coordinates) for each
        source in a `SourceCatalog`.

        The centroid is computed by fitting a 2D quadratic polynomial to a
        3x3 box around the brightest pixel in each source's segment cutout.
        Sources for which the fit fails fall back to the isophotal cutout
        centroid.

        Returns
        -------
        centroid_quad : 2D `~numpy.ndarray`
            The quadratic ``(x, y)`` centroid in cutout coordinates, with
            shape ``(n_labels, 2)``.
        """
        catalog = self._catalog
        # Precompute the pseudo-inverse for the 3x3 relative coordinate
        # design matrix [1, x, y, xy, x^2, y^2]. This is constant for all
        # sources and avoids per-source lstsq calls.
        xi = np.arange(3)
        x, y = np.meshgrid(xi, xi)
        x = x.ravel()
        y = y.ravel()
        coeff_matrix = np.empty((9, 6), dtype=float)
        coeff_matrix[:, 0] = 1
        coeff_matrix[:, 1] = x
        coeff_matrix[:, 2] = y
        coeff_matrix[:, 3] = x * y
        coeff_matrix[:, 4] = x * x
        coeff_matrix[:, 5] = y * y
        pinv = np.linalg.pinv(coeff_matrix)

        _nan = np.nan
        centroid_quad = []

        cutouts = catalog._data_cutouts
        if catalog.progress_bar:
            cutouts = add_progress_bar(cutouts, desc='centroid_quad')

        for cutout, mask in zip(cutouts, catalog._cutout_total_masks,
                                strict=True):
            ny, nx = cutout.shape

            # Cutout must be at least 3x3 for the quadratic fit
            if ny < 3 or nx < 3:
                centroid_quad.append((_nan, _nan))
                continue

            # Apply mask: _cutout_total_masks already includes non-finite
            # data values, so cutout[mask] = 0.0 handles both masked pixels
            # and non-finite values.
            cutout = np.array(cutout, dtype=float)
            cutout[mask] = 0.0

            # Find peak pixel
            yidx, xidx = np.unravel_index(np.argmax(cutout), cutout.shape)

            # If peak at edge of cutout, return peak position
            if xidx == 0 or xidx == nx - 1 or yidx == 0 or yidx == ny - 1:
                centroid_quad.append((float(xidx), float(yidx)))
                continue

            # Extract 3x3 box centered on peak (guaranteed to fit since
            # peak is not at edge)
            xidx0 = xidx - 1
            yidx0 = yidx - 1
            cutout_flat = cutout[yidx0:yidx0 + 3, xidx0:xidx0 + 3].ravel()

            # Compute polynomial coefficients via precomputed pseudo-inverse
            c = pinv @ cutout_flat
            c10, c01, c11, c20, c02 = c[1], c[2], c[3], c[4], c[5]

            det = 4.0 * c20 * c02 - c11 * c11
            if det <= 0 or c20 > 0:
                centroid_quad.append((_nan, _nan))
                continue

            # Maximum in relative coords, then convert to cutout coords
            xm = (c01 * c11 - 2.0 * c02 * c10) / det + xidx0
            ym = (c10 * c11 - 2.0 * c20 * c01) / det + yidx0

            if 0.0 < xm < (nx - 1.0) and 0.0 < ym < (ny - 1.0):
                centroid_quad.append((xm, ym))
            else:
                centroid_quad.append((_nan, _nan))

        centroid_quad = np.array(centroid_quad)

        # Use the segment barycenter if fit returned NaN
        nan_mask = (np.isnan(centroid_quad[:, 0])
                    | np.isnan(centroid_quad[:, 1]))
        if np.any(nan_mask):
            centroid_quad[nan_mask] = catalog.cutout_centroid[nan_mask]

        return centroid_quad
