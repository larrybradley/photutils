# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Helpers for `~photutils.aperture.ApertureStats`.

`_ApertureCutoutBuilder` is held by the `ApertureStats` instance via
composition (a `@lazyproperty` accessor) and constructs the aperture-
weighted data, variance, mask, and weight cutouts on demand for an
arbitrary list of aperture masks.
"""

import numpy as np

__all__ = []


class _ApertureCutoutBuilder:
    """
    Build aperture-weighted cutouts for an `ApertureStats` instance.

    Parameters
    ----------
    stats : `~photutils.aperture.ApertureStats`
        The host stats instance.
    """

    def __init__(self, stats):
        self._stats = stats

    def build(self, aperture_masks):
        """
        Make aperture-weighted cutouts for the data and variance, and
        cutouts for the total mask and aperture mask weights.

        Parameters
        ----------
        aperture_masks : list of `ApertureMask`
            A list of `ApertureMask` objects.

        Returns
        -------
        cutouts : list of tuple
            A list of ``(data, variance, mask, weights, overlap)`` tuples
            for each source/aperture position.
        """
        stats = self._stats
        data_cutouts = []
        variance_cutouts = []
        mask_cutouts = []
        weight_cutouts = []
        overlaps = []

        for (data_cutout, apermask, slices) in zip(stats._data_cutouts,
                                                   aperture_masks,
                                                   stats._overlap_slices,
                                                   strict=True):

            slc_large, slc_small = slices
            if slc_large is None:  # aperture does not overlap the data
                overlap = False
                data_cutout = np.array([np.nan])
                variance_cutout = np.array([np.nan])
                mask_cutout = np.array([False])
                weight_cutout = np.array([np.nan])
            else:
                # Create a mask of non-finite ``data`` values combined
                # with the input ``mask`` array
                data_mask = ~np.isfinite(data_cutout)
                if stats._mask is not None:
                    data_mask |= stats._mask[slc_large]

                overlap = True
                aperweight_cutout = apermask.data[slc_small]
                weight_cutout = aperweight_cutout * ~data_mask

                # Apply the aperture mask; for "exact" and "subpixel"
                # this is an expanded boolean mask using the aperture
                # mask zero values
                mask_cutout = (aperweight_cutout == 0) | data_mask

                data_cutout = data_cutout.copy()
                if stats.sigma_clip is None:
                    # data_cutout will have zeros where mask_cutout is
                    # True
                    data_cutout *= ~mask_cutout
                else:
                    # To input a mask, SigmaClip needs a MaskedArray
                    data_cutout_ma = np.ma.masked_array(data_cutout,
                                                        mask=mask_cutout)
                    data_sigclip = stats.sigma_clip(data_cutout_ma)

                    # Define a mask of only the sigma-clipped pixels
                    sigclip_mask = data_sigclip.mask & ~mask_cutout
                    weight_cutout *= ~sigclip_mask

                    mask_cutout = data_sigclip.mask
                    data_cutout = data_sigclip.filled(0.0)

                # Need to apply the aperture weights
                data_cutout *= aperweight_cutout

                if stats._error is None:
                    variance_cutout = None
                else:
                    # Apply the exact weights and total mask;
                    # error_cutout will have zeros where mask_cutout is
                    # True
                    variance = stats._error[slc_large]**2
                    variance_cutout = (variance * aperweight_cutout
                                       * ~mask_cutout)

            data_cutouts.append(data_cutout)
            variance_cutouts.append(variance_cutout)
            mask_cutouts.append(mask_cutout)
            weight_cutouts.append(weight_cutout)
            overlaps.append(overlap)

        # Use zip (instead of np.transpose) because these may contain
        # arrays that have different shapes
        return list(zip(data_cutouts, variance_cutouts, mask_cutouts,
                        weight_cutouts, overlaps, strict=True))
