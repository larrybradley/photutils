# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Local-background helpers for `~photutils.segmentation.SourceCatalog`.

Provides module-level implementations of the rectangular-annulus local
background estimation for `SourceCatalog`. The class methods remain in
`photutils.segmentation.catalog` as thin delegations.
"""

import numpy as np
from astropy.stats import SigmaClip

from photutils.aperture import RectangularAnnulus
from photutils.background import SExtractorBackground

__all__ = []


def _local_background_apertures(catalog):
    """
    Return the rectangular-annulus apertures used to estimate the local
    background for each source.
    """
    if catalog.local_bkg_width == 0:
        return catalog._null_objects

    apertures = []
    for bbox_ in catalog._bbox:
        xpos = 0.5 * (bbox_.ixmin + bbox_.ixmax - 1)
        ypos = 0.5 * (bbox_.iymin + bbox_.iymax - 1)
        scale = 1.5
        width_in = (bbox_.ixmax - bbox_.ixmin) * scale
        width_out = width_in + 2 * catalog.local_bkg_width
        height_in = (bbox_.iymax - bbox_.iymin) * scale
        height_out = height_in + 2 * catalog.local_bkg_width
        apertures.append(RectangularAnnulus((xpos, ypos), width_in,
                                            width_out, height_out,
                                            h_in=height_in, theta=0.0))
    return apertures


def _local_background(catalog):
    """
    Return the per-source local background estimate as a unitless 1D
    array. Sources that are completely masked have NaN entries.
    """
    if catalog.local_bkg_width == 0:
        local_bkgs = np.zeros(catalog.n_labels)
    else:
        sigma_clip = SigmaClip(sigma=3.0, cenfunc='median', maxiters=20)
        bkg_func = SExtractorBackground(sigma_clip=sigma_clip)
        bkg_apers = catalog._local_background_apertures

        local_bkgs = []
        for aperture in bkg_apers:
            aperture_mask = aperture.to_mask(method='center')
            slc_lg, slc_sm = aperture_mask.get_overlap_slices(
                catalog._data.shape)

            data_cutout = catalog._data[slc_lg].astype(float, copy=True)
            # All non-zero segment labels are masked
            segm_mask_cutout = (
                catalog._segmentation_image.data[slc_lg].astype(bool))
            if catalog._mask is None:
                mask_cutout = None
            else:
                mask_cutout = catalog._mask[slc_lg]
            data_mask_cutout = catalog._make_cutout_data_mask(data_cutout,
                                                              mask_cutout)
            data_mask_cutout |= segm_mask_cutout

            aperweight_cutout = aperture_mask.data[slc_sm]
            good_mask = (aperweight_cutout > 0) & ~data_mask_cutout

            data_cutout *= aperweight_cutout
            data_values = data_cutout[good_mask]  # 1D array

            # Check not enough unmasked pixels
            if len(data_values) < 10:
                local_bkgs.append(0.0)
                continue
            local_bkgs.append(bkg_func(data_values))
        local_bkgs = np.array(local_bkgs)

    local_bkgs[catalog._all_masked] = np.nan
    return local_bkgs
