# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Enhanced ImagePSF that provides _LegacyEPSFModel compatibility.
"""

import warnings

import numpy as np
from astropy.utils.exceptions import AstropyUserWarning

from photutils.aperture import CircularAperture
from photutils.psf.image_models import ImagePSF


class CompatibleImagePSF(ImagePSF):
    """
    An ImagePSF subclass that provides _LegacyEPSFModel compatibility.

    This class provides the same API as _LegacyEPSFModel but uses ImagePSF
    as the underlying implementation. This allows us to remove _LegacyEPSFModel
    while maintaining backward compatibility.

    Parameters
    ----------
    All parameters are the same as ImagePSF, plus:

    norm_radius : float, optional
        The radius inside which the ePSF is normalized by the sum over
        undersampled integer pixel values inside a circular aperture.
        Default is 5.5.

    normalize : bool, optional
        Whether to normalize the PSF data. Default is True.

    normalization_correction : float, optional
        A correction factor for the normalization. Default is 1.0.
    """

    def __init__(self, data, *, norm_radius=5.5, normalize=True,
                 normalization_correction=1.0, **kwargs):
        # Initialize the parent ImagePSF
        super().__init__(data, **kwargs)

        # Store additional parameters
        self._norm_radius = norm_radius
        self._normalize = normalize
        self._normalization_correction = normalization_correction
        self._normalization_status = 0
        self._img_norm = None

        # Apply normalization if requested
        if normalize:
            self._compute_normalization()

    @property
    def norm_radius(self):
        """
        The normalization radius.
        """
        return self._norm_radius

    @property
    def shape(self):
        """
        The shape of the PSF data (for _LegacyEPSFModel compatibility).
        """
        return self.data.shape

    @property
    def _data(self):
        """
        The PSF data (for _LegacyEPSFModel compatibility).
        """
        return self.data

    def _compute_raw_image_norm(self):
        """
        Compute normalization based on aperture photometry within a
        given radius.
        """
        xypos = (self.data.shape[1] / 2.0, self.data.shape[0] / 2.0)
        # How to generalize "radius" if oversampling is
        # different along x/y axes (ellipse?)
        radius = self._norm_radius * self.oversampling[0]
        aper = CircularAperture(xypos, r=radius)
        flux, _ = aper.do_photometry(self.data, method='exact')
        return flux[0] / np.prod(self.oversampling)

    def _compute_normalization(self):
        """
        Helper function that computes (corrected) normalization factor
        of the original image data.

        For the ePSF this is defined as the sum over the inner N
        (default=5.5) pixels of the non-oversampled image. Will
        renormalize the data to the value calculated.
        """
        if self._img_norm is None:
            if np.sum(self.data) == 0:
                self._img_norm = 1
            else:
                self._img_norm = self._compute_raw_image_norm()

        if (self._img_norm != 0.0 and np.isfinite(self._img_norm)):
            # Create a new normalized data array
            norm_factor = self._img_norm * self._normalization_correction
            self.data = self.data / norm_factor
            self._normalization_status = 0
        else:
            self._normalization_status = 1
            self._img_norm = 1
            warnings.warn('Overflow encountered while computing '
                          'normalization constant. Normalization '
                          'constant will be set to 1.', AstropyUserWarning)

    def evaluate(self, x, y, flux, x_0, y_0):
        """
        Evaluate the model with _LegacyEPSFModel-compatible coordinate
        convention.

        The main difference is that _LegacyEPSFModel doesn't apply
        oversampling in the coordinate transformation, while ImagePSF
        does.
        """
        # Use the _LegacyEPSFModel coordinate convention
        xi = np.asarray(x) - x_0 + self._origin[0]
        yi = np.asarray(y) - y_0 + self._origin[1]

        evaluated_model = flux * self.interpolator.ev(xi, yi)

        if self.fill_value is not None:
            # find indices of pixels that are outside the input pixel
            # grid and set these pixels to the 'fill_value':
            nx, ny = self.data.shape[1], self.data.shape[0]
            invalid = (((xi < 0) | (xi > (nx - 1) / self.oversampling[1]))
                       | ((yi < 0) | (yi > (ny - 1) / self.oversampling[0])))

            # Handle both scalar and array cases
            if np.isscalar(evaluated_model):
                if invalid:
                    evaluated_model = self.fill_value
            else:
                evaluated_model[invalid] = self.fill_value

        return evaluated_model

    def copy(self):
        """
        Return a copy of this model.
        """
        # Create a copy using the parent method, but return our type
        newcls = object.__new__(self.__class__)

        for key, val in self.__dict__.items():
            if key not in ['_interpolator']:
                newcls.__dict__[key] = val

        # Copy model parameters
        for param_name in self.param_names:
            getattr(newcls, param_name).value = getattr(self, param_name).value

        return newcls
