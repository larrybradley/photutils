# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared shape/morphology calculations from 2D-image central moments.

This module provides the `_ShapeProperties` helper class used by
`~photutils.segmentation.SourceCatalog` and
`~photutils.aperture.ApertureStats` to compute morphological
properties (covariance matrix, semi-axes, FWHM, orientation,
eccentricity, etc.) from a set of central image moments. The helper
is used via composition; the host classes own a `_ShapeProperties`
instance and delegate from their public properties.

All values returned by the helper are unitless (raw `~numpy.ndarray`).
The host class is responsible for applying any unit conventions.
"""

import warnings

import numpy as np
from astropy.utils import lazyproperty

__all__ = []


class _ShapeProperties:
    """
    Compute shape (morphology) properties from central image moments.

    This is a helper class used by `~photutils.segmentation.SourceCatalog`
    and `~photutils.aperture.ApertureStats` to share the morphology
    calculations. Each property is a unitless `~numpy.ndarray`. The
    host class is responsible for applying units (e.g., ``u.pix`` for
    semi-axes).

    Parameters
    ----------
    moments_central : `~numpy.ndarray`
        The central moments for each source. Either a 2D array of
        shape ``(>=3, >=3)`` (a single source) or a 3D array of shape
        ``(n, >=3, >=3)``. Only entries with row/column indices ``<=
        2`` are used.
    """

    def __init__(self, moments_central):
        moments = np.asarray(moments_central)
        if moments.ndim == 2:
            moments = moments[np.newaxis, :, :]
        if (moments.ndim != 3 or moments.shape[1] < 3
                or moments.shape[2] < 3):
            msg = ('moments_central must have shape (>=3, >=3) or '
                   '(n, >=3, >=3)')
            raise ValueError(msg)
        self._moments_central = moments

    @lazyproperty
    def n(self):
        """
        The number of sources.
        """
        return self._moments_central.shape[0]

    @lazyproperty
    def inertia_tensor(self):
        """
        The inertia tensor for rotation around the source center of
        mass.

        Shape: ``(n, 2, 2)``.
        """
        moments = self._moments_central
        mu_02 = moments[:, 0, 2]
        mu_11 = -moments[:, 1, 1]
        mu_20 = moments[:, 2, 0]
        tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
        return tensor.reshape((tensor.shape[0], 2, 2))

    @lazyproperty
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function with the
        same second-order moments as the source.

        Shape: ``(n, 2, 2)``.

        Implements the SourceExtractor prescription of incrementally
        increasing the diagonal elements by ``1/12`` for "infinitely"
        thin detections. Sources whose covariance determinant is
        negative (not positive semidefinite) are set to NaN.
        """
        moments = self._moments_central
        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu_norm = (moments
                       / moments[:, 0, 0][:, np.newaxis, np.newaxis])

        covar = np.array([mu_norm[:, 0, 2], mu_norm[:, 1, 1],
                          mu_norm[:, 1, 1],
                          mu_norm[:, 2, 0]]).swapaxes(0, 1)
        covar = covar.reshape((covar.shape[0], 2, 2))

        # Modify the covariance matrix in the case of "infinitely"
        # thin detections. This follows SourceExtractor's prescription
        # of incrementally increasing the diagonal elements by 1/12.
        delta = 1.0 / 12
        delta2 = delta**2
        # Ignore RuntimeWarning from NaN values in covar
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            covar_det = np.linalg.det(covar)

            # Covariance should be positive semidefinite
            idx = np.where(covar_det < 0)[0]
            covar[idx] = np.array([[np.nan, np.nan], [np.nan, np.nan]])

            idx = np.where(covar_det < delta2)[0]
            while idx.size > 0:
                covar[idx, 0, 0] += delta
                covar[idx, 1, 1] += delta
                covar_det = np.linalg.det(covar)
                idx = np.where(covar_det < delta2)[0]
        return covar

    @lazyproperty
    def covariance_eigvals(self):
        """
        The two eigenvalues of `covariance`, sorted in decreasing order.

        Shape: ``(n, 2)``.

        NaN values are returned for sources with non-finite or
        non-positive-semidefinite covariance matrices.
        """
        eigvals = np.empty((self.n, 2))
        eigvals.fill(np.nan)
        # np.linalg.eigvalsh requires finite input values
        idx = np.unique(np.where(np.isfinite(self.covariance))[0])
        eigvals[idx] = np.linalg.eigvalsh(self.covariance[idx])

        # Check for negative variance (just in case the covariance
        # matrix is not positive semidefinite)
        idx2 = np.unique(np.where(eigvals < 0)[0])
        eigvals[idx2] = (np.nan, np.nan)

        # Sort each eigenvalue pair in descending order
        # (eigvalsh returns values in ascending order)
        return np.fliplr(eigvals)

    @lazyproperty
    def semimajor_axis(self):
        """
        The 1-sigma standard deviation along the semimajor axis.
        """
        return np.sqrt(self.covariance_eigvals[:, 0])

    @lazyproperty
    def semiminor_axis(self):
        """
        The 1-sigma standard deviation along the semiminor axis.
        """
        return np.sqrt(self.covariance_eigvals[:, 1])

    @lazyproperty
    def fwhm(self):
        r"""
        The circularized FWHM of the equivalent 2D Gaussian.

        .. math::

           \mathrm{FWHM} = 2 \sqrt{\ln(2) \, (a^2 + b^2)}
        """
        return 2.0 * np.sqrt(np.log(2.0)
                             * (self.semimajor_axis**2
                                + self.semiminor_axis**2))

    @lazyproperty
    def orientation_radians(self):
        """
        The orientation angle in radians (range ``(-pi/2, pi/2]``).

        Host classes are expected to convert to degrees and apply any
        wrap convention (e.g., the segmentation catalog wraps to the
        ``[0, 360)`` range).
        """
        covar = self.covariance
        return 0.5 * np.arctan2(2.0 * covar[:, 0, 1],
                                (covar[:, 0, 0] - covar[:, 1, 1]))

    @lazyproperty
    def eccentricity(self):
        r"""
        The eccentricity of the equivalent 2D Gaussian.

        .. math::

            e = \sqrt{1 - \frac{b^2}{a^2}}
        """
        semimajor_var, semiminor_var = np.transpose(
            self.covariance_eigvals)
        return np.sqrt(1.0 - (semiminor_var / semimajor_var))

    @lazyproperty
    def elongation(self):
        """
        The ratio ``a / b`` of the semimajor and semiminor axes.
        """
        return self.semimajor_axis / self.semiminor_axis

    @lazyproperty
    def ellipticity(self):
        """
        ``1 - b/a`` (one minus the inverse of `elongation`).
        """
        return 1.0 - (self.semiminor_axis / self.semimajor_axis)

    @lazyproperty
    def covariance_xx(self):
        """
        The ``(0, 0)`` element of `covariance` (sigma_x**2).
        """
        return self.covariance[:, 0, 0]

    @lazyproperty
    def covariance_yy(self):
        """
        The ``(1, 1)`` element of `covariance` (sigma_y**2).
        """
        return self.covariance[:, 1, 1]

    @lazyproperty
    def covariance_xy(self):
        """
        The ``(0, 1)`` element of `covariance` (sigma_x * sigma_y).
        """
        return self.covariance[:, 0, 1]

    @lazyproperty
    def _orientation_trig(self):
        """
        ``(cos(theta), sin(theta))`` of the orientation angle.

        Cached so the ``ellipse_c*`` family avoids recomputing
        trigonometric functions.
        """
        return (np.cos(self.orientation_radians),
                np.sin(self.orientation_radians))

    @lazyproperty
    def ellipse_cxx(self):
        """
        Coefficient for ``x**2`` in the generalized ellipse equation.
        """
        cos_t, sin_t = self._orientation_trig
        return ((cos_t / self.semimajor_axis)**2
                + (sin_t / self.semiminor_axis)**2)

    @lazyproperty
    def ellipse_cyy(self):
        """
        Coefficient for ``y**2`` in the generalized ellipse equation.
        """
        cos_t, sin_t = self._orientation_trig
        return ((sin_t / self.semimajor_axis)**2
                + (cos_t / self.semiminor_axis)**2)

    @lazyproperty
    def ellipse_cxy(self):
        """
        Coefficient for ``x*y`` in the generalized ellipse equation.
        """
        cos_t, sin_t = self._orientation_trig
        return (2.0 * cos_t * sin_t
                * ((1.0 / self.semimajor_axis**2)
                   - (1.0 / self.semiminor_axis**2)))
