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

Geometry-bearing properties (``inertia_tensor``, ``semimajor_axis``,
``fwhm``, ``covariance_xx``, ``ellipse_cxx``, ...) return
`~astropy.units.Quantity` instances with appropriate ``u.pix`` powers.
``covariance`` is intentionally returned without units.  Dimensionless
ratios (``eccentricity``, ``elongation``, ``ellipticity``) and
``orientation_radians`` are returned as plain `~numpy.ndarray` so host
classes can apply their own angle-wrap conventions.
"""

import warnings

import astropy.units as u
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
        mass, in ``u.pix**2``.

        Shape: ``(n, 2, 2)``.
        """
        moments = self._moments_central
        mu_02 = moments[:, 0, 2]
        mu_11 = -moments[:, 1, 1]
        mu_20 = moments[:, 2, 0]
        tensor = np.array([mu_02, mu_11, mu_11, mu_20]).swapaxes(0, 1)
        return tensor.reshape((tensor.shape[0], 2, 2)) * u.pix**2

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
    def _covariance_eigvals_value(self):
        """
        Raw (unitless) covariance eigenvalues used internally to
        derive other shape properties.
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
    def covariance_eigvals(self):
        """
        The two eigenvalues of `covariance`, sorted in decreasing
        order, in ``u.pix**2``.

        Shape: ``(n, 2)``.

        NaN values are returned for sources with non-finite or
        non-positive-semidefinite covariance matrices.
        """
        return self._covariance_eigvals_value * u.pix**2

    @lazyproperty
    def _semimajor_axis_value(self):
        """Raw (unitless) ``semimajor_axis``."""
        return np.sqrt(self._covariance_eigvals_value[:, 0])

    @lazyproperty
    def _semiminor_axis_value(self):
        """Raw (unitless) ``semiminor_axis``."""
        return np.sqrt(self._covariance_eigvals_value[:, 1])

    @lazyproperty
    def semimajor_axis(self):
        """
        The 1-sigma standard deviation along the semimajor axis, in
        ``u.pix``.
        """
        return self._semimajor_axis_value * u.pix

    @lazyproperty
    def semiminor_axis(self):
        """
        The 1-sigma standard deviation along the semiminor axis, in
        ``u.pix``.
        """
        return self._semiminor_axis_value * u.pix

    @lazyproperty
    def fwhm(self):
        r"""
        The circularized FWHM of the equivalent 2D Gaussian, in
        ``u.pix``.

        .. math::

           \mathrm{FWHM} = 2 \sqrt{\ln(2) \, (a^2 + b^2)}
        """
        a = self._semimajor_axis_value
        b = self._semiminor_axis_value
        return 2.0 * np.sqrt(np.log(2.0) * (a**2 + b**2)) * u.pix

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
            self._covariance_eigvals_value)
        return np.sqrt(1.0 - (semiminor_var / semimajor_var))

    @lazyproperty
    def elongation(self):
        """
        The ratio ``a / b`` of the semimajor and semiminor axes.
        """
        return self._semimajor_axis_value / self._semiminor_axis_value

    @lazyproperty
    def ellipticity(self):
        """
        ``1 - b/a`` (one minus the inverse of `elongation`).
        """
        return 1.0 - (self._semiminor_axis_value
                      / self._semimajor_axis_value)

    @lazyproperty
    def covariance_xx(self):
        """
        The ``(0, 0)`` element of `covariance` (sigma_x**2), in
        ``u.pix**2``.
        """
        return self.covariance[:, 0, 0] * u.pix**2

    @lazyproperty
    def covariance_yy(self):
        """
        The ``(1, 1)`` element of `covariance` (sigma_y**2), in
        ``u.pix**2``.
        """
        return self.covariance[:, 1, 1] * u.pix**2

    @lazyproperty
    def covariance_xy(self):
        """
        The ``(0, 1)`` element of `covariance` (sigma_x * sigma_y),
        in ``u.pix**2``.
        """
        return self.covariance[:, 0, 1] * u.pix**2

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
        Coefficient for ``x**2`` in the generalized ellipse equation,
        in ``1 / u.pix**2``.
        """
        cos_t, sin_t = self._orientation_trig
        a = self._semimajor_axis_value
        b = self._semiminor_axis_value
        return ((cos_t / a)**2 + (sin_t / b)**2) / u.pix**2

    @lazyproperty
    def ellipse_cyy(self):
        """
        Coefficient for ``y**2`` in the generalized ellipse equation,
        in ``1 / u.pix**2``.
        """
        cos_t, sin_t = self._orientation_trig
        a = self._semimajor_axis_value
        b = self._semiminor_axis_value
        return ((sin_t / a)**2 + (cos_t / b)**2) / u.pix**2

    @lazyproperty
    def ellipse_cxy(self):
        """
        Coefficient for ``x*y`` in the generalized ellipse equation,
        in ``1 / u.pix**2``.
        """
        cos_t, sin_t = self._orientation_trig
        a = self._semimajor_axis_value
        b = self._semiminor_axis_value
        return (2.0 * cos_t * sin_t
                * ((1.0 / a**2) - (1.0 / b**2))) / u.pix**2
