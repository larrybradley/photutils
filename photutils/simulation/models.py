# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Models for simulating astronomical sources.
"""

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.models import Sersic2D
from scipy.special import gamma, gammaincinv

__all__ = ['SersicSource']

# The smallest positive normal float32 value (~1.2e-38), used as the
# lower bound of the strictly-positive model parameters.
_FLOAT_TINY = float(np.finfo(np.float32).tiny)


class SersicSource(Fittable2DModel):
    r"""
    A 2D Sersic profile normalized to a total flux.

    This model is `~astropy.modeling.models.Sersic2D` parameterized by
    the total ``flux`` instead of the ``amplitude`` at the half-light
    radius. The total flux is the analytical integral of the profile
    over the entire 2D plane.

    The model is evaluated by sampling the profile at the input
    coordinates (point sampling). There is no pixel-integrated version.
    The profile is cuspy at the center for large ``n``, so a point
    sample can differ substantially from the flux in a pixel. To
    integrate the model over pixels, render it with
    `~photutils.simulation.make_model_image` using the ``n_nodes``
    keyword or with `~photutils.simulation.make_convolved_model_image`.

    Parameters
    ----------
    flux : float or `~astropy.units.Quantity`, optional
        The total integrated flux over the entire profile.

    x_0 : float, optional
        The x position of the center of the profile.

    y_0 : float, optional
        The y position of the center of the profile.

    r_eff : float, optional
        The half-light radius along the semimajor axis. Half of the
        total flux is enclosed within the ellipse with this semimajor
        axis.

    n : float, optional
        The Sersic index. ``n=1`` is an exponential profile and ``n=4``
        is a de Vaucouleurs profile.

    ellip : float, optional
        The ellipticity, defined as ``1 - b / a`` where ``a`` and ``b``
        are the semimajor and semiminor axes.

    theta : float or `~astropy.units.Quantity`, optional
        The counterclockwise rotation angle of the semimajor axis from
        the positive x axis, either as a float in radians or as an
        angular `~astropy.units.Quantity`.

    **kwargs : dict, optional
        Additional keyword arguments passed to the
        `~astropy.modeling.Fittable2DModel` base class.

    Notes
    -----
    The amplitude of the underlying `~astropy.modeling.models.Sersic2D`
    profile (the surface brightness at ``r_eff``) is

    .. math::
        I_e = \frac{F}{2 \pi n \Gamma(2n) e^{b_n} b_n^{-2n}
        r_{\rm eff}^2 (1 - \epsilon)}

    where :math:`F` is the total flux, :math:`\Gamma` is the gamma
    function, :math:`\epsilon` is the ellipticity, and :math:`b_n` is
    defined by :math:`\gamma(2n, b_n) = \Gamma(2n) / 2`, with
    :math:`\gamma` the lower incomplete gamma function.

    The profile has no finite extent, so flux beyond the rendered
    cutout is lost. The default bounding box is a square with a
    half-width of ``10 * r_eff``. The fraction of the total flux of a
    circular profile that lies outside the circle inscribed in the
    default bounding box is:

    ===  ==================
    n    Fraction lost
    ===  ==================
    1    0
    2    0.3%
    4    3.9%
    6    8.0%
    ===  ==================

    Use the ``bbox_factor`` keyword of
    `~photutils.simulation.make_model_image` or the ``factor`` keyword
    of ``bounding_box`` to change the cutout size.

    Examples
    --------
    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from photutils.simulation import SersicSource

        model = SersicSource(flux=100, x_0=25, y_0=25, r_eff=5, n=2,
                             ellip=0.4, theta=0.5)
        yy, xx = np.mgrid[0:51, 0:51]
        data = model(xx, yy)
        fig, ax = plt.subplots()
        ax.imshow(data, origin='lower')
    """

    flux = Parameter(
        default=1, description='Total integrated flux over the profile')
    x_0 = Parameter(
        default=0, description='Position of the center along the x-axis')
    y_0 = Parameter(
        default=0, description='Position of the center along the y-axis')
    r_eff = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='Half-light radius along the semimajor axis')
    n = Parameter(
        default=1,
        bounds=(_FLOAT_TINY, None),
        fixed=True,
        description='Sersic index')
    ellip = Parameter(
        default=0,
        bounds=(0, 1),
        fixed=True,
        description='Ellipticity, 1 - b / a')
    theta = Parameter(
        default=0.0,
        fixed=True,
        description=('CCW rotation angle of the semimajor axis either as '
                     'a float (in radians) or a Quantity angle'))

    def __init__(self, *, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, r_eff=r_eff.default, n=n.default,
                 ellip=ellip.default, theta=theta.default, **kwargs):
        super().__init__(flux=flux, x_0=x_0, y_0=y_0, r_eff=r_eff, n=n,
                         ellip=ellip, theta=theta, **kwargs)

    def bounding_box(self, factor=10):
        """
        Calculate a square bounding box centered on the profile.

        Parameters
        ----------
        factor : float, optional
            The half-width of the bounding box in units of ``r_eff``.

        Returns
        -------
        bbox : tuple
            A bounding box defining the ((y_min, y_max), (x_min, x_max))
            limits of the model.
        """
        half_width = factor * self.r_eff.value
        return ((self.y_0.value - half_width, self.y_0.value + half_width),
                (self.x_0.value - half_width, self.x_0.value + half_width))

    @staticmethod
    def evaluate(x, y, flux, x_0, y_0, r_eff, n, ellip, theta):
        """
        Calculate the value of the Sersic model at the input
        coordinates for the given model parameters.

        Parameters
        ----------
        x, y : float or array_like
            The x and y coordinates at which to evaluate the model.

        flux : float or `~astropy.units.Quantity`
            The total integrated flux over the entire profile.

        x_0, y_0 : float
            The center of the profile along the x- and y-axes.

        r_eff : float
            The half-light radius along the semimajor axis.

        n : float
            The Sersic index.

        ellip : float
            The ellipticity, ``1 - b / a``.

        theta : float or `~astropy.units.Quantity`
            The counterclockwise rotation angle of the semimajor axis,
            either as a float in radians or as an angular
            `~astropy.units.Quantity`.

        Returns
        -------
        result : `~numpy.ndarray`
            The value of the model evaluated at the input coordinates.
            The output has the units of ``flux``.
        """
        b_n = gammaincinv(2 * n, 0.5)
        amplitude = flux / (2 * np.pi * n * gamma(2 * n) * np.exp(b_n)
                            * b_n**(-2 * n) * r_eff**2 * (1 - ellip))
        return Sersic2D.evaluate(x, y, amplitude, r_eff, n, x_0, y_0, ellip,
                                 theta)
