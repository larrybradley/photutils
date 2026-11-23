# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pill-shaped (stadium) apertures in both pixel and sky coordinates.
"""

import math
from functools import cached_property

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from photutils.aperture._batch_photometry import SHAPE_PILL
from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           PositiveScalarAngle, ScalarAngle,
                                           ScalarAngleOrValue,
                                           SkyCoordPositions)
from photutils.aperture.core import (PixelAperture, SkyAperture,
                                     _enable_batch_photometry,
                                     _RotatableApertureMixin,
                                     _update_method_subpixels_docstring)
from photutils.geometry._pill_overlap import pill_overlap_grid
from photutils.utils._wcs_helpers import (pixel_shape_to_sky_svd,
                                          sky_shape_to_pixel_svd)

__all__ = ['PillAperture', 'SkyPillAperture']


@_enable_batch_photometry
class PillAperture(_RotatableApertureMixin, PixelAperture):
    """
    A pill-shaped aperture defined in pixel coordinates.

    A pill, also called a stadium, is a rectangle with a semicircular
    cap on each end. It is the region within a fixed distance of a line
    segment. The aperture has a single fixed size/shape, but it can
    have multiple positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    length : float
        The full length of the pill in pixels, measured from tip to tip
        and including both semicircular ends. For ``theta=0`` the long
        axis is along the x-axis. It must be greater than or equal to
        ``width``.

    width : float
        The full width of the pill in pixels, which is also the diameter
        of the semicircular ends. For ``theta=0`` the width is along the
        y-axis.

    theta : float or `~astropy.units.Quantity`, optional
        The rotation angle of the long axis as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        value in radians (as a float) from the positive x-axis. The
        rotation angle increases counterclockwise.

    Raises
    ------
    ValueError : `ValueError`
        If either ``length`` or ``width`` is not positive, or if
        ``length`` is less than ``width``.

    Examples
    --------
    >>> from astropy.coordinates import Angle
    >>> from photutils.aperture import PillAperture

    >>> theta = Angle(80, 'deg')
    >>> aper = PillAperture([10.0, 20.0], 8.0, 3.0)
    >>> aper = PillAperture((10.0, 20.0), 8.0, 3.0, theta=theta)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = PillAperture([pos1, pos2, pos3], 8.0, 3.0)
    >>> aper = PillAperture((pos1, pos2, pos3), 8.0, 3.0, theta=theta)
    """

    _params = ('positions', 'length', 'width', 'theta')
    _ordered_pairs_inclusive = (('width', 'length'),)
    positions = PixelPositions('The center pixel position(s).')
    length = PositiveScalar('The full length in pixels, tip to tip.')
    width = PositiveScalar('The full width in pixels.')
    theta = ScalarAngleOrValue('The counterclockwise rotation angle as an '
                               'angular Quantity or a value in radians from '
                               'the positive x-axis.')

    def __init__(self, positions, length, width, *, theta=0.0):
        self.positions = positions
        self.length = length
        self.width = width
        self.theta = theta

    @cached_property
    def _xy_extents(self):
        """
        The half-width and half-height of the bounding box of the
        pill.
        """
        half_straight = 0.5 * (self.length - self.width)
        radius = 0.5 * self.width
        theta_rad = self._theta_rad
        return (half_straight * abs(math.cos(theta_rad)) + radius,
                half_straight * abs(math.sin(theta_rad)) + radius)

    def _batch_shape_params(self):
        return SHAPE_PILL, (self.length, self.width, self._theta_rad)

    @cached_property
    def area(self):
        """
        The exact geometric area of the aperture shape.
        """
        radius = 0.5 * self.width
        return (self.length - self.width) * self.width + np.pi * radius**2

    def _to_patch(self, *, origin=(0, 0), **kwargs):
        """
        Return a `~matplotlib.patches.Patch` for the aperture.

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or list of \
                `~matplotlib.patches.Patch`
            A patch for the aperture. If the aperture is scalar then a
            single `~matplotlib.patches.Patch` is returned, otherwise a
            list of `~matplotlib.patches.Patch` is returned.
        """
        # matplotlib is an optional dependency
        import matplotlib.patches as mpatches
        from matplotlib.path import Path

        xy_positions, patch_kwargs = self._define_patch_params(origin=origin,
                                                               **kwargs)

        half_straight = 0.5 * (self.length - self.width)
        radius = 0.5 * self.width

        # The outline in the pill frame is the right cap, the top side,
        # the left cap, and the bottom side.
        right = Path.arc(-90.0, 90.0)
        left = Path.arc(90.0, 270.0)
        vertices = np.concatenate((right.vertices * radius
                                   + [half_straight, 0.0],
                                   left.vertices * radius
                                   + [-half_straight, 0.0],
                                   [[0.0, 0.0]]))
        codes = np.concatenate((right.codes, left.codes, [Path.CLOSEPOLY]))
        # The second arc starts with a straight side, not a new subpath.
        codes[len(right.codes)] = Path.LINETO

        theta_rad = self._theta_rad
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        rotation = np.array([[cos_theta, sin_theta],
                             [-sin_theta, cos_theta]])
        vertices = vertices @ rotation

        patches = [mpatches.PathPatch(Path(vertices + xy_position, codes),
                                      **patch_kwargs)
                   for xy_position in xy_positions]

        if self.isscalar:
            return patches[0]

        return patches

    @_update_method_subpixels_docstring
    def _compute_overlap(self, edges, nx, ny, use_exact, subpixels):
        """
        Compute the overlap of the aperture on the pixel grid.

        <compute_overlap_docs>
        """
        return pill_overlap_grid(edges[0], edges[1], edges[2], edges[3],
                                 nx, ny, self.length, self.width,
                                 self._theta_rad, use_exact, subpixels)

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyPillAperture` object defined in
        celestial coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `SkyPillAperture` object
            A `SkyPillAperture` object.

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.

        The converted length is raised to the converted width when the
        WCS would make it smaller. This can happen when the length is
        equal or close to the width and the WCS has unequal pixel
        scales or shear. The round trip between pixel and sky
        apertures is then not preserved.
        """
        xpos, ypos = np.transpose(self.positions)
        positions = wcs.pixel_to_world(xpos, ypos)

        first_pos = np.atleast_2d(self.positions)[0]
        pixcoord = (float(first_pos[0]), float(first_pos[1]))
        sky_length, sky_width, sky_angle = pixel_shape_to_sky_svd(
            wcs, pixcoord, self.length, self.width, self._theta_rad)

        length = Angle(max(sky_length, sky_width), 'arcsec')
        width = Angle(sky_width, 'arcsec')
        return SkyPillAperture(positions=positions, length=length,
                               width=width, theta=sky_angle)


class SkyPillAperture(_RotatableApertureMixin, SkyAperture):
    """
    A pill-shaped aperture defined in sky coordinates.

    A pill, also called a stadium, is a rectangle with a semicircular
    cap on each end. It is the region within a fixed distance of a line
    segment. The aperture has a single fixed size/shape, but it can
    have multiple positions (see the ``positions`` input).

    Parameters
    ----------
    positions : `~astropy.coordinates.SkyCoord`
        The celestial coordinates of the aperture center(s). This can be
        either scalar coordinates or an array of coordinates.

    length : scalar `~astropy.units.Quantity`
        The full length of the pill in angular units, measured from tip
        to tip and including both semicircular ends. For ``theta=0``
        the long axis is along the North-South axis. It must be greater
        than or equal to ``width``.

    width : scalar `~astropy.units.Quantity`
        The full width of the pill in angular units, which is also the
        diameter of the semicircular ends. For ``theta=0`` the width is
        along the East-West axis.

    theta : scalar `~astropy.units.Quantity`, optional
        The position angle (in angular units) of the pill "long axis",
        measured from North (PA=0) toward East. This is counterclockwise
        on an image displayed with North up and East to the left. North
        and East are the directions of increasing latitude and longitude
        in the coordinate frame of ``positions`` (e.g., ICRS or
        Galactic). When the aperture is converted to pixels with a WCS
        whose celestial frame differs from that frame, the angle is
        converted to the equivalent angle in the WCS frame, so the
        aperture keeps the same orientation on the sky.

    Raises
    ------
    TypeError : `TypeError`
        If either ``length`` or ``width`` is not a
        `~astropy.units.Quantity`.

    ValueError : `ValueError`
        If either ``length`` or ``width`` is not positive, or if
        ``length`` is less than ``width``.

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u
    >>> from photutils.aperture import SkyPillAperture
    >>> positions = SkyCoord(ra=[10.0, 20.0], dec=[30.0, 40.0], unit='deg')
    >>> aper = SkyPillAperture(positions, 1.0*u.arcsec, 0.5*u.arcsec)
    """

    _params = ('positions', 'length', 'width', 'theta')
    _ordered_pairs_inclusive = (('width', 'length'),)
    positions = SkyCoordPositions('The center position(s) in sky coordinates.')
    length = PositiveScalarAngle('The full length in angular units, tip to '
                                 'tip.')
    width = PositiveScalarAngle('The full width in angular units.')
    theta = ScalarAngle('The position angle (in angular units) of the '
                        'pill long axis.')

    def __init__(self, positions, length, width, *, theta=0.0 * u.deg):
        self.positions = positions
        self.length = length
        self.width = width
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to a `PillAperture` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation that
            supports the `astropy shared interface for WCS
            <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_
            (e.g., `astropy.wcs.WCS`, `gwcs.wcs.WCS`).

        Returns
        -------
        aperture : `PillAperture` object
            A `PillAperture` object.

        Notes
        -----
        The aperture shape parameters are converted using the local WCS
        properties (pixel scale, rotation angle) evaluated at the first
        aperture position. Because aperture objects require scalar shape
        parameters, only a single reference position is used for the
        conversion. For apertures with multiple positions used with a
        WCS that has spatially-varying distortions, this may produce
        inaccurate results for positions far from the first position.

        The converted length is raised to the converted width when the
        WCS would make it smaller. This can happen when the length is
        equal or close to the width and the WCS has unequal pixel
        scales or shear. The round trip between pixel and sky
        apertures is then not preserved.
        """
        xpos, ypos = wcs.world_to_pixel(self.positions)
        positions = np.transpose((xpos, ypos))

        skypos = self.positions if self.isscalar else self.positions[0]
        first_pixcoord = tuple(np.atleast_2d(positions)[0])
        _, pix_length, pix_width, pix_angle = sky_shape_to_pixel_svd(
            wcs, skypos,
            self.length.to_value(u.arcsec),
            self.width.to_value(u.arcsec),
            self._theta_rad, pixcoord=first_pixcoord)

        return PillAperture(positions=positions,
                            length=max(pix_length, pix_width),
                            width=pix_width, theta=pix_angle)
