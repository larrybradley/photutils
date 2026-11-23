# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Pill-shaped (stadium) aperture in pixel coordinates.
"""

import math
from functools import cached_property

import numpy as np

from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
                                           ScalarAngleOrValue)
from photutils.aperture.core import (PixelAperture, _RotatableApertureMixin,
                                     _update_method_subpixels_docstring)
from photutils.geometry._pill_overlap import pill_overlap_grid

__all__ = ['PillAperture']


class PillAperture(_RotatableApertureMixin, PixelAperture):
    """
    A pill-shaped aperture defined in pixel coordinates.

    A pill, also called a stadium, is a rectangle with a semicircular
    cap on each short end. It is the region within a fixed distance of
    a line segment. The aperture has a single fixed size/shape, but it
    can have multiple positions (see the ``positions`` input).

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
        Convert the aperture to a sky aperture.

        This conversion is not implemented yet.

        Parameters
        ----------
        wcs : WCS object
            A world coordinate system (WCS) transformation.
        """
        msg = 'PillAperture.to_sky is not implemented yet'
        raise NotImplementedError(msg)
