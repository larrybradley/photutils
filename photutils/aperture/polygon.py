# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines circular and circular-annulus apertures in both
pixel and sky coordinates.
"""

import numpy as np
from astropy.utils import lazyproperty

from photutils.aperture.attributes import PixelPositions
#from photutils.aperture.attributes import (PixelPositions, PositiveScalar,
#                                           PositiveScalarAngle,
#                                           SkyCoordPositions)
from photutils.aperture.bounding_box import BoundingBox
from photutils.aperture.core import PixelAperture
# , SkyAperture
from photutils.aperture.mask import ApertureMask

# from photutils.geometry import circular_overlap_grid

__all__ = ['PolygonAperture']


def make_grid(polygon):
    from shapely.geometry import box

    xmin, ymin, xmax, ymax = polygon.bounds
    bbox = BoundingBox.from_float(xmin, xmax, ymin, ymax)
    #xmin = math.floor(xmin)
    #ymin = math.floor(ymin)
    #xmax = math.ceil(xmax)
    #ymax = math.ceil(ymax)
    xmin = bbox.ixmin
    xmax = bbox.ixmax
    ymin = bbox.iymin
    ymax = bbox.iymax

    cols = np.arange(xmin, xmax) - 0.5
    rows = np.arange(ymin, ymax) - 0.5

    poly_grid = []
    for y in rows:
        for x in cols:
            poly_grid.append(box(x, y, x + 1, y + 1))
    shape = (len(rows), len(cols))

    return poly_grid, shape, bbox


def make_mask(polygon, poly_grid, shape):
    import shapely
    areas = shapely.area(shapely.intersection(polygon, poly_grid))
    return areas.reshape(shape)


class PolygonAperture(PixelAperture):
    """
    A polygon aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    Parameters
    ----------
    positions : array_like
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

            * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
            * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs

    r : float
        The radius of the circle in pixels.

    Raises
    ------
    ValueError : `ValueError`
        If the input radius, ``r``, is negative.

    Examples
    --------
    >>> from photutils.aperture import CircularAperture
    >>> aper = CircularAperture([10.0, 20.0], 3.0)
    >>> aper = CircularAperture((10.0, 20.0), 3.0)

    >>> pos1 = (10.0, 20.0)  # (x, y)
    >>> pos2 = (30.0, 40.0)
    >>> pos3 = (50.0, 60.0)
    >>> aper = CircularAperture([pos1, pos2, pos3], 3.0)
    >>> aper = CircularAperture((pos1, pos2, pos3), 3.0)
    """

    _params = ('vertices')
    vertics = PixelPositions('The center pixel position(s).')

    def __init__(self, vertices):
        self.vertices = vertices

    @lazyproperty
    def _xy_extents(self):
        return 1, 1

    @lazyproperty
    def area(self):
        return 1

    def to_mask(self, method='exact', subpixels=5):

        use_exact, subpixels = self._translate_mask_mode(method, subpixels)

        if use_exact == 1:
            from shapely.geometry import Polygon

            # from shapely.strtree import STRtree

            poly = Polygon(self.vertices)
            poly_grid, grid_shape, bbox = make_grid(poly)

            # optimization to ignore polygons that don't intersect
            # TODO: figure out indexing to create ouput 2D mask array
            # tree = STRtree(poly_grid)
            # test_polygons = [poly]
            # idx = [tree.query(polygon) for polygon in test_polygons]
            # # this could be smaller list for concave polygons (or
            # # untrimmed grids)
            # intersecting_polys = tree.geometries[idx[0]]

            mask = make_mask(poly, poly_grid, grid_shape)

            return ApertureMask(mask, bbox)

    def _to_patch(self):
        pass

    def to_sky(self):
        pass

    def positions(self):
        pass
