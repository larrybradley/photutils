# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Private tools for rendering cutouts of 2D models.

The functions here evaluate a model on a grid of square cells, either
at the cell centers or as the mean over each cell using Gauss-Legendre
quadrature. The mean is refined around the source center, where the
profile of a typical source changes most rapidly, using pieces whose
boundaries are aligned to the source center.
"""

from functools import lru_cache
from math import ceil

import numpy as np

__all__ = []

# The largest number of elements in the arrays of quadrature node
# coordinates for which all of the nodes are evaluated in a single call
_MAX_QUADRATURE_ELEMENTS = 2**20

# The number of pieces per pixel along each axis into which the core
# cells are split. The core is the pixel that contains the source center
# and its eight neighbors.
_CORE_SPLIT = 8


@lru_cache
def pixel_nodes(n_nodes):
    """
    Return the Gauss-Legendre nodes and weights for the integral over a
    cell of unit width along one axis.

    Parameters
    ----------
    n_nodes : int
        The number of nodes.

    Returns
    -------
    offsets, weights : `~numpy.ndarray`
        The offsets of the nodes from the cell center and their weights,
        which sum to 1. The arrays are read-only because they are cached
        and shared.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    offsets = 0.5 * nodes
    weights = 0.5 * weights
    offsets.setflags(write=False)
    weights.setflags(write=False)
    return offsets, weights


def validate_n_nodes(n_nodes):
    """
    Validate the ``n_nodes`` keyword.

    Parameters
    ----------
    n_nodes : `None` or int
        The number of Gauss-Legendre nodes per axis.

    Returns
    -------
    n_nodes : `None` or int
        The validated value.

    Raises
    ------
    ValueError
        If ``n_nodes`` is not `None` or a positive integer.
    """
    if n_nodes is None:
        return None
    if (isinstance(n_nodes, bool)
            or not isinstance(n_nodes, (int, np.integer))
            or n_nodes < 1):
        msg = 'n_nodes must be None or a positive integer'
        raise ValueError(msg)
    return int(n_nodes)


def evaluate_model(model, x, y):
    """
    Evaluate a model on coordinate arrays of any number of dimensions.

    The model is evaluated on the flattened coordinates and the result
    is reshaped, so models that accept only 1D or 2D inputs also work.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The model to evaluate.

    x, y : `~numpy.ndarray`
        The coordinates. The arrays must have the same shape.

    Returns
    -------
    result : `~numpy.ndarray` or `~astropy.units.Quantity`
        The model values, with the shape of ``x``.
    """
    return model(x.ravel(), y.ravel()).reshape(x.shape)


def integrate_cells(model, x, y, cell_size, n_nodes):
    """
    Return the mean of a model over square cells.

    The mean is computed with ``n_nodes`` x ``n_nodes`` Gauss-Legendre
    nodes per cell. If the node array would be larger than 2**20
    elements, one row of nodes is evaluated at a time to bound the
    memory.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The model to integrate.

    x, y : `~numpy.ndarray`
        The cell centers. The arrays must have the same shape.

    cell_size : float
        The side length of each square cell.

    n_nodes : int
        The number of Gauss-Legendre nodes per axis.

    Returns
    -------
    result : `~numpy.ndarray` or `~astropy.units.Quantity`
        The mean of the model over each cell, with the shape of ``x``.
    """
    offsets, weights = pixel_nodes(n_nodes)
    offsets = offsets * cell_size
    weights_2d = np.outer(weights, weights)
    trailing = (1,) * x.ndim

    x_nodes = x + offsets.reshape((n_nodes, *trailing))

    # Small inputs are evaluated at all of the nodes in one call, which
    # is the fastest. Large inputs are evaluated one row of nodes at
    # a time so that the temporary arrays are n_nodes, and not
    # n_nodes**2, times the size of the input.
    n_elements = x_nodes.size * n_nodes
    step = n_nodes if n_elements <= _MAX_QUADRATURE_ELEMENTS else 1

    result = None
    for start in range(0, n_nodes, step):
        rows = slice(start, start + step)
        y_nodes = y + offsets[rows].reshape((-1, 1, *trailing))
        xx, yy = np.broadcast_arrays(x_nodes, y_nodes)
        weights_rows = weights_2d[rows].reshape((-1, n_nodes, *trailing))
        values = evaluate_model(model, xx, yy)
        term = np.sum(values * weights_rows, axis=(0, 1))
        result = term if result is None else result + term
    return result


def core_nodes(centers, center, cell_size, n_nodes):
    """
    Return the quadrature nodes and weights of the core cells along one
    axis.

    Each cell is cut at its edges and at every point ``center + k *
    step`` inside it, so the source center lies on a piece boundary.
    The ``step`` is ``cell_size / n_split``, where ``n_split`` is
    ``ceil(8 * cell_size)``. Each piece is integrated with ``n_nodes``
    Gauss-Legendre nodes over its own width. A cell has
    ``n_split`` or ``n_split + 1`` pieces. The arrays always hold
    ``n_split + 1`` pieces per cell, and a cell with ``n_split`` pieces
    has one extra piece of zero width and zero weight.

    Parameters
    ----------
    centers : 1D `~numpy.ndarray`
        The cell centers along the axis.

    center : float
        The source center along the axis.

    cell_size : float
        The side length of each square cell.

    n_nodes : int
        The number of Gauss-Legendre nodes per piece.

    Returns
    -------
    nodes, weights : 2D `~numpy.ndarray`
        The node coordinates and their weights, with one row per cell.
        The weights of each row sum to 1.
    """
    n_split = max(1, ceil(_CORE_SPLIT * cell_size))
    step = cell_size / n_split
    lower = centers[:, None] - 0.5 * cell_size
    upper = centers[:, None] + 0.5 * cell_size

    # The first point of the center-aligned lattice above the lower edge
    # of each cell, followed by the next n_split - 1 points. The last
    # point is beyond the upper edge if the lattice is aligned to the
    # cell edges, and it is then clipped to give a piece of zero width.
    first = center + (np.floor((lower - center) / step) + 1) * step
    cuts = np.clip(first + step * np.arange(n_split), lower, upper)
    edges = np.concatenate((lower, cuts, upper), axis=1)
    widths = np.diff(edges, axis=1)[..., None]
    middles = 0.5 * (edges[:, :-1] + edges[:, 1:])[..., None]

    # A zero-width piece lies on a cell edge, which can be the source
    # center. Its nodes have zero weight, so they are moved to the
    # middle of the first piece of the cell to avoid evaluating the model
    # where it may be infinite. The first piece always has positive
    # width, and the middle of a piece is never a piece boundary, so it
    # is never the source center.
    middles = np.where(widths == 0, middles[:, :1], middles)

    offsets, weights = pixel_nodes(n_nodes)
    n_cells = len(centers)
    nodes = (middles + widths * offsets).reshape(n_cells, -1)
    weights = (widths * weights / cell_size).reshape(n_cells, -1)
    return nodes, weights


def render_cells(model, x, y, x_0, y_0, *, cell_size=1.0, n_nodes):
    """
    Return the model values on a grid of square cells.

    If ``n_nodes`` is `None`, the model is evaluated at the cell
    centers. Otherwise, the mean over each cell is computed with
    Gauss-Legendre quadrature.

    The mean is refined for the core cells, whose centers lie in the
    pixel that contains the source center and its eight neighbors.
    Along each axis, each core cell is cut at its edges and at every
    point ``x_0 + k * step`` (``y_0 + k * step`` along y) inside it,
    where ``step`` is ``cell_size / ceil(8 * cell_size)``. This puts
    the source center on a piece boundary at any position. The pieces
    are at most 1/8 pixel wide and are narrower at the cell edges.
    Each piece is integrated with ``n_nodes`` x ``n_nodes`` nodes, and
    the cell mean is the area-weighted sum of the piece means. Core
    cells that are not in the grid are not refined.

    Parameters
    ----------
    model : 2D `astropy.modeling.Model`
        The model to render.

    x, y : 2D `~numpy.ndarray`
        The cell centers, in pixel units. The arrays must have the same
        shape.

    x_0, y_0 : float
        The source center, in the same coordinates as ``x`` and ``y``.

    cell_size : float, optional
        The side length of each square cell, in pixel units.

    n_nodes : `None` or int
        The number of Gauss-Legendre nodes per axis, or `None` to
        evaluate the model at the cell centers.

    Returns
    -------
    result : `~numpy.ndarray` or `~astropy.units.Quantity`
        The model values, with the shape of ``x``.
    """
    if n_nodes is None:
        return model(x, y)

    result = integrate_cells(model, x, y, cell_size, n_nodes)

    # Core cells: the pixel that contains the center and its neighbors
    cx = np.floor(x_0 + 0.5)
    cy = np.floor(y_0 + 0.5)
    in_core = (np.abs(x - cx) < 1.5) & (np.abs(y - cy) < 1.5)
    if not in_core.any():
        return result

    # The tensor product of the per-axis pieces gives the pieces of each
    # core cell
    x_nodes, x_weights = core_nodes(x[in_core], x_0, cell_size, n_nodes)
    y_nodes, y_weights = core_nodes(y[in_core], y_0, cell_size, n_nodes)
    xx, yy = np.broadcast_arrays(x_nodes[:, None, :], y_nodes[:, :, None])
    weights = y_weights[:, :, None] * x_weights[:, None, :]
    values = evaluate_model(model, xx, yy)
    result[in_core] = np.sum(values * weights, axis=(1, 2))
    return result
