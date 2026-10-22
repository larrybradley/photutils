# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Private tools for rendering cutouts of 2D models.

The functions here evaluate a model on a grid of square cells, either
at the cell centers or as the mean over each cell using Gauss-Legendre
quadrature. The mean is refined around the source center, where the
profile of a typical source changes most rapidly.
"""

from functools import lru_cache
from math import ceil

import numpy as np

__all__ = []

# The largest number of elements in the arrays of quadrature node
# coordinates for which all of the nodes are evaluated in a single call
_MAX_QUADRATURE_ELEMENTS = 2**20

# The number of pieces per axis into which each whole-pixel core cell is
# split. The core is the 3 x 3 pixels around the pixel that contains the
# source center.
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
        term = np.sum(model(xx, yy) * weights_rows, axis=(0, 1))
        result = term if result is None else result + term
    return result


def render_cells(model, x, y, x_0, y_0, *, cell_size=1.0, n_nodes):
    """
    Return the model values on a grid of square cells.

    If ``n_nodes`` is `None`, the model is evaluated at the cell
    centers. Otherwise, the mean over each cell is computed with
    Gauss-Legendre quadrature. The cells whose centers lie in the 3 x 3
    pixels around the pixel that contains the source center are each
    split into ``ceil(8 * cell_size)`` pieces per axis, and the mean
    of the piece means replaces the cell value. This is 8 x 8 pieces
    per pixel when ``cell_size`` is 1. Core cells that are not in the
    grid are not refined.

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

    # Core cells: the 3 x 3 pixels around the pixel containing the center
    cx = np.floor(x_0 + 0.5)
    cy = np.floor(y_0 + 0.5)
    in_core = (np.abs(x - cx) < 1.5) & (np.abs(y - cy) < 1.5)
    if not in_core.any():
        return result

    # Split each core cell into n_split x n_split pieces, integrate
    # each piece, and average the pieces
    n_split = max(1, ceil(_CORE_SPLIT * cell_size))
    piece_offsets = ((np.arange(n_split) + 0.5) / n_split - 0.5) * cell_size
    x_pieces = x[in_core][:, None, None] + piece_offsets[None, None, :]
    y_pieces = y[in_core][:, None, None] + piece_offsets[None, :, None]
    x_pieces, y_pieces = np.broadcast_arrays(x_pieces, y_pieces)
    pieces = integrate_cells(model, x_pieces, y_pieces, cell_size / n_split,
                             n_nodes)
    result[in_core] = pieces.mean(axis=(1, 2))
    return result
