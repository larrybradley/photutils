# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test that the Cython overlap functions are free-threading compatible and
return identical results when run concurrently from multiple threads.

This catches data races and shared-state issues introduced by ``with
nogil:`` sections in the functions.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.geometry import (circular_overlap_grid, elliptical_overlap_grid,
                                rectangular_overlap_grid)
from photutils.geometry._polygon_overlap import polygon_overlap_grid
from photutils.utils.tests.thread_test_helpers import fail_on_warning

N_THREADS = 8

pytestmark = pytest.mark.force_parallel_threads(N_THREADS)

GRID_SHAPES = ('circular', 'elliptical', 'rectangular', 'polygon')

_angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
_radii = 8.0 + 2.0 * np.sin(5.0 * _angles)
POLYGON_VX = np.ascontiguousarray(_radii * np.cos(_angles))
POLYGON_VY = np.ascontiguousarray(_radii * np.sin(_angles))


def _overlap_grid(shape, use_exact):
    """
    Return the overlap grid of an aperture of the given shape.
    """
    if shape == 'circular':
        return circular_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                     8.0, use_exact, 5)
    if shape == 'elliptical':
        return elliptical_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                       8.0, 5.0, 0.7, use_exact, 5)
    if shape == 'rectangular':
        return rectangular_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                        12.0, 7.0, 0.5, use_exact, 5)
    return polygon_overlap_grid(-15.0, 15.0, -15.0, 15.0, 200, 200,
                                POLYGON_VX, POLYGON_VY, use_exact, 8)


@pytest.fixture(name='expected_grids', scope='module')
def fixture_expected_grids():
    """
    Return the serial overlap grids, keyed by aperture shape and
    ``use_exact``.
    """
    return {(shape, use_exact): _overlap_grid(shape, use_exact)
            for shape in GRID_SHAPES for use_exact in (1, 0)}


@pytest.mark.iterations(4)
@pytest.mark.parametrize('use_exact', [1, 0])
@pytest.mark.parametrize('shape', GRID_SHAPES)
def test_overlap_grid_threadsafe(shape, use_exact, expected_grids):
    """
    Test that concurrent calls to each overlap function return results
    identical to a serial call.
    """
    with fail_on_warning():
        result = _overlap_grid(shape, use_exact)
    assert_array_equal(result, expected_grids[shape, use_exact])


@pytest.mark.iterations(8)
def test_mixed_functions_concurrent(expected_grids, thread_index,
                                    iteration_index):
    """
    Run all geometry function types concurrently from many threads.

    This is a more aggressive smoke test that mixes different functions
    so that if any of them shares mutable state (e.g., a module-level
    static buffer), interference between calls would surface. Each
    thread changes the function it calls in every iteration.
    """
    shape = GRID_SHAPES[(thread_index + iteration_index) % len(GRID_SHAPES)]
    with fail_on_warning():
        result = _overlap_grid(shape, 1)
    assert_array_equal(result, expected_grids[shape, 1])
