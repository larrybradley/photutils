# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests that the morphology functions are thread-safe and return
identical results when run concurrently from multiple threads.

Both ``data_properties`` and ``gini`` are pure functions with no
module-level mutable state. Each call operates only on its inputs
and local variables, and ``data_properties`` constructs a new
``SourceCatalog`` per call.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from photutils.morphology import data_properties, gini
from photutils.utils.tests.thread_test_helpers import fail_on_warning

pytestmark = [pytest.mark.force_parallel_threads(8),
              pytest.mark.iterations(4)]

DATA_PROPERTIES_COLUMNS = ['x_centroid', 'y_centroid', 'semimajor_axis',
                           'semiminor_axis', 'orientation', 'area']


@pytest.fixture(name='shared_inputs', scope='module')
def fixture_shared_inputs():
    """
    Return a shared image, mask, and background for the concurrent
    calls.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    mask : 2D `~numpy.ndarray`
        A boolean mask.

    background : 2D `~numpy.ndarray`
        A background image.
    """
    rng = np.random.default_rng(0)
    data = rng.random((51, 51))
    data[20:30, 20:30] += 50.0
    mask = np.zeros(data.shape, dtype=bool)
    mask[10, 10] = True
    background = np.full(data.shape, 0.5)
    return data, mask, background


def _data_properties_values(data, mask, background):
    """
    Return the data_properties values that the tests compare.
    """
    props = data_properties(data, mask=mask, background=background)
    values = [getattr(props, name).value
              for name in DATA_PROPERTIES_COLUMNS
              if name not in ('x_centroid', 'y_centroid')]
    values += [props.x_centroid, props.y_centroid, props.background_sum]
    return values


def _gini(data, mask, _background):
    """
    Return the Gini coefficient of the data.
    """
    return gini(data, mask=mask)


TASKS = (_data_properties_values, _gini)


@pytest.fixture(name='expected', scope='module')
def fixture_expected(shared_inputs):
    """
    Return the serial results, keyed by task.
    """
    return {task: task(*shared_inputs) for task in TASKS}


@pytest.mark.parametrize('task', TASKS, ids=['data_properties', 'gini'])
def test_concurrent_calls(shared_inputs, expected, task):
    """
    Test that concurrent data_properties and gini calls on shared input
    arrays return results identical to a serial call.
    """
    with fail_on_warning():
        result = task(*shared_inputs)
    assert_array_equal(result, expected[task])


def test_mixed_concurrent_calls(shared_inputs, expected, thread_index,
                                iteration_index):
    """
    Test data_properties and gini running interleaved in several
    threads.

    Each thread changes the function it calls in every iteration.
    """
    task = TASKS[(thread_index + iteration_index) % len(TASKS)]
    with fail_on_warning():
        result = task(*shared_inputs)
    assert_array_equal(result, expected[task])
