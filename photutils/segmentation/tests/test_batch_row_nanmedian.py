# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch row median kernel.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.segmentation._batch_catalog import batch_row_nanmedian


def test_matches_nanmedian():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(300, 41))
    values[rng.uniform(size=values.shape) < 0.3] = np.nan
    values[5] = np.nan  # a row with no finite value
    values[7] = np.nan
    values[7, 0] = 1.5  # a row with one finite value
    result = batch_row_nanmedian(np.ascontiguousarray(values))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanmedian(values, axis=1)
    assert_allclose(result, expected, equal_nan=True)
    assert np.isnan(result[5])
    assert result[7] == 1.5


@pytest.mark.parametrize('n_cols', [1, 2, 3, 4])
def test_small_rows(n_cols):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(50, n_cols))
    result = batch_row_nanmedian(np.ascontiguousarray(values))
    assert_allclose(result, np.median(values, axis=1))


def test_input_not_modified():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(20, 9))
    original = values.copy()
    batch_row_nanmedian(values)
    assert_allclose(values, original)


def test_empty():
    values = np.empty((0, 5))
    assert batch_row_nanmedian(values).shape == (0,)
