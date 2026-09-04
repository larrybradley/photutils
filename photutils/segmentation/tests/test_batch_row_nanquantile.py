# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch row quantile kernel.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.segmentation._batch_catalog import batch_row_nanquantile


@pytest.fixture
def values():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(300, 41))
    values[rng.uniform(size=values.shape) < 0.3] = np.nan
    values[5] = np.nan  # a row with no finite value
    values[7] = np.nan
    values[7, 0] = 1.5  # a row with one finite value
    return np.ascontiguousarray(values)


@pytest.mark.parametrize('quantile', [0.0, 0.25, 0.5, 0.9, 1.0])
def test_matches_nanquantile(values, quantile):
    result = batch_row_nanquantile(values, quantile=quantile)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanquantile(values, quantile, axis=1)
    assert_allclose(result, expected, equal_nan=True)
    assert np.isnan(result[5])
    assert result[7] == 1.5


@pytest.mark.parametrize('n_cols', [1, 2, 3, 4])
def test_small_rows(n_cols):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(50, n_cols))
    result = batch_row_nanquantile(values, quantile=0.5)
    assert_allclose(result, np.median(values, axis=1))


def test_input_not_modified():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(20, 9))
    original = values.copy()
    batch_row_nanquantile(values, quantile=0.25)
    assert_allclose(values, original)


def test_empty():
    values = np.empty((0, 5))
    assert batch_row_nanquantile(values, quantile=0.5).shape == (0,)


@pytest.mark.parametrize('quantile', [-0.1, 1.1, np.nan])
def test_invalid_quantile(quantile):
    values = np.zeros((2, 3))
    with pytest.raises(ValueError, match='quantile must be between'):
        batch_row_nanquantile(values, quantile=quantile)
