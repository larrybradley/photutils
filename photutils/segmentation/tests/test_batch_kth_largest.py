# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the batch k-th largest value kernel.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from photutils.segmentation._batch_catalog import batch_kth_largest


def reference(values, offsets, counts, k):
    """
    Return the k-th largest value of each source with a NumPy sort.
    """
    result = np.full(len(counts), np.nan)
    for i, count in enumerate(counts):
        if count >= k:
            source = values[offsets[i]:offsets[i] + count]
            result[i] = np.sort(source)[::-1][k - 1]
    return result


@pytest.fixture
def packed():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 40, 200)
    offsets = np.zeros(len(counts) + 1, dtype=np.intp)
    # A source with no pixels occupies one NaN placeholder
    offsets[1:] = np.cumsum(np.maximum(counts, 1))
    values = rng.normal(size=offsets[-1])
    values[offsets[:-1][counts == 0]] = np.nan
    return values, offsets, counts.astype(np.intp)


@pytest.mark.parametrize('k', [1, 3, 5, 12])
def test_matches_sort(packed, k):
    values, offsets, counts = packed
    result = batch_kth_largest(values, offsets=offsets, counts=counts, k=k)
    expected = reference(values, offsets, counts, k)
    assert_allclose(result, expected, equal_nan=True)


def test_ties():
    values = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 2.0])
    offsets = np.array([0, 3, 6], dtype=np.intp)
    counts = np.array([3, 3], dtype=np.intp)
    result = batch_kth_largest(values, offsets=offsets, counts=counts, k=2)
    assert_allclose(result, [3.0, 2.0])


def test_too_few_pixels_is_nan():
    values = np.array([5.0, 4.0, np.nan])
    offsets = np.array([0, 2, 3], dtype=np.intp)
    counts = np.array([2, 0], dtype=np.intp)
    result = batch_kth_largest(values, offsets=offsets, counts=counts, k=3)
    assert np.all(np.isnan(result))
    result = batch_kth_largest(values, offsets=offsets, counts=counts, k=2)
    assert_allclose(result, [4.0, np.nan], equal_nan=True)


def test_invalid_inputs(packed):
    values, offsets, counts = packed
    with pytest.raises(ValueError, match='k must be a positive integer'):
        batch_kth_largest(values, offsets=offsets, counts=counts, k=0)
    with pytest.raises(ValueError, match='offsets'):
        batch_kth_largest(values, offsets=offsets[:-1], counts=counts, k=1)
    bad = counts.copy()
    bad[0] = len(values) + 1
    with pytest.raises(ValueError, match='offsets and counts'):
        batch_kth_largest(values, offsets=offsets, counts=bad, k=1)
