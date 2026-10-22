# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the scoring module.
"""

import numpy as np
import pytest
from astropy.table import QTable
from numpy.testing import assert_array_equal

from photutils.segmentation import SegmentationImage
from photutils.simulation import match_segments


@pytest.fixture(name='segm')
def fixture_segm():
    """
    Make a 20 x 20 segmentation image with three segments.
    """
    data = np.zeros((20, 20), dtype=int)
    data[2:7, 2:7] = 1
    data[10:15, 10:15] = 2
    data[2:6, 15:19] = 3
    return SegmentationImage(data)


def make_sources(x, y, flux, **columns):
    """
    Make a table of input sources.
    """
    table = QTable()
    table['id'] = np.arange(len(x)) + 1
    table['x_0'] = np.asarray(x, dtype=float)
    table['y_0'] = np.asarray(y, dtype=float)
    table['flux'] = np.asarray(flux, dtype=float)
    for name, values in columns.items():
        table[name] = values
    return table


class TestMatchSegments:
    """
    Tests for ``match_segments``.
    """

    def test_blend(self, segm):
        """
        Test that the brightest source in a segment is detected and the
        others are blended.
        """
        sources = make_sources([3.0, 5.2], [3.0, 4.8], [10.0, 50.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['label'], [1, 1])
        assert_array_equal(result['detected'], [False, True])
        assert_array_equal(result['blended'], [True, False])

    def test_source_without_segment(self, segm):
        """
        Test a source that lies on no segment.
        """
        sources = make_sources([12.0, 8.0], [12.0, 17.0], [10.0, 20.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['label'], [2, 0])
        assert_array_equal(result['detected'], [True, False])
        assert_array_equal(result['blended'], [False, False])

    def test_segment_without_source(self, segm):
        """
        Test that segments covering no input position are listed in the
        table metadata.
        """
        sources = make_sources([3.0, 12.0], [3.0, 12.0], [10.0, 20.0])
        result = match_segments(segm, sources)
        assert_array_equal(result.meta['unmatched_labels'], [3])

    def test_priority_tie(self, segm):
        """
        Test that the earlier row wins a tie in priority.
        """
        sources = make_sources([3.0, 5.0, 4.0], [3.0, 5.0, 4.0],
                               [20.0, 20.0, 20.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['detected'], [True, False, False])
        assert_array_equal(result['blended'], [False, True, True])

    def test_priority_column(self, segm):
        """
        Test that ``priority`` selects the column used to credit a
        segment.
        """
        sources = make_sources([3.0, 5.0], [3.0, 5.0], [100.0, 10.0],
                               snr=[2.0, 8.0])
        result = match_segments(segm, sources, priority='snr')
        assert_array_equal(result['detected'], [False, True])

        match = 'params_table has no "snr2" column'
        with pytest.raises(ValueError, match=match):
            match_segments(segm, sources, priority='snr2')

    def test_outside_image(self, segm):
        """
        Test that positions outside the image or nonfinite positions get
        label 0.
        """
        sources = make_sources([-3.0, 25.0, np.nan, 3.0],
                               [3.0, 3.0, 3.0, 20.6],
                               [10.0, 10.0, 10.0, 10.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['label'], [0, 0, 0, 0])
        assert not np.any(result['detected'])
        assert not np.any(result['blended'])

    def test_pixel_rounding(self, segm):
        """
        Test that a position belongs to the pixel whose center is
        nearest, with half-pixel positions rounding up.
        """
        # Label 1 covers pixels 2 to 6, so x = 6.49 is in it and 6.5 is
        # not, while x = 1.5 rounds up into pixel 2
        sources = make_sources([6.49, 6.5, 1.5], [4.0, 4.0, 4.0],
                               [1.0, 1.0, 1.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['label'], [1, 0, 1])

    def test_columns_and_names(self, segm):
        """
        Test the output columns and custom position column names.
        """
        sources = make_sources([3.0], [3.0], [10.0])
        sources.rename_column('x_0', 'xc')
        sources.rename_column('y_0', 'yc')
        result = match_segments(segm, sources, x_name='xc', y_name='yc')
        assert result.colnames == ['id', 'xc', 'yc', 'flux', 'label',
                                   'detected', 'blended']
        assert 'label' not in sources.colnames

    def test_zero_rows(self, segm):
        """
        Test an input table with no rows.
        """
        sources = make_sources([], [], [])
        result = match_segments(segm, sources)
        assert len(result) == 0
        assert_array_equal(result.meta['unmatched_labels'], [1, 2, 3])

    def test_empty_segmentation(self):
        """
        Test a segmentation image with no segments.
        """
        segm = SegmentationImage(np.zeros((20, 20), dtype=int))
        sources = make_sources([3.0, 12.0], [3.0, 12.0], [10.0, 20.0])
        result = match_segments(segm, sources)
        assert_array_equal(result['label'], [0, 0])
        assert not np.any(result['detected'])
        assert len(result.meta['unmatched_labels']) == 0
