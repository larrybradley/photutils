# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the scoring module.
"""

import numpy as np
import pytest
from astropy.table import QTable
from numpy.testing import assert_allclose, assert_array_equal

from photutils.segmentation import SegmentationImage
from photutils.simulation import completeness, match_segments, match_sources


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


class TestMatchSources:
    """
    Tests for ``match_sources``.
    """

    def test_blend(self):
        """
        Test that one detection between two inputs is claimed by the
        brighter input and the fainter input is blended.
        """
        sources = make_sources([10.0, 12.0], [10.0, 10.0], [10.0, 50.0])
        result = match_sources([11.0], [10.0], sources, 2.0)
        assert_array_equal(result['detected'], [False, True])
        assert_array_equal(result['blended'], [True, False])
        assert_array_equal(result['match_index'], [-1, 0])
        assert_allclose(result['separation'][1], 1.0)
        assert np.isnan(result['separation'][0])

    def test_nearest_unclaimed(self):
        """
        Test that the brighter input takes its nearest detection and the
        fainter input gets the other one if it is within range.
        """
        sources = make_sources([10.0, 12.0], [10.0, 10.0], [10.0, 50.0])
        x_det = [11.8, 10.9]
        y_det = [10.0, 10.0]
        result = match_sources(x_det, y_det, sources, 1.5)
        assert_array_equal(result['detected'], [True, True])
        assert_array_equal(result['match_index'], [1, 0])
        assert_allclose(result['separation'], [0.9, 0.2])

        # The remaining detection is out of range of the fainter input
        result = match_sources(x_det, y_det, sources, 0.5)
        assert_array_equal(result['detected'], [False, True])
        assert_array_equal(result['blended'], [False, False])

    def test_no_detection_in_range(self):
        """
        Test an input with no detection within ``max_separation``.
        """
        sources = make_sources([10.0], [10.0], [10.0])
        result = match_sources([20.0], [20.0], sources, 3.0)
        assert_array_equal(result['detected'], [False])
        assert_array_equal(result['blended'], [False])
        assert_array_equal(result['match_index'], [-1])
        assert np.isnan(result['separation'][0])

    def test_unmatched_detection(self):
        """
        Test that detections claimed by no input are listed in the table
        metadata.
        """
        sources = make_sources([10.0], [10.0], [10.0])
        result = match_sources([10.5, 30.0, 50.0], [10.0, 30.0, 50.0],
                               sources, 2.0)
        assert_array_equal(result.meta['unmatched_indices'], [1, 2])

    def test_priority_tie(self):
        """
        Test that the earlier row wins a tie in priority.
        """
        sources = make_sources([10.0, 12.0], [10.0, 10.0], [20.0, 20.0])
        result = match_sources([11.0], [10.0], sources, 2.0)
        assert_array_equal(result['detected'], [True, False])
        assert_array_equal(result['blended'], [False, True])

    def test_zero_detections(self):
        """
        Test an empty list of detections.
        """
        sources = make_sources([10.0, 12.0], [10.0, 10.0], [20.0, 20.0])
        result = match_sources([], [], sources, 2.0)
        assert not np.any(result['detected'])
        assert not np.any(result['blended'])
        assert_array_equal(result['match_index'], [-1, -1])
        assert len(result.meta['unmatched_indices']) == 0

    def test_zero_rows(self):
        """
        Test an input table with no rows.
        """
        sources = make_sources([], [], [])
        result = match_sources([1.0, 2.0], [1.0, 2.0], sources, 2.0)
        assert len(result) == 0
        assert_array_equal(result.meta['unmatched_indices'], [0, 1])

    def test_errors(self):
        """
        Test the input validation of ``match_sources``.
        """
        sources = make_sources([10.0], [10.0], [10.0])
        match = 'params_table has no "snr" column'
        with pytest.raises(ValueError, match=match):
            match_sources([1.0], [1.0], sources, 2.0, priority='snr')

        match = 'x and y must have the same shape'
        with pytest.raises(ValueError, match=match):
            match_sources([1.0, 2.0], [1.0], sources, 2.0)

        match = 'max_separation must be positive'
        with pytest.raises(ValueError, match=match):
            match_sources([1.0], [1.0], sources, 0.0)

    def test_same_as_match_segments(self, segm):
        """
        Test that segmentation and source-list scores agree when each
        segment contains one detection.
        """
        sources = make_sources([3.0, 5.2, 12.0, 8.0], [3.0, 4.8, 12.0, 17.0],
                               [10.0, 50.0, 20.0, 30.0])
        # One detection at the center of each segment
        x_det = [4.0, 12.0, 16.5]
        y_det = [4.0, 12.0, 3.5]
        by_segment = match_segments(segm, sources)
        by_source = match_sources(x_det, y_det, sources, 2.0)
        assert_array_equal(by_source['detected'], by_segment['detected'])
        assert_array_equal(by_source['blended'], by_segment['blended'])


def make_bin_sample(n_detected, n_sources, center):
    """
    Make detected flags and values for one bin.
    """
    detected = np.zeros(n_sources, dtype=bool)
    detected[:n_detected] = True
    return detected, np.full(n_sources, center)


class TestCompleteness:
    """
    Tests for ``completeness``.
    """

    @pytest.mark.parametrize(('n_detected', 'n_sources', 'lower', 'upper'),
                             [(81, 263, 0.2553, 0.3662),
                              (15, 148, 0.0624, 0.1605),
                              (0, 20, 0.0000, 0.1611),
                              (1, 29, 0.0061, 0.1718)])
    def test_wilson_published(self, n_detected, n_sources, lower, upper):
        """
        Test the 95% Wilson score intervals against Newcombe (1998,
        Statistics in Medicine, 17, 857).
        """
        detected, values = make_bin_sample(n_detected, n_sources, 1.5)
        result = completeness(detected, values, [1, 2], confidence=0.95)
        assert_allclose(result['lower'][0], lower, atol=5e-5)
        assert_allclose(result['upper'][0], upper, atol=5e-5)

    def test_default_confidence(self):
        """
        Test the default 68.3% intervals.
        """
        det1, val1 = make_bin_sample(3, 10, 1.5)
        det2, val2 = make_bin_sample(28, 28, 2.5)
        result = completeness(np.concatenate((det1, det2)),
                              np.concatenate((val1, val2)), [1, 2, 3])
        assert_allclose(result['lower'], [0.178759, 0.965474], atol=1e-6)
        assert_allclose(result['upper'], [0.457647, 1.0], atol=1e-6)

    def test_counts(self):
        """
        Test the bin edges, counts, and completeness, with the last bin
        closed as in `numpy.histogram`.
        """
        values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 0.5, 3.5]
        detected = [True, False, True, True, False, True, True, True]
        result = completeness(detected, values, [1, 2, 3])
        assert result.colnames == ['bin_min', 'bin_max', 'n_sources',
                                   'n_detected', 'completeness', 'lower',
                                   'upper']
        assert_allclose(result['bin_min'], [1, 2])
        assert_allclose(result['bin_max'], [2, 3])
        assert_array_equal(result['n_sources'], [2, 4])
        assert_array_equal(result['n_detected'], [1, 3])
        assert_allclose(result['completeness'], [0.5, 0.75])

    def test_empty_bin_nan(self):
        """
        Test that an empty bin has NaN completeness and bounds.
        """
        result = completeness([True, False], [1.5, 1.6], [1, 2, 3])
        assert_array_equal(result['n_sources'], [2, 0])
        assert np.isnan(result['completeness'][1])
        assert np.isnan(result['lower'][1])
        assert np.isnan(result['upper'][1])

    def test_nan_values_dropped(self):
        """
        Test that NaN values are not counted in any bin.
        """
        result = completeness([True, False, True], [1.5, np.nan, 1.6],
                              [1, 2])
        assert_array_equal(result['n_sources'], [2])
        assert_array_equal(result['n_detected'], [2])

    def test_errors(self):
        """
        Test the input validation of ``completeness``.
        """
        match = 'detected and values must have the same shape'
        with pytest.raises(ValueError, match=match):
            completeness([True, False], [1.0], [0, 2])

        match = 'confidence must be between 0 and 1'
        with pytest.raises(ValueError, match=match):
            completeness([True], [1.0], [0, 2], confidence=1.5)
