# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the clean module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from numpy.testing import assert_equal

from photutils.segmentation import (SegmentationImage, detect_sources,
                                    get_spurious_labels,
                                    make_2dgaussian_kernel)

THRESHOLD = 5.0
N_PIXELS = 4


def make_scene(sources, shape=(100, 60)):
    """
    Return a noiseless image of Gaussian sources.

    Parameters
    ----------
    sources : list of tuple
        The ``(amplitude, x, y, sigma)`` of each source.

    shape : tuple of int, optional
        The image shape.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    data = np.zeros(shape)
    for amplitude, x, y, sigma in sources:
        data += Gaussian2D(amplitude, x, y, sigma, sigma)(xx, yy)
    return data


# A bright star with a faint blob inside its cleaning zone. The blob
# is well outside the star's isophote (its own segment) but the star's
# model wing at the blob exceeds the blob's n_pixels-th brightest
# pixel above the threshold.
STAR = (1000.0, 30.0, 70.0, 3.5)
NEAR_BLOB = (7.0, 30.0, 52.0, 2.5)

# A very faint blob within the near blob's cleaning zone. It is
# absorbed by the near blob, which is in turn absorbed by the star.
CHAIN_BLOB = (5.3, 30.0, 40.0, 3.0)

# A faint blob far from everything.
FAR_BLOB = (7.0, 30.0, 12.0, 2.5)


@pytest.fixture
def star_scene():
    data = make_scene([STAR, NEAR_BLOB])
    segm = detect_sources(data, THRESHOLD, N_PIXELS)
    return data, segm


def labels_at(segm, sources):
    """
    Return the segment label at the center of each source.
    """
    return [int(segm.data[int(y), int(x)]) for _, x, y, _ in sources]


class TestGetSpuriousLabels:
    def test_near_blob_is_absorbed(self, star_scene):
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert isinstance(result, QTable)
        assert result.colnames == ['label', 'absorbed_by']
        assert_equal(result['label'], [blob])
        assert_equal(result['absorbed_by'], [star])

    def test_far_blob_survives(self):
        data = make_scene([STAR, FAR_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert len(result) == 0
        assert result.colnames == ['label', 'absorbed_by']
        assert result['label'].dtype.kind == 'i'
        assert result['absorbed_by'].dtype.kind == 'i'

    def test_isolated_star(self):
        data = make_scene([STAR])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert len(result) == 0

    def test_absorbed_by_resolves_chain(self):
        data = make_scene([STAR, NEAR_BLOB, CHAIN_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        star, near, chain = labels_at(segm, [STAR, NEAR_BLOB, CHAIN_BLOB])
        # The raster label order matters for the sequential pass
        assert chain < near < star
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert_equal(result['label'], [chain, near])
        assert_equal(result['absorbed_by'], [star, star])

    def test_absorbed_source_does_not_absorb(self):
        # The star is labeled first and absorbs both blobs. The near
        # blob's wing would absorb the faint blob, but the near blob is
        # already absorbed when its own pairs are tested.
        star = (1000.0, 30.0, 20.0, 3.5)
        near = (7.0, 30.0, 38.0, 2.5)
        faint = (5.3, 30.0, 50.0, 3.0)
        data = make_scene([star, near, faint])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        assert labels_at(segm, [star, near, faint]) == [1, 2, 3]
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert_equal(result['label'], [2, 3])
        assert_equal(result['absorbed_by'], [1, 1])

    def test_result_sorted_by_label(self):
        data = make_scene([STAR, NEAR_BLOB, CHAIN_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert_equal(result['label'], np.sort(result['label']))

    def test_threshold_array(self, star_scene):
        data, segm = star_scene
        expected = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        threshold = np.full(data.shape, THRESHOLD)
        result = get_spurious_labels(data, segm, threshold, N_PIXELS)
        assert_equal(result['label'], expected['label'])
        assert_equal(result['absorbed_by'], expected['absorbed_by'])

    def test_convolved_data(self):
        data = make_scene([STAR, NEAR_BLOB])
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)
        segm = detect_sources(convolved_data, THRESHOLD, N_PIXELS)
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     convolved_data=convolved_data)
        assert_equal(result['label'], [blob])
        assert_equal(result['absorbed_by'], [star])

    def test_clean_param(self, star_scene):
        # A very steep model wing lets the blob survive
        data, segm = star_scene
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     clean_param=0.05)
        assert len(result) == 0

    def test_quantity_inputs(self, star_scene):
        data, segm = star_scene
        expected = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        unit = u.Jy
        result = get_spurious_labels(data << unit, segm, THRESHOLD << unit,
                                     N_PIXELS,
                                     convolved_data=data << unit)
        assert_equal(result['label'], expected['label'])
        assert_equal(result['absorbed_by'], expected['absorbed_by'])

    def test_quantity_mismatch(self, star_scene):
        data, segm = star_scene
        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data << u.Jy, segm, THRESHOLD << u.m,
                                N_PIXELS)
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data << u.Jy, segm, THRESHOLD, N_PIXELS)
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data << u.Jy, segm, THRESHOLD << u.Jy,
                                N_PIXELS, convolved_data=data)

    def test_nonpositive_flux_segment_survives(self, star_scene):
        # A segment whose moment-image flux is not positive has no
        # defined wing model and is never absorbed or an absorber
        data, segm = star_scene
        star, _ = labels_at(segm, [STAR, NEAR_BLOB])
        convolved_data = data.copy()
        convolved_data[segm.data == star] = -1.0
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     convolved_data=convolved_data)
        assert len(result) == 0

    def test_small_segment_mthresh(self, star_scene):
        # A segment with fewer than n_pixels pixels has a zero
        # comparison level and is absorbed by any positive wing
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, 1000)
        assert_equal(result['label'], [blob])
        assert_equal(result['absorbed_by'], [star])


class TestGetSpuriousLabelsInputs:
    def test_invalid_segmentation_image(self, star_scene):
        data, segm = star_scene
        match = 'segmentation_image must be a SegmentationImage'
        with pytest.raises(TypeError, match=match):
            get_spurious_labels(data, segm.data, THRESHOLD, N_PIXELS)

    def test_shape_mismatch(self, star_scene):
        data, segm = star_scene
        match = 'segmentation_image must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data[:-1], segm, THRESHOLD, N_PIXELS)

    def test_no_labels(self, star_scene):
        data, _ = star_scene
        segm = SegmentationImage(np.zeros(data.shape, dtype=int))
        match = 'segmentation_image must have at least one non-zero label'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)

    def test_convolved_data_shape(self, star_scene):
        data, segm = star_scene
        match = 'convolved_data must have the same shape as data'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                convolved_data=data[:, :-1])

    @pytest.mark.parametrize('n_pixels', [0, -1, 2.5])
    def test_invalid_n_pixels(self, star_scene, n_pixels):
        data, segm = star_scene
        match = 'n_pixels must be a positive integer'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, THRESHOLD, n_pixels)

    @pytest.mark.parametrize('clean_param', [0.0, -1.0, np.nan, np.inf])
    def test_invalid_clean_param(self, star_scene, clean_param):
        data, segm = star_scene
        match = 'clean_param must be a positive finite number'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                clean_param=clean_param)

    def test_threshold_shape(self, star_scene):
        data, segm = star_scene
        match = 'threshold must be a scalar or have the same shape as data'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, np.ones(3), N_PIXELS)

    @pytest.mark.parametrize('threshold', [0.0, -1.0])
    def test_nonpositive_threshold(self, star_scene, threshold):
        data, segm = star_scene
        match = 'threshold must be positive'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, threshold, N_PIXELS)
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, np.full(data.shape, threshold),
                                N_PIXELS)
