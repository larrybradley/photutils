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


def labels_at(segm, sources):
    """
    Return the segment label at the center of each source.

    Parameters
    ----------
    segm : `~photutils.segmentation.SegmentationImage`
        The segmentation image.

    sources : list of tuple
        The ``(amplitude, x, y, sigma)`` of each source.

    Returns
    -------
    labels : list of int
        The labels.
    """
    return [int(segm.data[int(y), int(x)]) for _, x, y, _ in sources]


def as_dict(result):
    """
    Return the result table as a label to absorber dictionary.

    Parameters
    ----------
    result : `~astropy.table.QTable`
        The ``get_spurious_labels`` result.

    Returns
    -------
    mapping : dict
        The mapping of each spurious label to its absorber.
    """
    return {int(label): int(absorber)
            for label, absorber in zip(result['label'],
                                       result['absorbed_by'], strict=True)}


# A bright star with a faint blob inside its cleaning zone. The blob
# is well outside the star's isophote (its own segment) but the star's
# model wing at the blob exceeds the blob's n_pixels-th brightest
# pixel above the threshold.
STAR = (1000.0, 30.0, 70.0, 3.5)
NEAR_BLOB = (7.0, 30.0, 52.0, 2.5)

# A very faint blob within the near blob's cleaning zone and within
# the star's cleaning zone.
FAINT_BLOB = (5.3, 30.0, 40.0, 3.0)

# A faint blob far from everything.
FAR_BLOB = (7.0, 30.0, 12.0, 2.5)


@pytest.fixture
def star_scene():
    data = make_scene([STAR, NEAR_BLOB])
    segm = detect_sources(data, THRESHOLD, N_PIXELS)
    return data, segm


class TestGetSpuriousLabels:
    def test_near_blob_is_absorbed(self, star_scene):
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert isinstance(result, QTable)
        assert result.colnames == ['label', 'absorbed_by']
        assert as_dict(result) == {blob: star}

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

    def test_faint_neighbor_absorbs(self):
        # Without the star, the near blob survives and its own wing
        # absorbs the faint blob
        data = make_scene([NEAR_BLOB, FAINT_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        near, faint = labels_at(segm, [NEAR_BLOB, FAINT_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {faint: near}

    def test_absorber_is_a_survivor(self):
        # The near blob is absorbed by the star, so it cannot absorb
        # the faint blob. The star's wing reaches the faint blob, so
        # both blobs are assigned to the star.
        data = make_scene([STAR, NEAR_BLOB, FAINT_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        star, near, faint = labels_at(segm, [STAR, NEAR_BLOB, FAINT_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {near: star, faint: star}

    def test_absorbed_source_does_not_absorb(self):
        # A broad faint blob is absorbed by the star. A very faint
        # blob within the broad blob's reach but outside the star's
        # zone is not absorbed, because its only potential absorber
        # does not survive.
        broad = (5.5, 30.0, 40.0, 6.0)
        faint = (5.3, 30.0, 22.0, 3.0)

        # Without the star, the broad blob's wing reaches the faint
        # blob
        data = make_scene([broad, faint])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        broad_label, faint_label = labels_at(segm, [broad, faint])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {faint_label: broad_label}

        data = make_scene([STAR, broad, faint])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        star, broad_label, faint_label = labels_at(segm,
                                                   [STAR, broad, faint])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {broad_label: star}
        assert faint_label not in result['label']

    def test_label_order_independence(self):
        data = make_scene([STAR, NEAR_BLOB, FAINT_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        expected = as_dict(get_spurious_labels(data, segm, THRESHOLD,
                                               N_PIXELS))

        # Reverse the label order
        new_labels = segm.labels[::-1]
        reversed_data = np.zeros_like(segm.data)
        for old, new in zip(segm.labels, new_labels, strict=True):
            reversed_data[segm.data == old] = new
        reversed_segm = SegmentationImage(reversed_data)
        result = as_dict(get_spurious_labels(data, reversed_segm,
                                             THRESHOLD, N_PIXELS))
        mapping = dict(zip(segm.labels, new_labels, strict=True))
        assert result == {mapping[label]: mapping[absorber]
                          for label, absorber in expected.items()}

    def test_non_consecutive_labels(self, star_scene):
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        segm = segm.copy()
        segm.reassign_label(star, 17)
        segm.reassign_label(blob, 5)
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {5: 17}

    def test_result_sorted_by_label(self):
        data = make_scene([STAR, NEAR_BLOB, FAINT_BLOB])
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert_equal(result['label'], np.sort(result['label']))

    def test_threshold_array(self, star_scene):
        data, segm = star_scene
        expected = as_dict(get_spurious_labels(data, segm, THRESHOLD,
                                               N_PIXELS))
        threshold = np.full(data.shape, THRESHOLD)
        result = get_spurious_labels(data, segm, threshold, N_PIXELS)
        assert as_dict(result) == expected

    def test_pixel_counts_use_pixel_thresholds(self):
        # A star with a broad halo, whose area correction is below its
        # cap, and a faint blob at the edge of its reach. Raising the
        # threshold on one half of the star's segment and lowering it
        # on the other keeps the mean threshold, so the wing model
        # normalization and the blob's level are unchanged, but fewer
        # star pixels exceed their own thresholds. The smaller area
        # correction lowers the wing model and the blob survives.
        core = (40.0, 30.0, 70.0, 1.0)
        halo = (7.0, 30.0, 70.0, 12.0)
        blob = (6.0, 30.0, 24.0, 2.0)
        data = make_scene([core, halo, blob], shape=(120, 60))
        segm = detect_sources(data, THRESHOLD, N_PIXELS)
        star_label, blob_label = labels_at(segm, [core, blob])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == {blob_label: star_label}

        _, xx = np.mgrid[0:120, 0:60]
        left = (segm.data == star_label) & (xx < 30)
        right = (segm.data == star_label) & (xx >= 30)
        threshold = np.full(data.shape, THRESHOLD)
        threshold[left] += 4.5
        threshold[right] -= (4.5 * np.count_nonzero(left)
                             / np.count_nonzero(right))
        assert np.isclose(threshold[segm.data == star_label].mean(),
                          THRESHOLD)
        result = get_spurious_labels(data, segm, threshold, N_PIXELS)
        assert len(result) == 0

    def test_convolved_data(self):
        data = make_scene([STAR, NEAR_BLOB])
        kernel = make_2dgaussian_kernel(3.0, size=5)
        convolved_data = convolve(data, kernel)
        segm = detect_sources(convolved_data, THRESHOLD, N_PIXELS)
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     convolved_data=convolved_data)
        assert as_dict(result) == {blob: star}

    def test_clean_param(self, star_scene):
        # A very steep model wing lets the blob survive
        data, segm = star_scene
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     clean_param=0.05)
        assert len(result) == 0

    def test_quantity_inputs(self, star_scene):
        data, segm = star_scene
        expected = as_dict(get_spurious_labels(data, segm, THRESHOLD,
                                               N_PIXELS))
        unit = u.Jy
        result = get_spurious_labels(data << unit, segm, THRESHOLD << unit,
                                     N_PIXELS,
                                     convolved_data=data << unit)
        assert as_dict(result) == expected

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
        assert as_dict(result) == {blob: star}

    @pytest.mark.parametrize('value', [np.nan, np.inf])
    def test_non_finite_pixels_ignored(self, star_scene, value):
        # A non-finite pixel in a segment is excluded from every
        # measurement rather than poisoning it
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        expected = {blob: star}

        # The faintest pixel of the star segment
        star_pixels = np.flatnonzero(segm.data == star)
        faintest = star_pixels[np.argmin(data.ravel()[star_pixels])]
        bad_data = data.copy()
        bad_data.ravel()[faintest] = value
        result = get_spurious_labels(bad_data, segm, THRESHOLD, N_PIXELS)
        assert as_dict(result) == expected
        result = get_spurious_labels(data, segm, THRESHOLD, N_PIXELS,
                                     convolved_data=bad_data)
        assert as_dict(result) == expected

    def test_non_finite_pixels_not_counted(self, star_scene):
        # Non-finite pixels do not count toward the n_pixels brightest
        # pixels that set the comparison level
        data, segm = star_scene
        star, blob = labels_at(segm, [STAR, NEAR_BLOB])
        n_blob = np.count_nonzero(segm.data == blob)
        convolved_data = data.copy()
        blob_pixels = np.flatnonzero(segm.data == blob)
        convolved_data.ravel()[blob_pixels[:-2]] = np.nan
        # With only two finite pixels, the level is zero for
        # n_pixels=3 and the blob is absorbed
        result = get_spurious_labels(data, segm, THRESHOLD, 3,
                                     convolved_data=convolved_data)
        assert as_dict(result) == {blob: star}
        assert n_blob > 3


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

    @pytest.mark.parametrize('threshold', [0.0, -1.0, np.nan, np.inf])
    def test_invalid_threshold_values(self, star_scene, threshold):
        data, segm = star_scene
        match = 'threshold must be positive and finite'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, threshold, N_PIXELS)
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(data, segm, np.full(data.shape, threshold),
                                N_PIXELS)

    @pytest.mark.parametrize(
        ('data_unit', 'threshold_unit', 'convolved_unit'),
        [(u.Jy, u.m, u.Jy), (u.Jy, None, u.Jy), (u.Jy, u.Jy, None),
         (None, u.Jy, None)])
    def test_quantity_mismatch(self, star_scene, data_unit, threshold_unit,
                               convolved_unit):
        data, segm = star_scene

        def with_unit(value, unit):
            return value if unit is None else value << unit

        match = 'must all have the same units'
        with pytest.raises(ValueError, match=match):
            get_spurious_labels(with_unit(data, data_unit), segm,
                                with_unit(THRESHOLD, threshold_unit),
                                N_PIXELS,
                                convolved_data=with_unit(data,
                                                         convolved_unit))
