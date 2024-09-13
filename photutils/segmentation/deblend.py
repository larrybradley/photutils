# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for deblending overlapping sources labeled in
a segmentation image.
"""

import warnings
from multiprocessing import cpu_count

import numpy as np
from astropy.units import Quantity
from astropy.utils import lazyproperty
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels

from photutils.segmentation.core import SegmentationImage
from photutils.segmentation.detect import _detect_sources
from photutils.segmentation.utils import _make_binary_structure
from photutils.utils._progress_bars import add_progress_bar
from photutils.utils._stats import nanmax, nanmin, nansum

__all__ = ['deblend_sources']


def deblend_sources(data, segment_img, npixels, *, labels=None, nlevels=32,
                    contrast=0.001, mode='exponential', connectivity=8,
                    relabel=True, nproc=1, progress_bar=True):
    """
    Deblend overlapping sources labeled in a segmentation image.

    Sources are deblended using a combination of multi-thresholding and
    `watershed segmentation
    <https://en.wikipedia.org/wiki/Watershed_(image_processing)>`_. In
    order to deblend sources, there must be a saddle between them.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The 2D array of the image. If filtering is desired, please input
        a convolved image here. This array should be the same array used
        in `~photutils.segmentation.detect_sources`.

    segment_img : `~photutils.segmentation.SegmentationImage`
        The segmentation image to deblend.

    npixels : int
        The minimum number of connected pixels, each greater than
        ``threshold``, that an object must have to be deblended.
        ``npixels`` must be a positive integer.

    labels : int or array_like of int, optional
        The label numbers to deblend. If `None` (default), then all
        labels in the segmentation image will be deblended.

    nlevels : int, optional
        The number of multi-thresholding levels to use for deblending.
        Each source will be re-thresholded at ``nlevels`` levels spaced
        between its minimum and maximum values (non-inclusive). The
        ``mode`` keyword determines how the levels are spaced.

    contrast : float, optional
        The fraction of the total source flux that a local peak must
        have (at any one of the multi-thresholds) to be deblended
        as a separate object. ``contrast`` must be between 0 and 1,
        inclusive. If ``contrast=0`` then every local peak will be made
        a separate object (maximum deblending). If ``contrast=1`` then
        no deblending will occur. The default is 0.001, which will
        deblend sources with a 7.5 magnitude difference.

    mode : {'exponential', 'linear', 'sinh'}, optional
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword) during
        deblending. The ``'exponential'`` and ``'sinh'`` modes have
        more threshold levels near the source minimum and less near
        the source maximum. The ``'linear'`` mode evenly spaces the
        threshold levels between the source minimum and maximum.
        The ``'exponential'`` and ``'sinh'`` modes differ in that
        the ``'exponential'`` levels are dependent on the source
        maximum/minimum ratio (smaller ratios are more linear; larger
        ratios are more exponential), while the ``'sinh'`` levels
        are not. Also, the ``'exponential'`` mode will be changed to
        ``'linear'`` for sources with non-positive minimum data values.

    connectivity : {8, 4}, optional
        The type of pixel connectivity used in determining how pixels
        are grouped into a detected source. The options are 8 (default)
        or 4. 8-connected pixels touch along their edges or corners.
        4-connected pixels touch along their edges. The ``connectivity``
        must be the same as that used to create the input segmentation
        image.

    relabel : bool, optional
        If `True` (default), then the segmentation image will be
        relabeled such that the labels are in consecutive order starting
        from 1.

    nproc : int, optional
        The number of processes to use for multiprocessing (if larger
        than 1). If set to 1, then a serial implementation is used
        instead of a parallel one. If `None`, then the number of
        processes will be set to the number of CPUs detected on the
        machine. Please note that due to overheads, multiprocessing may
        be slower than serial processing. This is especially true if one
        only has a small number of sources to deblend. The benefits of
        multiprocessing require ~1000 or more sources to deblend, with
        larger gains as the number of sources increase.

    progress_bar : bool, optional
        Whether to display a progress bar. Note that if multiprocessing
        is used (``nproc > 1``), the estimation times (e.g., time per
        iteration and time remaining, etc) may be unreliable. The
        progress bar requires that the `tqdm <https://tqdm.github.io/>`_
        optional dependency be installed. Note that the progress
        bar does not currently work in the Jupyter console due to
        limitations in ``tqdm``.

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :class:`photutils.segmentation.SourceFinder`
    """
    if isinstance(data, Quantity):
        data = data.value

    if not isinstance(segment_img, SegmentationImage):
        raise TypeError('segment_img must be a SegmentationImage')

    if segment_img.shape != data.shape:
        raise ValueError('The data and segmentation image must have '
                         'the same shape')

    if nlevels < 1:
        raise ValueError('nlevels must be >= 1')
    if contrast < 0 or contrast > 1:
        raise ValueError('contrast must be >= 0 and <= 1')

    if contrast == 1:  # no deblending
        return segment_img.copy()

    if mode not in ('exponential', 'linear', 'sinh'):
        raise ValueError('mode must be "exponential", "linear", or "sinh"')

    if labels is None:
        labels = segment_img.labels
    else:
        labels = np.atleast_1d(labels)
        segment_img.check_labels(labels)

    # include only sources that have at least (2 * npixels);
    # this is required for a source to be deblended into multiple
    # sources, each with a minimum of npixels
    mask = (segment_img.areas[segment_img.get_indices(labels)]
            >= (npixels * 2))
    labels = labels[mask]

    footprint = _make_binary_structure(data.ndim, connectivity)

    if nproc is None:
        nproc = cpu_count()  # pragma: no cover

    segm_deblended = segment_img.data.copy()

    indices = segment_img.get_indices(labels)
    if progress_bar:
        desc = 'Deblending'
        indices = add_progress_bar(indices, desc=desc)  # pragma: no cover

    max_label = segment_img.max_label + 1
    for label, idx in zip(labels, indices, strict=True):
        if progress_bar:
            indices.set_postfix_str(f'ID: {label}')
        source_slice = segment_img.slices[idx]
        source_data = data[source_slice]
        source_segment = segment_img.data[source_slice]
        source_deblended, warnings = _deblend_source(source_data,
                                                     source_segment,
                                                     label, npixels,
                                                     footprint, nlevels,
                                                     contrast, mode)

        if source_deblended is not None:
            source_mask = source_deblended > 0
            segm_deblended[source_slice][source_mask] = (
                source_deblended[source_mask] + max_label)
            nlabels = len(_get_labels(source_deblended))
            max_label += nlabels

    if relabel:
        segm_deblended = _relabel_array(segm_deblended, start_label=1)

    segm_img = object.__new__(SegmentationImage)
    segm_img._data = segm_deblended

    # TODO:
    #   - warnings

    return segm_img


def _deblend_source(data, segment_data, label, npixels, footprint, nlevels,
                    contrast, mode):
    """
    Convenience function to deblend a single labeled source.
    """
    deblender = _SingleSourceDeblender(data, segment_data, label, npixels,
                                       footprint, nlevels, contrast, mode)
    return deblender.deblend_source(), deblender.warnings


class _SingleSourceDeblender:
    """
    Class to deblend a single labeled source.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The cutout data array for a single source. ``data`` should
        also already be smoothed by the same filter used in
        :func:`~photutils.segmentation.detect_sources`, if applicable.

    segment_data : 2D int `~numpy.ndarray`
        The cutout segmentation image for a single source. Must have the
        same shape as ``data``.

    label : int
        The label of the source to deblend. This is needed because there
        may be more than one source label within the cutout.

    npixels : int
        The number of connected pixels, each greater than ``threshold``,
        that an object must have to be detected. ``npixels`` must be a
        positive integer.

    nlevels : int
        The number of multi-thresholding levels to use. Each source
        will be re-thresholded at ``nlevels`` levels spaced between its
        minimum and maximum values within the source segment. See the
        ``mode`` keyword for how the levels are spaced.

    contrast : float
        The fraction of the total (blended) source flux that a local
        peak must have (at any one of the multi-thresholds) to be
        considered as a separate object. ``contrast`` must be between 0
        and 1, inclusive. If ``contrast = 0`` then every local peak will
        be made a separate object (maximum deblending). If ``contrast =
        1`` then no deblending will occur. The default is 0.001, which
        will deblend sources with a 7.5 magnitude difference.

    mode : {'exponential', 'linear', 'sinh'}
        The mode used in defining the spacing between the
        multi-thresholding levels (see the ``nlevels`` keyword).

    Returns
    -------
    segment_image : `~photutils.segmentation.SegmentationImage`
        A segmentation image, with the same shape as ``data``, where
        sources are marked by different positive integer values. A value
        of zero is reserved for the background. Note that the returned
        `SegmentationImage` will have consecutive labels starting with
        1.
    """

    def __init__(self, data, segment_data, label, npixels, footprint, nlevels,
                 contrast, mode):

        self.data = data
        self.segment_data = segment_data
        self.label = label
        self.npixels = npixels
        self.footprint = footprint
        self.nlevels = nlevels
        self.contrast = contrast
        self.mode = mode

        self.segment_mask = segment_data == label
        data_values = data[self.segment_mask]
        self.source_min = nanmin(data_values)
        self.source_max = nanmax(data_values)
        self.source_sum = nansum(data_values)
        self.warnings = {}

    @lazyproperty
    def linear_thresholds(self):
        """
        Linearly spaced thresholds between the source minimum and
        maximum (inclusive).

        The source min/max are excluded later, giving nlevels thresholds
        between min and max (noninclusive).
        """
        return np.linspace(self.source_min, self.source_max, self.nlevels + 2)

    @lazyproperty
    def normalized_thresholds(self):
        """
        Normalized thresholds (from 0 to 1) between the source minimum
        and maximum (inclusive).
        """
        return ((self.linear_thresholds - self.source_min)
                / (self.source_max - self.source_min))

    def compute_thresholds(self):
        """
        Compute the multi-level detection thresholds for the source.

        Returns
        -------
        thresholds : 1D `~numpy.ndarray`
            The multi-level detection thresholds for the source.
        """
        if self.mode == 'exponential' and self.source_min <= 0:
            self.warnings['nonposmin'] = 'non-positive minimum'
            self.mode = 'linear'

        if self.mode == 'linear':
            thresholds = self.linear_thresholds
        elif self.mode == 'sinh':
            a = 0.25
            minval = self.source_min
            maxval = self.source_max
            thresholds = self.normalized_thresholds
            thresholds = np.sinh(thresholds / a) / np.sinh(1.0 / a)
            thresholds *= (maxval - minval)
            thresholds += minval
        elif self.mode == 'exponential':
            minval = self.source_min
            maxval = self.source_max
            thresholds = self.normalized_thresholds
            thresholds = minval * (maxval / minval) ** thresholds

        return thresholds[1:-1]  # do not include source min and max

    def multithreshold(self, deblend_mode=True):
        """
        Perform multithreshold detection for each source.

        Parameters
        ----------
        deblend_mode : bool, optional
            If `True` then only segmentation images with more than one
            label will be returned. If `False` then all segmentation
            images will be returned.

        Returns
        -------
        segments : list of 2D `~numpy.ndarray`
            A list of segmentation images, one for each threshold. If
            ``deblend_mode=True`` then only segmentation images with more
            than one label will be returned.
        """
        thresholds = self.compute_thresholds()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            segms = _detect_sources(self.data, thresholds, self.npixels,
                                    self.footprint, self.segment_mask,
                                    deblend_mode=deblend_mode)
            return [segm.data for segm in segms]

    def make_markers(self, segments, return_all=False):
        """
        Make markers (possible sources) for the watershed algorithm.

        Parameters
        ----------
        segments : list of 2D `~numpy.ndarray`
            A list of segmentation images, one for each threshold.

        return_all : bool, optional
            If `True` then return all segmentation images. If `False`
            then return only the last segmentation image.

        Returns
        -------
        markers : 2D `~numpy.ndarray` or list of 2D `~numpy.ndarray`
            A segmentation image that contain markers for possible
            sources. If ``return_all=True`` then a list of all
            segmentation images is returned.
        """
        for i in range(len(segments) - 1):
            segm_lower = segments[i]
            segm_upper = segments[i + 1]
            markers = segm_lower.astype(bool)
            new_markers = False
            # For a given label in the lower level, find the labels in
            # the upper level (higher threshold value) that are its
            # children (i.e., the labels within the same mask as the
            # lower level). If there are multiple children, then the
            # lower-level parent label is replaced by its children.
            # Parent labels that do not have multiple children in the
            # upper level are kept as is (maximizing the marker size).
            labels = _get_labels(segments[i])
            for label in labels:
                mask = (segm_lower == label)
                # find label mapping from the lower to upper level
                upper_labels = _get_labels(segm_upper[mask])
                if upper_labels.size >= 2:  # new child markers found
                    new_markers = True
                    markers[mask] = segm_upper[mask].astype(bool)

            if new_markers:
                # convert bool markers to integer labels
                # ndi_label(markers, structure=self.footprint, output=markers)
                segm_data, _ = ndi_label(markers, structure=self.footprint)
                segments[i + 1] = segm_data
            else:
                segments[i + 1] = segments[i]

        if return_all:
            return segments

        return segments[-1]

    def apply_watershed(self, markers):
        """
        Apply the watershed algorithm to the source markers.

        Parameters
        ----------
        markers : list of `~photutils.segmentation.SegmentationImage`
            A list of segmentation images that contain possible sources
            as markers. The last list element contains all of the
            potential source markers.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray`
            A 2D int array containing the deblended source labels. Note
            that the source labels may not be consecutive if a label was
            removed.
        """
        from skimage.segmentation import watershed

        # Deblend using watershed. If any source does not meet the contrast
        # criterion, then remove the faintest such source and repeat until
        # all sources meet the contrast criterion.
        remove_marker = True
        while remove_marker:
            markers = watershed(-self.data, markers, mask=self.segment_mask,
                                connectivity=self.footprint)

            labels = _get_labels(markers)
            if labels.size == 1:  # only 1 source left
                remove_marker = False
            else:
                flux_frac = (sum_labels(self.data, markers, index=labels)
                             / self.source_sum)
                remove_marker = any(flux_frac < self.contrast)

                if remove_marker:
                    # remove only the faintest source (one at a time)
                    # because several faint sources could combine to meet
                    # the contrast criterion
                    markers[markers == labels[np.argmin(flux_frac)]] = 0.0

        return markers

    def deblend_source(self):
        """
        Deblend a single labeled source.

        Returns
        -------
        segment_data : 2D int `~numpy.ndarray`
            A 2D int array containing the deblended source labels. The
            source labels are consecutive starting at 1.
        """
        if self.source_min == self.source_max:  # no deblending
            return None

        segments = self.multithreshold()
        if len(segments) == 0:  # no deblending
            return None

        # define the markers (possible sources) for the watershed algorithm
        markers = self.make_markers(segments)

        # If there are too many markers (e.g., due to low threshold
        # and/or small npixels), the watershed step can be very slow
        # (the threshold of 200 is arbitrary, but seems to work well).
        # This mostly affects the "exponential" mode, where there are
        # many levels at low thresholds, so here we try again with
        # "linear" mode.
        nlabels = len(_get_labels(markers))
        if self.mode != 'linear' and nlabels > 200:
            self.warnings['nmarkers'] = 'too many markers'
            self.mode = 'linear'
            segments = self.multithreshold()

            if len(segments) == 0:  # no deblending
                return None
            markers = self.make_markers(segments)

        # deblend using the watershed algorithm using the markers as seeds
        markers = self.apply_watershed(markers)

        if not np.array_equal(self.segment_mask, markers.astype(bool)):
            raise ValueError(f'Deblending failed for source "{self.label}". '
                             'Please ensure you used the same pixel '
                             'connectivity in detect_sources and '
                             'deblend_sources.')

        if len(_get_labels(markers)) == 1:  # no deblending
            return None

        # markers may not be consecutive if a label was removed due to
        # the contrast criterion
        return _relabel_array(markers, start_label=1)


def _get_labels(array):
    """
    Get the unique labels greater than zero in an array.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to get the unique labels from.

    Returns
    -------
    labels : int `~numpy.ndarray`
        The unique labels in the array.
    """
    # return np.unique(array[array != 0])
    labels = np.unique(array)
    return labels[labels != 0]


def _relabel_array(array, start_label=1):
    """
    Relabel an array such that the labels are consecutive integers
    starting from 1.

    Parameters
    ----------
    array : 2D `~numpy.ndarray`
        The 2D array to relabel.

    start_label : int, optional
        The starting label number. Must be >= 1. The default is 1.

    Returns
    -------
    relabeled_array : 2D `~numpy.ndarray`
        The relabeled array.
    """
    labels = _get_labels(array)

    # check if the labels are already consecutive starting from
    # start_label
    if (labels[0] == start_label
            and (labels[-1] - start_label + 1) == len(labels)):
        return array

    # Create an array to map old labels to new labels
    relabel_map = np.zeros(labels.max() + 1, dtype=array.dtype)
    relabel_map[labels] = np.arange(len(labels)) + start_label

    return relabel_map[array]
