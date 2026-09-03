# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module for identifying spurious detections in the wings of brighter
sources.
"""

import numpy as np
from astropy.table import QTable
from scipy.spatial import cKDTree

from photutils.segmentation.catalog import SourceCatalog
from photutils.segmentation.core import SegmentationImage
from photutils.utils._quantity_helpers import process_quantities

__all__ = ['get_spurious_labels']

# The neighbor search zone, in units of the sum of the semimajor axes
# of the two sources (the SourceExtractor CLEAN_ZONE constant)
CLEAN_ZONE = 10.0

# The wing model is treated as zero beyond this value of its radial
# argument, as in SourceExtractor
MAX_MODEL_ARG = 1e10


def get_spurious_labels(data, segmentation_image, threshold, n_pixels, *,
                        convolved_data=None, clean_param=1.0):
    """
    Identify the segments that are likely spurious detections in the
    wings of a brighter neighbor.

    This function implements the `SourceExtractor`_ CLEAN test. Each
    source is described by a Moffat-like wing model built from its
    isophotal measurements. The model amplitude comes from the source
    flux and ellipse area, the radial argument is measured in the
    source's own elliptical metric (its ``ellipse_cxx``,
    ``ellipse_cyy``, and ``ellipse_cxy`` coefficients), and the model
    is normalized to fall to the detection threshold at the source's
    isophotal extent. For each pair of sources whose centroids lie
    within ten times the sum of their semimajor axes, the wing model
    of the brighter source is evaluated at the centroid of the fainter
    one. If that value exceeds the height of the fainter source's
    ``n_pixels``-th brightest pixel above the threshold, the fainter
    source would not have been detected on its own and is flagged as
    spurious.

    The sources are considered in order of decreasing flux, with ties
    broken by label. A source is spurious if the wing model of any
    brighter surviving source exceeds its comparison level, and it is
    assigned to the surviving source whose wing model is highest at
    its centroid. A spurious source never absorbs another one, so
    every reported absorber is a surviving source and the result does
    not depend on the label order. The SourceExtractor and SEP
    implementations instead test the sources in label order and let a
    source that has already been absorbed absorb later ones, so the
    two codes can differ when three or more sources interact.

    The function does not modify the segmentation image. Use
    :meth:`~photutils.segmentation.SegmentationImage.reassign_labels`
    to merge each spurious segment into its absorber (the
    SourceExtractor behavior) or
    :meth:`~photutils.segmentation.SegmentationImage.remove_labels` to
    drop the spurious segments (the SEP behavior).

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`
        The 2D background-subtracted image. Its pixel values within
        each segment define the unfiltered peak and the counts of
        pixels above the threshold levels used by the wing-model area
        correction.

    segmentation_image : `~photutils.segmentation.SegmentationImage`
        The segmentation image, with the same shape as ``data``,
        typically the output of
        :func:`~photutils.segmentation.detect_sources` or
        :func:`~photutils.segmentation.deblend_sources`.

    threshold : float or 2D `~numpy.ndarray`
        The detection threshold used to make the segmentation image. A
        2D array must have the same shape as ``data``. All values must
        be positive. If ``data`` is a `~astropy.units.Quantity` array,
        then ``threshold`` must have the same units.

    n_pixels : int
        The minimum number of connected pixels used to detect the
        sources. The comparison level of each source is the height of
        its ``n_pixels``-th brightest pixel above the threshold.

    convolved_data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array used to detect the sources, i.e., the convolved
        image if one was used. Its pixel values within each segment
        define the source centroids, moments, fluxes, and comparison
        levels. If `None`, then ``data`` is used. If ``data`` is a
        `~astropy.units.Quantity` array, then ``convolved_data`` must
        have the same units.

    clean_param : float, optional
        The exponent of the wing model, i.e., the `SourceExtractor`_
        ``CLEAN_PARAM`` parameter. Larger values make the wings fall
        off more slowly and absorb more neighbors. Must be positive.

    Returns
    -------
    result : `~astropy.table.QTable`
        A table with a row for each spurious segment, sorted by label,
        with the columns:

        * ``'label'``: the label of the spurious segment
        * ``'absorbed_by'``: the label of the surviving source whose
          wing absorbed it

        The table is empty if no segment is spurious.

    See Also
    --------
    :func:`photutils.segmentation.detect_sources`
    :func:`photutils.segmentation.deblend_sources`

    Notes
    -----
    The ``data``, ``threshold``, and ``n_pixels`` inputs should match
    those used to make the segmentation image. The test is a heuristic
    tuned for stellar profiles. A faint real companion or a piece of
    galaxy substructure that sits within the modeled wing of a bright
    neighbor is flagged along with true spurious detections, so the
    result should be applied with care in crowded or extended fields.

    The comparison level of each source is the exact height of its
    ``n_pixels``-th brightest pixel above the threshold. The
    SourceExtractor and SEP implementations select that pixel with a
    heap whose sift-down descends into the wrong node, so their level
    is occasionally higher than the true value. A source near the
    decision boundary can therefore survive in those codes but be
    flagged here.

    Non-finite pixels are ignored. They are excluded from the moments,
    the fluxes, the pixel counts, and the comparison levels. Negative
    pixels within a segment are set to zero for the moments (as in
    `~photutils.segmentation.SourceCatalog`) but are included in the
    fluxes, whereas SourceExtractor weights the moments by the pixel
    values as they are. The two agree whenever the segmentation image
    was made from ``convolved_data`` (or ``data``) with the given
    ``threshold``, since every segment pixel then exceeds a positive
    threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.modeling.models import Gaussian2D
    >>> from photutils.segmentation import (detect_sources,
    ...                                     get_spurious_labels)
    >>> yy, xx = np.mgrid[0:100, 0:60]
    >>> data = (Gaussian2D(1000, 30, 70, 3.5, 3.5)(xx, yy)
    ...         + Gaussian2D(7, 30, 52, 2.5, 2.5)(xx, yy))
    >>> segment_map = detect_sources(data, 5.0, n_pixels=4)
    >>> segment_map.n_labels
    2
    >>> spurious = get_spurious_labels(data, segment_map, 5.0, n_pixels=4)
    >>> print(spurious)
    label absorbed_by
    ----- -----------
        1           2
    """
    inputs, _ = process_quantities((data, convolved_data, threshold),
                                   ('data', 'convolved_data', 'threshold'))
    data, convolved_data, threshold = inputs
    data = np.asarray(data, dtype=float)

    if not isinstance(segmentation_image, SegmentationImage):
        msg = 'segmentation_image must be a SegmentationImage'
        raise TypeError(msg)

    if segmentation_image.shape != data.shape:
        msg = 'segmentation_image must have the same shape as data'
        raise ValueError(msg)

    if segmentation_image.n_labels == 0:
        msg = 'segmentation_image must have at least one non-zero label'
        raise ValueError(msg)

    if convolved_data is None:
        convolved_data = data
    else:
        convolved_data = np.asarray(convolved_data, dtype=float)
        if convolved_data.shape != data.shape:
            msg = 'convolved_data must have the same shape as data'
            raise ValueError(msg)

    threshold = np.asarray(threshold, dtype=float)
    if threshold.ndim != 0 and threshold.shape != data.shape:
        msg = 'threshold must be a scalar or have the same shape as data'
        raise ValueError(msg)
    if not np.all(np.isfinite(threshold)) or np.any(threshold <= 0):
        msg = 'threshold must be positive and finite'
        raise ValueError(msg)

    if (n_pixels <= 0) or (int(n_pixels) != n_pixels):
        msg = f'n_pixels must be a positive integer, got {n_pixels!r}'
        raise ValueError(msg)

    if not np.isfinite(clean_param) or clean_param <= 0:
        msg = ('clean_param must be a positive finite number, got '
               f'{clean_param!r}')
        raise ValueError(msg)

    props = _measure_segments(data, convolved_data, segmentation_image,
                              threshold, int(n_pixels))
    absorbed_by = _find_absorbers(props, clean_param)

    labels = props['label']
    spurious = np.flatnonzero(absorbed_by >= 0)
    return QTable({'label': labels[spurious],
                   'absorbed_by': labels[absorbed_by[spurious]]})


def _measure_segments(data, convolved_data, segmentation_image, threshold,
                      n_pixels):
    """
    Measure the per-segment quantities used by the wing-model test.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The unfiltered image.

    convolved_data : 2D `~numpy.ndarray`
        The image whose pixel values define the moments, fluxes, and
        comparison levels.

    segmentation_image : `~photutils.segmentation.SegmentationImage`
        The segmentation image.

    threshold : `~numpy.ndarray`
        The detection threshold, as a 0D or 2D array.

    n_pixels : int
        The minimum number of connected pixels used for detection.

    Returns
    -------
    props : dict
        A dictionary of 1D arrays, one entry per label in increasing
        label order, with the keys ``'label'``, ``'x'``, ``'y'``,
        ``'a'``, ``'b'``, ``'cxx'``, ``'cyy'``, ``'cxy'``, ``'flux'``,
        ``'npix'``, ``'thresh'``, ``'abcor'``, and ``'mthresh'``.
        Non-finite pixels are ignored in every quantity.
    """
    catalog = SourceCatalog(convolved_data, segmentation_image)
    props = {'label': np.asarray(catalog.labels),
             'x': np.asarray(catalog.x_centroid),
             'y': np.asarray(catalog.y_centroid),
             'a': np.asarray(catalog.semimajor_axis.value),
             'b': np.asarray(catalog.semiminor_axis.value),
             'cxx': np.asarray(catalog.ellipse_cxx.value),
             'cyy': np.asarray(catalog.ellipse_cyy.value),
             'cxy': np.asarray(catalog.ellipse_cxy.value),
             'flux': np.asarray(catalog.segment_flux)}

    # Gather the segment pixels grouped by label, with the pixels of
    # each segment ordered by decreasing height above the threshold
    labels_flat = segmentation_image.data.ravel()
    idx = np.flatnonzero(labels_flat)
    if threshold.ndim == 0:
        pixel_thresh = np.full(idx.shape, float(threshold))
    else:
        pixel_thresh = threshold.ravel()[idx]
    excess = convolved_data.ravel()[idx] - pixel_thresh
    finite_excess = np.isfinite(excess)
    excess = np.where(finite_excess, excess, -np.inf)
    order = np.argsort(-excess)
    order = order[np.argsort(labels_flat[idx][order], kind='stable')]
    idx = idx[order]
    pixel_thresh = pixel_thresh[order]
    excess = excess[order]
    finite_excess = finite_excess[order]
    _, starts, counts = np.unique(labels_flat[idx], return_index=True,
                                  return_counts=True)
    n_finite = np.add.reduceat(finite_excess.astype(np.intp), starts)
    props['npix'] = n_finite

    values = data.ravel()[idx]
    values = np.where(np.isfinite(values), values, -np.inf)

    # The per-object threshold is the mean over the segment pixels
    thresh = np.add.reduceat(pixel_thresh, starts) / counts
    props['thresh'] = thresh

    # The unfiltered peak and the pixel counts above the threshold
    # and above the level midway between the threshold and the peak
    peak = np.maximum.reduceat(values, starts)
    thresh2 = (thresh + peak) / 2.0
    above_thresh = values > np.repeat(thresh, counts)
    above_thresh2 = values > np.repeat(thresh2, counts)
    n_above = np.add.reduceat(above_thresh.astype(np.intp), starts)
    n_above2 = np.add.reduceat(above_thresh2.astype(np.intp), starts)
    props['abcor'] = _area_correction(thresh, thresh2, n_above2 - n_above,
                                      props['a'], props['b'])

    # The comparison level is the height of the n_pixels-th brightest
    # pixel above the threshold (zero for segments with fewer pixels)
    mthresh = np.zeros(len(starts))
    enough = n_finite >= n_pixels
    mthresh[enough] = excess[starts[enough] + n_pixels - 1]
    props['mthresh'] = mthresh

    return props


def _area_correction(thresh, thresh2, darea, a, b):
    """
    Return the SourceExtractor ``abcor`` correction to the ellipse
    area of each source.

    The correction compares the drop in isophotal area between the
    threshold and the level midway to the peak with that of a Gaussian
    profile. It is capped at 1.

    Parameters
    ----------
    thresh : 1D `~numpy.ndarray`
        The per-object detection thresholds.

    thresh2 : 1D `~numpy.ndarray`
        The levels midway between the thresholds and the peaks.

    darea : 1D `~numpy.ndarray`
        The number of pixels above ``thresh2`` minus the number above
        ``thresh``.

    a, b : 1D `~numpy.ndarray`
        The semimajor and semiminor axes.

    Returns
    -------
    abcor : 1D `~numpy.ndarray`
        The area correction factors.
    """
    abcor = np.ones(len(thresh))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = thresh / thresh2
        valid = ratio > 0
        number = np.where(darea < 0, darea, -1.0)
        denom = (2.0 * np.pi * np.log(np.minimum(ratio, 0.99)) * a * b)
        corr = number / denom
    abcor[valid] = np.minimum(corr[valid], 1.0)
    return abcor


def _find_absorbers(props, clean_param):
    """
    Run the pairwise wing-model test and return the absorber of each
    source.

    Parameters
    ----------
    props : dict
        The per-segment quantities from `_measure_segments`.

    clean_param : float
        The wing model exponent.

    Returns
    -------
    absorbed_by : 1D `~numpy.ndarray`
        The index of the surviving source that absorbed each source,
        or -1 for the sources that survive.
    """
    x = props['x']
    y = props['y']
    a = props['a']
    flux = props['flux']
    n_sources = len(x)
    absorbed_by = np.full(n_sources, -1, dtype=np.intp)

    # Only sources with a defined position and size take part
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(a)
    finite_idx = np.flatnonzero(finite)
    if len(finite_idx) < 2:
        return absorbed_by

    i, j = _candidate_pairs(x[finite], y[finite], a[finite])
    i = finite_idx[i]
    j = finite_idx[j]

    # The brighter source of each pair may absorb the fainter one.
    # Equal fluxes are ordered by label.
    rank = np.empty(n_sources, dtype=np.intp)
    rank[np.lexsort((np.arange(n_sources), -flux))] = np.arange(n_sources)
    first = rank[i] < rank[j]
    eater = np.where(first, i, j)
    victim = np.where(first, j, i)
    model = _wing_model(props, eater, victim, clean_param)
    absorbs = model > props['mthresh'][victim]
    eater = eater[absorbs]
    victim = victim[absorbs]
    model = model[absorbs]

    # Resolve the victims in order of decreasing flux. Every absorber
    # of a victim is brighter and so has already been resolved, and
    # only surviving absorbers count. A victim with several surviving
    # absorbers is assigned to the one whose wing is highest.
    order = np.argsort(rank[victim], kind='stable')
    eater = eater[order]
    victim = victim[order]
    model = model[order]
    starts = np.flatnonzero(np.r_[True, victim[1:] != victim[:-1]])
    stops = np.r_[starts[1:], len(victim)]
    for start, stop in zip(starts, stops, strict=True):
        eaters = eater[start:stop]
        alive = absorbed_by[eaters] < 0
        if np.any(alive):
            best = np.argmax(np.where(alive, model[start:stop], -np.inf))
            absorbed_by[victim[start]] = eaters[best]

    return absorbed_by


def _candidate_pairs(x, y, a):
    """
    Return the index pairs of sources within each other's cleaning
    zone.

    A pair lies within the zone only if its separation is at most
    ``CLEAN_ZONE`` times the sum of the two semimajor axes, which is at
    most twice ``CLEAN_ZONE`` times the larger axis. A neighbor query
    around each source with that radius therefore finds every pair in
    which the source is the larger member, and the cost scales with
    the number of sources within each source's own zone rather than
    with the largest source in the field.

    Parameters
    ----------
    x, y : 1D `~numpy.ndarray`
        The source centroids.

    a : 1D `~numpy.ndarray`
        The source semimajor axes.

    Returns
    -------
    i, j : 1D `~numpy.ndarray`
        The indices of the pairs, with ``i < j``. Each pair appears
        once.
    """
    xy = np.column_stack((x, y))
    neighbors = cKDTree(xy).query_ball_point(xy, 2.0 * CLEAN_ZONE * a)
    counts = np.array([len(nbrs) for nbrs in neighbors], dtype=np.intp)
    i = np.repeat(np.arange(len(x)), counts)
    j = np.concatenate(neighbors).astype(np.intp)
    # Each pair can be found from both sides, so keep one copy of
    # each, with i < j, and drop the self pairs
    lo = np.minimum(i, j)
    hi = np.maximum(i, j)
    keys = np.unique((lo * len(x) + hi)[lo != hi])
    i = keys // len(x)
    j = keys % len(x)
    in_zone = ((x[i] - x[j])**2 + (y[i] - y[j])**2
               <= (CLEAN_ZONE * (a[i] + a[j]))**2)
    return i[in_zone], j[in_zone]


def _wing_model(props, eater, victim, clean_param):
    """
    Evaluate the wing model of each absorbing source at the centroid
    of its potential victim.

    The model is ``amp * (1 + alpha * r**2)**(-clean_param)``, where
    ``r`` is measured in the elliptical metric of the absorbing
    source, and it is zero beyond ``MAX_MODEL_ARG`` or where the
    radial argument is not above 1 (the victim lies at the center).
    Sources without a positive wing amplitude have a NaN model, which
    never absorbs.

    Parameters
    ----------
    props : dict
        The per-segment quantities from `_measure_segments`.

    eater, victim : 1D `~numpy.ndarray`
        The indices of the absorbing and potential victim sources of
        each pair.

    clean_param : float
        The wing model exponent.

    Returns
    -------
    model : 1D `~numpy.ndarray`
        The wing model values.
    """
    dx = props['x'][eater] - props['x'][victim]
    dy = props['y'][eater] - props['y'][victim]
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        unit_area = np.pi * props['a'] * props['b']
        amp = props['flux'] / (2.0 * unit_area * props['abcor'])
        alpha = (((amp / props['thresh'])**(1.0 / clean_param) - 1.0)
                 * unit_area / props['npix'])
        arg = 1.0 + alpha[eater] * (props['cxx'][eater] * dx**2
                                    + props['cyy'][eater] * dy**2
                                    + props['cxy'][eater] * dx * dy)
        return np.where((arg > 1.0) & (arg < MAX_MODEL_ARG),
                        amp[eater] * arg**(-clean_param), 0.0)
