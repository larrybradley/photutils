# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for scoring source-detection results against the input sources
of a simulated image.
"""

import warnings

import numpy as np
from astropy.table import QTable, Table

from photutils.simulation.noise import make_noise_image
from photutils.utils.exceptions import NoDetectionsWarning

__all__ = ['completeness', 'false_detection_rate', 'match_segments',
           'match_sources']


def match_segments(segment_img, params_table, *, x_name='x_0',
                   y_name='y_0', priority='flux'):
    """
    Match a segmentation image to the input sources of a simulated
    image.

    The label under each input position is read from the segmentation
    image. Each label is credited to one input source only, the one
    with the largest ``priority`` value, so a segment that covers
    several input sources (a blend) detects only one of them. Ties in
    ``priority`` go to the earlier row.

    Parameters
    ----------
    segment_img : `~photutils.segmentation.SegmentationImage`
        The segmentation image of the detected sources.

    params_table : `~astropy.table.Table`
        The table of input sources, with ``x_name`` and ``y_name``
        position columns in pixels and a ``priority`` column. Stack the
        tables of different populations (for example with
        `~astropy.table.vstack`) and score them in one call, so that
        blends between populations are credited correctly.

    x_name : str, optional
        The name of the x position column.

    y_name : str, optional
        The name of the y position column.

    priority : str, optional
        The name of the column that decides which input source a
        segment is credited to when it covers several of them.

    Returns
    -------
    result : `~astropy.table.Table` or `~astropy.table.QTable`
        A copy of ``params_table`` (of the same class) with three added
        columns:

        * ``'label'``: the segment label under the source position, or
          0 if there is none. A position belongs to the pixel whose
          center is nearest, so pixel ``i`` covers ``[i - 0.5, i +
          0.5)``. Positions outside the image or that are not finite
          get 0.
        * ``'detected'``: whether the source has its own segment, that
          is, whether its segment is credited to it.
        * ``'blended'``: whether the source lies in a segment that is
          credited to another source.

        The table ``meta['unmatched_labels']`` is a sorted array of the
        labels that cover no input position.

    Raises
    ------
    ValueError
        If ``params_table`` has no ``priority`` column.

    See Also
    --------
    match_sources

    Examples
    --------
    >>> import numpy as np
    >>> from astropy.table import QTable
    >>> from photutils.segmentation import SegmentationImage
    >>> from photutils.simulation import match_segments
    >>> data = np.zeros((10, 10), dtype=int)
    >>> data[1:5, 1:5] = 1
    >>> data[6:9, 6:9] = 2
    >>> segm = SegmentationImage(data)
    >>> truth = QTable({'x_0': [2.0, 3.0, 8.0], 'y_0': [2.0, 3.0, 1.0],
    ...                 'flux': [10.0, 50.0, 20.0]})
    >>> result = match_segments(segm, truth)
    >>> print(result['label', 'detected', 'blended'])
    label detected blended
    ----- -------- -------
        1    False    True
        1     True   False
        0    False   False
    >>> print(result.meta['unmatched_labels'])
    [2]
    """
    if priority not in params_table.colnames:
        msg = f'params_table has no "{priority}" column'
        raise ValueError(msg)

    data = np.asarray(segment_img.data)
    x = _as_float(params_table[x_name])
    y = _as_float(params_table[y_name])

    with np.errstate(invalid='ignore'):
        ix = np.floor(x + 0.5)
        iy = np.floor(y + 0.5)
    inside = (np.isfinite(ix) & np.isfinite(iy)
              & (ix >= 0) & (ix < data.shape[1])
              & (iy >= 0) & (iy < data.shape[0]))

    labels = np.zeros(len(params_table), dtype=int)
    labels[inside] = data[iy[inside].astype(int), ix[inside].astype(int)]

    detected = _credit_by_priority(labels, params_table[priority])

    result = params_table.copy()
    result['label'] = labels
    result['detected'] = detected
    result['blended'] = (labels > 0) & ~detected
    result.meta['unmatched_labels'] = np.setdiff1d(
        np.asarray(segment_img.labels, dtype=int), labels[labels > 0])

    return result


def match_sources(x, y, params_table, max_separation, *, x_name='x_0',
                  y_name='y_0', priority='flux'):
    """
    Match a list of detected source positions to the input sources of a
    simulated image.

    This is the source-list counterpart of `match_segments`, for
    results such as star-finder or PSF-photometry tables. Input sources
    are taken in order of decreasing ``priority``, and each claims its
    nearest unclaimed detection within ``max_separation``. A detection
    is therefore credited to one input source only, with the same blend
    rule as `match_segments`, so scores from segmentation images and
    from source lists are comparable. Ties in ``priority`` go to the
    earlier row.

    Parameters
    ----------
    x, y : 1D array_like
        The x and y pixel positions of the detected sources.

    params_table : `~astropy.table.Table`
        The table of input sources, with ``x_name`` and ``y_name``
        position columns in pixels and a ``priority`` column. Stack the
        tables of different populations and score them in one call.

    max_separation : float
        The largest distance in pixels between an input source and the
        detection it claims.

    x_name : str, optional
        The name of the x position column of ``params_table``.

    y_name : str, optional
        The name of the y position column of ``params_table``.

    priority : str, optional
        The name of the column that decides the order in which input
        sources claim detections.

    Returns
    -------
    result : `~astropy.table.Table` or `~astropy.table.QTable`
        A copy of ``params_table`` (of the same class) with four added
        columns:

        * ``'detected'``: whether the source claimed a detection.
        * ``'blended'``: whether the source had detections within
          ``max_separation`` but all of them were claimed by sources
          with a higher priority.
        * ``'match_index'``: the index into ``x`` and ``y`` of the
          claimed detection, or -1.
        * ``'separation'``: the distance in pixels to the claimed
          detection, or NaN.

        The table ``meta['unmatched_indices']`` is a sorted array of the
        indices of the detections that no input source claimed,
        including detections whose positions are not finite.

    Raises
    ------
    ValueError
        If ``params_table`` has no ``priority`` column, if ``x`` and
        ``y`` have different shapes, or if ``max_separation`` is not
        positive.

    See Also
    --------
    match_segments

    Examples
    --------
    >>> from astropy.table import QTable
    >>> from photutils.simulation import match_sources
    >>> truth = QTable({'x_0': [10.0, 12.0, 30.0], 'y_0': [10.0, 10.0, 5.0],
    ...                 'flux': [10.0, 50.0, 20.0]})
    >>> result = match_sources([11.5, 40.0], [10.0, 40.0], truth, 2.0)
    >>> print(result['detected', 'blended', 'match_index'])
    detected blended match_index
    -------- ------- -----------
       False    True          -1
        True   False           0
       False   False          -1
    >>> print(result.meta['unmatched_indices'])
    [1]
    """
    from scipy.spatial import KDTree

    if priority not in params_table.colnames:
        msg = f'params_table has no "{priority}" column'
        raise ValueError(msg)
    x = np.atleast_1d(_as_float(x))
    y = np.atleast_1d(_as_float(y))
    if x.shape != y.shape:
        msg = 'x and y must have the same shape'
        raise ValueError(msg)
    if not max_separation > 0:
        msg = 'max_separation must be positive'
        raise ValueError(msg)

    x_in = _as_float(params_table[x_name])
    y_in = _as_float(params_table[y_name])
    n_inputs = len(params_table)

    match_index = np.full(n_inputs, -1, dtype=int)
    separation = np.full(n_inputs, np.nan)
    has_candidates = np.zeros(n_inputs, dtype=bool)

    finite = np.isfinite(x_in) & np.isfinite(y_in)
    finite_det = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if finite_det.size > 0 and np.any(finite):
        tree = KDTree(np.column_stack((x[finite_det], y[finite_det])))
        hits = tree.query_ball_point(
            np.column_stack((x_in[finite], y_in[finite])), r=max_separation)
        candidates = np.empty(n_inputs, dtype=object)
        # Map the tree indices back to indices into x and y
        for idx, hit in zip(np.flatnonzero(finite), hits, strict=True):
            candidates[idx] = finite_det[np.asarray(hit, dtype=int)]
        claimed = np.zeros(x.size, dtype=bool)
        order = np.argsort(-_as_float(params_table[priority]),
                           kind='stable')
        for idx in order:
            if not finite[idx] or len(candidates[idx]) == 0:
                continue
            has_candidates[idx] = True
            free = np.sort(np.asarray(candidates[idx], dtype=int))
            free = free[~claimed[free]]
            if free.size == 0:
                continue
            dist = np.hypot(x[free] - x_in[idx], y[free] - y_in[idx])
            best = np.argmin(dist)
            claimed[free[best]] = True
            match_index[idx] = free[best]
            separation[idx] = dist[best]

    detected = match_index >= 0

    result = params_table.copy()
    result['detected'] = detected
    result['blended'] = has_candidates & ~detected
    result['match_index'] = match_index
    result['separation'] = separation
    result.meta['unmatched_indices'] = np.setdiff1d(
        np.arange(x.size), match_index[detected])

    return result


def completeness(detected, values, bins, *, confidence=0.683):
    """
    Calculate the detection completeness in bins of a source property.

    The completeness of a bin is the fraction of the input sources in
    the bin that were detected. Its lower and upper bounds are Wilson
    score intervals, which stay honest for small bins and for
    completeness near 0 or 1.

    Parameters
    ----------
    detected : 1D array_like of bool
        Whether each input source was detected, for example the
        ``'detected'`` column returned by `match_segments` or
        `match_sources`.

    values : 1D array_like
        The value of each input source used for binning, usually its
        optimal signal-to-noise ratio (see `optimal_snr`). Sources with
        NaN values are not counted.

    bins : int or 1D array_like
        The bins, as for `numpy.histogram`. A sequence gives the bin
        edges, including the rightmost edge. The last bin includes its
        right edge, and the others do not.

    confidence : float, optional
        The confidence level of the bounds. The default, 0.683, gives
        the equivalent of one-sigma bounds.

    Returns
    -------
    result : `~astropy.table.QTable`
        A table with one row per bin and the columns ``'bin_min'``,
        ``'bin_max'``, ``'n_sources'``, ``'n_detected'``,
        ``'completeness'``, ``'lower'``, and ``'upper'``. The last three
        are NaN for an empty bin.

    Raises
    ------
    ValueError
        If ``detected`` and ``values`` have different shapes, or if
        ``confidence`` is not between 0 and 1.

    See Also
    --------
    match_segments, match_sources

    Examples
    --------
    >>> import numpy as np
    >>> from photutils.simulation import completeness
    >>> rng = np.random.default_rng(0)
    >>> snr = rng.uniform(1, 20, 500)
    >>> detected = rng.uniform(size=500) < snr / 10
    >>> result = completeness(detected, snr, [1, 3, 5, 10, 20])
    >>> for col in ('completeness', 'lower', 'upper'):
    ...     result[col].info.format = '.3f'
    >>> print(result)
    bin_min bin_max n_sources n_detected completeness lower upper
    ------- ------- --------- ---------- ------------ ----- -----
        1.0     3.0        47          5        0.106 0.069 0.160
        3.0     5.0        50         21        0.420 0.352 0.491
        5.0    10.0       110         80        0.727 0.683 0.768
       10.0    20.0       293        293        1.000 0.997 1.000
    """
    from astropy.stats import binom_conf_interval

    detected = np.asarray(detected, dtype=bool)
    values = _as_float(values)
    if detected.shape != values.shape:
        msg = 'detected and values must have the same shape'
        raise ValueError(msg)
    if not 0 < confidence < 1:
        msg = 'confidence must be between 0 and 1'
        raise ValueError(msg)

    finite = np.isfinite(values)
    n_sources, edges = np.histogram(values[finite], bins=bins)
    edges = edges.astype(float)
    n_detected, _ = np.histogram(values[finite & detected], bins=edges)

    fraction = np.full(len(n_sources), np.nan)
    lower = np.full(len(n_sources), np.nan)
    upper = np.full(len(n_sources), np.nan)
    nonempty = n_sources > 0
    if np.any(nonempty):
        fraction[nonempty] = n_detected[nonempty] / n_sources[nonempty]
        bounds = binom_conf_interval(n_detected[nonempty],
                                     n_sources[nonempty],
                                     confidence_level=confidence,
                                     interval='wilson')
        lower[nonempty] = bounds[0]
        upper[nonempty] = bounds[1]

    result = QTable()
    result['bin_min'] = edges[:-1]
    result['bin_max'] = edges[1:]
    result['n_sources'] = n_sources
    result['n_detected'] = n_detected
    result['completeness'] = fraction
    result['lower'] = lower
    result['upper'] = upper
    return result


def false_detection_rate(detect, shape, *, n_trials=10, stddev=1.0,
                         kernel=None, seed=None):
    """
    Measure the rate of false detections on pure-noise images.

    A detection algorithm is run on ``n_trials`` images that contain
    only Gaussian noise with zero mean, so every detection is false.

    Parameters
    ----------
    detect : callable
        The detection algorithm, called as ``detect(image)`` with a 2D
        noise image. It must return a
        `~photutils.segmentation.SegmentationImage` (or any object with
        an ``n_labels`` attribute), an `~astropy.table.Table` with one
        row per detection, or `None` for no detections. A
        `~photutils.utils.exceptions.NoDetectionsWarning` raised by
        ``detect`` is suppressed, since no detections is the expected
        result on a noise image.

    shape : 2-tuple of int
        The shape of the noise images.

    n_trials : int, optional
        The number of noise images.

    stddev : float, optional
        The per-pixel standard deviation of the noise.

    kernel : 2D `~numpy.ndarray` or `~astropy.convolution.Kernel2D`, optional
        If given, the noise is correlated with this kernel (see
        `make_noise_image`), which mimics resampled images.

    seed : int, optional
        A seed for the noise. Each trial uses its own seed, derived from
        ``seed`` with `numpy.random.SeedSequence`, so the trials are
        independent and the result is reproducible. If `None`, fresh,
        unpredictable entropy is pulled from the OS.

    Returns
    -------
    result : `~astropy.table.QTable`
        A table with one row per trial and the columns ``'trial'``
        (starting at 1), ``'n_detections'``, and
        ``'rate_per_megapixel'`` (the number of detections per million
        pixels).

    Raises
    ------
    ValueError
        If ``n_trials`` is not a positive integer.

    TypeError
        If ``detect`` returns an object of another type.

    See Also
    --------
    make_noise_image, match_segments

    Examples
    --------
    Correlated noise, as in a resampled image, gives false detections
    with a threshold that assumes uncorrelated noise:

    >>> from astropy.convolution import Gaussian2DKernel
    >>> from photutils.segmentation import detect_sources
    >>> from photutils.simulation import false_detection_rate
    >>> def detect(image):
    ...     return detect_sources(image, threshold=3.0, n_pixels=5)
    >>> result = false_detection_rate(detect, (500, 500), n_trials=3,
    ...                               kernel=Gaussian2DKernel(1.5),
    ...                               seed=0)
    >>> print(result)
    trial n_detections rate_per_megapixel
    ----- ------------ ------------------
        1           33              132.0
        2           25              100.0
        3           22               88.0
    """
    if (isinstance(n_trials, bool)
            or not isinstance(n_trials, int | np.integer) or n_trials < 1):
        msg = 'n_trials must be a positive integer'
        raise ValueError(msg)

    seeds = np.random.SeedSequence(seed).spawn(n_trials)
    n_detections = np.zeros(n_trials, dtype=int)
    for trial, trial_seed in enumerate(seeds):
        image = make_noise_image(shape, mean=0.0, stddev=stddev,
                                 kernel=kernel, seed=trial_seed)
        with warnings.catch_warnings():
            # No detections is the expected result on a noise image
            warnings.simplefilter('ignore', NoDetectionsWarning)
            detections = detect(image)
        n_detections[trial] = _count_detections(detections)

    result = QTable()
    result['trial'] = np.arange(n_trials) + 1
    result['n_detections'] = n_detections
    result['rate_per_megapixel'] = n_detections / np.prod(shape) * 1e6
    return result


def _count_detections(result):
    """
    Count the detections returned by a detection algorithm.

    Parameters
    ----------
    result : object
        A segmentation image (or any object with an ``n_labels``
        attribute), a table, or `None`.

    Returns
    -------
    n_detections : int
        The number of detections.
    """
    if result is None:
        return 0
    if hasattr(result, 'n_labels'):
        return int(result.n_labels)
    if isinstance(result, Table):
        return len(result)
    msg = 'detect must return a SegmentationImage, a table, or None'
    raise TypeError(msg)


def _credit_by_priority(labels, priority):
    """
    Credit each label to the source with the largest priority.

    Parameters
    ----------
    labels : 1D `~numpy.ndarray` of int
        The label of each source, with 0 meaning no label.

    priority : array_like
        The priority of each source. Ties go to the earlier source.

    Returns
    -------
    credited : 1D `~numpy.ndarray` of bool
        Whether each source is credited with its label.
    """
    order = np.argsort(-_as_float(priority), kind='stable')
    ordered = labels[order]
    _, first = np.unique(ordered, return_index=True)
    credited = np.zeros(len(labels), dtype=bool)
    credited[order[first]] = True
    credited &= labels > 0
    return credited


def _as_float(values):
    """
    Return values as a float array with masked entries set to NaN.

    Masked entries come, for example, from stacking tables of
    populations with different columns using `~astropy.table.vstack`.

    Parameters
    ----------
    values : array_like
        The values, possibly a masked column or array.

    Returns
    -------
    result : `~numpy.ndarray`
        The values as floats.
    """
    result = np.array(values, dtype=float)
    mask = np.ma.getmaskarray(values)
    if np.any(mask):
        result[mask] = np.nan
    return result
