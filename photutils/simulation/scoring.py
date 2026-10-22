# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for scoring source-detection results against the input sources
of a simulated image.
"""

import numpy as np

__all__ = ['match_segments']


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
    result : `~astropy.table.QTable`
        A copy of ``params_table`` with three added columns:

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
    x = np.asarray(params_table[x_name], dtype=float)
    y = np.asarray(params_table[y_name], dtype=float)

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
    order = np.argsort(-np.asarray(priority, dtype=float), kind='stable')
    ordered = labels[order]
    _, first = np.unique(ordered, return_index=True)
    credited = np.zeros(len(labels), dtype=bool)
    credited[order[first]] = True
    credited &= labels > 0
    return credited
