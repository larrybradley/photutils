# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for calculating the properties of sources
defined by a segmentation image.
"""

import time

from copy import copy
import inspect
import warnings

from astropy.coordinates import SkyCoord
from astropy.table import QTable
import astropy.units as u
from astropy.utils import lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
import numpy as np

from .core import SegmentationImage
from ..aperture import BoundingBox
from ..utils._convolution import _filter_data
from ..utils._moments import _moments, _moments_central
from ..utils._wcs_helpers import _pixel_to_world

#__all__ = ['SourceProperties', 'source_properties', 'SourceCatalog']
__all__ = ['SourceProperties']

#__doctest_requires__ = {('SourceProperties', 'SourceProperties.*',
#                         'SourceCatalog', 'SourceCatalog.*',
#                         'source_properties', 'properties_table'):
#                        ['scipy']}

# default table columns for `to_table()` output
DEFAULT_COLUMNS = ['id', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'sky_centroid_icrs', 'source_sum', 'source_sum_err',
                   'background_sum', 'background_mean',
                   'background_at_centroid', 'bbox_xmin', 'bbox_xmax',
                   'bbox_ymin', 'bbox_ymax', 'min_value', 'max_value',
                   'minval_xpos', 'minval_ypos', 'maxval_xpos', 'maxval_ypos',
                   'area', 'equivalent_radius', 'perimeter',
                   'semimajor_axis_sigma', 'semiminor_axis_sigma',
                   'orientation', 'eccentricity', 'ellipticity', 'elongation',
                   'covar_sigx2', 'covar_sigxy', 'covar_sigy2', 'cxx', 'cxy',
                   'cyy', 'gini']


#    def source_properties(data, segment_img, error=None, mask=None,
#                      background=None, filter_kernel=None, wcs=None,
#                      labels=None):


import functools


def unpacked(method):
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        results = method(*args, **kwargs)
        return results if len(results) != 1 else results[0]
    return _decorator

def as_scalar_if_possible(func):
    def wrapper(arr):
        print('as_scalar wrapper')

        arr = func(arr)
        #return arr if arr.shape else np.asscalar(arr)
        try:
            return arr if len(arr) != 1 else arr[0]
        except TypeError:
            return arr
    return wrapper

def as_scalar(method):
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return result if len(result) != 1 else result[0]
        except TypeError:
            return result
    return _decorator


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


class SourceProperties:
    def __init__(self, data, segment_img, error=None, mask=None,
                 kernel=None, background=None, wcs=None):

        #self._cache = {}
        self._data_unit = None
        data, error, background = self._process_quantities(data, error,
                                                           background)
        self._data = data
        self._segment_img = self._validate_segment_img(segment_img)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._kernel = kernel
        self._background = self._validate_array(background, 'background')
        self._wcs = wcs

    def _process_quantities(self, data, error, background):
        """
        Check units of input arrays.

        If any of the input arrays have units then they all must have
        units and the units must be the same.

        Return unitless ndarrays with the array unit set in
        self._data_unit.
        """
        inputs = (data, error, background)
        has_unit = [hasattr(x, 'unit') for x in inputs if x is not None]
        use_units = all(has_unit)
        if any(has_unit) and not use_units:
            raise ValueError('If any of data, error, or background has units, '
                             'then they all must all have units.')
        if use_units:
            self._data_unit = data.unit
            data = data.value
            if error is not None:
                if error.unit != self._data_unit:
                    raise ValueError('error must have the same units as data')
                error = error.value
            if background is not None:
                if background.unit != self._data_unit:
                    raise ValueError('background must have the same units as '
                                     'data')
                background = background.value
        return data, error, background

    def _validate_segment_img(self, segment_img):
        if not isinstance(segment_img, SegmentationImage):
            raise ValueError('segment_img must be a SegmentationImage')
        if segment_img.shape != self._data.shape:
            raise ValueError('segment_img and data must have the same shape.')
        return segment_img

    def _validate_array(self, array, name, check_units=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            array = np.asanyarray(array)
            if array.shape != self._data.shape:
                raise ValueError(f'error and {name} must have the same shape.')
        return array

    #@lazyproperty
    # def _properties(self):
    #     properties = []
    #     for label in self._segment_img.labels:
    #         properties.append(_SourceProperties(
    #             self._data, self._convolved_data, self._segment_img, label,
    #             error=self._error, mask=self._mask,
    #             background=self._background, data_unit=self._data_unit))
    #     return properties

    # def __getattr__(self, attr):
    #     # called only if attr explicitly defined in this cls
    #     if attr not in self._cache:
    #         values = [getattr(source, attr) for source in self._properties]

    #         if isinstance(values[0], u.Quantity) and np.isscalar(values[0]):
    #             # turn list of Quantities into a Quantity array
    #             values = u.Quantity(values)
    #         #if isinstance(values[0], SkyCoord):  # pragma: no cover
    #         #    # failsafe: turn list of SkyCoord into a SkyCoord array
    #         #    values = SkyCoord(values)

    #         # TODO: add other properties as arrays
    #         if attr in ('moments', 'moments_central'):
    #             values = np.array(values)

    #         self._cache[attr] = values

    #     return self._cache[attr]

    @property
    def _lazyproperties(self):
        """
        Return all lazyproperties (even in superclasses).
        """
        def islazyproperty(object):
            return isinstance(object, lazyproperty)

        return [i[0] for i in inspect.getmembers(self.__class__,
                                                 predicate=islazyproperty)]

#zzzzz
    def __getitem__(self, index):
        newcls = object.__new__(self.__class__)

        segm = copy(self._segment_img)  # TODO (copy method?)
        # TODO fix for non-consecutive labels
        segm.keep_labels(segm.labels[index])
        newcls._segment_img = segm

        # attributes defined in __init__ (_segment_img set above)
        init_attr = ('_data', '_error', '_mask', '_kernel', '_background',
                     '_wcs', '_data_unit')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        ref_attr = ('_convolved_data', '_data_mask')

        # slice any evaluated lazyproperty objects
        #print(self._get_lazyproperties())
        for key, value in self.__dict__.items():
            print(key, key in self._lazyproperties)
            if key in self._lazyproperties:
                print(value)
                if key in ref_attr:  # do not slice
                    newcls.__dict__[key] = value
                else:
                    # skip copy if value is not an array/list for each
                    # source (e.g., nlabels)
                    if not np.isscalar(value):
                        # TODO: this doesn't work for list objects
                        # with fancy indexing, e.g. index = [3, 1, 2]
                        # FIXME
                        #newcls.__dict__[key] = copy(value[index])
                        try:
                            newcls.__dict__[key] = value[index]
                        except TypeError:
                            val = [value[i] for i in index]
                            newcls.__dict__[key] = val

        return newcls

    def __len__(self):
        return self.nlabels

    def isscalar(self):
        return self.nlabels == 1

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    #labels, slices

    @lazyproperty
    @as_scalar
    def _null_object(self):
        """
        Return an array of None values.

        Used for SkyCoord properties if ``wcs`` is `None`.
        """
        return np.array([None] * len(self))

    @lazyproperty
    @as_scalar
    def _null_value(self):
        """
        Return an array of np.nan values.

        Used for background properties if ``background`` is `None`.
        """
        values = np.empty(len(self))
        values.fill(np.nan)
        return values

    @lazyproperty
    def _convolved_data(self):
        if self._kernel is None:
            return self._data
        return _filter_data(self._data, self._kernel, mode='constant',
                            fill_value=0.0, check_normalization=True)

    @lazyproperty
    def _data_mask(self):
        mask = ~np.isfinite(self._data)
        if self._mask is not None:
            mask |= self._mask
        return mask

    @lazyproperty
    @as_scalar
    def _cutout_segment_mask(self):
        label = self.label
        slices = self.slices
        if self.isscalar:
            label = (label,)
            slices = (slices,)
        return [self._segment_img.data[slice_] != label_
                for label_, slice_ in zip(label, slices)]

    @lazyproperty
    @as_scalar
    def _cutout_total_mask(self):
        """
        Boolean mask representing the combination of
        ``_cutout_segment_mask``, ``_cutout_nonfinite_mask``, and
        ``_cutout_input_mask``.

        All pixels with the value of source label are `False`. All others
        are `True`.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """
        slices = self.slices
        mask = self._cutout_segment_mask
        if self.isscalar:
            slices = (slices,)
            mask = (mask,)
        masks = []
        for segm_mask, slice_ in zip(mask, slices):
            masks.append(segm_mask | self._data_mask[slice_])
        return masks

    @as_scalar
    def _make_cutout(self, array, units=True, masked=False):
        slices = self.slices
        if self.isscalar:
            slices = (slices,)
        cutouts = [array[slice_] for slice_ in slices]
        if units and self._data_unit is not None:
            cutouts = [(cutout << self._data_unit) for cutout in cutouts]
        if masked:
            mask = self._cutout_total_mask
            if self.isscalar:
                mask = (mask,)
            result = [np.ma.masked_array(cutout, mask=mask_)
                      for cutout, mask_ in zip(cutouts, mask)]
            return result
        return cutouts

    @lazyproperty
    @as_scalar
    def _cutout_moment_data(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the input
        ``convolved_data``. The following pixels are set to zero in
        these arrays:

            * any masked pixels
            * invalid values (NaN and +/- inf)
            * negative data values - negative pixels (especially at
              large radii) can give image moments that have negative
              variances.

        These arrays are used to derive moment-based properties.
        """
        mask = ~np.isfinite(self._convolved_data) | (self._convolved_data < 0)
        if self._mask is not None:
            mask |= self._mask

        label = self.label
        slices = self.slices
        cutout = self.convdata_cutout
        if self.isscalar:
            label = (label,)
            slices = (slices,)
            cutout = (cutout,)

        cutouts = []
        for label_, slice_, convdata in zip(label, slices, cutout):
            mask2 = (self._segment_img.data[slice_] != label_) | mask[slice_]
            cutout = convdata.copy()
            cutout[mask2] = 0.
            cutouts.append(cutout)
        return cutouts

    def to_table(self, columns=None, exclude_columns=None):
        return _properties_table(self, columns=columns,
                                 exclude_columns=exclude_columns)
    @lazyproperty
    def nlabels(self):
        return self._segment_img.nlabels

    @lazyproperty
    @as_scalar
    def label(self):
        return self._segment_img.labels

    @lazyproperty
    @as_scalar
    def id(self):
        """
        The source identification number corresponding to the object
        label in the segmentation image.
        """
        return self.label

    @lazyproperty
    def _slices(self):
        return self._segment_img.slices

    @lazyproperty
    @as_scalar
    def slices(self):
        return self._segment_img.slices

    @lazyproperty
    @as_scalar
    def segm_cutout(self):
        return [segm.data for segm in self._segment_img.segments]

    @lazyproperty
    @as_scalar
    def segm_cutout_ma(self):
        return [segm.data_ma for segm in self._segment_img.segments]

    @lazyproperty
    @as_scalar
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source segment.
        """
        return self._make_cutout(self._data, units=True, masked=False)

    @lazyproperty
    @as_scalar
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """
        return self._make_cutout(self._data, units=False, masked=True)

    @lazyproperty
    @as_scalar
    def convdata_cutout(self):
        return self._make_cutout(self._convolved_data, units=True,
                                  masked=False)

    @lazyproperty
    @as_scalar
    def convdata_cutout_ma(self):
        return self._make_cutout(self._convolved_data, units=False,
                                  masked=True)

    @lazyproperty
    @as_scalar
    def error_cutout(self):
        if self._error is None:
            return self._null_objects
        return self._make_cutout(self._error, units=True,
                                  masked=False)

    @lazyproperty
    @as_scalar
    def error_cutout_ma(self):
        if self._error is None:
            return self._null_objects
        return self._make_cutout(self._error, units=False,
                                  masked=True)

    @lazyproperty
    @as_scalar
    def background_cutout(self):
        if self._background is None:
            return self._null_objects
        return self._make_cutout(self._background, units=True,
                                  masked=False)

    @lazyproperty
    @as_scalar
    def background_cutout_ma(self):
        if self._error is None:
            return self._null_objects
        return self._make_cutout(self._background, units=False,
                                  masked=True)

    @lazyproperty
    @as_scalar
    def _data_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``data`` values within the
        source segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked) via the ``_data_mask``.

        If all pixels are masked, an empty array will be returned.

        This array is used for ``source_sum``, ``area``, ``min_value``,
        ``max_value``, etc.
        """
        return [array.compressed() if len(array.compressed()) > 0 else np.nan
                for array in self.data_cutout_ma]

    @lazyproperty
    @as_scalar
    def _error_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``error`` values within
        the source segment.

        This array is used for ``source_sum_err``.
        """
        #if self._error is None:
        #    return self._null_values
        return [array.compressed() if len(array.compressed()) > 0 else np.nan
                for array in self.error_cutout_ma]

    @lazyproperty
    @as_scalar
    def _background_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``background`` values
        within the source segment.

        This array is used for ``background_sum`` and
        ``background_mean``.
        """
        return [array.compressed() if len(array.compressed()) > 0 else np.nan
                for array in self.background_cutout_ma]

    @lazyproperty
    @as_scalar
    def moments(self):
        """Spatial moments up to 3rd order of the source."""
        cutout = self._cutout_moment_data
        if self.isscalar:
            cutout = (cutout,)
        return np.array([_moments(arr, order=3) for arr in cutout])

    @lazyproperty
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        cutout = self._cutout_moment_data
        xcen = self.xcentroid
        ycen = self.ycentroid
        if self.isscalar:
            cutout = (cutout,)
            xcen = (xcen,)
            ycen = (ycen,)

        return np.array([_moments_central(arr, center=(xcen_, ycen_), order=3)
                         for arr, xcen_, ycen_ in zip(cutout, xcen, ycen)])

    @lazyproperty
    @as_scalar
    def _cutout_yxcentroid(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of
        the centroid within the source segment.
        """
        moments = self.moments
        if self.isscalar:
            moments = np.expand_dims(moments, axis=0)
        mu_00 = moments[:, 0, 0]
        badmask = (mu_00 == 0)
        ycentroid = np.where(badmask, np.nan, moments[:, 1, 0] / mu_00)
        xcentroid = np.where(badmask, np.nan, moments[:, 0, 1] / mu_00)
        return ycentroid, xcentroid

    @lazyproperty
    @as_scalar
    def cutout_centroid(self):
        return np.transpose(self._cutout_yxcentroid)

    @lazyproperty
    @as_scalar
    def _yxcentroid(self):
        return (self._cutout_yxcentroid[0] + self.bbox_ymin,
                self._cutout_yxcentroid[1] + self.bbox_xmin)

    @lazyproperty
    @as_scalar
    def centroid(self):
        """
        The ``(y, x)`` coordinate of the centroid within the source
        segment.
        """
        return np.transpose(self._yxcentroid)

    @lazyproperty
    @as_scalar
    def xcentroid(self):
        """
        The ``x`` coordinate of the centroid within the source segment.
        """
        return self._yxcentroid[1]

    @lazyproperty
    @as_scalar
    def ycentroid(self):
        """
        The ``y`` coordinate of the centroid within the source segment.
        """
        return self._yxcentroid[0]

    @lazyproperty
    def sky_centroid(self):
        """
        The sky coordinates of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input WCS.
        """
        if self._wcs is None:
            return self._null_objects
        return self._wcs.pixel_to_world(self.xcentroid, self.ycentroid)

    @lazyproperty
    def sky_centroid_icrs(self):
        """
        The sky coordinates, in the International Celestial Reference
        System (ICRS) frame, of the centroid within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.
        """
        if self._wcs is None:
            return self._null_objects
        return self.sky_centroid.icrs

    @lazyproperty
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices]

    @lazyproperty
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel location within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices])

    @lazyproperty
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices])

    @lazyproperty
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel location within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices])

    @lazyproperty
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel location within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices])

    @lazyproperty
    def _bbox_corner_ll(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox_ in bbox:
            xypos.append((bbox_.ixmin - 0.5, bbox_.iymin - 0.5))
        return xypos

    @lazyproperty
    def _bbox_corner_ul(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox in self.bbox:
            xypos.append((bbox.ixmin - 0.5, bbox.iymax + 0.5))
        return xypos

    @lazyproperty
    def _bbox_corner_lr(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox in self.bbox:
            xypos.append((bbox.ixmax + 0.5, bbox.iymin - 0.5))
        return xypos

    @lazyproperty
    def _bbox_corner_ur(self):
        bbox = self.bbox
        if self.isscalar:
            bbox = (bbox,)
        xypos = []
        for bbox in self.bbox:
            xypos.append((bbox.ixmax + 0.5, bbox.iymax + 0.5))
        return xypos

    @lazyproperty
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_objects
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ll))

    @lazyproperty
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_objects
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ul))

    @lazyproperty
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_objects
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_lr))

    @lazyproperty
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right vertex of the minimal
        bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all of the source segment pixels in
        their entirety, thus the vertices are at the pixel *corners*.
        """
        if self._wcs is None:
            return self._null_objects
        return self._wcs.pixel_to_world(*np.transpose(self._bbox_corner_ur))

    @lazyproperty
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """
        values = np.array([np.min(array) for array in self._data_values])
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value of the ``data`` within the source
        segment.
        """
        values = np.array([np.max(array) for array in self._data_values])
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    def minval_cutout_index(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        idx = []
        for arr in self.data_cutout_ma:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmin(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    def maxval_cutout_index(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        idx = []
        for arr in self.data_cutout_ma:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmax(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    def minval_index(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        out = []
        for idx, slc in zip(self.minval_cutout_index, self.slices):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    def maxval_index(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        out = []
        for idx, slc in zip(self.maxval_cutout_index, self.slices):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    def minval_xindex(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        return np.transpose(self.minval_index)[1]

    @lazyproperty
    def minval_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """
        return np.transpose(self.minval_index)[0]

    @lazyproperty
    def maxval_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        return np.transpose(self.maxval_index)[1]

    @lazyproperty
    def maxval_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """
        return np.transpose(self.maxval_index)[0]





#zzzzzzzz

#class _SourceCutouts:
class _SourceProperties:
    """
    Class to calculate photometry and morphological properties of a
    single labeled source.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.  If ``convolved_data`` is input, then it will be used
        instead of ``data`` to calculate the source centroid and
        morphological properties.  Source photometry is always measured
        from ``data``.  For accurate source properties and photometry,
        ``data`` should be background-subtracted.  Non-finite ``data``
        values (NaN and +/- inf) are automatically masked.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    label : int
        The label number of the source whose properties are calculated.

    convolved_data : array-like or `~astropy.units.Quantity`, optional
        The convolved version of the background-subtracted ``data`` from
        which to calculate the source centroid and morphological
        properties.  The kernel used to perform the filtering should be
        the same one used in defining the source segments (e.g., see
        :func:`~photutils.segmentation.detect_sources`).  If ``data`` is
        a `~astropy.units.Quantity` array then ``convolved_data`` must be
        a `~astropy.units.Quantity` array (and vise versa) with
        identical units.  Non-finite ``convolved_data`` values (NaN and
        +/- inf) are not automatically masked, unless they are at the
        same position of non-finite values in the input ``data`` array.
        Such pixels can be masked using the ``mask`` keyword.  If
        `None`, then the unconvolved ``data`` will be used instead.

    error : array_like or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data`` array.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vise versa) with identical
        units.  Non-finite ``error`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array.  Such pixels can
        be masked using the ``mask`` keyword.  See the Notes section
        below for details on the error propagation.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.  Non-finite
        values (NaN and +/- inf) in the input ``data`` are automatically
        masked.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  If ``data`` is
        a `~astropy.units.Quantity` array then ``background`` must be a
        `~astropy.units.Quantity` array (and vise versa) with identical
        units.  Inputting the ``background`` merely allows for its
        properties to be measured within each source segment.  The input
        ``background`` does *not* get subtracted from the input
        ``data``, which should already be background-subtracted.
        Non-finite ``background`` values (NaN and +/- inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array.  Such pixels can
        be masked using the ``mask`` keyword.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).  If `None`, then all sky-based
        properties will be set to `None`.

    Notes
    -----
    ``data`` (and optional ``convolved_data``) should be
    background-subtracted for accurate source photometry and properties.

    `SExtractor`_'s centroid and morphological parameters are always
    calculated from a convolved "detection" image, i.e. the image used to
    define the segmentation image.  The usual downside of the filtering
    is the sources will be made more circular than they actually are.
    If you wish to reproduce `SExtractor`_ centroid and morphology
    results, then input a filtered and background-subtracted "detection"
    image into the ``filtered_data`` keyword.  If ``filtered_data`` is
    `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values (``filtered_data`` or ``data``) within the
    source segment are set to zero when calculating morphological
    properties based on image moments.  Negative values could occur, for
    example, if the segmentation image was defined from a different
    image (e.g., different bandpass) or if the background was
    oversubtracted. Note that
    `~photutils.segmentation.SourceProperties.source_sum` always
    includes the contribution of negative ``data`` values.

    The input ``error`` array is assumed to include *all* sources of
    error, including the Poisson error of the sources.
    `~photutils.segmentation.SourceProperties.source_sum_err` is simply
    the quadrature sum of the pixel-wise total errors over the
    non-masked pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.segmentation.SourceProperties.source_sum_err`, :math:`S`
    are the non-masked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    Custom errors for source segments can be calculated using the
    `~photutils.segmentation.SourceProperties.error_cutout_ma` and
    `~photutils.segmentation.SourceProperties.background_cutout_ma`
    properties, which are 2D `~numpy.ma.MaskedArray` cutout versions of
    the input ``error`` and ``background``.  The mask is `True` for
    pixels outside of the source segment, masked pixels from the
    ``mask`` input, or any non-finite ``data`` values (NaN and +/- inf).

    .. _SExtractor: https://www.astromatic.net/software/sextractor
    """

    def __init__(self, data, convolved_data, segment_img, label,
                 error=None, mask=None, background=None, data_unit=None):

        self._data_unit = data_unit
        self.label = label
        self._data = data
        self._convolved_data = convolved_data
        self._segment_img = segment_img
        self._error = error
        self._mask = mask
        self._background = background

        self.segment = segment_img[segment_img.get_index(label)]  # TODO
        self.slices = self.segment.slices

    def __str__(self):
        cls_name = '<{0}.{1}>'.format(self.__class__.__module__,
                                      self.__class__.__name__)
        cls_info = []
        params = ['label']
        for param in params:
            cls_info.append((param, getattr(self, param)))
        fmt = (['{0}: {1}'.format(key, val) for key, val in cls_info])
        fmt.insert(1, 'centroid (x, y): ({0:0.4f}, {1:0.4f})'
                   .format(self.xcentroid.value, self.ycentroid.value))
        return '{}\n'.format(cls_name) + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    @lazyproperty
    def _segment_mask(self):
        """
        Boolean mask for the source segment.

        ``_segment_mask`` is `False` only for pixels whose value equals
        the label number of this source.  All other pixels are `True`
        (masked).
        """
        return self._segment_img.data[self.slices] != self.label

    @lazyproperty
    def _input_mask(self):
        """
        Cutout of the user-input boolean mask.
        """
        if self._mask is None:
            return None
        return self._mask[self.slices]

    @lazyproperty
    def _data_mask(self):
        """
        Boolean mask for non-finite (NaN and +/- inf) ``data`` values.
        """
        return ~np.isfinite(self.data_cutout)

    @lazyproperty
    def _total_mask(self):
        """
        Boolean mask representing the combination of
        ``_segment_mask``, ``_input_mask``, and ``_data_mask``.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """
        mask = self._segment_mask | self._data_mask
        if self._input_mask is not None:
            mask |= self._input_mask
        return mask

    @lazyproperty
    def _is_completely_masked(self):
        """
        Boolean indicating if all pixels within the source segment are
        masked.
        """
        return np.all(self._total_mask)

    @lazyproperty
    def _convolved_data_zeroed(self):
        """
        A 2D `~numpy.ndarray` cutout from the input ``convolved_data``
        where any masked pixels (from ``_total_mask``) are set to zero.

        Invalid values (NaN and +/- inf) are set to zero. Any units are
        dropped on the input ``convolved_data``.

        Negative data values are also set to zero because negative
        pixels (especially at large radii) can result in image moments
        that result in negative variances.

        This array is used for moment-based properties.
        """
        convolved_data = self._convolved_data[self.slices]
        if isinstance(convolved_data, u.Quantity):
            convolved_data = convolved_data.value

        convolved_data = np.where(self._total_mask, 0., convolved_data)  # copy
        convolved_data[convolved_data < 0] = 0.
        return convolved_data.astype(float)  # TODO

    # based on centroid and size (Cutout2D)
    #def make_cutout(self, data, masked=False):

    # TODO: add pad option
    def make_bbox_cutout(self, data, masked=False):
        """
        Create a (masked) cutout array from the input ``data`` using the
        minimal bounding box of the source segment.

        If ``masked`` is `False` (default), then the returned cutout
        array is simply a `~numpy.ndarray`. The returned cutout is a
        view (not a copy) of the input ``data``. No pixels are altered
        (e.g., set to zero) within the bounding box.

        If ``masked_array` is `True`, then the returned cutout array is
        a `~numpy.ma.MaskedArray`. The mask is `True` for pixels outside
        of the source segment (labeled region of interest), masked
        pixels from the input ``mask``, and any non-finite ``data``
        values (NaN and +/- inf). The data part of the masked array is a
        view (not a copy) of the input ``data``.

        Parameters
        ----------
        data : array-like (2D)
            The data array from which to create the cutout array.
            ``data`` must have the same shape as the segmentation image.

        masked : bool, optional
            If `True` then a `~numpy.ma.MaskedArray` will be returned,
            where the mask is `True` for pixels outside of the source
            segment (labeled region of interest), masked pixels from
            the ``mask`` input, or any non-finite ``data`` values (NaN
            and +/- inf). If `False`, then a `~numpy.ndarray` will be
            returned.

        Returns
        -------
        result : 2D `~numpy.ndarray` or `~numpy.ma.MaskedArray`
            The 2D cutout array.
        """
        data = np.asanyarray(data)
        if data.shape != self._segment_img.shape:
            raise ValueError('data must have the same shape as the '
                             'segmentation image input to SourceProperties')

        if masked:
            return np.ma.masked_array(data[self.slices],
                                      mask=self._total_mask)
        else:
            return data[self.slices]

    @lazyproperty
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source segment.
        """
        cutout = self._data[self.slices]
        if self._data_unit is not None:
            cutout <<= self._data_unit
        return cutout

    @lazyproperty
    def data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """
        cutout = np.ma.masked_array(self._data[self.slices],
                                    mask=self._total_mask)
        if self._data_unit is not None:
            cutout <<= self._data_unit
        return cutout

    @lazyproperty
    def convolved_data_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the ``convolved_data``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).
        """
        cutout = np.ma.masked_array(self._convolved_data[self.slices],
                                    mask=self._total_mask)
        if self._data_unit is not None:
            cutout <<= self._data_unit
        return cutout

    @lazyproperty
    def error_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input ``error``
        image.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).

        If ``error`` is `None`, then ``error_cutout_ma`` is also `None`.
        """
        if self._error is None:
            return None
        cutout = np.ma.masked_array(self._error[self.slices],
                                    mask=self._total_mask)
        if self._data_unit is not None:
            cutout <<= self._data_unit
        return cutout

    @lazyproperty
    def background_cutout_ma(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the input
        ``background``.

        The mask is `True` for pixels outside of the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and +/- inf).

        If ``background`` is `None`, then ``background_cutout_ma`` is
        also `None`.
        """
        if self._background is None:
            return None
        cutout = np.ma.masked_array(self._background[self.slices],
                                    mask=self._total_mask)
        if self._data_unit is not None:
            cutout <<= self._data_unit
        return cutout

    @lazyproperty
    def _data_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``data`` values within the
        source segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked) via the ``_data_mask``.

        If all pixels are masked, an empty array will be returned.

        This array is used for ``source_sum``, ``area``, ``min_value``,
        ``max_value``, etc.
        """
        values = self.data_cutout_ma.compressed()
        if self._data_unit is not None:
            values = values.value
        return values

    @lazyproperty
    def _error_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``error`` values within
        the source segment.

        This array is used for ``source_sum_err``.
        """
        values = self.error_cutout_ma.compressed()
        if self._data_unit is not None:
            values = values.value
        return values

    @lazyproperty
    def _background_values(self):
        """
        A 1D `~numpy.ndarray` of the unmasked ``background`` values
        within the source segment.

        This array is used for ``background_sum`` and
        ``background_mean``.
        """
        values = self.background_cutout_ma.compressed()
        if self._data_unit is not None:
            values = values.value
        return values

    @lazyproperty
    def indices(self):
        """
        A tuple of two `~numpy.ndarray` containing the ``y`` and ``x``
        pixel indices, respectively, of unmasked pixels within the
        source segment.

        Non-finite ``data`` values (NaN and +/- inf) are excluded.

        If all ``data`` pixels are masked, a tuple of two empty arrays
        will be returned.
        """
        yindices, xindices = np.nonzero(self.data_cutout_ma)
        return (yindices + self.slices[0].start,
                xindices + self.slices[1].start)

    @lazyproperty
    def moments(self):
        """Spatial moments up to 3rd order of the source."""
        return _moments(self._convolved_data_zeroed, order=3)

    @lazyproperty
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        ycentroid, xcentroid = self.cutout_centroid.value
        return _moments_central(self._convolved_data_zeroed,
                                center=(xcentroid, ycentroid), order=3)


    # @lazyproperty
    # def centroid(self):
    #     """
    #     The ``(y, x)`` coordinate of the centroid within the source
    #     segment.
    #     """
    #     ycen, xcen = self.cutout_centroid.value
    #     return (ycen + self.slices[0].start,
    #             xcen + self.slices[1].start) * u.pix

    # @lazyproperty
    # def xcentroid(self):
    #     """
    #     The ``x`` coordinate of the centroid within the source segment.
    #     """
    #     return self.centroid[1]

    # @lazyproperty
    # def ycentroid(self):
    #     """
    #     The ``y`` coordinate of the centroid within the source segment.
    #     """

    #     return self.centroid[0]

    # @lazyproperty
    # def sky_centroid(self):
    #     """
    #     The sky coordinates of the centroid within the source segment,
    #     returned as a `~astropy.coordinates.SkyCoord` object.

    #     The output coordinate frame is the same as the input WCS.
    #     """

    #     return _pixel_to_world(self.xcentroid.value, self.ycentroid.value,
    #                            self._wcs)

    # @lazyproperty
    # def sky_centroid_icrs(self):
    #     """
    #     The sky coordinates, in the International Celestial Reference
    #     System (ICRS) frame, of the centroid within the source segment,
    #     returned as a `~astropy.coordinates.SkyCoord` object.
    #     """

    #     if self._wcs is None:
    #         return None
    #     else:
    #         return self.sky_centroid.icrs

    # @lazyproperty
    # def bbox(self):
    #     """
    #     The `~photutils.aperture.BoundingBox` of the minimal rectangular
    #     region containing the source segment.
    #     """

    #     return BoundingBox(self.slices[1].start, self.slices[1].stop,
    #                        self.slices[0].start, self.slices[0].stop)

    # @lazyproperty
    # def bbox_xmin(self):
    #     """
    #     The minimum ``x`` pixel location within the minimal bounding box
    #     containing the source segment.
    #     """

    #     return self.bbox.ixmin * u.pix

    # @lazyproperty
    # def bbox_xmax(self):
    #     """
    #     The maximum ``x`` pixel location within the minimal bounding box
    #     containing the source segment.

    #     Note that this value is inclusive, unlike numpy slice indices.
    #     """

    #     return (self.bbox.ixmax - 1) * u.pix

    # @lazyproperty
    # def bbox_ymin(self):
    #     """
    #     The minimum ``y`` pixel location within the minimal bounding box
    #     containing the source segment.
    #     """

    #     return self.bbox.iymin * u.pix

    # @lazyproperty
    # def bbox_ymax(self):
    #     """
    #     The maximum ``y`` pixel location within the minimal bounding box
    #     containing the source segment.

    #     Note that this value is inclusive, unlike numpy slice indices.
    #     """

    #     return (self.bbox.iymax - 1) * u.pix

    # @lazyproperty
    # def sky_bbox_ll(self):
    #     """
    #     The sky coordinates of the lower-left vertex of the minimal
    #     bounding box of the source segment, returned as a
    #     `~astropy.coordinates.SkyCoord` object.

    #     The bounding box encloses all of the source segment pixels in
    #     their entirety, thus the vertices are at the pixel *corners*.
    #     """

    #     return _calc_sky_bbox_corner(self.bbox, 'll', self._wcs)

    # @lazyproperty
    # def sky_bbox_ul(self):
    #     """
    #     The sky coordinates of the upper-left vertex of the minimal
    #     bounding box of the source segment, returned as a
    #     `~astropy.coordinates.SkyCoord` object.

    #     The bounding box encloses all of the source segment pixels in
    #     their entirety, thus the vertices are at the pixel *corners*.
    #     """

    #     return _calc_sky_bbox_corner(self.bbox, 'ul', self._wcs)

    # @lazyproperty
    # def sky_bbox_lr(self):
    #     """
    #     The sky coordinates of the lower-right vertex of the minimal
    #     bounding box of the source segment, returned as a
    #     `~astropy.coordinates.SkyCoord` object.

    #     The bounding box encloses all of the source segment pixels in
    #     their entirety, thus the vertices are at the pixel *corners*.
    #     """

    #     return _calc_sky_bbox_corner(self.bbox, 'lr', self._wcs)

    # @lazyproperty
    # def sky_bbox_ur(self):
    #     """
    #     The sky coordinates of the upper-right vertex of the minimal
    #     bounding box of the source segment, returned as a
    #     `~astropy.coordinates.SkyCoord` object.

    #     The bounding box encloses all of the source segment pixels in
    #     their entirety, thus the vertices are at the pixel *corners*.
    #     """

    #     return _calc_sky_bbox_corner(self.bbox, 'ur', self._wcs)

    @lazyproperty
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit
        else:
            return np.min(self._data_values)

    @lazyproperty
    def max_value(self):
        """
        The maximum pixel value of the ``data`` within the source
        segment.
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit
        else:
            return np.max(self._data_values)

    @lazyproperty
    def minval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            arr = self.data_cutout_ma
            # multiplying by unit converts int to float, but keep as
            # float in case the array contains a NaN
            return np.asarray(np.unravel_index(np.argmin(arr),
                                               arr.shape)) * u.pix

    @lazyproperty
    def maxval_cutout_pos(self):
        """
        The ``(y, x)`` coordinate, relative to the `data_cutout`, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            arr = self.data_cutout_ma
            # multiplying by unit converts int to float, but keep as
            # float in case the array contains a NaN
            return np.asarray(np.unravel_index(np.argmax(arr),
                                               arr.shape)) * u.pix

    @lazyproperty
    def minval_pos(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            yposition, xposition = self.minval_cutout_pos.value
            return (yposition + self.slices[0].start,
                    xposition + self.slices[1].start) * u.pix

    @lazyproperty
    def maxval_pos(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        if self._is_completely_masked:
            return (np.nan, np.nan) * u.pix
        else:
            yposition, xposition = self.maxval_cutout_pos.value
            return (yposition + self.slices[0].start,
                    xposition + self.slices[1].start) * u.pix

    @lazyproperty
    def minval_xpos(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        return self.minval_pos[1]

    @lazyproperty
    def minval_ypos(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurence is returned.
        """

        return self.minval_pos[0]

    @lazyproperty
    def maxval_xpos(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        return self.maxval_pos[1]

    @lazyproperty
    def maxval_ypos(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurence is returned.
        """

        return self.maxval_pos[0]

    @lazyproperty
    def source_sum(self):
        """
        The sum of the unmasked ``data`` values within the source segment.

        .. math:: F = \\sum_{i \\in S} (I_i - B_i)

        where :math:`F` is ``source_sum``, :math:`(I_i - B_i)` is the
        ``data``, and :math:`S` are the unmasked pixels in the source
        segment.

        Non-finite pixel values (NaN and +/- inf) are excluded
        (automatically masked).
        """

        if self._is_completely_masked:
            return np.nan * self._data_unit  # table output needs unit
        else:
            return np.sum(self._data_values)

    @lazyproperty
    def source_sum_err(self):
        """
        The uncertainty of
        `~photutils.segmentation.SourceProperties.source_sum`,
        propagated from the input ``error`` array.

        ``source_sum_err`` is the quadrature sum of the total errors
        over the non-masked pixels within the source segment:

        .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
                  \\sigma_{\\mathrm{tot}, i}^2}

        where :math:`\\Delta F` is ``source_sum_err``,
        :math:`\\sigma_{\\mathrm{tot, i}}` are the pixel-wise total
        errors, and :math:`S` are the non-masked pixels in the source
        segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the error array.
        """

        if self._error is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # table output needs unit
            else:
                return np.sqrt(np.sum(self._error_values ** 2))
        else:
            return None

    @lazyproperty
    def background_sum(self):
        """
        The sum of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the background array.
        """

        if self._background is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # unit for table
            else:
                return np.sum(self._background_values)
        else:
            return None

    @lazyproperty
    def background_mean(self):
        """
        The mean of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and +/- inf) that are
        automatically masked, are also masked in the background array.
        """

        if self._background is not None:
            if self._is_completely_masked:
                return np.nan * self._data_unit  # unit for table
            else:
                return np.mean(self._background_values)
        else:
            return None

    @lazyproperty
    def background_at_centroid(self):
        """
        The value of the ``background`` at the position of the source
        centroid.

        The background value at fractional position values are
        determined using bilinear interpolation.
        """

        if self._background is not None:
            from scipy.ndimage import map_coordinates

            # centroid can be NaN if segment is completely masked or if
            # all data values are <= 0
            if np.any(~np.isfinite(self.centroid)):
                return np.nan * self._data_unit  # unit for table
            else:
                value = map_coordinates(self._background,
                                        [[self.ycentroid.value],
                                         [self.xcentroid.value]], order=1,
                                        mode='nearest')[0]
                return value * self._data_unit
        else:
            return None

    @lazyproperty
    def area(self):
        """
        The total unmasked area of the source segment in units of
        pixels**2.

        Note that the source area may be smaller than its segment area
        if a mask is input to `SourceProperties` or `source_properties`,
        or if the ``data`` within the segment contains invalid values
        (NaN and +/- inf).
        """

        if self._is_completely_masked:
            return np.nan * u.pix**2
        else:
            return len(self._data_values) * u.pix**2

    @lazyproperty
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """

        return np.sqrt(self.area / np.pi)

    @lazyproperty
    def perimeter(self):
        """
        The perimeter of the source segment, approximated as the total
        length of lines connecting the centers of the border pixels
        defined by a 4-pixel connectivity.

        If any masked pixels make holes within the source segment, then
        the perimeter around the inner hole (e.g. an annulus) will also
        contribute to the total perimeter.

        References
        ----------
        .. [1] K. Benkrid, D. Crookes, and A. Benkrid.  "Design and FPGA
               Implementation of a Perimeter Estimator".  Proceedings of
               the Irish Machine Vision and Image Processing Conference,
               pp. 51-57 (2000).
               http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
        """

        if self._is_completely_masked:
            return np.nan * u.pix  # unit for table
        else:
            from scipy.ndimage import binary_erosion, convolve

            data = ~self._total_mask
            selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            data_eroded = binary_erosion(data, selem, border_value=0)
            border = np.logical_xor(data, data_eroded).astype(int)

            kernel = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]])
            perimeter_data = convolve(border, kernel, mode='constant', cval=0)

            size = 34
            perimeter_hist = np.bincount(perimeter_data.ravel(),
                                         minlength=size)

            weights = np.zeros(size, dtype=float)
            weights[[5, 7, 15, 17, 25, 27]] = 1.
            weights[[21, 33]] = np.sqrt(2.)
            weights[[13, 23]] = (1 + np.sqrt(2.)) / 2.

            return (perimeter_hist[0:size] @ weights) * u.pix

    @lazyproperty
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """

        moments = self.moments_central
        mu_02 = moments[0, 2]
        mu_11 = -moments[1, 1]
        mu_20 = moments[2, 0]
        return np.array([[mu_02, mu_11], [mu_11, mu_20]]) * u.pix**2

    @lazyproperty
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """

        moments = self.moments_central
        if moments[0, 0] != 0:
            mu_norm = moments / moments[0, 0]
            covariance = self._check_covariance(
                np.array([[mu_norm[0, 2], mu_norm[1, 1]],
                          [mu_norm[1, 1], mu_norm[2, 0]]]))
            return covariance * u.pix**2
        else:
            return np.empty((2, 2)) * np.nan * u.pix**2

    @staticmethod
    def _check_covariance(covariance):
        """
        Check and modify the covariance matrix in the case of
        "infinitely" thin detections.  This follows SExtractor's
        prescription of incrementally increasing the diagonal elements
        by 1/12.
        """

        increment = 1. / 12  # arbitrary SExtractor value
        value = (covariance[0, 0] * covariance[1, 1]) - covariance[0, 1]**2
        if value >= increment**2:
            return covariance
        else:
            covar = np.copy(covariance)
            while value < increment**2:
                covar[0, 0] += increment
                covar[1, 1] += increment
                value = (covar[0, 0] * covar[1, 1]) - covar[0, 1]**2
            return covar

    @lazyproperty
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """

        unit = u.pix**2  # eigvals unit
        if np.any(~np.isfinite(self.covariance.value)):
            return (np.nan, np.nan) * unit
        else:
            eigvals = np.linalg.eigvals(self.covariance.value)
            if np.any(eigvals < 0):  # negative variance
                return (np.nan, np.nan) * unit  # pragma: no cover
            return (np.max(eigvals), np.min(eigvals)) * unit

    @lazyproperty
    def semimajor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """

        # this matches SExtractor's A parameter
        return np.sqrt(self.covariance_eigvals[0])

    @lazyproperty
    def semiminor_axis_sigma(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """

        # this matches SExtractor's B parameter
        return np.sqrt(self.covariance_eigvals[1])

    @lazyproperty
    def eccentricity(self):
        """
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math:: e = \\sqrt{1 - \\frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """

        semimajor_var, semiminor_var = self.covariance_eigvals
        if semimajor_var == 0:
            return 0.  # pragma: no cover
        return np.sqrt(1. - (semiminor_var / semimajor_var))

    @lazyproperty
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.  The angle increases in the counter-clockwise direction.
        """

        covar_00, covar_01, _, covar_11 = self.covariance.flat
        if covar_00 < 0 or covar_11 < 0:  # negative variance
            return np.nan * u.deg  # pragma: no cover

        # Quantity output in radians because inputs are Quantities
        orient_radians = 0.5 * np.arctan2(2. * covar_01,
                                          (covar_00 - covar_11))
        return orient_radians.to(u.deg)

    @lazyproperty
    def elongation(self):
        """
        The ratio of the lengths of the semimajor and semiminor axes:

        .. math:: \\mathrm{elongation} = \\frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

        Note that this is the same as `SExtractor`_'s elongation
        parameter.
        """

        return self.semimajor_axis_sigma / self.semiminor_axis_sigma

    @lazyproperty
    def ellipticity(self):
        """
        ``1`` minus the ratio of the lengths of the semimajor and
        semiminor axes (or ``1`` minus the `elongation`):

        .. math:: \\mathrm{ellipticity} = 1 - \\frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.

        Note that this is the same as `SExtractor`_'s ellipticity
        parameter.
        """

        return 1.0 - (self.semiminor_axis_sigma / self.semimajor_axis_sigma)

    @lazyproperty
    def covar_sigx2(self):
        """
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\\sigma_x^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s X2 parameter.
        """

        return self.covariance[0, 0]

    @lazyproperty
    def covar_sigy2(self):
        """
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\\sigma_y^2`, in units of pixel**2.

        Note that this is the same as `SExtractor`_'s Y2 parameter.
        """

        return self.covariance[1, 1]

    @lazyproperty
    def covar_sigxy(self):
        """
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\\sigma_x \\sigma_y`, in units of
        pixel**2.

        Note that this is the same as `SExtractor`_'s XY parameter.
        """

        return self.covariance[0, 1]

    @lazyproperty
    def cxx(self):
        """
        `SExtractor`_'s CXX ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return ((np.cos(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.sin(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def cyy(self):
        """
        `SExtractor`_'s CYY ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return ((np.sin(self.orientation) / self.semimajor_axis_sigma)**2 +
                (np.cos(self.orientation) / self.semiminor_axis_sigma)**2)

    @lazyproperty
    def cxy(self):
        """
        `SExtractor`_'s CXY ellipse parameter in units of pixel**(-2).

        The ellipse is defined as

            .. math::
                cxx (x - \\bar{x})^2 + cxy (x - \\bar{x}) (y - \\bar{y}) +
                cyy (y - \\bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).  `SExtractor`_ reports that the
        isophotal limit of a source is well represented by :math:`R
        \\approx 3`.
        """

        return (2. * np.cos(self.orientation) * np.sin(self.orientation) *
                ((1. / self.semimajor_axis_sigma**2) -
                 (1. / self.semiminor_axis_sigma**2)))

    @lazyproperty
    def gini(self):
        """
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

        The Gini coefficient is calculated using the prescription from
        `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        as:

        .. math::
            G = \\frac{1}{\\left | \\bar{x} \\right | n (n - 1)}
            \\sum^{n}_{i} (2i - n - 1) \\left | x_i \\right |

        where :math:`\\bar{x}` is the mean over all pixel values
        :math:`x_i`.

        The Gini coefficient is a way of measuring the inequality in a
        given set of values.  In the context of galaxy morphology, it
        measures how the light of a galaxy image is distributed among
        its pixels.  A Gini coefficient value of 0 corresponds to a
        galaxy image with the light evenly distributed over all pixels
        while a Gini coefficient value of 1 represents a galaxy image
        with all its light concentrated in just one pixel.
        """

        npix = np.size(self._data_values)
        normalization = (np.abs(np.mean(self._data_values)) * npix *
                         (npix - 1))
        kernel = ((2. * np.arange(1, npix + 1) - npix - 1) *
                  np.abs(np.sort(self._data_values)))

        return np.sum(kernel) / normalization


def source_properties(data, segment_img, error=None, mask=None,
                      background=None, filter_kernel=None, wcs=None,
                      labels=None):
    """
    Calculate photometry and morphological properties of sources defined
    by a labeled segmentation image.

    Parameters
    ----------
    data : array_like or `~astropy.units.Quantity`
        The 2D array from which to calculate the source photometry and
        properties.  ``data`` should be background-subtracted.
        Non-finite ``data`` values (NaN and +/- inf) are automatically
        masked.

    segment_img : `SegmentationImage` or array_like (int)
        A 2D segmentation image, either as a `SegmentationImage` object
        or an `~numpy.ndarray`, with the same shape as ``data`` where
        sources are labeled by different positive integer values.  A
        value of zero is reserved for the background.

    error : array_like or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data`` array.
        ``error`` is assumed to include *all* sources of error,
        including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`) .  ``error`` must have the
        same shape as the input ``data``.  Non-finite ``error`` values
        (NaN and +/- inf) are not automatically masked, unless they are
        at the same position of non-finite values in the input ``data``
        array.  Such pixels can be masked using the ``mask`` keyword.
        See the Notes section below for details on the error
        propagation.

    mask : array_like (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations.  Non-finite
        values (NaN and +/- inf) in the input ``data`` are automatically
        masked.

    background : float, array_like, or `~astropy.units.Quantity`, optional
        The background level that was *previously* present in the input
        ``data``.  ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``.  Inputting the
        ``background`` merely allows for its properties to be measured
        within each source segment.  The input ``background`` does *not*
        get subtracted from the input ``data``, which should already be
        background-subtracted.  Non-finite ``background`` values (NaN
        and +/- inf) are not automatically masked, unless they are at
        the same position of non-finite values in the input ``data``
        array.  Such pixels can be masked using the ``mask`` keyword.

    filter_kernel : array-like (2D) or `~astropy.convolution.Kernel2D`, optional
        The 2D array of the kernel used to filter the data prior to
        calculating the source centroid and morphological parameters.
        The kernel should be the same one used in defining the source
        segments, i.e. the detection image (e.g., see
        :func:`~photutils.segmentation.detect_sources`).  If `None`,
        then the unfiltered ``data`` will be used instead.

    wcs : `None` or WCS object, optional
        A world coordinate system (WCS) transformation that supports the
        `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`).  If `None`, then all sky-based
        properties will be set to `None`.

    labels : int, array-like (1D, int)
        The segmentation labels for which to calculate source
        properties.  If `None` (default), then the properties will be
        calculated for all labeled sources.

    Returns
    -------
    output : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.

    Notes
    -----
    `SExtractor`_'s centroid and morphological parameters are always
    calculated from a filtered "detection" image, i.e. the image used to
    define the segmentation image.  The usual downside of the filtering
    is the sources will be made more circular than they actually are.
    If you wish to reproduce `SExtractor`_ centroid and morphology
    results, then input a filtered and background-subtracted "detection"
    image into the ``filtered_data`` keyword.  If ``filtered_data`` is
    `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values (``filtered_data`` or ``data``) within the
    source segment are set to zero when calculating morphological
    properties based on image moments.  Negative values could occur, for
    example, if the segmentation image was defined from a different
    image (e.g., different bandpass) or if the background was
    oversubtracted. Note that
    `~photutils.segmentation.SourceProperties.source_sum` always
    includes the contribution of negative ``data`` values.

    The input ``error`` is assumed to include *all* sources of error,
    including the Poisson error of the sources.
    `~photutils.segmentation.SourceProperties.source_sum_err` is simply
    the quadrature sum of the pixel-wise total errors over the
    non-masked pixels within the source segment:

    .. math:: \\Delta F = \\sqrt{\\sum_{i \\in S}
              \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.segmentation.SourceProperties.source_sum_err`, :math:`S`
    are the non-masked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    .. _SExtractor: https://www.astromatic.net/software/sextractor

    See Also
    --------
    SegmentationImage, SourceProperties, detect_sources

    Examples
    --------
    >>> import numpy as np
    >>> from photutils import SegmentationImage, source_properties
    >>> image = np.arange(16.).reshape(4, 4)
    >>> print(image)  # doctest: +SKIP
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]
     [12. 13. 14. 15.]]
    >>> segm = SegmentationImage([[1, 1, 0, 0],
    ...                           [1, 0, 0, 2],
    ...                           [0, 0, 2, 2],
    ...                           [0, 2, 2, 0]])
    >>> props = source_properties(image, segm)

    Print some properties of the first object (labeled with ``1`` in the
    segmentation image):

    >>> props[0].id  # id corresponds to segment label number
    1
    >>> props[0].centroid  # doctest: +FLOAT_CMP
    <Quantity [0.8, 0.2] pix>
    >>> props[0].source_sum  # doctest: +FLOAT_CMP
    5.0
    >>> props[0].area  # doctest: +FLOAT_CMP
    <Quantity 3. pix2>
    >>> props[0].max_value  # doctest: +FLOAT_CMP
    4.0

    Print some properties of the second object (labeled with ``2`` in
    the segmentation image):

    >>> props[1].id  # id corresponds to segment label number
    2
    >>> props[1].centroid  # doctest: +FLOAT_CMP
    <Quantity [2.36363636, 2.09090909] pix>
    >>> props[1].perimeter  # doctest: +FLOAT_CMP
    <Quantity 5.41421356 pix>
    >>> props[1].orientation  # doctest: +FLOAT_CMP
    <Quantity -42.4996777 deg>
    """

    if not isinstance(segment_img, SegmentationImage):
        segment_img = SegmentationImage(segment_img)

    if segment_img.shape != data.shape:
        raise ValueError('segment_img and data must have the same shape.')

    # filter the data once, instead of repeating for each source
    if filter_kernel is not None:
        filtered_data = _filter_data(data, filter_kernel, mode='constant',
                                     fill_value=0.0, check_normalization=True)
    else:
        filtered_data = None

    if labels is None:
        labels = segment_img.labels
    labels = np.atleast_1d(labels)

    sources_props = []
    for label in labels:
        if label not in segment_img.labels:
            warnings.warn('label {} is not in the segmentation image.'
                          .format(label), AstropyUserWarning)
            continue  # skip invalid labels

        sources_props.append(SourceProperties(
            data, segment_img, label, filtered_data=filtered_data,
            error=error, mask=mask, background=background, wcs=wcs))

    if not sources_props:
        raise ValueError('No sources are defined.')

    return SourceCatalog(sources_props, wcs=wcs)


class _SourceCatalog:
    """
    Class to hold source catalogs.
    """

    def __init__(self, properties_list, wcs=None):
        if isinstance(properties_list, SourceProperties):
            self._data = [properties_list]
        elif isinstance(properties_list, list):
            if not properties_list:
                raise ValueError('properties_list must not be an empty list.')
            self._data = properties_list
        else:
            raise ValueError('invalid input.')

        self.wcs = wcs
        self._cache = {}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __delitem__(self, index):
        del self._data[index]

    def __iter__(self):
        for i in self._data:
            yield i

    def __str__(self):
        cls_name = '<{0}.{1}>'.format(self.__class__.__module__,
                                      self.__class__.__name__)
        fmt = ['Catalog length: {0}'.format(len(self))]

        return '{}\n'.format(cls_name) + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        if attr not in self._cache:
            values = [getattr(p, attr) for p in self._data]

            if isinstance(values[0], u.Quantity):
                # turn list of Quantities into a Quantity array
                values = u.Quantity(values)
            if isinstance(values[0], SkyCoord):  # pragma: no cover
                # failsafe: turn list of SkyCoord into a SkyCoord array
                values = SkyCoord(values)

            self._cache[attr] = values

        return self._cache[attr]

    @lazyproperty
    def _null_values(self):
        """
        Return an array of np.nan values.

        Used by SkyCoord properties if ``wcs`` is `None`.
        """
        values = np.empty(len(self))
        values.fill(np.nan)
        return values

    @lazyproperty
    def background_at_centroid(self):
        background = self._data[0]._background
        if background is None:
            return self._none_list
        else:
            from scipy.ndimage import map_coordinates

            values = map_coordinates(background,
                                     [[self.ycentroid.value],
                                      [self.xcentroid.value]], order=1,
                                     mode='nearest')[0]

            mask = np.isfinite(self.xcentroid) & np.isfinite(self.ycentroid)
            values[~mask] = np.nan

            return values * self._data[0]._data_unit

    @lazyproperty
    def sky_centroid(self):
        if self.wcs is None:
            return self._none_list
        else:
            # For a large catalog, it's much faster to calculate world
            # coordinates using the complete list of (x, y) instead of
            # looping through the individual (x, y).  It's also much
            # faster to recalculate the world coordinates than to create a
            # SkyCoord array from a loop-generated SkyCoord list.  The
            # assumption here is that the wcs is the same for each
            # SourceProperties instance.
            return _pixel_to_world(self.xcentroid.value, self.ycentroid.value,
                                   self.wcs)

    @lazyproperty
    def sky_centroid_icrs(self):
        if self.wcs is None:
            return self._none_list
        else:
            return self.sky_centroid.icrs

    @lazyproperty
    def sky_bbox_ll(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'll', self.wcs)

    @lazyproperty
    def sky_bbox_ul(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'ul', self.wcs)

    @lazyproperty
    def sky_bbox_lr(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'lr', self.wcs)

    @lazyproperty
    def sky_bbox_ur(self):
        if self.wcs is None:
            return self._none_list
        else:
            return _calc_sky_bbox_corner(self.bbox, 'ur', self.wcs)

    def to_table(self, columns=None, exclude_columns=None):
        """
        Construct a `~astropy.table.QTable` of source properties from a
        `SourceCatalog` object.

        If ``columns`` or ``exclude_columns`` are not input, then the
        `~astropy.table.QTable` will include a default list of
        scalar-valued properties.

        Multi-dimensional properties, e.g.
        `~photutils.segmentation.SourceProperties.data_cutout`, can be
        included in the ``columns`` input, but they will not be
        preserved when writing the table to a file.  This is a
        limitation of multi-dimensional columns in astropy tables.

        Parameters
        ----------
        columns : str or list of str, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`.  The allowed column names are any
            of the attributes of `SourceProperties`.

        exclude_columns : str or list of str, optional
            Names of columns to exclude from the default columns in the
            output `~astropy.table.QTable`.  The default columns are
            defined in the
            ``photutils.segmentation.properties.DEFAULT_COLUMNS``
            variable.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of source properties with one row per source.

        See Also
        --------
        SegmentationImage, SourceProperties, source_properties, detect_sources

        Examples
        --------
        >>> import numpy as np
        >>> from photutils import source_properties
        >>> image = np.arange(16.).reshape(4, 4)
        >>> print(image)  # doctest: +SKIP
        [[ 0.  1.  2.  3.]
         [ 4.  5.  6.  7.]
         [ 8.  9. 10. 11.]
         [12. 13. 14. 15.]]
        >>> segm = SegmentationImage([[1, 1, 0, 0],
        ...                           [1, 0, 0, 2],
        ...                           [0, 0, 2, 2],
        ...                           [0, 2, 2, 0]])
        >>> cat = source_properties(image, segm)
        >>> columns = ['id', 'xcentroid', 'ycentroid', 'source_sum']
        >>> tbl = cat.to_table(columns=columns)
        >>> tbl['xcentroid'].info.format = '.10f'  # optional format
        >>> tbl['ycentroid'].info.format = '.10f'  # optional format
        >>> print(tbl)
        id  xcentroid    ycentroid   source_sum
                pix          pix
        --- ------------ ------------ ----------
        1 0.2000000000 0.8000000000        5.0
        2 2.0909090909 2.3636363636       55.0
        """

        return _properties_table(self, columns=columns,
                                 exclude_columns=exclude_columns)


def _properties_table(obj, columns=None, exclude_columns=None):
    """
    Construct a `~astropy.table.QTable` of source properties from a
    `SourceProperties` or `SourceCatalog` object.

    Parameters
    ----------
    obj : `SourceProperties` or `SourceCatalog` instance
        The object containing the source properties.

    columns : str or list of str, optional
        Names of columns, in order, to include in the output
        `~astropy.table.QTable`.  The allowed column names are any
        of the attributes of `SourceProperties`.

    exclude_columns : str or list of str, optional
        Names of columns to exclude from the default columns in the
        output `~astropy.table.QTable`.  The default columns are defined
        in the ``photutils.segmentation.properties.DEFAULT_COLUMNS``
        variable.

    Returns
    -------
    table : `~astropy.table.QTable`
        A table of source properties with one row per source.
    """

    # start with the default columns
    columns_all = DEFAULT_COLUMNS

    table_columns = None
    if exclude_columns is not None:
        table_columns = [s for s in columns_all if s not in exclude_columns]
    if columns is not None:
        table_columns = np.atleast_1d(columns)
    if table_columns is None:
        table_columns = columns_all

    tbl = QTable()
    for column in table_columns:
        values = getattr(obj, column)

        if isinstance(obj, SourceProperties):
            # turn scalar values into length-1 arrays because QTable
            # column assignment requires an object with a length
            values = np.atleast_1d(values)

            # Unfortunately np.atleast_1d creates an array of SkyCoord
            # instead of a SkyCoord array (Quantity does work correctly
            # with np.atleast_1d).  Here we make a SkyCoord array for
            # the output table column.
            if isinstance(values[0], SkyCoord):
                values = SkyCoord(values)  # length-1 SkyCoord array

        tbl[column] = values

    return tbl



