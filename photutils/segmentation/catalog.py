# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tools for calculating the properties of sources defined by a
segmentation image.
"""

import functools
import inspect
import warnings
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.utils import lazyproperty
from scipy.ndimage import map_coordinates

from photutils.aperture import BoundingBox
from photutils.morphology import gini as gini_func
from photutils.segmentation._catalog_centroid_refine import _CentroidRefiner
from photutils.segmentation._catalog_local_bkg import _LocalBackground
from photutils.segmentation._catalog_photometry import _Photometry
from photutils.segmentation.core import SegmentationImage
from photutils.utils._catalog_helpers import (as_scalar, get_lazyproperties,
                                              slice_composition_helper)
from photutils.utils._deprecation import (_get_future_column_names,
                                          create_empty_deprecated_qtable,
                                          deprecated_getattr,
                                          deprecated_positional_kwargs,
                                          deprecated_renamed_argument)
from photutils.utils._misc import _get_meta
from photutils.utils._quantity_helpers import process_quantities
from photutils.utils._shape_properties import _ShapeProperties
from photutils.utils.cutouts import CutoutImage

__all__ = ['SourceCatalog']


# Default table columns for `to_table()` output
DEFAULT_COLUMNS = ['label', 'x_centroid', 'y_centroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_axis', 'semiminor_axis', 'orientation',
                   'eccentricity', 'min_value', 'max_value', 'segment_flux',
                   'segment_flux_err', 'kron_flux', 'kron_flux_err']

# Remove in 4.0
_DEPRECATED_ATTRIBUTES = {
    'add_extra_property': 'add_property',
    'apermask_method': 'aperture_mask_method',
    'background': 'background_cutout',
    'background_ma': 'background_cutout_masked',
    'convdata': 'conv_data_cutout',
    'convdata_ma': 'conv_data_cutout_masked',
    'covar_sigx2': 'covariance_xx',
    'covar_sigxy': 'covariance_xy',
    'covar_sigy2': 'covariance_yy',
    'cutout_maxval_index': 'cutout_max_value_index',
    'cutout_minval_index': 'cutout_min_value_index',
    'cxx': 'ellipse_cxx',
    'cxy': 'ellipse_cxy',
    'cyy': 'ellipse_cyy',
    'data': 'data_cutout',
    'data_ma': 'data_cutout_masked',
    'error': 'error_cutout',
    'error_ma': 'error_cutout_masked',
    'extra_properties': 'custom_properties',
    'fluxfrac_radius': 'flux_radius',
    'get_label': 'select_label',
    'get_labels': 'select_labels',
    'kron_fluxerr': 'kron_flux_err',
    'localbkg_width': 'local_bkg_width',
    'maxval_index': 'max_value_index',
    'maxval_xindex': 'max_value_xindex',
    'maxval_yindex': 'max_value_yindex',
    'minval_index': 'min_value_index',
    'minval_xindex': 'min_value_xindex',
    'minval_yindex': 'min_value_yindex',
    'nlabels': 'n_labels',
    'remove_extra_properties': 'remove_properties',
    'remove_extra_property': 'remove_property',
    'rename_extra_property': 'rename_property',
    'segment': 'segment_cutout',
    'segment_fluxerr': 'segment_flux_err',
    'segment_ma': 'segment_cutout_masked',
    'semimajor_sigma': 'semimajor_axis',
    'semiminor_sigma': 'semiminor_axis',
    'xcentroid': 'x_centroid',
    'xcentroid_quad': 'x_centroid_quad',
    'xcentroid_win': 'x_centroid_win',
    'ycentroid': 'y_centroid',
    'ycentroid_quad': 'y_centroid_quad',
    'ycentroid_win': 'y_centroid_win',
}

_DEPRECATED_META_KEYS = {
    'localbkg_width': 'local_bkg_width',
    'apermask_method': 'aperture_mask_method',
}


def use_detcat(method):
    """
    Return a decorated method where it will return the value from the
    detection image catalog instead of using the method to calculate it.

    Parameters
    ----------
    method : function
        The method to be decorated.

    Returns
    -------
    decorator : function
        The decorated method.
    """
    @functools.wraps(method)
    def _use_detcat(self, *args, **kwargs):
        if self._detection_catalog is None:
            return method(self, *args, **kwargs)

        return getattr(self._detection_catalog, method.__name__)

    return _use_detcat


class SourceCatalog:
    """
    Class to create a catalog of photometry and morphological properties
    for sources defined by a segmentation image.

    Parameters
    ----------
    data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array from which to calculate the source photometry and
        properties. If ``convolved_data`` is input, then a convolved
        version of ``data`` will be used instead of ``data`` to
        calculate the source centroid and morphological properties.
        Source photometry is always measured from ``data``. For
        accurate source properties and photometry, ``data`` should be
        background-subtracted. Non-finite ``data`` values (NaN and inf)
        are automatically masked.

    segmentation_image : `~photutils.segmentation.SegmentationImage`
        A `~photutils.segmentation.SegmentationImage` object defining
        the sources.

    convolved_data : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The 2D array used to calculate the source centroid and
        morphological properties. Typically, ``convolved_data``
        should be the input ``data`` array convolved by the
        same smoothing kernel that was applied to the detection
        image when deriving the source segments (e.g., see
        :func:`~photutils.segmentation.detect_sources`). If
        ``convolved_data`` is `None`, then the unconvolved ``data`` will
        be used instead. Non-finite ``convolved_data`` values (NaN and
        inf) are not automatically masked, unless they are at the same
        position of non-finite values in the input ``data`` array. Such
        pixels can be masked using the ``mask`` keyword.

    error : 2D `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        The total error array corresponding to the input ``data``
        array. ``error`` is assumed to include *all* sources of
        error, including the Poisson error of the sources (see
        `~photutils.utils.calc_total_error`). ``error`` must have
        the same shape as the input ``data``. If ``data`` is a
        `~astropy.units.Quantity` array then ``error`` must be a
        `~astropy.units.Quantity` array (and vice versa) with identical
        units. Non-finite ``error`` values (NaN and inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array. Such pixels can
        be masked using the ``mask`` keyword. See the Notes section
        below for details on the error propagation.

    mask : 2D `~numpy.ndarray` (bool), optional
        A boolean mask with the same shape as ``data`` where a `True`
        value indicates the corresponding element of ``data`` is masked.
        Masked data are excluded from all calculations. Non-finite
        values (NaN and inf) in the input ``data`` are automatically
        masked.

    background : float, 2D `~numpy.ndarray`, or `~astropy.units.Quantity`, \
            optional
        The background level that was *previously* present in the input
        ``data``. ``background`` may either be a scalar value or a 2D
        image with the same shape as the input ``data``. If ``data``
        is a `~astropy.units.Quantity` array then ``background`` must
        be a `~astropy.units.Quantity` array (and vice versa) with
        identical units. Inputing the ``background`` merely allows
        for its properties to be measured within each source segment.
        The input ``background`` does *not* get subtracted from the
        input ``data``, which should already be background-subtracted.
        Non-finite ``background`` values (NaN and inf) are not
        automatically masked, unless they are at the same position of
        non-finite values in the input ``data`` array. Such pixels can
        be masked using the ``mask`` keyword.

    wcs : WCS object or `None`, optional
        A world coordinate system (WCS) transformation that
        supports the `astropy shared interface for WCS
        <https://docs.astropy.org/en/stable/wcs/wcsapi.html>`_ (e.g.,
        `astropy.wcs.WCS`, `gwcs.wcs.WCS`). If `None`, then all
        sky-based properties will be set to `None`. This keyword will be
        ignored if ``detection_catalog`` is input.

    local_bkg_width : int, optional
        The width of the rectangular annulus used to compute a
        local background around each source. If zero, then no local
        background subtraction is performed. The local background
        affects the ``min_value``, ``max_value``, ``segment_flux``,
        ``kron_flux``, and ``flux_radius`` properties. It is also used
        when calculating circular and Kron aperture photometry (i.e.,
        `circular_photometry` and `kron_photometry`). It does not affect
        the moment-based morphological properties of the source.

    aperture_mask_method : {'correct', 'mask', 'none'}, optional
        The method used to handle neighboring sources when performing
        aperture photometry (e.g., circular apertures or elliptical Kron
        apertures). This parameter also affects the Kron radius. The
        options are:

        * 'correct': replace pixels assigned to neighboring sources by
          replacing them with pixels on the opposite side of the source
          center (equivalent to MASK_TYPE=CORRECT in SourceExtractor).

        * 'mask': mask pixels assigned to neighboring sources
          (equivalent to MASK_TYPE=BLANK in SourceExtractor).

        * 'none': do not mask any pixels (equivalent to MASK_TYPE=NONE
          in SourceExtractor).

        This keyword will be ignored if ``detection_catalog`` is
        input. In that case, the ``aperture_mask_method`` set in the
        ``detection_catalog`` will be used.

    kron_params : tuple of 2 or 3 floats, optional
        A list of parameters used to determine the Kron aperture.
        The first item is the scaling parameter of the unscaled Kron
        radius and the second item represents the minimum value for
        the unscaled Kron radius in pixels. The optional third item is
        the minimum circular radius in pixels. If ``kron_params[0]``
        * `kron_radius` * sqrt(`semimajor_axis` * `semiminor_axis`)
        is less than or equal to this radius, then the Kron aperture
        will be a circle with this minimum radius. This keyword will be
        ignored if ``detection_catalog`` is input.

    detection_catalog : `SourceCatalog`, optional
        A `SourceCatalog` object for the detection image. The
        segmentation image used to create the detection catalog must be
        the same one input to ``segmentation_image``. If input, then
        the detection catalog source centroids and morphological/shape
        properties will be returned instead of calculating them from
        the input ``data``. The detection catalog centroids and shape
        properties will also be used to perform aperture photometry
        (i.e., circular and Kron). If ``detection_catalog`` is
        input, then the input ``wcs``, ``aperture_mask_method``, and
        ``kron_params`` keywords will be ignored. This keyword affects
        `circular_photometry` (including returned apertures), all Kron
        parameters (Kron radius, flux, flux errors, apertures, and
        custom `kron_photometry`), and `flux_radius` (which is based on
        the Kron flux).

    progress_bar : bool, optional
        Whether to display a progress bar when calculating
        some properties (e.g., ``kron_radius``, ``kron_flux``,
        ``flux_radius``, ``circular_photometry``, ``centroid_win``,
        ``centroid_quad``). The progress bar requires that the `tqdm
        <https://tqdm.github.io/>`_ optional dependency be installed.

    Notes
    -----
    ``data`` should be background-subtracted for accurate source
    photometry and properties. The previously-subtracted background can
    be passed into this class to calculate properties of the background
    for each source.

    Note that this class does not convert input data in
    surface-brightness units to flux or counts. Conversion from
    surface-brightness units should be performed before using this
    class.

    `SourceExtractor`_'s centroid and morphological parameters are
    always calculated from a convolved, or filtered, "detection"
    image (``convolved_data``), i.e., the image used to define the
    segmentation image. The usual downside of the filtering is the
    sources will be made more circular than they actually are. If
    you wish to reproduce `SourceExtractor`_ centroid and morphology
    results, then input the ``convolved_data``. If ``convolved_data``
    is `None`, then the unfiltered ``data`` will be used for the source
    centroid and morphological parameters.

    Negative data values within the source segment are set to zero
    when calculating morphological properties based on image moments.
    Negative values could occur, for example, if the segmentation
    image was defined from a different image (e.g., different
    bandpass) or if the background was oversubtracted. However,
    `~photutils.segmentation.SourceCatalog.segment_flux` always includes
    the contribution of negative ``data`` values.

    The input ``error`` array is assumed to include *all* sources
    of error, including the Poisson error of the sources.
    `~photutils.segmentation.SourceCatalog.segment_flux_err` is simply
    the quadrature sum of the pixel-wise total errors over the
    unmasked pixels within the source segment:

    .. math::

        \\Delta F = \\sqrt{\\sum_{i \\in S}
            \\sigma_{\\mathrm{tot}, i}^2}

    where :math:`\\Delta F` is
    `~photutils.segmentation.SourceCatalog.segment_flux_err`,
    :math:`S` are the unmasked pixels in the source segment, and
    :math:`\\sigma_{\\mathrm{tot}, i}` is the input ``error`` array.

    Custom errors for source segments can be calculated using the
    `~photutils.segmentation.SourceCatalog.error_cutout_masked` and
    `~photutils.segmentation.SourceCatalog.background_cutout_masked`
    properties, which are 2D `~numpy.ma.MaskedArray` cutout versions of
    the input ``error`` and ``background`` arrays. The mask is `True`
    for pixels outside the source segment, masked pixels from the
    ``mask`` input, or any non-finite ``data`` values (NaN and inf).

    **Scalar vs. Multi-source Catalogs**

    A `SourceCatalog` can represent a single source or multiple
    sources. Most properties adapt their return type accordingly: for
    a multi-source catalog, properties return arrays or lists (one
    element per source); for a single-source (scalar) catalog, the
    same properties return a scalar value or a single object. For
    example, `kron_aperture` returns a list of aperture objects for a
    multi-source catalog, but a single aperture object for a scalar
    catalog. Similarly, `data_cutout` returns a list of 2D cutout arrays
    for a multi-source catalog, but a single 2D array for a scalar
    catalog. A scalar catalog is created when the a multi-source catalog
    is indexed to select a single source.

    .. _SourceExtractor: https://sextractor.readthedocs.io/en/latest/
    """

    @deprecated_renamed_argument('segment_img', 'segmentation_image', '3.0',
                                 until='4.0')
    @deprecated_renamed_argument('localbkg_width', 'local_bkg_width',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('apermask_method', 'aperture_mask_method',
                                 '3.0', until='4.0')
    @deprecated_renamed_argument('detection_cat', 'detection_catalog', '3.0',
                                 until='4.0')
    def __init__(self, data, segmentation_image, *, convolved_data=None,
                 error=None, mask=None, background=None, wcs=None,
                 local_bkg_width=0, aperture_mask_method='correct',
                 kron_params=(2.5, 1.4, 0.0), detection_catalog=None,
                 progress_bar=False):

        inputs = (data, convolved_data, error, background)
        names = ('data', 'convolved_data', 'error', 'background')
        inputs, unit = process_quantities(inputs, names)
        (data, convolved_data, error, background) = inputs

        self._data_unit = unit
        self._data = self._validate_array(data, 'data', shape=False)
        self._convolved_data = self._validate_array(convolved_data,
                                                    'convolved_data')
        self._segmentation_image = self._validate_segmentation_image(
            segmentation_image)
        self._error = self._validate_array(error, 'error')
        self._mask = self._validate_array(mask, 'mask')
        self._background = self._validate_array(background, 'background')
        self.wcs = wcs
        self.local_bkg_width = self._validate_local_bkg_width(
            local_bkg_width)
        self.aperture_mask_method = self._validate_aperture_mask_method(
            aperture_mask_method)
        self.kron_params = self._validate_kron_params(kron_params)
        self.progress_bar = progress_bar

        # Needed for ordering and isscalar
        # NOTE: calculate slices before labels for performance.
        #       _labels is initially always a non-scalar array, but
        #       it can become a numpy scalar after indexing/slicing.
        self._slices = self._segmentation_image.slices
        self._labels = self._segmentation_image.labels

        if self._labels.shape == (0,):
            msg = 'segmentation_image must have at least one non-zero label'
            raise ValueError(msg)

        self._detection_catalog = self._validate_detection_catalog(
            detection_catalog)
        attrs = ('wcs', 'aperture_mask_method', 'kron_params')
        if self._detection_catalog is not None:
            for attr in attrs:
                setattr(self, attr, getattr(self._detection_catalog, attr))

        if convolved_data is None:
            self._convolved_data = self._data

        self._aperture_mask_kwargs = {
            'circ': {'method': 'exact'},
            'kron': {'method': 'exact'},
            'flux_radius': {'method': 'exact'},
            'cen_win': {'method': 'center'},
        }

        self.default_columns = DEFAULT_COLUMNS
        self._custom_properties = []
        self._flux_radius_cache = {}
        self.meta = _get_meta()
        self._update_meta()

    def _validate_segmentation_image(self, segmentation_image):
        if not isinstance(segmentation_image, SegmentationImage):
            msg = 'segmentation_image must be a SegmentationImage'
            raise TypeError(msg)
        if segmentation_image.shape != self._data.shape:
            msg = 'segmentation_image and data must have the same shape'
            raise ValueError(msg)
        return segmentation_image

    def _validate_array(self, array, name, *, shape=True):
        if name == 'mask' and array is np.ma.nomask:
            array = None
        if array is not None:
            # UFuncTypeError is raised when subtracting float
            # local_background from int data; convert to float
            array = np.asanyarray(array)
            if array.ndim != 2:
                msg = f'{name} must be a 2D array'
                raise ValueError(msg)
            if shape and array.shape != self._data.shape:
                msg = f'data and {name} must have the same shape'
                raise ValueError(msg)
        return array

    @staticmethod
    def _validate_local_bkg_width(local_bkg_width):
        if local_bkg_width < 0:
            msg = 'local_bkg_width must be >= 0'
            raise ValueError(msg)
        local_bkg_width_int = int(local_bkg_width)
        if local_bkg_width_int != local_bkg_width:
            msg = 'local_bkg_width must be an integer'
            raise ValueError(msg)
        return local_bkg_width_int

    @staticmethod
    def _validate_aperture_mask_method(aperture_mask_method):
        if aperture_mask_method not in ('none', 'mask', 'correct'):
            msg = 'Invalid aperture_mask_method value'
            raise ValueError(msg)
        return aperture_mask_method

    @staticmethod
    def _validate_kron_params(kron_params):
        if np.ndim(kron_params) != 1:
            msg = 'kron_params must be 1D'
            raise ValueError(msg)
        nparams = len(kron_params)
        if nparams not in (2, 3):
            msg = 'kron_params must have 2 or 3 elements'
            raise ValueError(msg)
        if kron_params[0] <= 0:
            msg = 'kron_params[0] must be > 0'
            raise ValueError(msg)
        if kron_params[1] <= 0:
            msg = 'kron_params[1] must be > 0'
            raise ValueError(msg)
        if nparams == 3 and kron_params[2] < 0:
            msg = 'kron_params[2] must be >= 0'
            raise ValueError(msg)
        return tuple(kron_params)

    def _validate_detection_catalog(self, detection_catalog):
        if detection_catalog is None:
            return None

        if not isinstance(detection_catalog, SourceCatalog):
            msg = 'detection_catalog must be a SourceCatalog instance'
            raise TypeError(msg)
        if not np.array_equal(detection_catalog._segmentation_image,
                              self._segmentation_image):
            msg = ('detection_catalog must have same segmentation_image '
                   'as the input segmentation_image')
            raise ValueError(msg)
        return detection_catalog

    def _update_meta(self):
        meta_values = {}
        attrs = ('local_bkg_width', 'aperture_mask_method', 'kron_params')
        for attr in attrs:
            meta_values[attr] = getattr(self, attr)

        if not _get_future_column_names():
            for old_name, new_name in _DEPRECATED_META_KEYS.items():
                if new_name in meta_values:
                    meta_values[old_name] = meta_values[new_name]

        self.meta.update(meta_values)

    def _set_semode(self):
        # SE emulation
        self._aperture_mask_kwargs = {
            'circ': {'method': 'subpixel', 'subpixels': 5},
            'kron': {'method': 'center'},
            'flux_radius': {'method': 'subpixel', 'subpixels': 5},
            'cen_win': {'method': 'subpixel', 'subpixels': 11},
        }

    @property
    def _properties(self):
        """
        A list of all class properties, include lazyproperties (even in
        superclasses).

        The result is cached on the class to avoid repeated
        introspection via `inspect.getmembers`.
        """
        cls = self.__class__
        attr = '_cached_properties'
        # Subclasses get their own property list
        if attr not in cls.__dict__:
            def isproperty(obj):
                return isinstance(obj, property)

            setattr(cls, attr,
                    [i[0] for i in inspect.getmembers(
                        cls, predicate=isproperty)])
        return getattr(cls, attr)

    @property
    def properties(self):
        """
        A list of built-in source properties.
        """
        lazyproperties = [name for name in self._lazyproperties if not
                          name.startswith('_')]
        lazyproperties.remove('isscalar')
        lazyproperties.remove('n_labels')
        lazyproperties.extend(['label', 'labels', 'slices'])
        lazyproperties.sort()
        return lazyproperties

    @property
    def _lazyproperties(self):
        """
        A list of all class lazyproperties (even in superclasses).
        """
        return get_lazyproperties(self.__class__)

    @staticmethod
    def _index_object_list(lst, index):
        """
        Index a list of heterogeneous objects using numpy-style
        indexing.

        A numpy object array is used to support fancy and boolean
        indices on lists of tuples or other structured objects.

        A sentinel ``None`` is appended (and then removed) to prevent
        numpy from recursing into nested sequences (e.g., tuples of
        slices).

        Parameters
        ----------
        lst : list
            The list of objects to index.

        index : int, slice, list, or array
            The index to apply to the list.

        Returns
        -------
        result : list or object
            A list for array results or the element itself for scalar
            (integer) indices.
        """
        result = np.array([*lst, None], dtype=object)[:-1][index]
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def __getitem__(self, index):
        if self.isscalar:
            msg = (f'A scalar {self.__class__.__name__!r} object cannot '
                   'be indexed')
            raise TypeError(msg)

        newcls = object.__new__(self.__class__)

        # Attributes defined in __init__ that are copied directly to the
        # new class
        init_attr = ('_data', '_segmentation_image', '_error', '_mask',
                     '_background', 'wcs', '_data_unit', '_convolved_data',
                     'local_bkg_width', 'aperture_mask_method',
                     'kron_params', 'default_columns', '_custom_properties',
                     'meta', '_aperture_mask_kwargs', 'progress_bar')
        for attr in init_attr:
            setattr(newcls, attr, getattr(self, attr))

        # _labels determines ordering and isscalar
        attr = '_labels'
        setattr(newcls, attr, getattr(self, attr)[index])

        # Need to slice detection_catalog, if input
        attr = '_detection_catalog'
        if getattr(self, attr) is None:
            setattr(newcls, attr, None)
        else:
            setattr(newcls, attr, getattr(self, attr)[index])

        attr = '_slices'
        value = self._index_object_list(getattr(self, attr), index)
        setattr(newcls, attr, value)

        # Slice the flux_radius cache values
        newcls._flux_radius_cache = {key: value[index]
                                     for key, value
                                     in self._flux_radius_cache.items()}

        # Evaluated lazyproperty objects and extra properties
        keys = (set(self.__dict__.keys())
                & (set(self._lazyproperties) | set(self._custom_properties)))
        # Composition helpers hold a reference back to the parent
        # catalog (or to a per-source array); they are handled
        # separately below so that any cached @lazyproperty values
        # stored on the helper are sliced and preserved on the new
        # instance (avoiding expensive recomputation on slicing).
        helper_specs = (
            ('_shape', _ShapeProperties,
             lambda: _ShapeProperties(newcls.moments_central)),
            ('_local_bkg', _LocalBackground,
             lambda: _LocalBackground(newcls)),
            ('_centroid_refiner', _CentroidRefiner,
             lambda: _CentroidRefiner(newcls)),
            ('_photometry', _Photometry,
             lambda: _Photometry(newcls)),
        )
        for helper_key, _, _ in helper_specs:
            keys.discard(helper_key)
        for key in keys:
            value = self.__dict__[key]

            # Do not insert attributes that are always scalar (e.g.,
            # isscalar, n_labels), i.e., not an array/list for each
            # source
            if np.isscalar(value):
                continue

            try:
                # Keep _<attr> lazyproperties as length-1 iterables;
                # _<attr> lazyproperties should not have @as_scalar applied
                if newcls.isscalar and key.startswith('_'):
                    if isinstance(value, np.ndarray):
                        val = value[:, np.newaxis][index]
                    else:
                        val = [value[index]]
                else:
                    val = value[index]
            except TypeError:
                # Apply fancy indices (e.g., array/list or bool
                # mask) to lists
                val = self._index_object_list(value, index)

            newcls.__dict__[key] = val

        # Rebind composition helpers to the sliced instance, copying
        # sliced cached lazyproperty values from the original helpers
        # so expensive per-source results (e.g., centroid_win,
        # local_background values) are preserved.
        for helper_key, helper_cls, helper_factory in helper_specs:
            old_helper = self.__dict__.get(helper_key)
            new_helper = slice_composition_helper(
                old_helper, helper_factory, helper_cls, index,
                isscalar_new=newcls.isscalar,
                index_object_list=self._index_object_list)
            if new_helper is not None:
                newcls.__dict__[helper_key] = new_helper
        return newcls

    def __str__(self):
        cls_name = f'<{self.__class__.__module__}.{self.__class__.__name__}>'
        with np.printoptions(threshold=25, edgeitems=5):
            fmt = [f'Length: {self.n_labels}', f'labels: {self.labels}']
        return f'{cls_name}\n' + '\n'.join(fmt)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        if self.isscalar:
            msg = f'Scalar {self.__class__.__name__!r} object has no len()'
            raise TypeError(msg)
        return self.n_labels

    def __iter__(self):
        for item in range(len(self)):
            yield self.__getitem__(item)

    # Remove in 4.0
    def __getattr__(self, name):
        return deprecated_getattr(self, name, _DEPRECATED_ATTRIBUTES,
                                  since='3.0', until='4.0')

    @lazyproperty
    def isscalar(self):
        """
        Whether the instance is scalar (e.g., a single source).
        """
        return self._labels.shape == ()

    @staticmethod
    def _has_len(value):
        if isinstance(value, str):
            return False
        try:
            # NOTE: cannot just check for __len__ attribute, because
            # it could exist, but raise an Exception for scalar objects
            len(value)
        except TypeError:
            return False
        return True

    def copy(self):
        """
        Return a deep copy of this SourceCatalog.

        Returns
        -------
        result : `SourceCatalog`
            A deep copy of this object.
        """
        return deepcopy(self)

    @property
    def custom_properties(self):
        """
        A list of the user-defined source properties.
        """
        return self._custom_properties

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def add_property(self, name, value, overwrite=False):
        """
        Add a user-defined property as a new attribute.

        For example, the property ``name`` can then be included in the
        `to_table` ``columns`` keyword list to output the results in the
        table.

        The complete list of user-defined properties is stored in the
        `custom_properties` attribute.

        Parameters
        ----------
        name : str
            The name of property. The name must not conflict with any of
            the built-in property names or attributes.

        value : array_like or float
            The value to assign.

        overwrite : bool, option
            If `True`, will overwrite the existing property ``name``.
        """
        internal_attributes = ((set(self.__dict__.keys())
                               | set(self._properties))
                               - set(self.custom_properties))
        if name in internal_attributes:
            msg = f'{name} cannot be set because it is a built-in attribute'
            raise ValueError(msg)

        if not overwrite:
            if hasattr(self, name):
                msg = (f'{name} already exists as an attribute. Set '
                       'overwrite=True to overwrite an existing attribute.')
                raise ValueError(msg)
            if name in self._custom_properties:
                msg = (f'{name} already exists in the custom_properties '
                       'attribute list.')
                raise ValueError(msg)

        property_error = False
        if self.isscalar:
            # This allows flux_radius to add len-1 array values for
            # scalar self
            if self._has_len(value) and len(value) == 1:
                value = value[0]

            if hasattr(value, 'isscalar'):
                # e.g., Quantity, SkyCoord, Time
                if not value.isscalar:
                    property_error = True
            elif not np.isscalar(value):
                property_error = True
        elif not self._has_len(value) or len(value) != self.n_labels:
            property_error = True
        if property_error:
            msg = ('value must have the same number of elements as the '
                   'catalog in order to add it as a property.')
            raise ValueError(msg)

        setattr(self, name, value)
        if name not in self._custom_properties:
            self._custom_properties.append(name)

    def remove_property(self, name):
        """
        Remove a user-defined property.

        The property must have been defined using `add_property`. The
        complete list of user-defined properties is stored in the
        `custom_properties` attribute.

        Parameters
        ----------
        name : str
            The name of the property to remove.
        """
        self.remove_properties(name)

    def remove_properties(self, names):
        """
        Remove user-defined properties.

        The properties must have been defined using `add_property`.
        The complete list of user-defined properties is stored in the
        `custom_properties` attribute.

        Parameters
        ----------
        names : list of str or str
            The names of the properties to remove.
        """
        if isinstance(names, str):
            names = [names]

        # We copy the list here to prevent changing the list in-place
        # during the for loop below, e.g., in case a user inputs
        # self.custom_properties to ``names``
        custom_properties = self._custom_properties.copy()

        for name in names:
            if name in custom_properties:
                delattr(self, name)
                custom_properties.remove(name)
            else:
                msg = f'{name} is not a defined property'
                raise ValueError(msg)
        self._custom_properties = custom_properties

    def rename_property(self, name, new_name):
        """
        Rename a user-defined property.

        The renamed property will remain at the same index in the
        `custom_properties` list.

        Parameters
        ----------
        name : str
            The old attribute name.

        new_name : str
            The new attribute name.
        """
        self.add_property(new_name, getattr(self, name))
        idx = self.custom_properties.index(name)
        self.remove_property(name)
        # Preserve the order of self.custom_properties
        self.custom_properties.remove(new_name)
        self.custom_properties.insert(idx, new_name)

    @lazyproperty
    def _null_objects(self):
        """
        Return `None` values.

        For example, this is used for SkyCoord properties if ``wcs`` is
        `None`.
        """
        return np.array([None] * self.n_labels)

    @lazyproperty
    def _null_values(self):
        """
        Return np.nan values.

        For example, this is used for background properties if
        ``background`` is `None`.
        """
        values = np.empty(self.n_labels)
        values.fill(np.nan)
        return values

    @lazyproperty
    def _data_cutouts(self):
        """
        A list of data cutouts using the segmentation image slices.
        """
        return [self._data[slc] for slc in self._slices_iter]

    @lazyproperty
    def _segmentation_image_cutouts(self):
        """
        A list of segmentation image cutouts using the segmentation
        image slices.
        """
        return [self._segmentation_image.data[slc]
                for slc in self._slices_iter]

    @lazyproperty
    def _mask_cutouts(self):
        """
        A list of mask cutouts using the segmentation image slices.

        If the input ``mask`` is None then a list of None is returned.
        """
        if self._mask is None:
            return self._null_objects
        return [self._mask[slc] for slc in self._slices_iter]

    @lazyproperty
    def _error_cutouts(self):
        """
        A list of error cutouts using the segmentation image slices.

        If the input ``error`` is None then a list of None is returned.
        """
        if self._error is None:
            return self._null_objects
        return [self._error[slc] for slc in self._slices_iter]

    @lazyproperty
    def _convdata_cutouts(self):
        """
        A list of convolved data cutouts using the segmentation image
        slices.
        """
        return [self._convolved_data[slc] for slc in self._slices_iter]

    @lazyproperty
    def _background_cutouts(self):
        """
        A list of background cutouts using the segmentation image
        slices.
        """
        if self._background is None:
            return self._null_objects
        return [self._background[slc] for slc in self._slices_iter]

    @staticmethod
    def _make_cutout_data_mask(data_cutout, mask_cutout):
        """
        Make a cutout data mask, combining both the input ``mask`` and
        non-finite ``data`` values.
        """
        data_mask = ~np.isfinite(data_cutout)
        if mask_cutout is not None:
            data_mask |= mask_cutout
        return data_mask

    def _make_cutout_data_masks(self, data_cutouts, mask_cutouts):
        """
        Make a list of cutout data masks, combining both the input
        ``mask`` and non-finite ``data`` values for each source.
        """
        data_masks = []
        for (data_cutout, mask_cutout) in zip(data_cutouts, mask_cutouts,
                                              strict=True):
            data_masks.append(self._make_cutout_data_mask(data_cutout,
                                                          mask_cutout))
        return data_masks

    @lazyproperty
    def _cutout_segment_masks(self):
        """
        Cutout boolean mask for source segment.

        The mask is `True` for all pixels (background and from other
        source segments) outside the source segment.
        """
        return [segm != label
                for label, segm
                in zip(self.labels,
                       self._segmentation_image_cutouts,
                       strict=True)]

    @lazyproperty
    def _cutout_data_masks(self):
        """
        Cutout boolean mask of non-finite ``data`` values combined with
        the input ``mask`` array.

        The mask is `True` for non-finite ``data`` values and where the
        input ``mask`` is `True`.
        """
        return self._make_cutout_data_masks(self._data_cutouts,
                                            self._mask_cutouts)

    @lazyproperty
    def _cutout_total_masks(self):
        """
        Boolean mask representing the combination of
        ``_cutout_segment_masks`` and ``_cutout_data_masks``.

        This mask is applied to ``data``, ``error``, and ``background``
        inputs when calculating properties.
        """
        masks = []
        for mask1, mask2 in zip(self._cutout_segment_masks,
                                self._cutout_data_masks, strict=True):
            masks.append(mask1 | mask2)
        return masks

    @lazyproperty
    def _moment_data_cutouts(self):
        """
        A list of 2D `~numpy.ndarray` cutouts from the (convolved) data.

        The following pixels are set to zero in these arrays:

        * pixels outside the source segment
        * any masked pixels from the input ``mask``
        * invalid convolved data values (NaN and inf)
        * negative convolved data values; negative pixels (especially
          at large radii) can give image moments that have negative
          variances.

        These arrays are used to derive moment-based properties.
        """
        cutouts = []
        for convdata_cutout, mask_cutout, segmmask_cutout in zip(
                self._convdata_cutouts, self._mask_cutouts,
                self._cutout_segment_masks, strict=True):

            convdata_mask = (~np.isfinite(convdata_cutout)
                             | (convdata_cutout < 0) | segmmask_cutout)

            if self._mask is not None:
                convdata_mask |= mask_cutout

            cutout = convdata_cutout.copy()
            cutout[convdata_mask] = 0.0
            cutouts.append(cutout)
        return cutouts

    def _prepare_cutouts(self, arrays, *, units=True, masked=False,
                         dtype=None):
        """
        Prepare cutouts by applying optional units, masks, or dtype.
        """
        if units and masked:
            msg = 'Both units and masked cannot be True'
            raise ValueError(msg)

        if dtype is not None:
            cutouts = [cutout.astype(dtype, copy=True) for cutout in arrays]
        else:
            cutouts = arrays

        if units and self._data_unit is not None:
            cutouts = [(cutout << self._data_unit) for cutout in cutouts]
        if masked:
            return [np.ma.masked_array(cutout, mask=mask)
                    for cutout, mask in zip(cutouts, self._cutout_total_masks,
                                            strict=True)]

        return cutouts

    def select_label(self, label):
        """
        Return a new `SourceCatalog` object for the input ``label``
        only.

        Parameters
        ----------
        label : int
            The source label.

        Returns
        -------
        cat : `SourceCatalog`
            A new `SourceCatalog` object containing only the source with
            the input ``label``.
        """
        return self.select_labels(label)

    def select_labels(self, labels):
        """
        Return a new `SourceCatalog` object for the input ``labels``
        only.

        Parameters
        ----------
        labels : list, tuple, or `~numpy.ndarray` of int
            The source label(s).

        Returns
        -------
        cat : `SourceCatalog`
            A new `SourceCatalog` object containing only the sources with
            the input ``labels``.
        """
        self._segmentation_image.check_labels(labels)
        sorter = np.argsort(self.labels)
        indices = sorter[np.searchsorted(self.labels, labels, sorter=sorter)]
        return self[indices]

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def to_table(self, columns=None):
        """
        Create a `~astropy.table.QTable` of source properties.

        Parameters
        ----------
        columns : str, list of str, `None`, optional
            Names of columns, in order, to include in the output
            `~astropy.table.QTable`. The allowed column names are any of
            the `SourceCatalog` properties or custom properties added
            using `add_property`. If ``columns`` is `None`, then a
            default list of scalar-valued properties (as defined by the
            ``default_columns`` attribute) will be used.

        Returns
        -------
        table : `~astropy.table.QTable`
            A table of sources properties with one row per source.
        """
        if columns is None:
            table_columns = self.default_columns
        elif isinstance(columns, str):
            table_columns = [columns]
        else:
            table_columns = columns

        # Replace with QTable() in 4.0
        tbl = create_empty_deprecated_qtable(
            _DEPRECATED_ATTRIBUTES, since='3.0', until='4.0')

        tbl.meta.update(self.meta)  # keep tbl.meta type
        for column in table_columns:
            values = getattr(self, column)

            # Column assignment requires an object with a length
            if self.isscalar:
                values = (values,)

            tbl[column] = values
        return tbl

    @lazyproperty
    def n_labels(self):
        """
        The number of source labels.
        """
        return len(self.labels)

    @property
    @as_scalar
    def label(self):
        """
        The source label number(s).

        This label number corresponds to the assigned pixel value in the
        `~photutils.segmentation.SegmentationImage`.

        Returns an array for multi-source catalogs, or a scalar for a
        single-source catalog.
        """
        return self._labels

    @property
    def labels(self):
        """
        The source label number(s), always as an iterable
        `~numpy.ndarray`.

        This label number corresponds to the assigned pixel value in the
        `~photutils.segmentation.SegmentationImage`.
        """
        labels = self.label
        if self.isscalar:
            labels = np.array((labels,))
        return labels

    @property
    @as_scalar
    def slices(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source.

        Returns a list for multi-source catalogs, or a single tuple for
        a single-source catalog.
        """
        return self._slices

    @lazyproperty
    def _slices_iter(self):
        """
        A tuple of slice objects defining the minimal bounding box of
        the source, always as an iterable.
        """
        _slices = self.slices
        if self.isscalar:
            _slices = (_slices,)
        return _slices

    @lazyproperty
    @as_scalar
    def segment_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout of the segmentation image using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(
            self._segmentation_image_cutouts, units=False,
            masked=False)

    @lazyproperty
    @as_scalar
    def segment_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout of the segmentation image
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(
            self._segmentation_image_cutouts, units=False,
            masked=True)

    @lazyproperty
    @as_scalar
    def data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the data using the minimal
        bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._data_cutouts, units=True,
                                     masked=False, dtype=float)

    @lazyproperty
    @as_scalar
    def data_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the data using the
        minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._data_cutouts, units=False,
                                     masked=True, dtype=float)

    @lazyproperty
    @as_scalar
    def conv_data_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the convolved data using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._convdata_cutouts, units=True,
                                     masked=False, dtype=float)

    @lazyproperty
    @as_scalar
    def conv_data_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the convolved data
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        return self._prepare_cutouts(self._convdata_cutouts, units=False,
                                     masked=True, dtype=float)

    @lazyproperty
    @as_scalar
    def error_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the error array using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._error is None:
            return self._null_objects
        return self._prepare_cutouts(self._error_cutouts, units=True,
                                     masked=False)

    @lazyproperty
    @as_scalar
    def error_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the error array using
        the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._error is None:
            return self._null_objects
        return self._prepare_cutouts(self._error_cutouts, units=False,
                                     masked=True)

    @lazyproperty
    @as_scalar
    def background_cutout(self):
        """
        A 2D `~numpy.ndarray` cutout from the background array using the
        minimal bounding box of the source.

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._background is None:
            return self._null_objects
        return self._prepare_cutouts(self._background_cutouts, units=True,
                                     masked=False)

    @lazyproperty
    @as_scalar
    def background_cutout_masked(self):
        """
        A 2D `~numpy.ma.MaskedArray` cutout from the background array.
        using the minimal bounding box of the source.

        The mask is `True` for pixels outside the source segment
        (labeled region of interest), masked pixels from the ``mask``
        input, or any non-finite ``data`` values (NaN and inf).

        Returns a list of arrays for multi-source catalogs, or a single
        array for a single-source catalog.
        """
        if self._background is None:
            return self._null_objects
        return self._prepare_cutouts(self._background_cutouts, units=False,
                                     masked=True)

    @lazyproperty
    def _all_masked(self):
        """
        True if all pixels over the source segment are masked.
        """
        return np.array([np.all(mask) for mask in self._cutout_total_masks])

    def _get_values(self, array):
        """
        Get a 1D array of unmasked values from the input array within
        the source segment.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self.isscalar:
            array = (array,)
        return [arr.compressed() if len(arr.compressed()) > 0
                else np.array([np.nan]) for arr in array]

    @staticmethod
    def _reduceat(values, ufunc, *, transform=None):
        """
        Apply ``ufunc.reduceat`` to a list of arrays.

        This is significantly faster than a list comprehension with
        individual NumPy calls for each array.

        Parameters
        ----------
        values : list of 1D `~numpy.ndarray`
            A list of 1D arrays.

        ufunc : `~numpy.ufunc`
            The NumPy ufunc to apply (e.g., `~numpy.add`,
            `~numpy.minimum`, `~numpy.maximum`).

        transform : callable or None, optional
            An optional transformation to apply to the concatenated
            array before reducing (e.g., `~numpy.square`).

        Returns
        -------
        result : `~numpy.ndarray`
            The reduceat result.

        sizes : `~numpy.ndarray`
            The sizes of the input arrays.
        """
        if not values:
            return np.array([]), np.array([], dtype=int)

        sizes = np.array([len(arr) for arr in values])
        splits = np.concatenate(([0], np.cumsum(sizes[:-1])))
        concat = np.concatenate(values)
        if transform is not None:
            concat = transform(concat)
        return ufunc.reduceat(concat, splits), sizes

    @lazyproperty
    def _data_values(self):
        """
        A 1D array of unmasked data values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        return self._get_values(self.data_cutout_masked)

    @lazyproperty
    def _error_values(self):
        """
        A 1D array of unmasked error values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self._error is None:
            return self._null_objects
        return self._get_values(self.error_cutout_masked)

    @lazyproperty
    def _background_values(self):
        """
        A 1D array of unmasked background values.

        An array with a single NaN is returned for completely-masked
        sources.
        """
        if self._background is None:
            return self._null_objects
        return self._get_values(self.background_cutout_masked)

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments(self):
        """
        Spatial moments up to 3rd order of the source.
        """
        result = []
        for arr in self._moment_data_cutouts:
            ny, nx = arr.shape
            y = np.arange(ny, dtype=float)
            x = np.arange(nx, dtype=float)
            yp = np.column_stack([np.ones(ny), y, y * y, y ** 3])
            xp = np.column_stack([np.ones(nx), x, x * x, x ** 3])
            result.append(yp.T @ arr @ xp)
        return np.array(result)

    @lazyproperty
    @use_detcat
    @as_scalar
    def moments_central(self):
        """
        Central moments (translation invariant) of the source up to 3rd
        order.
        """
        cutout_centroid = self.cutout_centroid
        if self.isscalar:
            cutout_centroid = cutout_centroid[np.newaxis, :]
        result = []
        for arr, xcen, ycen in zip(self._moment_data_cutouts,
                                   cutout_centroid[:, 0],
                                   cutout_centroid[:, 1], strict=True):
            ny, nx = arr.shape
            yc = np.arange(ny, dtype=float) - ycen
            xc = np.arange(nx, dtype=float) - xcen
            yp = np.column_stack([np.ones(ny), yc, yc * yc, yc ** 3])
            xp = np.column_stack([np.ones(nx), xc, xc * xc, xc ** 3])
            result.append(yp.T @ arr @ xp)
        return np.array(result)

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        centroid within the isophotal source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        moments = self.moments
        if self.isscalar:
            moments = moments[np.newaxis, :]

        # Ignore divide-by-zero RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            y_centroid = moments[:, 1, 0] / moments[:, 0, 0]
            x_centroid = moments[:, 0, 1] / moments[:, 0, 0]
        return np.transpose((x_centroid, y_centroid))

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid(self):
        """
        The ``(x, y)`` coordinate of the centroid within the isophotal
        source segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid + origin

    @lazyproperty
    @use_detcat
    def _x_centroid(self):
        """
        The ``x`` coordinate of the `centroid` within the source
        segment, always as an iterable.
        """
        if self.isscalar:
            x_centroid = self.centroid[0:1]  # scalar array
        else:
            x_centroid = self.centroid[:, 0]
        return x_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid(self):
        """
        The ``x`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._x_centroid

    @lazyproperty
    @use_detcat
    def _y_centroid(self):
        """
        The ``y`` coordinate of the `centroid` within the source
        segment, always as an iterable.
        """
        if self.isscalar:
            y_centroid = self.centroid[1:2]  # scalar array
        else:
            y_centroid = self.centroid[:, 1]
        return y_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid(self):
        """
        The ``y`` coordinate of the `centroid` within the source
        segment.

        The centroid is computed as the center of mass of the unmasked
        pixels within the source segment.
        """
        return self._y_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_win(self):
        """
        The ``(x, y)`` coordinate of the "windowed" centroid.

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE and YWIN_IMAGE parameters.

        Notes
        -----
        During each iteration, the centroid is calculated using all
        pixels within a circular aperture of ``4 * sigma`` from the
        current position, weighting pixel values with a 2D Gaussian with
        a standard deviation of ``sigma``. ``sigma`` is the half-light
        radius (i.e., ``flux_radius(0.5)``) times (2.0 / 2.35). A
        minimum half-light radius of 0.5 pixels is used. Iteration stops
        when the change in centroid position falls below a pre-defined
        threshold or a maximum number of iterations is reached.

        If the windowed centroid falls outside the 1-sigma ellipse
        shape based on the image moments, then the isophotal `centroid`
        will be used instead. If the half-light radius is not finite
        (e.g., due to a non-finite Kron radius), then ``np.nan`` will be
        returned.
        """
        return self._centroid_refiner.centroid_win

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid_win(self):
        """
        The ``x`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE parameters.
        """
        if self.isscalar:
            x_centroid = self.centroid_win[0]  # scalar array
        else:
            x_centroid = self.centroid_win[:, 0]
        return x_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid_win(self):
        """
        The ``y`` coordinate of the "windowed" centroid
        (`centroid_win`).

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s YWIN_IMAGE parameters.
        """
        if self.isscalar:
            y_centroid = self.centroid_win[1]  # scalar array
        else:
            y_centroid = self.centroid_win[:, 1]
        return y_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_win(self):
        """
        The ``(x, y)`` coordinate, relative to the cutout data, of the
        "windowed" centroid.

        The window centroid is computed using an iterative algorithm
        to derive a more accurate centroid. It is equivalent to
        `SourceExtractor`_'s XWIN_IMAGE and YWIN_IMAGE parameters. See
        `centroid_win` for further details about the algorithm.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.centroid_win - origin

    @lazyproperty
    @use_detcat
    @as_scalar
    def cutout_centroid_quad(self):
        """
        The ``(x, y)`` centroid coordinate, relative to the cutout data,
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.

        Notes
        -----
        `~photutils.centroids.centroid_quadratic` is used to calculate
        the centroid with ``fit_boxsize=3``.

        Because this centroid is based on fitting data, it can fail for
        many reasons including:

        * quadratic fit failed
        * quadratic fit does not have a maximum
        * quadratic fit maximum falls outside image
        * not enough unmasked data points (6 are required)

        In these cases, then the isophotal `centroid` will be used
        instead.

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        return self._centroid_refiner.cutout_centroid_quad

    @lazyproperty
    @use_detcat
    @as_scalar
    def centroid_quad(self):
        """
        The ``(x, y)`` centroid coordinate, calculated by fitting a 2D
        quadratic polynomial to the unmasked pixels in the source
        segment.

        Notes
        -----
        `~photutils.centroids.centroid_quadratic` is used to calculate
        the centroid with ``fit_boxsize=3``.

        Because this centroid is based on fitting data, it can fail for
        many reasons, returning (np.nan, np.nan):

        * quadratic fit failed
        * quadratic fit does not have a maximum
        * quadratic fit maximum falls outside image
        * not enough unmasked data points (6 are required)

        Also note that a fit is not performed if the maximum data value
        is at the edge of the source segment. In this case, the position
        of the maximum pixel will be returned.
        """
        origin = np.transpose((self.bbox_xmin, self.bbox_ymin))
        return self.cutout_centroid_quad + origin

    @lazyproperty
    @use_detcat
    @as_scalar
    def x_centroid_quad(self):
        """
        The ``x`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        if self.isscalar:
            x_centroid = self.centroid_quad[0]  # scalar array
        else:
            x_centroid = self.centroid_quad[:, 0]
        return x_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def y_centroid_quad(self):
        """
        The ``y`` coordinate of the centroid (`centroid_quad`),
        calculated by fitting a 2D quadratic polynomial to the unmasked
        pixels in the source segment.
        """
        if self.isscalar:
            y_centroid = self.centroid_quad[1]  # scalar array
        else:
            y_centroid = self.centroid_quad[:, 1]
        return y_centroid

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid(self):
        """
        The sky coordinate of the `centroid` within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.x_centroid, self.y_centroid)

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_icrs(self):
        """
        The sky coordinate in the International Celestial Reference
        System (ICRS) frame of the `centroid` within the source segment,
        returned as a `~astropy.coordinates.SkyCoord` object.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.sky_centroid.icrs

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_win(self):
        """
        The sky coordinate of the "windowed" centroid (`centroid_win`)
        within the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.x_centroid_win,
                                       self.y_centroid_win)

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_centroid_quad(self):
        """
        The sky coordinate of the centroid (`centroid_quad`), calculated
        by fitting a 2D quadratic polynomial to the unmasked pixels in
        the source segment.

        The output coordinate frame is the same as the input ``wcs``.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(self.x_centroid_quad,
                                       self.y_centroid_quad)

    @lazyproperty
    @use_detcat
    def _bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment, always as an iterable.
        """
        return [BoundingBox(ixmin=slc[1].start, ixmax=slc[1].stop,
                            iymin=slc[0].start, iymax=slc[0].stop)
                for slc in self._slices_iter]

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox(self):
        """
        The `~photutils.aperture.BoundingBox` of the minimal rectangular
        region containing the source segment.

        Returns a list for multi-source catalogs, or a single
        `~photutils.aperture.BoundingBox` for a single-source catalog.
        """
        return self._bbox

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmin(self):
        """
        The minimum ``x`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[1].start for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_xmax(self):
        """
        The maximum ``x`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[1].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymin(self):
        """
        The minimum ``y`` pixel index within the minimal bounding box
        containing the source segment.
        """
        return np.array([slc[0].start for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    @as_scalar
    def bbox_ymax(self):
        """
        The maximum ``y`` pixel index within the minimal bounding box
        containing the source segment.

        Note that this value is inclusive, unlike numpy slice indices.
        """
        return np.array([slc[0].stop - 1 for slc in self._slices_iter])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ll(self):
        """
        Lower-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ul(self):
        """
        Upper-left *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmin - 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_lr(self):
        """
        Lower-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymin - 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    def _bbox_corner_ur(self):
        """
        Upper-right *outside* pixel corner location (not index).
        """
        return np.array([(bbox_.ixmax + 0.5, bbox_.iymax + 0.5)
                         for bbox_ in self._bbox])

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ll(self):
        """
        The sky coordinates of the lower-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ll))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ul(self):
        """
        The sky coordinates of the upper-left corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ul))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_lr(self):
        """
        The sky coordinates of the lower-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_lr))

    @lazyproperty
    @use_detcat
    @as_scalar
    def sky_bbox_ur(self):
        """
        The sky coordinates of the upper-right corner vertex of the
        minimal bounding box of the source segment, returned as a
        `~astropy.coordinates.SkyCoord` object.

        The bounding box encloses all the source segment pixels in their
        entirety, thus the vertices are at the pixel *corners*, not
        their centers.

        `None` if ``wcs`` is not input.
        """
        if self.wcs is None:
            return self._null_objects
        return self.wcs.pixel_to_world(*np.transpose(self._bbox_corner_ur))

    @lazyproperty
    @as_scalar
    def min_value(self):
        """
        The minimum pixel value of the ``data`` within the source
        segment.
        """
        values, _ = self._reduceat(self._data_values, np.minimum)
        values -= self._local_background
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    @as_scalar
    def max_value(self):
        """
        The maximum pixel value of the ``data`` within the source
        segment.
        """
        values, _ = self._reduceat(self._data_values, np.maximum)
        values -= self._local_background
        if self._data_unit is not None:
            values <<= self._data_unit
        return values

    @lazyproperty
    @as_scalar
    def cutout_min_value_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        minimum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        data = self.data_cutout_masked
        if self.isscalar:
            data = (data,)
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmin(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    @as_scalar
    def cutout_max_value_index(self):
        """
        The ``(y, x)`` coordinate, relative to the cutout data, of the
        maximum pixel value of the ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        data = self.data_cutout_masked
        if self.isscalar:
            data = (data,)
        idx = []
        for arr in data:
            if np.all(arr.mask):
                idx.append((np.nan, np.nan))
            else:
                idx.append(np.unravel_index(np.argmax(arr), arr.shape))
        return np.array(idx)

    @lazyproperty
    @as_scalar
    def min_value_index(self):
        """
        The ``(y, x)`` coordinate of the minimum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        index = self.cutout_min_value_index
        if self.isscalar:
            index = (index,)
        out = []
        for idx, slc in zip(index, self._slices_iter, strict=True):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    @as_scalar
    def max_value_index(self):
        """
        The ``(y, x)`` coordinate of the maximum pixel value of the
        ``data`` within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        index = self.cutout_max_value_index
        if self.isscalar:
            index = (index,)
        out = []
        for idx, slc in zip(index, self._slices_iter, strict=True):
            out.append((idx[0] + slc[0].start, idx[1] + slc[1].start))
        return np.array(out)

    @lazyproperty
    @as_scalar
    def min_value_xindex(self):
        """
        The ``x`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            xidx = self.min_value_index[1]
        else:
            xidx = self.min_value_index[:, 1]
        return xidx

    @lazyproperty
    @as_scalar
    def min_value_yindex(self):
        """
        The ``y`` coordinate of the minimum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the minimum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            yidx = self.min_value_index[0]
        else:
            yidx = self.min_value_index[:, 0]
        return yidx

    @lazyproperty
    @as_scalar
    def max_value_xindex(self):
        """
        The ``x`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            xidx = self.max_value_index[1]
        else:
            xidx = self.max_value_index[:, 1]
        return xidx

    @lazyproperty
    @as_scalar
    def max_value_yindex(self):
        """
        The ``y`` coordinate of the maximum pixel value of the ``data``
        within the source segment.

        If there are multiple occurrences of the maximum value, only the
        first occurrence is returned.
        """
        if self.isscalar:
            yidx = self.max_value_index[0]
        else:
            yidx = self.max_value_index[:, 0]
        return yidx

    @lazyproperty
    @as_scalar
    def segment_flux(self):
        r"""
        The sum of the unmasked ``data`` values within the source
        segment.

        .. math::

            F = \sum_{i \in S} I_i

        where :math:`F` is ``segment_flux``, :math:`I_i` is the
        background-subtracted ``data``, and :math:`S` are the unmasked
        pixels in the source segment.

        Non-finite pixel values (NaN and inf) are excluded
        (automatically masked).
        """
        localbkg = self._local_background
        if self.isscalar:
            localbkg = localbkg[0]
        source_sum, _ = self._reduceat(self._data_values, np.add)
        source_sum -= self.area.value * localbkg
        if self._data_unit is not None:
            source_sum <<= self._data_unit
        return source_sum

    @lazyproperty
    @as_scalar
    def segment_flux_err(self):
        r"""
        The uncertainty of `segment_flux`, propagated from the input
        ``error`` array.

        ``segment_flux_err`` is the quadrature sum of the total errors
        over the unmasked pixels within the source segment:

        .. math::

            \Delta F = \sqrt{\sum_{i \in S} \sigma_{\mathrm{tot}, i}^2}

        where :math:`\Delta F` is the `segment_flux_err`,
        :math:`\sigma_{\mathrm{tot, i}}` are the pixel-wise total errors
        (``error``), and :math:`S` are the unmasked pixels in the source
        segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the error array.
        """
        if self._error is None:
            err = self._null_values
        else:
            err_sq, _ = self._reduceat(self._error_values, np.add,
                                       transform=np.square)
            err = np.sqrt(err_sq)

        if self._data_unit is not None:
            err <<= self._data_unit
        return err

    @lazyproperty
    @as_scalar
    def background_sum(self):
        """
        The sum of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the background array.
        """
        if self._background is None:
            bkg_sum = self._null_values
        else:
            bkg_sum, _ = self._reduceat(
                self._background_values, np.add)

        if self._data_unit is not None:
            bkg_sum <<= self._data_unit
        return bkg_sum

    @lazyproperty
    @as_scalar
    def background_mean(self):
        """
        The mean of ``background`` values within the source segment.

        Pixel values that are masked in the input ``data``, including
        any non-finite pixel values (NaN and inf) that are automatically
        masked, are also masked in the background array.
        """
        if self._background is None:
            bkg_mean = self._null_values
        else:
            bkg_sum, sizes = self._reduceat(
                self._background_values, np.add)
            bkg_mean = bkg_sum / sizes

        if self._data_unit is not None:
            bkg_mean <<= self._data_unit
        return bkg_mean

    @lazyproperty
    @as_scalar
    def background_centroid(self):
        """
        The value of the per-pixel ``background`` at the position of the
        isophotal (center-of-mass) `centroid`.

        If ``detection_catalog`` is input, then its `centroid` will be
        used.

        The background values at fractional position values are
        determined using bilinear interpolation.
        """
        if self._background is None:
            bkg = self._null_values
        else:
            xcen = self._x_centroid
            ycen = self._y_centroid
            bkg = map_coordinates(self._background, (ycen, xcen), order=1,
                                  mode='nearest')

            mask = np.isfinite(xcen) & np.isfinite(ycen)
            bkg[~mask] = np.nan

        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    @lazyproperty
    @use_detcat
    @as_scalar
    def segment_area(self):
        """
        The total area of the source segment in units of pixels**2.

        This area is simply the area of the source segment from the
        input ``segmentation_image``. It does not take into account
        any data masking (i.e., a ``mask`` input to `SourceCatalog` or
        invalid ``data`` values).
        """
        areas = []
        for label, slices in zip(self.labels, self._slices_iter,
                                 strict=True):
            areas.append(np.count_nonzero(
                self._segmentation_image[slices] == label))
        return np.array(areas) << (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def area(self):
        """
        The total unmasked area of the source in units of pixels**2.

        Note that the source area may be smaller than its `segment_area`
        if a mask is input to `SourceCatalog` or if the ``data`` within
        the segment contains invalid values (NaN and inf).
        """
        areas = np.array([arr.size for arr in self._data_values]).astype(float)
        areas[self._all_masked] = np.nan
        return areas << (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def equivalent_radius(self):
        """
        The radius of a circle with the same `area` as the source
        segment.
        """
        return np.sqrt(self.area / np.pi)

    @lazyproperty
    @use_detcat
    @as_scalar
    def perimeter(self):
        """
        The perimeter of the source segment, approximated as the total
        length of lines connecting the centers of the border pixels
        defined by a 4-pixel connectivity.

        If any masked pixels make holes within the source segment, then
        the perimeter around the inner hole (e.g., an annulus) will also
        contribute to the total perimeter.

        References
        ----------
        .. [1] K. Benkrid, D. Crookes, and A. Benkrid. "Design and FPGA
               Implementation of a Perimeter Estimator". Proceedings of
               the Irish Machine Vision and Image Processing Conference,
               pp. 51-57 (2000).
        """
        size = 34
        weights = np.zeros(size, dtype=float)
        weights[[5, 7, 15, 17, 25, 27]] = 1.0
        weights[[21, 33]] = np.sqrt(2.0)
        weights[[13, 23]] = (1 + np.sqrt(2.0)) / 2.0

        perimeter = []
        for mask in self._cutout_total_masks:
            if np.all(mask):
                perimeter.append(np.nan)
                continue

            ny, nx = mask.shape

            # Pad source array with zeros (border_value=0)
            padded = np.zeros((ny + 2, nx + 2), dtype=np.int8)
            padded[1:-1, 1:-1] = ~mask

            # Binary erosion with cross footprint (4-connectivity):
            # a pixel is eroded if any 4-connected neighbor is 0
            p = padded
            eroded = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                      & p[1:-1, :-2] & p[1:-1, 2:])

            # Border pixels are source pixels that were eroded away
            border = np.zeros((ny + 2, nx + 2), dtype=np.int8)
            border[1:-1, 1:-1] = padded[1:-1, 1:-1] & ~eroded

            # Convolution with kernel [[10,2,10], [2,1,2], [10,2,10]]
            b = border
            conv = (10 * b[:-2, :-2] + 2 * b[:-2, 1:-1]
                    + 10 * b[:-2, 2:] + 2 * b[1:-1, :-2]
                    + b[1:-1, 1:-1] + 2 * b[1:-1, 2:]
                    + 10 * b[2:, :-2] + 2 * b[2:, 1:-1]
                    + 10 * b[2:, 2:])

            hist = np.bincount(conv.ravel(), minlength=size)
            perimeter.append(hist[:size] @ weights)

        return np.array(perimeter) * u.pix

    @lazyproperty
    @use_detcat
    def _shape(self):
        """
        The `_ShapeProperties` helper computing the morphology
        properties from the central moments.
        """
        moments = self.moments_central
        if self.isscalar:
            moments = moments[np.newaxis, :]
        return _ShapeProperties(moments)

    @lazyproperty
    @use_detcat
    @as_scalar
    def inertia_tensor(self):
        """
        The inertia tensor of the source for the rotation around its
        center of mass.
        """
        return self._shape.inertia_tensor * u.pix**2

    @lazyproperty
    @use_detcat
    def _covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source, always as an iterable.
        """
        return self._shape.covariance

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance(self):
        """
        The covariance matrix of the 2D Gaussian function that has the
        same second-order moments as the source.
        """
        return self._covariance * (u.pix**2)

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_eigvals(self):
        """
        The two eigenvalues of the `covariance` matrix in decreasing
        order.
        """
        return self._shape.covariance_eigvals * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def semimajor_axis(self):
        """
        The 1-sigma standard deviation along the semimajor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        # This matches SourceExtractor's A parameter
        return self._shape.semimajor_axis * u.pix

    @lazyproperty
    @use_detcat
    @as_scalar
    def semiminor_axis(self):
        """
        The 1-sigma standard deviation along the semiminor axis of the
        2D Gaussian function that has the same second-order central
        moments as the source.
        """
        # This matches SourceExtractor's B parameter
        return self._shape.semiminor_axis * u.pix

    @lazyproperty
    @use_detcat
    @as_scalar
    def fwhm(self):
        r"""
        The circularized full width at half maximum (FWHM) of the 2D
        Gaussian function that has the same second-order central moments
        as the source.

        .. math::

           \mathrm{FWHM} & = 2 \sqrt{2 \ln(2)} \sqrt{0.5 (a^2 + b^2)}
           \\
                         & = 2 \sqrt{\ln(2) \ (a^2 + b^2)}

        where :math:`a` and :math:`b` are the 1-sigma lengths of the
        semimajor (`semimajor_axis`) and semiminor (`semiminor_axis`)
        axes, respectively.
        """
        return self._shape.fwhm * u.pix

    @lazyproperty
    @use_detcat
    @as_scalar
    def orientation(self):
        """
        The angle between the ``x`` axis and the major axis of the 2D
        Gaussian function that has the same second-order moments as the
        source.

        The angle increases in the counter-clockwise direction and
        will be in the range [0, 360) degrees.
        """
        return (np.rad2deg(self._shape.orientation_radians) % 360) << u.deg

    @lazyproperty
    @use_detcat
    @as_scalar
    def eccentricity(self):
        r"""
        The eccentricity of the 2D Gaussian function that has the same
        second-order moments as the source.

        The eccentricity is the fraction of the distance along the
        semimajor axis at which the focus lies.

        .. math::

            e = \sqrt{1 - \frac{b^2}{a^2}}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self._shape.eccentricity

    @lazyproperty
    @use_detcat
    @as_scalar
    def elongation(self):
        r"""
        The ratio of the lengths of the semimajor and semiminor axes.

        .. math::

            \mathrm{elongation} = \frac{a}{b}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self._shape.elongation

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipticity(self):
        r"""
        1.0 minus the ratio of the lengths of the semimajor and
        semiminor axes.

        .. math::

            \mathrm{ellipticity} = \frac{a - b}{a} = 1 - \frac{b}{a}

        where :math:`a` and :math:`b` are the lengths of the semimajor
        and semiminor axes, respectively.
        """
        return self._shape.ellipticity

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_xx(self):
        r"""
        The ``(0, 0)`` element of the `covariance` matrix, representing
        :math:`\sigma_x^2`, in units of pixel**2.
        """
        return self._shape.covariance_xx * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_yy(self):
        r"""
        The ``(1, 1)`` element of the `covariance` matrix, representing
        :math:`\sigma_y^2`, in units of pixel**2.
        """
        return self._shape.covariance_yy * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def covariance_xy(self):
        r"""
        The ``(0, 1)`` and ``(1, 0)`` elements of the `covariance`
        matrix, representing :math:`\sigma_x \sigma_y`, in units of
        pixel**2.
        """
        return self._shape.covariance_xy * u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cxx(self):
        r"""
        Coefficient for ``x**2`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return self._shape.ellipse_cxx / u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cyy(self):
        r"""
        Coefficient for ``y**2`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return self._shape.ellipse_cyy / u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def ellipse_cxy(self):
        r"""
        Coefficient for ``x * y`` in the generalized ellipse equation in
        units of pixel**(-2).

        The ellipse is defined as

        .. math::

            cxx (x - \bar{x})^2 + cxy (x - \bar{x}) (y - \bar{y}) +
            cyy (y - \bar{y})^2 = R^2

        where :math:`R` is a parameter which scales the ellipse (in
        units of the axes lengths).

        `SourceExtractor`_ reports that the isophotal limit of a source
        is well represented by :math:`R \approx 3`.
        """
        return self._shape.ellipse_cxy / u.pix**2

    @lazyproperty
    @use_detcat
    @as_scalar
    def gini(self):
        r"""
        The `Gini coefficient
        <https://en.wikipedia.org/wiki/Gini_coefficient>`_ of the
        source.

        The Gini coefficient of the distribution of absolute flux values
        is calculated using the prescription from `Lotz et al. 2004
        <https://ui.adsabs.harvard.edu/abs/2004AJ....128..163L/abstract>`_
        (Eq. 6) as:

        .. math::

            G = \frac{1}{\overline{|x|} \, n \, (n - 1)}
                \sum^{n}_{i} (2i - n - 1) \left | x_i \right |

        where :math:`\overline{|x|}` is the mean of the absolute value
        of all pixel values :math:`x_i`. If the sum of all pixel values
        is zero, the Gini coefficient is zero.

        Negative pixel values are used via their absolute value. Invalid
        values (NaN and inf) in the input are automatically excluded
        from the calculation. If only a single finite pixel remains
        after filtering, the Gini coefficient is 0.0.
        """
        return np.array([gini_func(arr) for arr in self._data_values])

    @lazyproperty
    def _local_bkg(self):
        """
        Composition helper that computes the rectangular-annulus local
        background apertures and per-source values.
        """
        return _LocalBackground(self)

    @lazyproperty
    def _centroid_refiner(self):
        """
        Composition helper that computes the windowed and quadratic
        centroid refinements.
        """
        return _CentroidRefiner(self)

    @lazyproperty
    def _photometry(self):
        """
        Composition helper that provides circular, Kron, and
        flux-radius photometry implementations.
        """
        return _Photometry(self)

    @lazyproperty
    def _local_background_apertures(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.
        """
        return self._local_bkg.apertures

    @lazyproperty
    @use_detcat
    @as_scalar
    def local_background_aperture(self):
        """
        The `~photutils.aperture.RectangularAnnulus` aperture used to
        estimate the local background.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._local_background_apertures

    @lazyproperty
    def _local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.

        Pixels are masked where the input ``mask`` is `True`, where the
        input ``data`` is non-finite, and within any non-zero pixel
        label in the segmentation image.

        This property is always an `~numpy.ndarray` without units.
        """
        return self._local_bkg.values

    @lazyproperty
    @as_scalar
    def local_background(self):
        """
        The local background value (per pixel) estimated using a
        rectangular annulus aperture around the source.
        """
        bkg = self._local_background
        if self._data_unit is not None:
            bkg <<= self._data_unit
        return bkg

    def _aperture_to_mask(self, aperture, **kwargs):
        """
        Call ``aperture.to_mask()``, but first check that the aperture
        bounding box is not larger than the input data to prevent
        out-of-memory errors from pathologically large apertures.

        Returns `None` if the aperture mask would be unreasonably large.
        """
        return self._photometry._aperture_to_mask(aperture, **kwargs)

    def _make_aperture_data(self, label, x_centroid, y_centroid,
                            aperture_bbox, local_background, *,
                            make_error=True):
        """
        Make cutouts of data, error, and mask arrays for aperture
        photometry (e.g., circular or Kron).

        Neighboring sources can be included, masked, or corrected based
        on the ``aperture_mask_method`` keyword.
        """
        return self._photometry._make_aperture_data(
            label, x_centroid, y_centroid, aperture_bbox,
            local_background, make_error=make_error)

    @as_scalar
    def make_circular_apertures(self, radius):
        """
        Make circular aperture for each source.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        Returns
        -------
        result : `~photutils.aperture.CircularAperture` or \
                 list of `~photutils.aperture.CircularAperture`
            The circular aperture for each source. The aperture will be
            `None` where the source `centroid` position is not finite or
            where the source is completely masked.
        """
        return self._photometry._make_circular_apertures(radius)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_circular_apertures(self, radius, ax=None, origin=(0, 0),
                                **kwargs):
        """
        Plot circular apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        An aperture will not be plotted for sources where the source
        `centroid` position is not finite or where the source is
        completely masked.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : `~matplotlib.patches.Patch` or \
                list of `~matplotlib.patches.Patch`
            The matplotlib patch for each plotted aperture. The patches
            can be used, for example, when adding a plot legend.
        """
        return self._photometry._plot_circular_apertures(
            radius, ax, origin, **kwargs)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def circular_photometry(self, radius, name=None, overwrite=False):
        """
        Perform circular aperture photometry for each source.

        The circular aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        Parameters
        ----------
        radius : float
            The radius of the circle in pixels.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the flux and flux error. The attribute names
            ``[name]_flux`` and ``[name]_flux_err`` will store the
            photometry results. For example, these names can then be
            included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, flux_err : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position is not finite or the source is
            completely masked).
        """
        return self._photometry._circular_photometry(radius, name, overwrite)

    def _make_elliptical_apertures(self, *, scale=6.0):
        """
        Return a list of elliptical apertures based on the scaled
        isophotal shape of the sources.

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its source `centroid` and shape parameters will be used.

        If scale is zero (due to a minimum circular radius set in
        ``kron_params``) then a circular aperture will be returned with
        the minimum circular radius.

        Parameters
        ----------
        scale : float or `~numpy.ndarray`, optional
            The scale factor to apply to the ellipse major and minor
            axes. The default value of 6.0 is roughly two times the
            isophotal extent of the source. A `~numpy.ndarray` input
            must be a 1D array of length ``n_labels``.

        Returns
        -------
        result : list of `~photutils.aperture.EllipticalAperture`
            A list of `~photutils.aperture.EllipticalAperture`
            instances. The aperture will be `None` where the source
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked.
        """
        return self._photometry._make_elliptical_apertures(scale=scale)

    @lazyproperty
    @use_detcat
    def _measured_kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius, always as an array
        (without units).

        The returned value is the measured Kron radius without applying
        any minimum Kron or circular radius.
        """
        return self._photometry._measured_kron_radius()

    @as_scalar
    def _calc_kron_radius(self, kron_params):
        """
        Calculate the *unscaled* first-moment Kron radius, applying any
        minimum Kron or circular radius to the measured Kron radius.

        Returned as a Quantity array or scalar (if self isscalar) with
        pixel units.
        """
        return self._photometry._calc_kron_radius(kron_params)

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_radius(self):
        r"""
        The *unscaled* first-moment Kron radius.

        The *unscaled* first-moment Kron radius is given by:

        .. math::

            r_k = \frac{\sum_{i \in A} \ r_i I_i}{\sum_{i \in A} I_i}

        where :math:`I_i` are the data values and the sum is over
        pixels in an elliptical aperture whose axes are defined by
        six times the semimajor (`semimajor_axis`) and semiminor
        axes (`semiminor_axis`) at the calculated `orientation` (all
        properties derived from the central image moments of the
        source). :math:`r_i` is the elliptical "radius" to the pixel
        given by:

        .. math::

            r_i^2 = cxx (x_i - \bar{x})^2 +
                cxy (x_i - \bar{x})(y_i - \bar{y}) +
                cyy (y_i - \bar{y})^2

        where :math:`\bar{x}` and :math:`\bar{y}` represent the source
        `centroid` and the coefficients are based on image moments
        (`ellipse_cxx`, `ellipse_cxy`, and `ellipse_cyy`).

        The `kron_radius` value is the unscaled moment value. The
        minimum unscaled radius can be set using the second element of
        the `SourceCatalog` ``kron_params`` keyword. If the measured
        unscaled Kron radius exceeds 6.0 (the measurement aperture
        scale factor), ``np.nan`` will be returned. Such values are
        unphysical, typically caused by near-cancellation in the
        denominator of the Kron formula due to outlier pixels or noise.

        If either the numerator or denominator above is less than
        or equal to 0, then the minimum unscaled Kron radius
        (``kron_params[1]``) will be used.

        The Kron aperture is calculated for each source using its shape
        parameters, `kron_radius`, and the ``kron_params`` scaling and
        minimum values input into `SourceCatalog`. The Kron aperture is
        used to compute the Kron photometry.

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis` *
        `semiminor_axis`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron radius will be set to
        zero and the Kron aperture will be a circle with this minimum
        radius.

        If the source is completely masked, then ``np.nan`` will be
        returned for both the Kron radius and Kron flux (the Kron
        aperture will be `None`).

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its ``kron_radius`` will be returned.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.
        """
        return self._calc_kron_radius(self.kron_params)

    def _make_kron_apertures(self, kron_params):
        """
        Make Kron apertures for each source, always returned as a list.
        """
        return self._photometry._make_kron_apertures(kron_params)

    @lazyproperty
    @use_detcat
    @as_scalar
    def kron_aperture(self):
        r"""
        The elliptical (or circular) Kron aperture.

        The Kron aperture is calculated for each source using its shape
        parameters, `kron_radius`, and the ``kron_params`` scaling and
        minimum values input into `SourceCatalog`. The Kron aperture is
        used to compute the Kron photometry.

        If ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis` *
        `semiminor_axis`) is less than or equal to the minimum circular
        radius (``kron_params[2]``), then the Kron aperture will be a
        circle with this minimum radius.

        The aperture will be `None` where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.

        If a ``detection_catalog`` was input to `SourceCatalog`, then
        its ``kron_aperture`` will be returned.

        Returns a list of apertures for multi-source catalogs, or a
        single aperture for a single-source catalog.
        """
        return self._make_kron_apertures(self.kron_params)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def make_kron_apertures(self, kron_params=None):
        """
        Make Kron apertures for each source.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_flux_err`) in the source catalog. This method should
        be used only to explore alternative ``kron_params`` with a
        detection image.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats or `None`, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius. If `None`, then the ``kron_params`` input into
            `SourceCatalog` will be used (the apertures will be the same
            as those in `kron_aperture`).

        Returns
        -------
        result : `~photutils.aperture.PixelAperture` \
                 or list of `~photutils.aperture.PixelAperture`
            The Kron apertures for each source. Each aperture will
            either be a `~photutils.aperture.EllipticalAperture`,
            `~photutils.aperture.CircularAperture`, or `None`. The
            aperture will be `None` where the source `centroid` position
            or elliptical shape parameters are not finite or where the
            source is completely masked.
        """
        if kron_params is None:
            return self.kron_aperture
        return self._make_kron_apertures(kron_params)

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def plot_kron_apertures(self, kron_params=None, ax=None, origin=(0, 0),
                            **kwargs):
        """
        Plot Kron apertures for each source on a matplotlib
        `~matplotlib.axes.Axes` instance.

        The aperture for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        An aperture will not be plotted for sources where the source
        `centroid` position or elliptical shape parameters are not
        finite or where the source is completely masked.

        Note that changing ``kron_params`` from the values
        input into `SourceCatalog` does not change the Kron
        apertures (`kron_aperture`) and photometry (`kron_flux` and
        `kron_flux_err`) in the source catalog. This method should be
        used only to visualize/explore alternative ``kron_params`` with
        a detection image.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats or `None`, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius. If `None`, then the ``kron_params`` input into
            `SourceCatalog` will be used (the apertures will be the same
            as those in `kron_aperture`).

        ax : `matplotlib.axes.Axes` or `None`, optional
            The matplotlib axes on which to plot. If `None`, then the
            current `~matplotlib.axes.Axes` instance is used.

        origin : array_like, optional
            The ``(x, y)`` position of the origin of the displayed
            image.

        **kwargs : dict, optional
            Any keyword arguments accepted by
            `matplotlib.patches.Patch`.

        Returns
        -------
        patch : list of `~matplotlib.patches.Patch`
            A list of matplotlib patches for the plotted aperture. The
            patches can be used, for example, when adding a plot legend.
        """
        return self._photometry._plot_kron_apertures(
            kron_params, ax, origin, **kwargs)

    def _aperture_photometry(self, apertures, *, desc='', **kwargs):
        """
        Perform aperture photometry on cutouts of the data and optional
        error arrays.

        The appropriate ``aperture_mask_method`` is applied to the
        cutouts to handle neighboring sources.

        Parameters
        ----------
        apertures : list of `PixelAperture`
            A list of the apertures.

        desc : str, optional
            The description displayed before the progress bar.

        **kwargs : dict, optional
            Additional keyword arguments passed to the aperture
            ``to_mask`` method.

        Returns
        -------
        flux, flux_err : 1D `~numpy.ndaray`
            The flux and flux error arrays.
        """
        return self._photometry._aperture_photometry(
            apertures, desc=desc, **kwargs)

    def _calc_kron_photometry(self, *, kron_params=None):
        """
        Calculate the flux and flux error in the Kron aperture (without
        units).

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned.

        If ``detection_catalog`` is input, then its `centroid` values
        will be used.

        Returns
        -------
        kron_flux, kron_flux_err : tuple of `~numpy.ndarray`
            The Kron flux and flux error.
        """
        return self._photometry._calc_kron_photometry(kron_params=kron_params)

    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def kron_photometry(self, kron_params, name=None, overwrite=False):
        """
        Perform photometry for each source using an elliptical Kron
        aperture.

        This method can be used to calculate the Kron photometry using
        alternate ``kron_params`` (e.g., different scalings of the Kron
        radius).

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        Parameters
        ----------
        kron_params : list of 2 or 3 floats, optional
            A list of parameters used to determine the Kron aperture.
            The first item is the scaling parameter of the unscaled
            Kron radius and the second item represents the minimum
            value for the unscaled Kron radius in pixels. The optional
            third item is the minimum circular radius in pixels. If
            ``kron_params[0]`` * `kron_radius` * sqrt(`semimajor_axis`
            * `semiminor_axis`) is less than or equal to this radius,
            then the Kron aperture will be a circle with this minimum
            radius.

        name : str or `None`, optional
            The prefix name which will be used to define attribute
            names for the Kron flux and flux error. The attribute
            names ``[name]_flux`` and ``[name]_flux_err`` will store
            the photometry results. For example, these names can then
            be included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        flux, flux_err : float or `~numpy.ndarray` of floats
            The aperture fluxes and flux errors. NaN will be returned
            where the aperture is `None` (e.g., where the source
            `centroid` position or elliptical shape parameters are not
            finite or where the source is completely masked).
        """
        return self._photometry._kron_photometry_table(
            kron_params, name, overwrite)

    @lazyproperty
    def _kron_photometry(self):
        """
        The flux and flux error in the Kron aperture (without units).

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        return np.transpose(self._calc_kron_photometry(kron_params=None))

    @lazyproperty
    @as_scalar
    def kron_flux(self):
        """
        The flux in the Kron aperture.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_flux = self._kron_photometry[:, 0]
        if self._data_unit is not None:
            kron_flux <<= self._data_unit
        return kron_flux

    @lazyproperty
    @as_scalar
    def kron_flux_err(self):
        """
        The flux error in the Kron aperture.

        See the `SourceCatalog` ``aperture_mask_method`` keyword for
        options to mask neighboring sources.

        If the Kron aperture is `None`, then ``np.nan`` will be
        returned. This will occur where the source `centroid` position
        or elliptical shape parameters are not finite or where the
        source is completely masked.
        """
        kron_flux_err = self._kron_photometry[:, 1]
        if self._data_unit is not None:
            kron_flux_err <<= self._data_unit
        return kron_flux_err

    @lazyproperty
    @use_detcat
    def _max_circular_kron_radius(self):
        """
        The maximum circular Kron radius used as the upper limit of
        ``flux_radius``.
        """
        return self._photometry._max_circular_kron_radius()

    @lazyproperty
    @use_detcat
    def _flux_radius_optimizer_args(self):
        """
        Per-source pre-computed argument tuples used by ``flux_radius``
        root finding.
        """
        return self._photometry._flux_radius_optimizer_args()

    @as_scalar
    @deprecated_positional_kwargs(since='3.0', until='4.0')
    def flux_radius(self, fraction, name=None, overwrite=False):
        """
        Calculate the circular radius that encloses the specified
        fraction of the Kron flux.

        To estimate the half-light radius, use ``fraction = 0.5``.

        Parameters
        ----------
        fraction : float
            The fraction of the Kron flux at which to find the circular
            radius.

        name : str or `None`, optional
            The attribute name which will be assigned to the value
            of the output array. For example, this name can then be
            included in the `to_table` ``columns`` keyword list to
            output the results in the table.

        overwrite : bool, optional
            If True, overwrite the attribute ``name`` if it exists.

        Returns
        -------
        radius : 1D `~numpy.ndarray`
            The circular radius that encloses the specified fraction of
            the Kron flux. NaN is returned where no solution was found
            or where the Kron flux is zero or non-finite.
        """
        return self._photometry._flux_radius(fraction, name, overwrite)

    @as_scalar
    def make_cutouts(self, shape, *, array=None, mode='partial',
                     fill_value=np.nan):
        """
        Make cutout arrays for each source.

        The cutout for each source will be centered at its
        `centroid` position. If a ``detection_catalog`` was input to
        `SourceCatalog`, then its `centroid` values will be used.

        Parameters
        ----------
        shape : 2-tuple
            The cutout shape along each axis in ``(ny, nx)`` order.

        array : `None` or 2D `~numpy.ndarray`
            A 2D array with the same shape as the ``data`` array input
            to `~photutils.segmentation.SourceCatalog`. If `None` then
            the ``data`` array will be used. If any cutout arrays
            are not fully contained within the ``array`` array and
            ``mode='partial'`` with ``fill_value=np.nan``, then the
            input ``array`` must have a float data type.

        mode : {'partial', 'trim'}, optional
            The mode used for extracting the cutout array. In
            ``'partial'`` mode, positions in the cutout array that
            do not overlap with the large array will be filled with
            ``fill_value``. In ``'trim'`` mode, only the overlapping
            elements are returned, thus the resulting small array may be
            smaller than the requested ``shape``.

        fill_value : number, optional
            If ``mode='partial'``, the value to fill pixels in the
            extracted cutout array that do not overlap with the input
            ``array_large``. ``fill_value`` will be changed to have
            the same ``dtype`` as the ``array_large`` array, with
            one exception. If ``array_large`` has integer type and
            ``fill_value`` is ``np.nan``, then a `ValueError` will be
            raised.

        Returns
        -------
        cutouts : `~photutils.utils.CutoutImage` \
                  or list of `~photutils.utils.CutoutImage`
            The `~photutils.utils.CutoutImage` for each source. The
            cutout will be `None` where the source `centroid` position
            is not finite or where the source is completely masked.
        """
        if array is None:
            array = self._data
        elif array.shape != self._data.shape:
            msg = 'array must have the same shape as data'
            raise ValueError(msg)

        if mode not in ('partial', 'trim'):
            msg = "mode must be 'partial' or 'trim'"
            raise ValueError(msg)

        cutouts = []
        for (xcen, ycen, all_masked) in zip(self._x_centroid,
                                            self._y_centroid,
                                            self._all_masked, strict=True):

            if all_masked or np.any(~np.isfinite((xcen, ycen))):
                cutouts.append(None)
                continue

            cutouts.append(CutoutImage(array, (ycen, xcen), shape,
                                       mode=mode, fill_value=fill_value))

        return cutouts
