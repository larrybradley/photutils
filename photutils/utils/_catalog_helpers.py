# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Shared helpers for catalog-style classes (e.g., `SourceCatalog`,
`ApertureStats`).

This module provides utilities that are reused by classes which expose
both scalar (single-source) and array (multi-source) modes via a
`isscalar` attribute and a set of `~astropy.utils.lazyproperty`
attributes.
"""

import functools
import inspect

from astropy.utils import lazyproperty

__all__ = []


def as_scalar(method):
    """
    Return a decorated method where it will always return a scalar value
    (instead of a length-1 tuple/list/array) if the class is scalar.

    Note that lazyproperties that begin with ``'_'`` should not have
    this decorator applied. Such properties are assumed to always be
    iterable and when slicing (see ``__getitem__``) from a cached
    multi-object catalog to create a single-object catalog, they will
    no longer be scalar.

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
    def _as_scalar(*args, **kwargs):
        result = method(*args, **kwargs)
        try:
            return (result[0] if args[0].isscalar and len(result) == 1
                    else result)
        except TypeError:  # if result has no len
            return result

    return _as_scalar


def get_lazyproperties(cls):
    """
    Return the list of all lazyproperty attribute names defined on a
    class (including its superclasses).

    The result is cached on the class itself (under the
    ``_cached_lazyproperties`` attribute) to avoid repeated
    introspection via `inspect.getmembers`. Subclasses receive their
    own cached list.

    Parameters
    ----------
    cls : type
        The class to inspect.

    Returns
    -------
    names : list of str
        The names of the lazyproperty attributes on ``cls``.
    """
    attr = '_cached_lazyproperties'
    # Subclasses get their own lazyproperty list
    if attr not in cls.__dict__:
        def _islazyproperty(obj):
            return isinstance(obj, lazyproperty)

        setattr(cls, attr,
                [i[0] for i in inspect.getmembers(
                    cls, predicate=_islazyproperty)])
    return getattr(cls, attr)


def slice_composition_helper(old_helper, helper_factory, helper_cls, index,
                             *, isscalar_new, index_object_list):
    """
    Build a new composition helper via ``helper_factory`` and copy
    sliced cached lazyproperty values from ``old_helper`` into it.

    This preserves expensive per-source computations cached on
    composition helpers (e.g., `_LocalBackground`, `_CentroidRefiner`,
    `_ShapeProperties`) when the host catalog is sliced.

    Parameters
    ----------
    old_helper : helper instance or None
        The previously-cached helper instance on the parent catalog,
        or `None` if it was never computed.

    helper_factory : callable
        A zero-argument callable that returns a fresh helper instance
        bound to the (already-sliced) host.

    helper_cls : type
        The helper class (used to introspect cached
        `~astropy.utils.lazyproperty` names).

    index : array_like or slice
        The index applied to the parent catalog.

    isscalar_new : bool
        Whether the new host is scalar (single source).

    index_object_list : callable
        Fallback ``(value, index) -> sliced_value`` for object lists
        that do not support direct indexing (e.g., fancy indices on
        Python lists).

    Returns
    -------
    new_helper : helper instance or None
        The new helper with sliced caches, or `None` if ``old_helper``
        was `None`.
    """
    import numpy as np
    if old_helper is None:
        return None

    new_helper = helper_factory()
    cached_keys = (set(old_helper.__dict__.keys())
                   & set(get_lazyproperties(helper_cls)))
    for key in cached_keys:
        value = old_helper.__dict__[key]
        if np.isscalar(value):
            new_helper.__dict__[key] = value
            continue

        # Tuples of arrays (e.g., ``(cos_theta, sin_theta)``) are
        # cached as a unit; slice each element individually.
        if isinstance(value, tuple):
            new_helper.__dict__[key] = tuple(
                _slice_one(item, index, isscalar_new, index_object_list)
                for item in value)
            continue

        new_helper.__dict__[key] = _slice_one(
            value, index, isscalar_new, index_object_list)

    return new_helper


def _slice_one(value, index, isscalar_new, index_object_list):
    """Slice a single cached lazyproperty value using the same
    convention as the catalog ``__getitem__`` loop.
    """
    import numpy as np
    if np.isscalar(value):
        return value
    try:
        if isscalar_new:
            if isinstance(value, np.ndarray):
                return value[:, np.newaxis][index]
            return [value[index]]
        return value[index]
    except TypeError:
        return index_object_list(value, index)
