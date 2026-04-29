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
