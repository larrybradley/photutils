# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module to create Astropy Tables with deprecated column names.

This module provides a robust, performant solution for creating tables
with built-in column name deprecation. It is designed to create new
table objects from raw data, rather than modifying existing tables.

The primary function, ``create_deprecated_table``, handles the data
renaming and constructs an instance of a custom ``Table`` or ``QTable``
subclass that correctly handles all deprecated name access.
"""

import warnings

from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyDeprecationWarning


class DeprecatedColumnMixin:
    """
    A mixin to handle deprecated column names in Astropy tables.

    This mixin overrides common table methods to intercept calls
    using old column names. It translates them to new names, issues a
    deprecation warning, and then calls the original parent method via
    ``super()``. This works correctly because instances are created from
    this class directly, ensuring a valid method resolution order.
    """

    _deprecation_map = {}

    def _translate_one_name(self, name):
        """
        Translate a single name, issue a warning, and return the new
        name.

        Parameters
        ----------
        name : str
            The column name to be translated.

        Returns
        -------
        result : str
            The translated new column name, or the original name if it
            is not deprecated.
        """
        if name in self._deprecation_map:
            new_name = self._deprecation_map[name]
            msg = (
                f"The column name '{name}' is deprecated; using "
                f"'{new_name}' instead. This will be removed in a "
                'future version.'
            )
            warnings.warn(msg, AstropyDeprecationWarning, stacklevel=5)
            return new_name
        return name

    def _translate_names(self, names):
        """
        Translate a single name or a list/tuple of names.

        Parameters
        ----------
        names : str or list or tuple
            The column name(s) to be translated.

        Returns
        -------
        str or list or tuple
            The translated new column name(s).
        """
        if isinstance(names, (list, tuple)):
            return [self._translate_one_name(name) for name in names]

        if not isinstance(names, str):
            return names

        return self._translate_one_name(names)

    def __getitem__(self, item):
        """
        Override for item access.
        """
        if isinstance(item, (str, list, tuple)):
            item = self._translate_names(item)
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        """
        Override for item assignment.
        """
        if isinstance(item, str):
            item = self._translate_names(item)
        super().__setitem__(item, value)

    def __delitem__(self, item):
        """
        Override for item deletion.
        """
        if isinstance(item, str):
            item = self._translate_names(item)
        super().__delitem__(item)

    def keep_columns(self, names):
        """
        Override for keeping specified columns.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of column names to keep.
        """
        names = self._translate_names(names)
        super().keep_columns(names)

    def remove_column(self, name):
        """
        Override for column removal.

        Parameters
        ----------
        name : str
            The name of the column to be removed.
        """
        name = self._translate_names(name)
        super().remove_column(name)

    def remove_columns(self, names):
        """
        Override for multiple column removal.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of column names to be removed.
        """
        names = self._translate_names(names)
        super().remove_columns(names)

    def rename_column(self, name, new_name):
        """
        Override for column renaming.

        Parameters
        ----------
        name : str
            The current name of the column to be renamed.

        new_name : str
            The new name for the column.
        """
        name = self._translate_names(name)
        super().rename_column(name, new_name)

    def rename_columns(self, names, new_names):
        """
        Override for multiple column renaming.

        Parameters
        ----------
        names : list or tuple
            A list or tuple of current column names to be renamed.

        new_names : list or tuple
            A list or tuple of new names for the columns.
        """
        names = self._translate_names(names)
        super().rename_columns(names, new_names)

    def replace_column(self, name, new_col):
        """
        Override for column replacement.

        Parameters
        ----------
        name : str
            The current name of the column to be replaced.

        new_col : `Column` or `MaskedColumn`
            The new column to replace the existing one.
        """
        name = self._translate_names(name)
        super().replace_column(name, new_col)

    def add_index(self, names):
        """
        Override for index addition.

        Parameters
        ----------
        names : str or list or tuple
            The name(s) of the column(s) to be indexed.
        """
        names = self._translate_names(names)
        super().add_index(names)

    def remove_indices(self, names):
        """
        Override for index removal.

        Parameters
        ----------
        names : str or list or tuple
            The name(s) of the column(s) whose indices are to be removed.
        """
        names = self._translate_names(names)
        super().remove_indices(names)

    def sort(self, keys):
        """
        Override for sorting.

        Parameters
        ----------
        keys : str or list or tuple
            The name(s) of the column(s) to sort by.
        """
        keys = self._translate_names(keys)
        super().sort(keys)

    def group_by(self, keys):
        """
        Override for grouping.

        Parameters
        ----------
        keys : str or list or tuple
            The name(s) of the column(s) to group by.
        """
        keys = self._translate_names(keys)
        return super().group_by(keys)


class DeprecatedColumnTable(DeprecatedColumnMixin, Table):
    """
    An Astropy Table with built-in support for deprecated names.
    """


class DeprecatedColumnQTable(DeprecatedColumnMixin, QTable):
    """
    An Astropy QTable with built-in support for deprecated names.
    """


def create_deprecated_table(data, deprecation_map, use_qtable=False, **kwargs):
    """
    Create a new table from scratch with deprecated column name support.

    This function takes raw data and a deprecation map, renames the
    data keys internally, and constructs the appropriate ``Table`` or
    ``QTable`` subclass. All other keywords are passed directly to the
    underlying table constructor.

    Parameters
    ----------
    data : dict
        A dictionary of data for the table, using the OLD (soon to be
        deprecated) column names as keys.

    deprecation_map : dict
        A dictionary mapping old (deprecated) names to new names.

    use_qtable : bool, optional
        If ``True``, a ``DeprecatedColumnQTable`` will be created.
        Defaults to ``False``.

    **kwargs : dict, optional
        Any other keywords accepted by the ``astropy.table.Table``
        constructor (e.g., ``masked=True``, ``meta={...}``).

    Returns
    -------
    table : `DeprecatedColumnTable` or `DeprecatedColumnQTable`
        A new table instance with deprecation behavior.
    """
    table_class = (DeprecatedColumnQTable if use_qtable
                   else DeprecatedColumnTable)

    # Rename the keys in the data dictionary before creation
    renamed_data = {
        deprecation_map.get(k, k): v for k, v in data.items()
    }

    # Create the table instance
    table = table_class(renamed_data, **kwargs)
    table._deprecation_map = deprecation_map
    return table
