# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A module to test the deprecation of old column names in Astropy tables.
"""

import numpy as np
import pytest
from astropy.table import Table, join, unique
from astropy.table.np_utils import TableMergeError
from astropy.utils.exceptions import AstropyDeprecationWarning

from photutils.utils._deprecation import (DeprecatedColumnQTable,
                                          DeprecatedColumnTable,
                                          create_deprecated_table)

DEPRECATION_MAP = {'old': 'new', 'old_b': 'new_b'}


@pytest.fixture
def raw_data():
    """
    Provide a raw data dictionary for table creation.
    """
    return {'old': [3, 2, 1], 'old_b': [4, 5, 6], 'stable': [7, 8, 9]}


def test_creation_and_type(raw_data):
    """
    Test that the factory creates the correct object type.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)
    assert isinstance(table, DeprecatedColumnTable)
    assert not isinstance(table, DeprecatedColumnQTable)
    assert set(table.colnames) == {'new', 'new_b', 'stable'}

    qtable = create_deprecated_table(raw_data, DEPRECATION_MAP,
                                     use_qtable=True)
    assert isinstance(qtable, DeprecatedColumnQTable)


def test_masked_creation(raw_data):
    """
    Test that kwargs like "masked" are passed through correctly.
    """
    table = create_deprecated_table(
        raw_data, DEPRECATION_MAP, masked=True,
    )
    assert isinstance(table, DeprecatedColumnTable)
    assert table.masked is True
    table['new'].mask[0] = True
    assert np.all(table['new'].mask == [True, False, False])


def test_getitem_access(raw_data):
    """
    Test deprecated access via __getitem__.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)

    match_old = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match_old):
        col = table['old']
    assert np.all(col == table['new'])

    match_old_b = "'old_b' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match_old_b):
        sub_table = table[['stable', 'old_b']]
    assert sub_table.colnames == ['stable', 'new_b']


def test_setitem_assignment(raw_data):
    """
    Test deprecated assignment via __setitem__.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table['old'] = [100, 200, 300]
    assert np.all(table['new'] == [100, 200, 300])


def test_delitem_and_remove(raw_data):
    """
    Test deprecated deletion via __delitem__ and remove methods.
    """
    table1 = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        del table1['old']
    assert 'new' not in table1.colnames

    table2 = create_deprecated_table(raw_data, DEPRECATION_MAP)
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table2.remove_column('old')
    assert 'new' not in table2.colnames


def test_keep_columns(raw_data):
    """
    Test deprecated use in keep_columns.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table.keep_columns(['stable', 'old'])
    assert set(table.colnames) == {'stable', 'new'}


def test_rename_methods(raw_data):
    """
    Test deprecated use in rename_column and rename_columns.
    """
    table1 = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match1 = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match1):
        table1.rename_column('old', 'final_name_1')
    assert 'final_name_1' in table1.colnames
    assert 'new' not in table1.colnames

    table2 = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match2 = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match2):
        table2.rename_columns(['old', 'old_b'], ['final1', 'final2'])
    assert set(table2.colnames) == {'final1', 'final2', 'stable'}


def test_data_operations(raw_data):
    """
    Test deprecated use in sort, group_by, and unique.
    """
    table_sort = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match_sort = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match_sort):
        table_sort.sort('old')
    assert table_sort['new'][0] == 1

    table_group = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match_group = "'old_b' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match_group):
        groups = table_group.group_by('old_b')
    assert len(groups.groups) == 3

    table_unique = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match_unique = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match_unique):
        unique_table = unique(table_unique, keys='old')
    assert len(unique_table) == 3


def test_join(raw_data, recwarn):
    """
    Test deprecated use in the standalone join function.
    """
    table1 = create_deprecated_table(raw_data, DEPRECATION_MAP)
    table2 = Table({'new': [1, 3], 'extra': [1.1, 3.3]})

    msg = "Left table does not have key column 'old'"
    with pytest.raises(TableMergeError, match=msg):
        join(table1, table2, keys='old')

    # Test that it works correctly with the new name and issues no warnings.
    joined = join(table1, table2, keys='new')
    assert 'extra' in joined.colnames
    assert len(joined) == 2
    assert len(recwarn) == 0


def test_indexing(raw_data, recwarn):
    """
    Test deprecated use in add_index and remove_indices.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)
    match = "'old' is deprecated"
    with pytest.warns(AstropyDeprecationWarning, match=match):
        table.add_index('old')

    assert len(table.indices) == 1
    assert table.indices[0].columns[0].name == 'new'

    # The `pytest.warns` context manager consumes the warning. Now we can
    # test that the next operation issues no new warnings.
    table.remove_indices('new')
    assert not table.indices
    assert len(recwarn) == 0


def test_non_string_access_no_warning(raw_data, recwarn):
    """
    Test that non-string access does not trigger warnings.

    This ensures that row access via integers or slices does not
    incorrectly engage the name translation logic.
    """
    table = create_deprecated_table(raw_data, DEPRECATION_MAP)

    # Access a row by integer
    row = table[0]
    assert row['new'] == 3

    # Slice rows
    sliced = table[0:2]
    assert len(sliced) == 2

    # Assert that no warnings were recorded during these operations.
    assert len(recwarn) == 0
