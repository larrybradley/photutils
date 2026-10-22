# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the _testing module.
"""

from unittest.mock import patch

import pytest

from photutils.utils._testing import get_path


@pytest.mark.parametrize('cache', [True, False])
def test_get_path(cache):
    """
    Test that get_path downloads from the photutils-datasets repository
    and passes the cache keyword through.
    """
    with patch('photutils.utils._testing.download_file',
               return_value='local_path') as mock_download:
        result = get_path('isophote/M51.fits', cache=cache)

    assert result == 'local_path'
    mock_download.assert_called_once()
    args, kwargs = mock_download.call_args
    assert args[0].endswith('photutils-datasets/raw/main/data/isophote/'
                            'M51.fits')
    assert kwargs['cache'] is cache


def test_get_path_default_cache():
    """
    Test that get_path caches by default.
    """
    with patch('photutils.utils._testing.download_file',
               return_value='path') as mock_download:
        get_path('isophote/M51.fits')

    assert mock_download.call_args.kwargs['cache'] is True
