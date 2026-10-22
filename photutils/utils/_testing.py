# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Private helpers for the test suite.
"""

from astropy.utils.data import download_file

__all__ = []


def get_path(filename, *, cache=True):
    """
    Get the local path for a file in the photutils-datasets repository.

    Parameters
    ----------
    filename : str
        File name in the photutils-datasets data folder.

    cache : bool, optional
        Whether to cache the contents of the remote URL. Default is
        `True`.

    Returns
    -------
    path : str
        The local path of the file.
    """
    url = ('https://github.com/astropy/photutils-datasets/raw/'
           f'main/data/{filename}')
    return download_file(url, cache=cache)
