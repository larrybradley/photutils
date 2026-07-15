# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Root pytest configuration.

This module patches the pytest-doctestplus output checker so
that doctest comparisons of printed astropy tables ignore the
lengths of the dashed column-separator rows. Together with the
FLOAT_CMP and NORMALIZE_WHITESPACE doctest options (enabled in
pyproject.toml), this allows documentation examples to display
tables with truncated float values without needing to set a
column ``info.format`` for consistent table output.
"""

import re

from pytest_doctestplus.output_checker import OutputChecker

_DASH_RUN = re.compile(r'-{2,}')
_check_output = OutputChecker.check_output


def _check_output_ignore_dashes(self, want, got, flags):
    """
    Check doctest output, ignoring the lengths of dash runs.

    If the standard check fails, retry with every run of two or
    more hyphens (e.g., astropy table column separators, whose
    widths depend on the formatted column values) collapsed to a
    fixed length in both the expected and actual output.
    """
    if _check_output(self, want, got, flags):
        return True

    return _check_output(self, _DASH_RUN.sub('--', want),
                         _DASH_RUN.sub('--', got), flags)


OutputChecker.check_output = _check_output_ignore_dashes
