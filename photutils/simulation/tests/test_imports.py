# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the import order and signatures of photutils.simulation.
"""

import inspect
import subprocess  # nosec B404
import sys

import pytest

from photutils import simulation

# photutils.simulation defines no package-level __all__, so collect the
# public callables defined in its submodules
PUBLIC_FUNCTIONS = sorted(
    name for name in dir(simulation)
    if (not name.startswith('_')
        and callable(getattr(simulation, name))
        and getattr(simulation, name).__module__.startswith(
            'photutils.simulation.')))


@pytest.mark.parametrize('code', [
    'import photutils.simulation',
    'import photutils.psf',
    'import photutils.psf; import photutils.simulation',
    'import photutils.simulation; import photutils.psf',
    'import photutils.datasets'])
def test_import_order(code):
    """
    Test that the subpackages import in any order without a circular
    import error.
    """
    subprocess.run(  # noqa: S603  # nosec B603
        [sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('name', PUBLIC_FUNCTIONS)
def test_optional_args_keyword_only(name):
    """
    Test that every parameter with a default is keyword-only.
    """
    sig = inspect.signature(getattr(simulation, name))
    for param in sig.parameters.values():
        if param.default is not inspect.Parameter.empty:
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, param.name


def test_public_functions_collected():
    """
    Test that the public function collection is not empty.
    """
    assert 'make_model_image' in PUBLIC_FUNCTIONS
    assert len(PUBLIC_FUNCTIONS) >= 10
