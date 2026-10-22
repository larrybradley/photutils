# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the import order and signatures of photutils.simulation.
"""

import inspect
import subprocess  # nosec B404
import sys

import pytest


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


@pytest.mark.parametrize('name', [
    'make_model_params',
    'make_random_models_table',
    'make_model_image',
    'make_psf_model_image',
    'make_4gaussians_image',
    'make_100gaussians_image',
    'make_noise_image',
    'apply_poisson_noise',
    'make_wcs',
    'make_gwcs'])
def test_optional_args_keyword_only(name):
    """
    Test that every parameter with a default is keyword-only.
    """
    from photutils import simulation

    sig = inspect.signature(getattr(simulation, name))
    for param in sig.parameters.values():
        if param.default is not inspect.Parameter.empty:
            assert param.kind is inspect.Parameter.KEYWORD_ONLY, param.name
