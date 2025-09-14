#!/usr/bin/env python3
"""
Test the CompatibleImagePSF class functionality.
"""

import numpy as np

from photutils.psf.compatible_epsf import CompatibleImagePSF


def test_compatible_epsf_basic():
    """
    Test basic CompatibleImagePSF functionality.
    """
    # Create test data
    size = 21
    y, x = np.mgrid[0:size, 0:size]
    center = size // 2

    # Create a simple Gaussian-like PSF
    sigma = 2.0
    data = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))

    # Test basic initialization
    origin = (center, center)  # (y, x) format for origin
    oversampling = 2

    model = CompatibleImagePSF(data, origin=origin,
                               oversampling=oversampling,
                               norm_radius=5.5, normalize=False)

    # Test basic properties
    assert model.shape == data.shape
    assert np.allclose(model._data, data)
    assert np.allclose(model.origin, origin)

    # Test evaluation at the center
    print(f"Model origin: {model._origin}")
    print(f"Input origin: {origin}")
    print(f"Model oversampling: {model.oversampling}")

    # The model coordinate system: evaluate at (0, 0) to get the origin value
    flux_val = model.evaluate(0, 0, flux=1.0, x_0=0.0, y_0=0.0)
    print(f"Flux val at (0, 0): {flux_val}")
    print(f"Data at center: {data[center, center]}")
    assert flux_val > 0  # Should be positive at center

    # Test evaluation with array inputs
    x_test = np.array([0, 1])  # Evaluate at origin and slightly offset
    y_test = np.array([0, 1])
    vals = model.evaluate(x_test, y_test, flux=1.0, x_0=0.0, y_0=0.0)
    print(f"Values: {vals}")
    assert len(vals) == 2
    assert vals[0] > vals[1]  # Origin should be brighter than offset


def test_compatible_epsf_normalization():
    """
    Test CompatibleImagePSF normalization functionality.
    """
    size = 11
    y, x = np.mgrid[0:size, 0:size]
    center = size // 2

    # Create test data
    data = np.exp(-((x - center)**2 + (y - center)**2) / 8.0)

    # Test with normalization enabled
    model = CompatibleImagePSF(data, norm_radius=3.0, normalize=True)

    # The model should have normalization applied
    assert hasattr(model, '_normalization_constant')
    assert model._normalization_constant > 0

    # Test that normalized data is different from original
    assert not np.allclose(model._data, data)


def test_compatible_epsf_compatibility_api():
    """
    Test that CompatibleImagePSF provides the expected API.
    """
    size = 11
    data = np.ones((size, size))

    model = CompatibleImagePSF(data)

    # Test that it has the expected properties for compatibility
    assert hasattr(model, 'shape')
    assert hasattr(model, '_data')
    assert hasattr(model, 'origin')
    assert hasattr(model, 'oversampling')

    # Test that shape and _data work as expected
    assert model.shape == data.shape
    assert isinstance(model._data, np.ndarray)


if __name__ == '__main__':
    test_compatible_epsf_basic()
    test_compatible_epsf_normalization()
    test_compatible_epsf_compatibility_api()
    print('All tests passed!')
