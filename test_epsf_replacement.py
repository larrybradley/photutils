#!/usr/bin/env python3
"""
Simple test to check if our CompatibleImagePSF replacement works.
"""

from astropy.nddata import NDData
from astropy.table import Table

from photutils.datasets import make_100gaussians_image
from photutils.psf.epsf import EPSFBuilder
from photutils.psf.epsf_stars import extract_stars


def test_epsf_builder():
    """
    Test that EPSFBuilder works with our CompatibleImagePSF replacement.
    """
    print('Creating test data...')

    # Create a simple test image with some stars
    data = make_100gaussians_image()

    # Create some simple star positions
    stars_tbl = Table()
    stars_tbl['x'] = [100, 200, 150]
    stars_tbl['y'] = [100, 200, 150]

    print('Extracting stars...')
    try:
        nddata = NDData(data)
        stars = extract_stars(nddata, stars_tbl, size=11)
        print(f"Extracted {len(stars)} stars")

        print('Creating EPSFBuilder...')
        epsf_builder = EPSFBuilder(oversampling=1, maxiters=2,
                                   progress_bar=False, norm_radius=3)

        print('Building ePSF...')
        epsf, fitted_stars = epsf_builder(stars)

        print(f"Success! Built ePSF with data shape {epsf.data.shape}")
        print(f"ePSF type: {type(epsf)}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_epsf_builder()
    if success:
        print('✓ Test passed! Our CompatibleImagePSF replacement works.')
    else:
        print('✗ Test failed.')
