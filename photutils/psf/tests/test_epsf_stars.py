# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the epsf_stars module.
"""

import warnings

import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_array_equal

from photutils.psf.epsf_stars import (EPSFStar, EPSFStars,
                                      _compute_mean_sky_coordinate,
                                      _create_weights_cutout,
                                      _prepare_uncertainty_info, extract_stars)
from photutils.psf.functional_models import CircularGaussianPRF
from photutils.psf.image_models import ImagePSF


class TestExtractStars:
    def setup_class(self):
        stars_tbl = Table()
        stars_tbl['x'] = [15, 15, 35, 35]
        stars_tbl['y'] = [15, 35, 40, 10]
        self.stars_tbl = stars_tbl

        yy, xx = np.mgrid[0:51, 0:55]
        self.data = np.zeros(xx.shape)
        for (xi, yi) in zip(stars_tbl['x'], stars_tbl['y'], strict=True):
            m = Moffat2D(100, xi, yi, 3, 3)
            self.data += m(xx, yy)

        self.nddata = NDData(data=self.data)

    def test_extract_stars(self):
        size = 11
        stars = extract_stars(self.nddata, self.stars_tbl, size=size)
        assert len(stars) == 4
        assert isinstance(stars, EPSFStars)
        assert isinstance(stars[0], EPSFStars)
        assert stars[0].data.shape == (size, size)
        assert stars.n_stars == stars.n_all_stars
        assert stars.n_stars == stars.n_good_stars
        assert stars.center.shape == (len(stars), 2)

    def test_extract_stars_inputs(self):
        match = 'data must be a single NDData object or list of NDData objects'
        with pytest.raises(TypeError, match=match):
            extract_stars(np.ones(3), self.stars_tbl)

        match = 'All catalog elements must be Table objects'
        with pytest.raises(TypeError, match=match):
            extract_stars(self.nddata, [(1, 1), (2, 2), (3, 3)])

        match = 'number of catalogs must match the number of input images'
        with pytest.raises(ValueError, match=match):
            extract_stars(self.nddata, [self.stars_tbl, self.stars_tbl])

        match = 'the catalog must have a "skycoord" column'
        with pytest.raises(ValueError, match=match):
            extract_stars([self.nddata, self.nddata], self.stars_tbl)


def test_epsf_star_residual_image():
    """
    Test to ensure ``compute_residual_image`` gives correct residuals.
    """
    size = 100
    yy, xx, = np.mgrid[0:size + 1, 0:size + 1] / 4
    gmodel = CircularGaussianPRF().evaluate(xx, yy, 1, 12.5, 12.5, 2.5)
    epsf = ImagePSF(gmodel, oversampling=4)
    _size = 25
    data = np.zeros((_size, _size))
    _yy, _xx, = np.mgrid[0:_size, 0:_size]
    data += epsf.evaluate(x=_xx, y=_yy, flux=16, x_0=12, y_0=12)
    tbl = Table()
    tbl['x'] = [12]
    tbl['y'] = [12]
    stars = extract_stars(NDData(data), tbl, size=23)
    residual = stars[0].compute_residual_image(epsf)
    # As current EPSFStar instances cannot accept CircularGaussianPRF
    # as input, we have to accept some loss of precision from the
    # conversion to ePSF, and spline fitting (twice), so assert_allclose
    # cannot be more precise than 0.001 currently.
    assert_allclose(np.sum(residual), 0.0, atol=1.0e-3, rtol=1e-3)


def test_stars_pickleable():
    """
    Verify that EPSFStars can be successfully pickled/unpickled for use
    multiprocessing.
    """
    from multiprocessing.reduction import ForkingPickler

    # Doesn't need to actually contain anything useful
    stars = EPSFStars([1])
    # This should not blow up
    ForkingPickler.loads(ForkingPickler.dumps(stars))


class TestEPSFStar:
    """
    Test EPSFStar class functionality.
    """

    def test_basic_initialization(self):
        """
        Test basic EPSFStar initialization.
        """
        data = np.ones((11, 11))
        star = EPSFStar(data)

        assert star.data.shape == (11, 11)
        assert star.cutout_center is not None
        assert star.weights.shape == data.shape
        assert star.flux > 0

    def test_invalid_data_input(self):
        """
        Test EPSFStar initialization with invalid data.
        """
        # Test None data
        with pytest.raises(ValueError, match='Input data cannot be None'):
            EPSFStar(None)

        # Test 1D data
        match = 'Input data must be 2-dimensional'
        with pytest.raises(ValueError, match=match):
            EPSFStar(np.ones(10))

        # Test 3D data
        with pytest.raises(ValueError, match=match):
            EPSFStar(np.ones((5, 5, 5)))

        # Test empty data
        with pytest.raises(ValueError, match='Input data cannot be empty'):
            EPSFStar(np.array([]).reshape(0, 0))

    def test_weights_validation(self):
        """
        Test weight validation in EPSFStar.
        """
        data = np.ones((5, 5))

        # Test mismatched weights shape
        wrong_weights = np.ones((3, 3))
        match = 'Weights shape .* must match data shape'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, weights=wrong_weights)

        # Test non-finite weights (should generate warning)
        bad_weights = np.ones((5, 5))
        bad_weights[2, 2] = np.inf
        bad_weights[1, 1] = np.nan

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star = EPSFStar(data, weights=bad_weights)
            assert len(w) == 1
            assert issubclass(w[0].category, AstropyUserWarning)
            assert 'Non-finite weight values' in str(w[0].message)

        # Check that non-finite weights were set to zero
        assert star.weights[2, 2] == 0.0
        assert star.weights[1, 1] == 0.0

    def test_invalid_data_handling(self):
        """
        Test handling of invalid pixel values.
        """
        data = np.ones((5, 5))
        data[1, 1] = np.nan
        data[2, 2] = np.inf

        # Should warn about high percentage of invalid data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star = EPSFStar(data)
            # Should mask invalid pixels
            assert star.mask[1, 1]
            assert star.mask[2, 2]
            assert star.weights[1, 1] == 0.0
            assert star.weights[2, 2] == 0.0
            # Check that warning was issued about invalid data
            assert len(w) > 0

    def test_cutout_center_validation(self):
        """
        Test cutout_center validation.
        """
        data = np.ones((5, 5))
        star = EPSFStar(data)

        # Test invalid shape
        with pytest.raises(ValueError, match='must have exactly 2 elements'):
            star.cutout_center = [1, 2, 3]

        # Test non-finite values
        with pytest.raises(ValueError, match='must be finite'):
            star.cutout_center = [np.nan, 2.0]

        # Test bounds warnings (should warn but not raise)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            star.cutout_center = [-1, 2]  # Outside bounds
            assert len(w) >= 1
            # Check that warning mentions coordinates outside bounds
            warning_messages = [str(warning.message) for warning in w]
            assert any('outside the cutout bounds' in msg
                       for msg in warning_messages)

    def test_origin_validation(self):
        """
        Test origin parameter validation.
        """
        data = np.ones((5, 5))

        # Test invalid origin shape
        match = 'Origin must have exactly 2 elements'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, origin=[1, 2, 3])

        # Test non-finite origin
        match = 'Origin coordinates must be finite'
        with pytest.raises(ValueError, match=match):
            EPSFStar(data, origin=[np.inf, 2])

    def test_estimate_flux_masked_data(self):
        """
        Test flux estimation with masked data.
        """
        data = np.ones((5, 5)) * 10

        # Create weights that mask some pixels
        weights = np.ones((5, 5))
        weights[1:3, 1:3] = 0  # Mask central 2x2 region

        star = EPSFStar(data, weights=weights)

        # Flux should be estimated via interpolation
        assert star.flux > 0
        # Should be close to total flux despite masking
        assert star.flux == pytest.approx(250, rel=0.1)  # 5*5*10 = 250


class TestHelperFunctions:
    """
    Test new helper functions.
    """

    def test_compute_mean_sky_coordinate(self):
        """
        Test spherical coordinate averaging.
        """
        # Test simple case with coordinates around equator
        coords = np.array([
            [0.0, 0.0],    # lon=0, lat=0
            [90.0, 0.0],   # lon=90, lat=0
            [180.0, 0.0],  # lon=180, lat=0
            [270.0, 0.0],   # lon=270, lat=0
        ])

        mean_lon, mean_lat = _compute_mean_sky_coordinate(coords)

        # Mean latitude should be close to 0
        assert abs(mean_lat) < 1e-10
        # Mean longitude is tricky due to wraparound, but should be reasonable

    def test_prepare_uncertainty_info(self):
        """
        Test uncertainty info preparation.
        """
        # Test with no uncertainty
        data = NDData(np.ones((5, 5)))
        info = _prepare_uncertainty_info(data)
        assert info['type'] == 'none'

        # Test with weight-like uncertainty by creating custom uncertainty
        class WeightsUncertainty(StdDevUncertainty):
            @property
            def uncertainty_type(self):
                return 'weights'

        weights = np.ones((5, 5)) * 2
        data.uncertainty = WeightsUncertainty(weights)

        info = _prepare_uncertainty_info(data)
        assert info['type'] == 'weights'
        assert_array_equal(info['array'], weights)

    def test_create_weights_cutout(self):
        """
        Test weights cutout creation.
        """
        # Test with no uncertainty
        info = {'type': 'none'}
        slices = (slice(1, 4), slice(1, 4))  # 3x3 cutout
        mask = None

        weights = _create_weights_cutout(info, mask, slices)
        assert weights.shape == (3, 3)
        assert_array_equal(weights, np.ones((3, 3)))

        # Test with mask
        full_mask = np.zeros((5, 5), dtype=bool)
        full_mask[2, 2] = True  # Mask center of cutout

        weights = _create_weights_cutout(info, full_mask, slices)
        assert weights[1, 1] == 0.0  # Should be masked


class TestExtractStarsEdgeCases:
    """
    Test extract_stars edge cases and error conditions.
    """

    def setup_method(self):
        """
        Set up test data.
        """
        self.data = np.ones((50, 50))
        self.nddata = NDData(self.data)

    def test_empty_catalog(self):
        """
        Test extraction with empty catalog.
        """
        empty_table = Table()
        empty_table['x'] = []
        empty_table['y'] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(self.nddata, empty_table)
            assert len(stars) == 0
            # Should warn about empty catalog
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('empty' in msg.lower() for msg in warning_messages)

    def test_stars_outside_image(self):
        """
        Test extraction with stars outside image bounds.
        """
        table = Table()
        table['x'] = [-10, 100]  # Outside image bounds
        table['y'] = [25, 25]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            stars = extract_stars(self.nddata, table, size=11)
            assert len(stars) == 0
            # Should warn about excluded stars
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any('not extracted' in msg for msg in warning_messages)

    def test_invalid_input_types(self):
        """
        Test extraction with invalid input types.
        """
        table = Table()
        table['x'] = [25]
        table['y'] = [25]

        # Test invalid data type
        with pytest.raises(TypeError, match='must be a single NDData object'):
            extract_stars('not_nddata', table)

        # Test invalid catalog type
        with pytest.raises(TypeError, match='must be a single Table object'):
            extract_stars(self.nddata, 'not_table')

    def test_coordinate_validation(self):
        """
        Test coordinate system validation.
        """
        table = Table()
        table['x'] = [25]
        table['y'] = [25]

        # Test missing skycoord for multiple images
        with pytest.raises(ValueError, match='must have a "skycoord" column'):
            extract_stars([self.nddata, self.nddata], table)

        # Test missing coordinate columns
        bad_table = Table()
        bad_table['flux'] = [100]  # No x, y, or skycoord

        with pytest.raises(ValueError, match='must have either'):
            extract_stars(self.nddata, bad_table)
