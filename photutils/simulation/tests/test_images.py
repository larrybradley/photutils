# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Box2D, Moffat2D, Polynomial2D
from astropy.nddata import NDData
from astropy.table import QTable, Table
from numpy.testing import assert_allclose, assert_array_equal

from photutils.psf import (AiryDiskPSF, CircularGaussianPRF,
                           CircularGaussianPSF, CircularGaussianSigmaPRF,
                           GaussianPRF, GaussianPSF, GriddedPSFModel, ImagePSF,
                           MoffatPRF, MoffatPSF, make_epsf_from_psf)
from photutils.simulation import (SersicSource, make_convolved_model_image,
                                  make_model_image)
from photutils.simulation.images import _model_shape_from_bbox


def test_make_model_image():
    """
    Test the basic functionality of make_model_image.
    """
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # Test variable model shape
    params['model_shape'] = [9, 7, 11]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # Test local_bkg
    params['local_bkg'] = [1, 2, 3]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # Test scalar shape
    del params['model_shape']
    del params['local_bkg']
    image = make_model_image(300, model, params, model_shape=model_shape)
    assert image.shape == (300, 300)


def test_make_model_image_variable_shape_pairs():
    """
    Test a model_shape column containing (ny, nx) pairs.
    """
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    params['model_shape'] = np.array([[9, 7], [7, 9], [11, 11]])
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert image.sum() > 1


def test_make_model_image_variable_shape_invalid():
    """
    Test that invalid values in a model_shape column raise an error.
    """
    params = QTable()
    params['x_0'] = [50, 70]
    params['y_0'] = [50, 50]
    params['gamma'] = [1.7, 2.32]
    params['alpha'] = [2.9, 5.7]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)

    match = 'model_shape must be > 0'
    for model_shapes in ([0, 11], [-5, 11]):
        params['model_shape'] = model_shapes
        with pytest.raises(ValueError, match=match):
            make_model_image(shape, model, params)

    match = 'model_shape must be a finite value'
    params['model_shape'] = [np.nan, 11.0]
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params)


def test_make_model_image_nonfinite_local_bkg():
    """
    Test that sources with a nonfinite local_bkg value are skipped.
    """
    params = QTable()
    params['x_0'] = [50.0, 70.0, 90.0]
    params['y_0'] = [50.0, 50.0, 50.0]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (100, 150)
    model_shape = (11, 11)
    params['local_bkg'] = [0.0, 0.0, 0.0]
    image0 = make_model_image(shape, model, params, model_shape=model_shape)

    params['local_bkg'] = [np.nan, 0.0, np.inf]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert not np.any(np.isnan(image))
    assert image[50, 50] == 0  # first source skipped
    assert image[50, 90] == 0  # third source skipped
    assert_allclose(image[50, 70], image0[50, 70])


def test_make_model_image_units():
    """
    Test that the model image is created with the correct units when the
    flux column has units.
    """
    unit = u.Jy
    params = QTable()
    params['x_0'] = [30, 50, 70.5]
    params['y_0'] = [50, 50, 50.5]
    params['flux'] = [1, 2, 3] * unit
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert model.flux == 1.0  # Default flux (unchanged)

    params['local_bkg'] = [0.1, 0.2, 0.3] * unit
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit

    match = 'The local_bkg column must have the same flux units'
    params['local_bkg'] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params, model_shape=model_shape)


def test_make_model_image_units_no_overlap():
    """
    Test that the model image is created with the correct units when
    there is no overlap between the model and the image.
    """
    unit = u.Jy
    params = QTable()
    params['x_0'] = [50, 70.5]
    params['y_0'] = [50, 50.5]
    params['flux'] = [2, 3] * unit
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (10, 12)
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert model.flux == 1.0  # Default flux (unchanged)

    params['flux'] = [2, 3]
    image = make_model_image(shape, model, params)
    assert image.shape == shape
    assert not isinstance(image, u.Quantity)
    assert model.flux == 1.0  # Default flux (unchanged)


def test_make_model_image_units_no_sources_rendered():
    """
    Test that the output units do not depend on whether any sources
    are actually rendered.
    """
    unit = u.Jy
    model = CircularGaussianSigmaPRF(sigma=1.5)
    shape = (10, 12)

    # Zero-row table
    params = QTable()
    params['x_0'] = np.array([], dtype=float)
    params['y_0'] = np.array([], dtype=float)
    params['flux'] = np.array([], dtype=float) * unit
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert np.all(image.value == 0)

    # All rows nonfinite
    params = QTable()
    params['x_0'] = [np.nan]
    params['y_0'] = [5.0]
    params['flux'] = [1.0] * unit
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert isinstance(image, u.Quantity)
    assert image.unit == unit
    assert np.all(image.value == 0)

    # Zero-row table without units
    params = QTable()
    params['x_0'] = np.array([], dtype=float)
    params['y_0'] = np.array([], dtype=float)
    image = make_model_image(shape, model, params, model_shape=(5, 5))
    assert not isinstance(image, u.Quantity)
    assert np.all(image == 0)


def test_make_model_image_no_overlap():
    """
    Test the model image when there is no overlap between the model and
    the image.
    """
    params = QTable()
    params['x_0'] = [50]
    params['y_0'] = [50]
    params['gamma'] = [1.7]
    params['alpha'] = [2.9]
    model = Moffat2D(amplitude=1)
    shape = (10, 10)
    model_shape = (3, 3)
    data = make_model_image(shape, model, params, model_shape=model_shape)
    assert data.shape == shape
    assert np.sum(data) == 0


@pytest.mark.parametrize('per_source', [False, True])
def test_make_model_image_edge_no_overlap(per_source):
    """
    Test that a source whose model box ends exactly at the lower edge of
    the image is skipped.

    Regression test for the array-valued ``model_shape`` being compared
    to a tuple in ``overlap_slices``.
    """
    params = QTable()
    params['x_0'] = [50.0, -16.0, 50.0]
    params['y_0'] = [50.0, 50.0, -16.0]
    params['flux'] = [100.0, 100.0, 100.0]
    model = CircularGaussianPSF(fwhm=2.0)
    shape = (100, 100)
    model_shape = (31, 31)

    ref_image = make_model_image(shape, model, params[:1],
                                 model_shape=model_shape)
    if per_source:
        params['model_shape'] = [31, 31, 31]
        image = make_model_image(shape, model, params)
    else:
        image = make_model_image(shape, model, params,
                                 model_shape=model_shape)
    assert_allclose(image, ref_image)


def test_make_model_image_inputs():
    """
    Test that the appropriate exceptions are raised for invalid inputs.
    """
    match = 'shape must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100, 3), Moffat2D(), QTable())

    match = 'shape must be > 0'
    with pytest.raises(ValueError, match=match):
        make_model_image((-100, 100), Moffat2D(), QTable())

    match = 'model must be a Model instance'
    with pytest.raises(TypeError, match=match):
        make_model_image((100, 100), None, QTable())

    match = 'model must be a 2D model'
    model = Moffat2D()
    model.n_inputs = 1
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable())

    match = 'params_table must be an astropy Table'
    model = Moffat2D()
    with pytest.raises(TypeError, match=match):
        make_model_image((100, 100), model, None)

    match = 'not in model parameter names'
    model = Moffat2D()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable(), x_name='invalid')

    match = 'not in params_table column names'
    model = Moffat2D()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable(), y_name='invalid')

    model = Moffat2D()
    params = QTable()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, params)

    model = Moffat2D()
    params = QTable()
    params['x_0'] = [50, 70, 90]
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, params)

    match = 'model_shape must be specified if the model does not have'
    params = QTable()
    params['x_0'] = [50]
    params['y_0'] = [50]
    params['gamma'] = [1.7]
    params['alpha'] = [2.9]
    model = Moffat2D(amplitude=1)
    shape = (100, 100)
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params)


def test_make_model_image_table_units():
    """
    Test that a column with units in a plain Table raises a clear
    error.
    """
    params = Table()
    params['x_0'] = [10.0]
    params['y_0'] = [10.0]
    params['amplitude'] = [100.0] * u.Jy
    match = 'params_table must be a QTable'
    with pytest.raises(ValueError, match=match):
        make_model_image((21, 21), Moffat2D(), params, model_shape=(11, 11))

    image = make_model_image((21, 21), Moffat2D(), QTable(params),
                             model_shape=(11, 11))
    assert image.unit == u.Jy


def test_make_model_image_bbox():
    """
    Test the model image when using a PSF model that has a bounding box
    and the bbox_factor keyword to control the size of the bounding box.
    """
    model1 = CircularGaussianPSF(x_0=50, y_0=50, fwhm=10)
    yy, xx = np.mgrid[:101, :101]
    model2 = ImagePSF(model1(xx, yy), x_0=50, y_0=50)

    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    shape = (100, 151)
    image1 = make_model_image(shape, model2, params, bbox_factor=10)
    image2 = make_model_image(shape, model2, params, bbox_factor=None)
    assert_allclose(image1, image2)

    image3 = make_model_image(shape, model1, params, bbox_factor=10)
    image4 = make_model_image(shape, model1, params, bbox_factor=None)
    assert_allclose(image3, image4)

    model1.bbox_factor = 10
    image5 = make_model_image(shape, model1, params)
    assert np.sum(image5) > np.sum(image4)


def test_make_model_image_params_map():
    """
    Test the model image when using a parameter mapping to map the model
    parameter names to different column names in the input table.
    """
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)

    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma2'] = [1.7, 2.32, 5.8]
    params['alpha4'] = [2.9, 5.7, 4.6]
    params_map = {'gamma': 'gamma2', 'alpha': 'alpha4'}
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image2 = make_model_image(shape, model, params, model_shape=model_shape,
                              params_map=params_map)
    assert_allclose(image, image2)


def test_make_model_image_nonfinite():
    """
    Test the model image when the input table contains nonfinite
    values.
    """
    params = QTable()
    params['x_0'] = [50, np.nan, 90, 100]
    params['y_0'] = [50, 50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8, np.inf]
    params['alpha'] = [2.9, 5.7, 4.6, 3.1]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() < 33
    assert image[50, 100] == 0

    # All invalid sources
    params = QTable()
    params['x_0'] = [50, np.nan, 90, 100]
    params['y_0'] = [-np.inf, 50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8, np.inf]
    params['alpha'] = [2.9, 5.7, np.nan, 3.1]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() == 0


def test_make_model_image_progress_bar():
    """
    Test the model image with progress_bar=True.
    """
    pytest.importorskip('tqdm')
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    image = make_model_image(shape, model, params, model_shape=model_shape,
                             progress_bar=True)
    assert image.shape == shape
    assert image.sum() > 1


def test_model_shape_from_bbox_no_bbox():
    """
    Test that _model_shape_from_bbox raises an error when the model does
    not have a bounding_box attribute.
    """
    model = Polynomial2D(degree=2)
    match = 'model does not have a bounding_box attribute'
    with pytest.raises(ValueError, match=match):
        _model_shape_from_bbox(model)


def test_make_model_image_n_nodes_none_unchanged():
    """
    Test that n_nodes=None is the same as the default.
    """
    params = QTable()
    params['x_0'] = [20.3, 45.6, 70.1]
    params['y_0'] = [30.2, 55.5, 20.8]
    params['gamma'] = [1.7, 2.32, 1.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=100)
    kwargs = {'model_shape': (15, 15)}
    default = make_model_image((80, 100), model, params, **kwargs)
    explicit = make_model_image((80, 100), model, params, n_nodes=None,
                                **kwargs)
    assert_array_equal(explicit, default)


def test_make_model_image_n_nodes_gaussian():
    """
    Test that integrating a GaussianPSF matches the pixel-integrated
    GaussianPRF.
    """
    params = QTable()
    params['x_0'] = [10.0, 20.5, 29.3]
    params['y_0'] = [10.0, 20.5, 14.8]
    params['flux'] = [100.0, 250.0, 175.0]
    shape = (30, 40)
    kwargs = {'model_shape': (15, 15)}
    psf = GaussianPSF(x_fwhm=2.5, y_fwhm=2.5)
    prf = GaussianPRF(x_fwhm=2.5, y_fwhm=2.5)
    result = make_model_image(shape, psf, params, n_nodes=4, **kwargs)
    expected = make_model_image(shape, prf, params, **kwargs)
    assert_allclose(result, expected, rtol=0, atol=1e-6 * expected.max())


@pytest.mark.parametrize('n_nodes', [0, -1, 2.5, True])
def test_make_model_image_n_nodes_invalid(n_nodes):
    """
    Test that an invalid n_nodes raises a ValueError.
    """
    params = QTable()
    params['x_0'] = [10.0]
    params['y_0'] = [10.0]
    match = 'n_nodes must be None or a positive integer'
    with pytest.raises(ValueError, match=match):
        make_model_image((20, 20), Moffat2D(), params, model_shape=7,
                         n_nodes=n_nodes)


def test_make_model_image_n_nodes_units():
    """
    Test that n_nodes works with a flux column that has units.
    """
    params = QTable()
    params['x_0'] = [10.0, 20.3]
    params['y_0'] = [10.0, 14.8]
    params['flux'] = [100.0, 250.0]
    shape = (30, 40)
    kwargs = {'model_shape': (15, 15), 'n_nodes': 4}
    psf = GaussianPSF(x_fwhm=2.5, y_fwhm=2.5)
    expected = make_model_image(shape, psf, params, **kwargs)
    params['flux'] <<= u.Jy
    result = make_model_image(shape, psf, params, **kwargs)
    assert result.unit == u.Jy
    assert_allclose(result.value, expected, rtol=1e-14)


def test_make_model_image_n_nodes_2d_only_model():
    """
    Test n_nodes with a model that accepts only 1D or 2D inputs.

    A GriddedPSFModel with a single ePSF gives the same values as an
    ImagePSF made from that ePSF, which accepts inputs of any shape.
    """
    yy, xx = np.mgrid[0:11, 0:11]
    data = CircularGaussianPRF(flux=1.0, x_0=5.0, y_0=5.0, fwhm=2.5)(xx, yy)
    meta = {'grid_xypos': [(0, 0)], 'oversampling': 1}
    gridded = GriddedPSFModel(NDData(data[np.newaxis], meta=meta))
    params = QTable()
    params['x_0'] = [10.3, 22.6]
    params['y_0'] = [9.7, 15.2]
    params['flux'] = [100.0, 250.0]
    shape = (25, 30)
    result = make_model_image(shape, gridded, params, n_nodes=2)
    expected = make_model_image(shape, ImagePSF(data), params, n_nodes=2)
    assert_allclose(result, expected, rtol=1e-14)


def make_image_psf():
    """
    Make an ImagePSF, oversampled by 4, from a circular Gaussian PSF
    with a FWHM of 1.6 pixels.
    """
    yy, xx = np.mgrid[-48:49, -48:49] / 4
    data = CircularGaussianPSF(fwhm=1.6)(xx, yy)
    return ImagePSF(make_epsf_from_psf(data, oversampling=4), oversampling=4)


def make_gridded_psf():
    """
    Make a GriddedPSFModel from four circular Gaussian ePSFs.
    """
    yy, xx = np.mgrid[0:25, 0:25]
    data = [CircularGaussianPRF(x_0=12, y_0=12, fwhm=fwhm)(xx, yy)
            for fwhm in (2.0, 2.2, 2.4, 2.6)]
    meta = {'grid_xypos': [(0, 0), (60, 0), (0, 60), (60, 60)],
            'oversampling': 1}
    return GriddedPSFModel(NDData(np.array(data), meta=meta))


def make_sersic_params():
    """
    Make a table with two Sersic sources.
    """
    params = QTable()
    params['x_0'] = [20.3, 41.6]
    params['y_0'] = [25.8, 34.2]
    params['flux'] = [100.0, 250.0]
    params['r_eff'] = [1.5, 2.5]
    params['n'] = [1.0, 2.5]
    params['ellip'] = [0.2, 0.4]
    params['theta'] = [0.3, 1.2]
    return params


@pytest.mark.parametrize('psf_model', [GaussianPSF(), CircularGaussianPSF(),
                                       MoffatPSF(), AiryDiskPSF()])
def test_convolved_psf_model_rejected(psf_model):
    """
    Test that a PSF model that is not integrated over the pixels is
    rejected.
    """
    match = 'not integrated over the pixels'
    with pytest.raises(TypeError, match=match):
        make_convolved_model_image((60, 60), SersicSource(),
                                   make_sersic_params(), psf_model)


@pytest.mark.parametrize('psf_model', [make_image_psf(), make_gridded_psf(),
                                       CircularGaussianPRF(fwhm=2.5),
                                       MoffatPRF(alpha=3.0)])
def test_convolved_psf_model_accepted(psf_model):
    """
    Test that ePSF and PRF models are accepted.
    """
    params = make_sersic_params()
    image = make_convolved_model_image((60, 60), SersicSource(), params,
                                       psf_model)
    assert np.all(np.isfinite(image))
    assert 0.9 * params['flux'].sum() < image.sum() < params['flux'].sum()


@pytest.mark.parametrize('psf_model', [
    CircularGaussianPRF(fwhm=1.6, bbox_factor=10),
    GaussianPRF(x_fwhm=1.6, y_fwhm=2.6, theta=30.0, bbox_factor=10),
    make_image_psf()])
def test_point_source_matches_make_model_image(psf_model):
    """
    Test that a source that fills a single grid cell renders exactly as
    the PSF model does with make_model_image.

    The cell integral of the box source is its flux, so the convolved
    image is the flux times the PSF model at the box center. The two
    functions truncate an analytic PSF model on different pixels at the
    edge of its bounding box, so the bounding box is made large enough
    that the PSF model is negligible there. The ImagePSF is zero beyond
    its data, which both functions cover.
    """
    stars = QTable()
    stars['x_0'] = [10.25, 30.5, 51.75]
    stars['y_0'] = [12.5, 40.75, 25.0]
    stars['flux'] = [100.0, 200.0, 300.0]
    boxes = stars.copy()
    boxes['amplitude'] = stars['flux'] * 16
    del boxes['flux']
    shape = (60, 64)
    source = Box2D(x_width=0.25, y_width=0.25)
    result = make_convolved_model_image(shape, source, boxes, psf_model,
                                        oversample=4)
    expected = make_model_image(shape, psf_model, stars)
    assert_allclose(result, expected, rtol=0, atol=1e-12 * expected.max())


def render_gaussian_identity(oversample, box_term):
    """
    Render a Gaussian source convolved with a Gaussian PRF, and the
    Gaussian PRF of the summed variances that the result is expected to
    match.
    """
    params = QTable()
    params['x_0'] = [20.0, 41.37]
    params['y_0'] = [22.5, 30.81]
    params['flux'] = [100.0, 250.0]
    shape = (50, 64)
    source = GaussianPSF(x_fwhm=3.5, y_fwhm=3.5)
    psf_model = CircularGaussianPRF(fwhm=3.0)
    result = make_convolved_model_image(shape, source, params, psf_model,
                                        oversample=oversample, n_nodes=4,
                                        model_shape=35)
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    variance = (3.5**2 + 3.0**2) * fwhm_to_sigma**2
    if box_term:
        variance += 1.0 / (12 * oversample**2)
    model = CircularGaussianSigmaPRF(sigma=np.sqrt(variance))
    expected = make_model_image(shape, model, params, model_shape=35)
    return result, expected


@pytest.mark.parametrize('oversample', [3, 4])
def test_gaussian_source_identity(oversample):
    """
    Test that a Gaussian source convolved with a Gaussian PRF is the
    Gaussian PRF of the summed variances, including the variance of the
    box of width 1 / oversample.
    """
    result, expected = render_gaussian_identity(oversample, box_term=True)
    assert_allclose(result, expected, rtol=0, atol=1e-4 * expected.max())


def test_gaussian_identity_needs_box_term():
    """
    Test that the Gaussian identity fails the tolerance without the
    variance of the box of width 1 / oversample.
    """
    result, expected = render_gaussian_identity(4, box_term=False)
    error = np.max(np.abs(result - expected))
    assert error > 1e-4 * expected.max()


def test_convolved_flux_conserved():
    """
    Test that the convolved image keeps the flux inside the source
    cutout.
    """
    params = QTable()
    params['x_0'] = [64.3]
    params['y_0'] = [63.6]
    params['flux'] = [100.0]
    params['r_eff'] = [2.0]
    params['n'] = [1.0]
    shape = (128, 128)
    source = SersicSource()
    psf_model = CircularGaussianPRF(fwhm=3.0)
    image = make_convolved_model_image(shape, source, params, psf_model)
    cutout_flux = make_model_image(shape, source, params, n_nodes=4).sum()
    assert_allclose(image.sum(), cutout_flux, rtol=1e-3)


def test_convolved_off_image():
    """
    Test sources that are partly or entirely outside the image.
    """
    params = QTable()
    params['x_0'] = [-100.0]
    params['y_0'] = [20.0]
    params['flux'] = [100.0]
    shape = (40, 50)
    source = GaussianPSF(x_fwhm=1.0, y_fwhm=1.0)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    kwargs = {'model_shape': 5}
    image = make_convolved_model_image(shape, source, params, psf_model,
                                       **kwargs)
    assert image.shape == shape
    assert_array_equal(image, 0)

    # A source whose cutout is outside the image, but whose light
    # reaches the image through the PSF
    params['x_0'] = [-4.0]
    image = make_convolved_model_image(shape, source, params, psf_model,
                                       **kwargs)
    assert 0 < image.sum() < 1
    assert_array_equal(image[:, 5:], 0)

    # A source centered on the image edge is trimmed to half its flux
    params['x_0'] = [-0.5]
    image = make_convolved_model_image(shape, source, params, psf_model,
                                       **kwargs)
    assert_allclose(image.sum(), 50.0, rtol=1e-3)


def test_convolved_large_source_matches_padded_image():
    """
    Test that a source much larger than the image, which straddles the
    image edge, renders as in a larger image that is cropped.

    The padded image is large enough to hold the whole source cutout, so
    its render is not trimmed.
    """
    shape = (40, 50)
    pad = 30
    params = QTable()
    params['x_0'] = [2.3, 47.6]
    params['y_0'] = [12.7, 38.1]
    params['flux'] = [100.0, 250.0]
    params['r_eff'] = [6.0, 4.0]
    params['n'] = [1.0, 2.0]
    params['ellip'] = [0.2, 0.4]
    params['theta'] = [0.3, 1.2]
    source = SersicSource()
    psf_model = CircularGaussianPRF(fwhm=2.5)
    result = make_convolved_model_image(shape, source, params, psf_model)

    padded = params.copy()
    padded['x_0'] += pad
    padded['y_0'] += pad
    padded_shape = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    expected = make_convolved_model_image(padded_shape, source, padded,
                                          psf_model)
    expected = expected[pad:pad + shape[0], pad:pad + shape[1]]
    assert result.sum() > 0.5 * params['flux'].sum()
    assert_allclose(result, expected, rtol=0, atol=1e-12 * expected.max())


def test_convolved_nonfinite_skipped():
    """
    Test that rows with a nonfinite parameter are skipped.
    """
    params = make_sersic_params()
    expected = make_convolved_model_image((60, 60), SersicSource(),
                                          params[:1], CircularGaussianPRF())
    params['flux'][1] = np.nan
    result = make_convolved_model_image((60, 60), SersicSource(), params,
                                        CircularGaussianPRF())
    assert_array_equal(result, expected)


def test_convolved_units_error():
    """
    Test that a params_table column with units raises an error.
    """
    params = make_sersic_params()
    params['flux'] <<= u.Jy
    match = 'make_convolved_model_image does not support units'
    with pytest.raises(ValueError, match=match):
        make_convolved_model_image((60, 60), SersicSource(), params,
                                   CircularGaussianPRF())


def test_convolved_local_bkg_error():
    """
    Test that a local_bkg column raises an error.
    """
    params = make_sersic_params()
    params['local_bkg'] = [1.0, 2.0]
    match = 'make_convolved_model_image does not support a local_bkg'
    with pytest.raises(ValueError, match=match):
        make_convolved_model_image((60, 60), SersicSource(), params,
                                   CircularGaussianPRF())


@pytest.mark.parametrize('oversample', [0, -2, 2.5, True])
def test_convolved_oversample_invalid(oversample):
    """
    Test that an invalid oversample raises a ValueError.
    """
    match = 'oversample must be a positive integer'
    with pytest.raises(ValueError, match=match):
        make_convolved_model_image((60, 60), SersicSource(),
                                   make_sersic_params(),
                                   CircularGaussianPRF(),
                                   oversample=oversample)


def test_convolved_psf_model_no_bbox():
    """
    Test that a PSF model without a bounding box raises an error.
    """
    psf_model = CircularGaussianPRF()
    psf_model.bounding_box = None
    match = 'psf_model must have a bounding_box attribute'
    with pytest.raises(ValueError, match=match):
        make_convolved_model_image((60, 60), SersicSource(),
                                   make_sersic_params(), psf_model)


def test_convolved_progress_bar():
    """
    Test make_convolved_model_image with progress_bar=True.
    """
    pytest.importorskip('tqdm')
    params = make_sersic_params()
    expected = make_convolved_model_image((60, 60), SersicSource(), params,
                                          CircularGaussianPRF())
    result = make_convolved_model_image((60, 60), SersicSource(), params,
                                        CircularGaussianPRF(),
                                        progress_bar=True)
    assert_array_equal(result, expected)


def test_convolved_gridded_psf_local():
    """
    Test that a GriddedPSFModel is evaluated at the source position.

    A source near a grid point renders as with an ImagePSF made from
    the ePSF at that grid point.
    """
    gridded = make_gridded_psf()
    params = make_sersic_params()[:1]
    params['x_0'] = [60.2]
    params['y_0'] = [59.7]
    shape = (70, 70)
    result = make_convolved_model_image(shape, SersicSource(), params,
                                        gridded)
    epsf = ImagePSF(gridded.data[3])
    expected = make_convolved_model_image(shape, SersicSource(), params,
                                          epsf)
    assert_allclose(result, expected, rtol=0, atol=1e-12 * expected.max())


def test_make_model_image_ellip_no_warning():
    """
    Test that make_model_image does not warn for a SersicSource whose
    ellipticity is in params_table.

    The output units are found by evaluating the model with parameter
    values of 1, which is an invalid ellipticity.
    """
    params = make_sersic_params()
    image = make_model_image((60, 60), SersicSource(), params, n_nodes=4)
    assert np.all(np.isfinite(image))
