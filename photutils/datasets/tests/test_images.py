# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the images module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.models import Moffat2D
from astropy.table import QTable

from photutils.datasets import make_model_image, params_table_to_models
from photutils.psf import CircularGaussianPSF, CircularGaussianSigmaPRF


def test_make_model_image():
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

    # test variable model shape
    params['model_shape'] = [9, 7, 11]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1

    # test local_bkg
    params['local_bkg'] = [1, 2, 3]
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert image.sum() > 1


def test_make_model_image_units():
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
    assert model.flux == 1.0  # default flux (unchanged)

    params['local_bkg'] = [0.1, 0.2, 0.3] * unit
    image = make_model_image(shape, model, params, model_shape=model_shape)
    assert image.shape == shape
    assert isinstance(image, u.Quantity)
    assert image.unit == unit

    match = 'The local_bkg column must have the same flux units'
    params['local_bkg'] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match=match):
        make_model_image(shape, model, params, model_shape=model_shape)


def test_make_model_image_discretize_method():
    params = QTable()
    params['x_0'] = [50, 70, 90]
    params['y_0'] = [50, 50, 50]
    params['gamma'] = [1.7, 2.32, 5.8]
    params['alpha'] = [2.9, 5.7, 4.6]
    model = Moffat2D(amplitude=1)
    shape = (300, 500)
    model_shape = (11, 11)
    for method in ('interp', 'oversample'):
        image = make_model_image(shape, model, params, model_shape=model_shape,
                                 discretize_method=method)
        assert image.shape == shape
        assert image.sum() > 1


def test_make_model_image_no_overlap():
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


def test_make_model_image_inputs():
    match = 'shape must be a 2-tuple'
    with pytest.raises(ValueError, match=match):
        make_model_image(100, Moffat2D(), QTable())

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

    model = Moffat2D()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, QTable(), y_name='invalid')

    match = '"x_0" not in psf_params column names'
    model = Moffat2D()
    params = QTable()
    with pytest.raises(ValueError, match=match):
        make_model_image((100, 100), model, params)

    match = '"y_0" not in psf_params column names'
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


def test_params_table_to_models():
    tbl = QTable()
    tbl['x_0'] = [1, 2, 3]
    tbl['y_0'] = [4, 5, 6]
    tbl['flux'] = [100, 200, 300]
    tbl['name'] = ['a', 'b', 'c']
    model = CircularGaussianPSF()
    models = params_table_to_models(tbl, model)

    assert len(models) == 3
    for i in range(len(models)):
        assert models[i].x_0 == tbl['x_0'][i]
        assert models[i].y_0 == tbl['y_0'][i]
        assert models[i].flux == tbl['flux'][i]
        assert models[i].fwhm == model.fwhm
        assert models[i].name == tbl['name'][i]

    tbl = QTable()
    tbl['invalid1'] = [1, 2, 3]
    tbl['invalid2'] = [4, 5, 6]
    match = 'No matching model parameter names found in params_table'
    with pytest.raises(ValueError, match=match):
        params_table_to_models(tbl, model)
