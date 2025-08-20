# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for the photometry module.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.modeling.fitting import (LMLSQFitter, SimplexLSQFitter,
                                      TRFLSQFitter)
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.nddata import NDData, StdDevUncertainty
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyUserWarning
from numpy.testing import assert_allclose, assert_equal

from photutils.background import LocalBackground, MMMBackground
from photutils.datasets import make_model_image, make_noise_image
from photutils.detection import DAOStarFinder
from photutils.psf import (CircularGaussianPRF, IterativePSFPhotometry,
                           PSFPhotometry, SourceGrouper, make_psf_model,
                           make_psf_model_image)
from photutils.utils.exceptions import NoDetectionsWarning


@pytest.fixture(name='test_data')
def fixture_test_data():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    return data, error, true_params


def test_invalid_inputs():
    model = CircularGaussianPRF(fwhm=1.0)

    match = 'psf_model must be an Astropy Model subclass'
    with pytest.raises(TypeError, match=match):
        _ = PSFPhotometry(1, 3)

    match = 'psf_model must be two-dimensional'
    psf_model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'psf_model must be two-dimensional'
    psf_model = Gaussian1D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'Invalid PSF model - could not find PSF parameter names'
    psf_model = Gaussian2D()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(psf_model, 3)

    match = 'fit_shape must have an odd value for both axes'
    for shape in ((0, 0), (4, 3)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    match = 'fit_shape must be >= 1'
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, (-1, 1))

    match = 'fit_shape must be a finite value'
    for shape in ((np.nan, 3), (5, np.inf)):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, shape)

    kwargs = {'finder': 1, 'fitter': 1}
    for key, val in kwargs.items():
        match = f"'{key}' must be a callable object"
        with pytest.raises(TypeError, match=match):
            _ = PSFPhotometry(model, 1, **{key: val})

    match = 'localbkg_estimator must be a LocalBackground instance'
    localbkg = MMMBackground()
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, 1, localbkg_estimator=localbkg)

    match = 'aperture_radius must be a strictly-positive scalar'
    for radius in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match=match):
            _ = PSFPhotometry(model, 1, aperture_radius=radius)

    match = 'grouper must be a SourceGrouper instance'
    with pytest.raises(ValueError, match=match):
        _ = PSFPhotometry(model, (5, 5), grouper=1)

    match = 'data must be a 2D array'
    psfphot = PSFPhotometry(model, (3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(np.arange(3))

    match = 'data and error must have the same shape'
    data = np.ones((11, 11))
    error = np.ones((3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error)

    match = 'data and mask must have the same shape'
    data = np.ones((11, 11))
    mask = np.ones((3, 3))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, mask=mask)

    match = 'init_params must be an astropy Table'
    data = np.ones((11, 11))
    with pytest.raises(TypeError, match=match):
        _ = psfphot(data, init_params=1)

    match = ('init_param must contain valid column names for the x and y '
             'source positions')
    tbl = Table()
    tbl['a'] = np.arange(3)
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=tbl)

    # test no finder or init_params
    match = 'finder must be defined if init_params is not input'
    psfphot = PSFPhotometry(model, (3, 3), aperture_radius=5)
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data)

    # data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    psfphot2 = PSFPhotometry(model, (3, 3), aperture_radius=3)
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot2(data, init_params=init_params)

    # mask is input, but data has unmasked non-finite value
    match = 'Input data contains unmasked non-finite values'
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[7, 7] = True
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot2(data, mask=mask, init_params=init_params)

    match = 'init_params local_bkg column contains non-finite values'
    tbl = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    init_params['local_bkg'] = [0.1, np.inf]
    data = np.ones((11, 11))
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, init_params=init_params)


def test_psf_photometry(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)

    assert isinstance(psfphot.finder_results, QTable)
    assert isinstance(phot, QTable)
    assert isinstance(psfphot.results, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape
    assert phot.colnames[:4] == ['id', 'group_id', 'group_size', 'local_bkg']
    # test that error columns are ordered correctly
    assert phot['x_err'].max() > 0.0062
    assert phot['y_err'].max() > 0.0065
    assert phot['flux_err'].max() > 2.5

    keys = ('fit_infos', 'fit_error_indices')
    for key in keys:
        assert key in psfphot.fit_info

    # test that repeated calls reset the results
    phot = psfphot(data, error=error)
    assert len(psfphot.fit_info['fit_infos']) == len(phot)

    # test units
    unit = u.Jy
    finderu = DAOStarFinder(6.0 * unit, 2.0)
    psfphotu = PSFPhotometry(psf_model, fit_shape, finder=finderu,
                             aperture_radius=4)
    photu = psfphotu(data * unit, error=error * unit)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert photu[col].unit == unit
    resid_datau = psfphotu.make_residual_image(data << unit,
                                               psf_shape=fit_shape)
    assert resid_datau.unit == unit
    colnames = ('qfit', 'cfit')
    for col in colnames:
        assert not isinstance(col, u.Quantity)


@pytest.mark.parametrize('fit_fwhm', [False, True])
def test_psf_photometry_forced(test_data, fit_fwhm):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    if fit_fwhm:
        psf_model.fwhm.fixed = False
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)

    assert isinstance(psfphot.finder_results, QTable)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape
    assert phot.colnames[:4] == ['id', 'group_id', 'group_size', 'local_bkg']
    assert_equal(phot['x_init'], phot['x_fit'])

    if fit_fwhm:
        col = 'fwhm'
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes]
        for colname in colnames:
            assert colname in phot.colnames


def test_psf_photometry_nddata(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)

    # test NDData input
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot1 = psfphot(data, error=error)
    phot2 = psfphot(nddata)
    resid_data1 = psfphot.make_residual_image(data, psf_shape=fit_shape)
    resid_data2 = psfphot.make_residual_image(nddata, psf_shape=fit_shape)

    assert np.all(phot1 == phot2)
    assert isinstance(resid_data2, NDData)
    assert resid_data2.data.shape == data.shape
    assert_allclose(resid_data1, resid_data2.data)

    # test NDData input with units
    unit = u.Jy
    finderu = DAOStarFinder(6.0 * unit, 2.0)
    psfphotu = PSFPhotometry(psf_model, fit_shape, finder=finderu,
                             aperture_radius=4)
    photu = psfphotu(data * unit, error=error * unit)
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty, unit=unit)
    photu = psfphotu(nddata)
    assert photu['flux_init'].unit == unit
    assert photu['flux_fit'].unit == unit
    assert photu['flux_err'].unit == unit
    resid_data3 = psfphotu.make_residual_image(nddata, psf_shape=fit_shape)
    assert resid_data3.unit == unit


def test_psf_photometry_finite_weights(test_data):
    data, _, _ = test_data
    error = np.zeros_like(data)

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    match1 = 'divide by zero'
    match2 = 'Fit weights contain a non-finite value'
    with (pytest.warns(RuntimeWarning, match=match1),
          pytest.raises(ValueError, match=match2)):
        _ = psfphot(data, error=error)


def test_model_residual_image(test_data):
    data, error, _ = test_data

    data = data + 10
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(16.0, 2.0)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4,
                            localbkg_estimator=localbkg_estimator)
    psfphot(data, error=error)

    psf_shape = (25, 25)
    model1 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=True)
    resid1 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=False)
    resid2 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=True)

    x, y = 0, 100
    assert model1[y, x] < 0.1
    assert model2[y, x] > 9
    assert resid1[y, x] > 9
    assert resid2[y, x] < 0

    x, y = 0, 80
    assert model1[y, x] < 0.1
    assert model2[y, x] > 18
    assert resid1[y, x] > 9
    assert resid2[y, x] < -9


@pytest.mark.parametrize('fit_stddev', [False, True])
def test_psf_photometry_compound_psfmodel(test_data, fit_stddev):
    """
    Test compound models output from ``make_psf_model``.
    """
    data, error, sources = test_data
    x_stddev = y_stddev = 1.2
    psf_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev,
                          y_stddev=y_stddev)
    psf_model = make_psf_model(psf_func, x_name='x_mean', y_name='y_mean')
    if fit_stddev:
        psf_model.x_stddev_2.fixed = False
        psf_model.y_stddev_2.fixed = False

    fit_shape = (5, 5)
    finder = DAOStarFinder(5.0, 3.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)

    if fit_stddev:
        cols = ('x_stddev_2', 'y_stddev_2')
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes for col in cols]
        for colname in colnames:
            assert colname in phot.colnames

    # test model and residual images
    psf_shape = (9, 9)
    model1 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=False)
    resid1 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=True)
    resid2 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=True)
    assert model1.shape == data.shape
    assert model2.shape == data.shape
    assert resid1.shape == data.shape
    assert resid2.shape == data.shape
    assert_equal(data - model1, resid1)
    assert_equal(data - model2, resid2)

    # test with init_params
    init_params = psfphot.fit_params
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)

    if fit_stddev:
        cols = ('x_stddev_2', 'y_stddev_2')
        suffixes = ('_init', '_fit', '_err')
        colnames = [col + suffix for suffix in suffixes for col in cols]
        for colname in colnames:
            assert colname in phot.colnames

    # test results when fit does not converge (fitter_maxiters=3)
    match = r'One or more fit\(s\) may not have converged.'
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4, fitter_maxiters=3)
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error)
    assert len(phot) == len(sources)


@pytest.mark.parametrize('mode', ['new', 'all'])
def test_iterative_psf_photometry_compound(mode):
    x_stddev = y_stddev = 1.7
    psf_func = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=x_stddev,
                          y_stddev=y_stddev)
    psf_model = make_psf_model(psf_func, x_name='x_mean', y_name='y_mean')
    psf_model.x_stddev_2.fixed = False
    psf_model.y_stddev_2.fixed = False

    other_params = {psf_model.flux_name: (500, 700)}

    model_shape = (9, 9)
    n_sources = 10
    shape = (101, 101)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             **other_params,
                                             min_separation=10, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
    data += noise
    error = np.abs(noise)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 3.0)
    grouper = SourceGrouper(min_separation=2)

    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, aperture_radius=4,
                                     sub_shape=fit_shape,
                                     mode=mode, maxiters=2)
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(true_params)

    cols = ('x_stddev_2', 'y_stddev_2')
    suffixes = ('_init', '_fit', '_err')
    colnames = [col + suffix for suffix in suffixes for col in cols]
    for colname in colnames:
        assert colname in phot.colnames

    # test model and residual images
    psf_shape = (9, 9)
    model1 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=False)
    resid1 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=False)
    model2 = psfphot.make_model_image(data.shape, psf_shape=psf_shape,
                                      include_localbkg=True)
    resid2 = psfphot.make_residual_image(data, psf_shape=psf_shape,
                                         include_localbkg=True)
    assert model1.shape == data.shape
    assert model2.shape == data.shape
    assert resid1.shape == data.shape
    assert resid2.shape == data.shape
    assert_equal(data - model1, resid1)
    assert_equal(data - model2, resid2)

    # test with init_params
    init_params = psfphot.fit_results[-1].fit_params
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == len(true_params)

    cols = ('x_stddev_2', 'y_stddev_2')
    suffixes = ('_init', '_fit', '_err')
    colnames = [col + suffix for suffix in suffixes for col in cols]
    for colname in colnames:
        assert colname in phot.colnames


def test_psf_photometry_mask(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    # test np.ma.nomask
    phot = psfphot(data, error=error, mask=None)
    photm = psfphot(data, error=error, mask=np.ma.nomask)
    assert np.all(phot == photm)

    # masked near source at ~(63, 49)
    data_orig = data.copy()
    data = data.copy()
    data[55, 60:70] = np.nan

    match = 'Input data contains unmasked non-finite values'
    with pytest.warns(AstropyUserWarning, match=match):
        phot1 = psfphot(data, error=error, mask=None)
    assert len(phot1) == len(sources)

    mask = ~np.isfinite(data)
    phot2 = psfphot(data, error=error, mask=mask)
    assert np.all(phot1 == phot2)

    # unmasked NaN with mask not None
    match = 'Input data contains unmasked non-finite values'
    mask = ~np.isfinite(data)
    mask[55, 65] = False
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error, mask=mask)
    assert len(phot) == len(sources)

    # mask all True; finder returns no sources
    match = 'No sources were found'
    mask = np.ones(data.shape, dtype=bool)
    with pytest.warns(NoDetectionsWarning, match=match):
        psfphot(data, mask=mask)

    # completely masked source should return NaNs and not raise
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.ones(data.shape, dtype=bool)
    phot_masked = psfphot(data_orig, mask=mask, init_params=init_params)
    assert len(phot_masked) == 1
    for col in ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err', 'qfit', 'cfit'):
        assert np.isnan(phot_masked[col][0])
    assert phot_masked['npixfit'][0] == 0
    assert phot_masked['group_size'][0] == 0

    # masked central pixel
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.zeros(data.shape, dtype=bool)
    mask[49, 63] = True
    phot = psfphot(data_orig, mask=mask, init_params=init_params)
    assert len(phot) == 1
    assert np.isnan(phot['cfit'][0])

    # this should not raise a warning because the non-finite pixel was
    # explicitly masked
    psfphot = PSFPhotometry(psf_model, (3, 3), aperture_radius=3)
    data = np.ones((11, 11))
    data[5, 5] = np.nan
    mask = np.zeros(data.shape, dtype=bool)
    mask[5, 5] = True
    init_params = Table()
    init_params['x_init'] = [1, 2]
    init_params['y_init'] = [1, 2]
    psfphot(data, mask=mask, init_params=init_params)


def test_psf_photometry_init_params(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1

    match = 'aperture_radius must be defined if a flux column is not'
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=None)
    with pytest.raises(ValueError, match=match):
        _ = psfphot(data, error=error, init_params=init_params)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)
    init_params['flux'] = 650
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    init_params['group_id'] = 1
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == 1

    colnames = ('flux', 'local_bkg')
    for col in colnames:
        init_params2 = init_params.copy()
        init_params2.remove_column('flux')
        init_params2[col] = [650 * u.Jy]
        match = 'column has units, but the input data does not have units'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data, error=error, init_params=init_params2)

        init_params2[col] = [650 * u.Jy]
        match = 'column has units that are incompatible with the input data'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.m, init_params=init_params2)

        init_params2[col] = [650]
        match = 'The input data has units, but the init_params'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.Jy, init_params=init_params2)

    # no-overlap source should return NaNs and not raise; also test too-few-pixels
    init_params = QTable()
    init_params['x'] = [-63]
    init_params['y'] = [-49]
    init_params['flux'] = [100]
    phot_no_overlap = psfphot(data, init_params=init_params)
    assert len(phot_no_overlap) == 1
    for col in ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err', 'qfit', 'cfit'):
        assert np.isnan(phot_no_overlap[col][0])
    assert phot_no_overlap['npixfit'][0] == 0
    assert phot_no_overlap['group_size'][0] == 0

    # too-few pixels (unmasking only 2 pixels < 3 free params) should give NaNs
    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    mask = np.ones(data.shape, dtype=bool)
    mask[49, 63] = False
    mask[49, 64] = False
    phot_few = psfphot(data, error=error, mask=mask, init_params=init_params)
    assert len(phot_few) == 1
    for col in ('x_fit', 'y_fit', 'flux_fit', 'x_err', 'y_err', 'flux_err', 'qfit', 'cfit'):
        assert np.isnan(phot_few[col][0])
    assert phot_few['npixfit'][0] == 2
    assert phot_few['group_size'][0] == 0

    # check that the first matching column name is used
    init_params = QTable()
    x = 63
    y = 49
    flux = 680
    init_params['x'] = [x]
    init_params['y'] = [y]
    init_params['flux'] = [flux]
    init_params['x_cen'] = [x + 0.1]
    init_params['y_cen'] = [y + 0.1]
    init_params['flux0'] = [flux + 0.1]
    phot = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1
    assert phot['x_init'][0] == x
    assert phot['y_init'][0] == y
    assert phot['flux_init'][0] == flux


def test_psf_photometry_init_params_units(test_data):
    data, error, _ = test_data
    data2 = data.copy()
    error2 = error.copy()

    unit = u.Jy
    data2 <<= unit
    error2 <<= unit

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4)

    init_params = QTable()
    init_params['x'] = [63]
    init_params['y'] = [49]
    init_params['flux'] = [650 * unit]
    init_params['local_bkg'] = [0.001 * unit]
    phot = psfphot(data2, error=error2, init_params=init_params)
    assert isinstance(phot, QTable)
    assert len(phot) == 1

    for val in (True, False):
        im = psfphot.make_model_image(data2.shape, psf_shape=fit_shape,
                                      include_localbkg=val)
        assert isinstance(im, u.Quantity)
        assert im.unit == unit
        resid = psfphot.make_residual_image(data2, psf_shape=fit_shape,
                                            include_localbkg=val)
        assert isinstance(resid, u.Quantity)
        assert resid.unit == unit

    # test invalid units
    colnames = ('flux', 'local_bkg')
    for col in colnames:
        init_params2 = init_params.copy()
        init_params2.remove_column('flux')
        init_params2.remove_column('local_bkg')
        init_params2[col] = [650 * u.Jy]
        match = 'column has units, but the input data does not have units'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data, error=error, init_params=init_params2)

        init_params2[col] = [650 * u.Jy]
        match = 'column has units that are incompatible with the input data'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.m, init_params=init_params2)

        init_params2[col] = [650]
        match = 'The input data has units, but the init_params'
        with pytest.raises(ValueError, match=match):
            _ = psfphot(data << u.Jy, init_params=init_params2)


def test_psf_photometry_init_params_columns(test_data):
    data, error, _ = test_data
    data = data.copy()

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder)

    xy_suffixes = ('_init', 'init', 'centroid', '_centroid', '_peak', '',
                   'cen', '_cen', 'pos', '_pos', '_0', '0')
    flux_cols = ['flux_init', 'flux_0', 'flux0', 'flux', 'source_sum',
                 'segment_flux', 'kron_flux']
    pad = len(xy_suffixes) - len(flux_cols)
    flux_cols += flux_cols[0:pad]  # pad to have same length as xy_suffixes

    xcols = ['x' + i for i in xy_suffixes]
    ycols = ['y' + i for i in xy_suffixes]

    phots = []
    for xcol, ycol, fluxcol in zip(xcols, ycols, flux_cols, strict=True):
        init_params = QTable()
        init_params[xcol] = [42]
        init_params[ycol] = [36]
        init_params[fluxcol] = [680]
        phot = psfphot(data, error=error, init_params=init_params)
        assert isinstance(phot, QTable)
        assert len(phot) == 1
        phots.append(phot)

    for phot in phots[1:]:
        assert_allclose(phot['x_fit'], phots[0]['x_fit'])
        assert_allclose(phot['y_fit'], phots[0]['y_fit'])
        assert_allclose(phot['flux_fit'], phots[0]['flux_fit'])


def test_grouper(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4)
    phot = psfphot(data, error=error)
    assert isinstance(phot, QTable)
    assert len(phot) == len(sources)
    assert_equal(phot['group_id'], (1, 2, 3, 4, 5, 5, 5, 6, 6, 7))
    assert_equal(phot['group_size'], (1, 1, 1, 1, 3, 3, 3, 2, 2, 1))


def test_grouper_init_params(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4)
    phot0 = psfphot(data, error=error)

    init_params = QTable()
    init_params['id'] = phot0['id']
    init_params['group_id'] = 1
    init_params['x'] = phot0['x_init']
    init_params['y'] = phot0['y_init']
    init_params['flux'] = phot0['flux_init']
    phot1 = psfphot(data, error=error, init_params=init_params)
    nsources = len(phot1)
    assert isinstance(phot1, QTable)
    assert_equal(phot1['group_id'], np.ones(nsources, dtype=int))
    assert_equal(phot1['group_size'], np.ones(nsources, dtype=int) * nsources)

    # test with grouper=None
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=None, aperture_radius=4)
    phot2 = psfphot(data, error=error, init_params=init_params)
    assert isinstance(phot2, QTable)
    assert_equal(phot1['group_id'], np.ones(nsources, dtype=int))
    assert_equal(phot1['group_size'], np.ones(nsources, dtype=int) * nsources)


def test_large_group_warning():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2)
    grouper = SourceGrouper(min_separation=50)
    model_shape = (5, 5)
    fit_shape = (5, 5)
    n_sources = 50
    shape = (301, 301)
    data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                             model_shape=model_shape,
                                             flux=(500, 700),
                                             min_separation=10, seed=0)
    match = 'Some groups have more than'
    psfphot = PSFPhotometry(psf_model, fit_shape, grouper=grouper)
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot(data, init_params=true_params)


def test_local_bkg(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    grouper = SourceGrouper(min_separation=20)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            grouper=grouper, aperture_radius=4,
                            localbkg_estimator=localbkg_estimator)
    phot = psfphot(data, error=error)
    assert np.count_nonzero(phot['local_bkg']) == len(sources)


def test_fixed_params(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = True
    psf_model.y_0.fixed = True
    psf_model.flux.fixed = True
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    match = r'`bounds` must contain 2 elements'
    with pytest.raises(ValueError, match=match):
        psfphot(data, error=error)


def test_fixed_params_units(test_data):
    data, error, _ = test_data
    unit = u.nJy

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.x_0.fixed = False
    psf_model.y_0.fixed = False
    psf_model.flux.fixed = True
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0 * unit, 2.0)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=4)

    phot = psfphot(data << unit, error=error << unit)
    assert phot['local_bkg'].unit == unit
    assert phot['flux_init'].unit == unit
    assert phot['flux_fit'].unit == unit
    assert phot['flux_err'].unit == unit


def test_fit_warning(test_data):
    data, _, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.flux.fixed = False
    psf_model.fwhm.bounds = (None, None)
    fit_shape = (5, 5)
    fitter = LMLSQFitter()  # uses "status" instead of "ierr"
    finder = DAOStarFinder(6.0, 2.0)
    # set fitter_maxiters = 1 so that the fit error status is set
    psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                            fitter_maxiters=1, finder=finder,
                            aperture_radius=4)

    match = r'One or more fit\(s\) may not have converged.'
    with pytest.warns(AstropyUserWarning, match=match):
        _ = psfphot(data)
    assert len(psfphot.fit_info['fit_error_indices']) > 0


def test_fitter_no_maxiters_no_metrics(test_data):
    """
    Test with a fitter that does not have a maxiters parameter and does
    not produce a residual array.
    """
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    psf_model.flux.fixed = False
    fit_shape = (5, 5)
    fitter = SimplexLSQFitter()  # does not produce residual array
    finder = DAOStarFinder(6.0, 2.0)
    match = '"maxiters" will be ignored because it is not accepted by'
    with pytest.warns(AstropyUserWarning, match=match):
        psfphot = PSFPhotometry(psf_model, fit_shape, fitter=fitter,
                                finder=finder, aperture_radius=4)
    phot = psfphot(data, error=error)
    assert np.all(np.isnan(phot['qfit']))
    assert np.all(np.isnan(phot['cfit']))


def test_xy_bounds(test_data):
    data, error, _ = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    init_params = QTable()
    init_params['x'] = [65]
    init_params['y'] = [51]
    xy_bounds = (1, 1)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(phot) == len(init_params)
    assert_allclose(phot['x_fit'], 64.0)  # at lower bound
    assert_allclose(phot['y_fit'], 50.0)  # at lower bound

    psfphot2 = PSFPhotometry(psf_model, fit_shape, finder=None,
                             aperture_radius=4, xy_bounds=1)
    phot2 = psfphot2(data, error=error, init_params=init_params)
    cols = ('x_fit', 'y_fit', 'flux_fit')
    for col in cols:
        assert np.all(phot[col] == phot2[col])

    xy_bounds = (None, 1)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    phot = psfphot(data, error=error, init_params=init_params)
    assert phot['x_fit'] < 64.0
    assert_allclose(phot['y_fit'], 50.0)  # at lower bound

    xy_bounds = (1, None)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    phot = psfphot(data, error=error, init_params=init_params)
    assert_allclose(phot['x_fit'], 64.0)  # at lower bound
    assert phot['y_fit'] < 50.0

    xy_bounds = (None, None)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=None,
                            aperture_radius=4, xy_bounds=xy_bounds)
    init_params['x'] = [63]
    init_params['y'] = [49]
    phot = psfphot(data, error=error, init_params=init_params)
    assert phot['x_fit'] < 63.3
    assert phot['y_fit'] < 48.7
    assert phot['flags'] == 0

    # test invalid inputs
    match = 'xy_bounds must have 1 or 2 elements'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=(1, 2, 3))
    match = 'xy_bounds must be a 1D array'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=np.ones((1, 1)))
    match = 'xy_bounds must be strictly positive'
    with pytest.raises(ValueError, match=match):
        PSFPhotometry(psf_model, fit_shape, xy_bounds=(-1, 2))


def test_iterative_psf_photometry_mode_new(test_data):
    data, error, sources = test_data

    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    bkgstat = MMMBackground()
    localbkg_estimator = LocalBackground(5, 10, bkgstat)
    finder = DAOStarFinder(10.0, 2.0)

    init_params = QTable()
    init_params['x'] = [54, 29, 80]
    init_params['y'] = [8, 26, 29]
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    phot = psfphot(data, error=error, init_params=init_params)
    cols = ['id', 'group_id', 'group_size', 'iter_detected', 'local_bkg']
    assert phot.colnames[:5] == cols
    assert len(psfphot.fit_results) == 2

    assert 'iter_detected' in phot.colnames
    assert len(phot) == len(sources)

    resid_data = psfphot.make_residual_image(data, psf_shape=fit_shape)
    assert isinstance(resid_data, np.ndarray)
    assert resid_data.shape == data.shape

    nddata = NDData(data)
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.data.shape == data.shape

    # test that repeated calls reset the results
    phot = psfphot(data, error=error, init_params=init_params)
    assert len(psfphot.fit_results) == 2

    # test NDData without units
    uncertainty = StdDevUncertainty(error)
    nddata = NDData(data, uncertainty=uncertainty)
    phot0 = psfphot(nddata, init_params=init_params)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert_allclose(phot0[col], phot[col])
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert_equal(resid_nddata.data, resid_data)

    # test with units and mode='new'
    unit = u.Jy
    finder_units = DAOStarFinder(10.0 * unit, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape,
                                     finder=finder_units, mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)

    phot2 = psfphot(data << unit, error=error << unit, init_params=init_params)
    assert phot2['flux_fit'].unit == unit
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot2[col].unit == unit
        assert_allclose(phot2[col].value, phot[col])

    # test NDData with units
    uncertainty = StdDevUncertainty(error << unit)
    nddata = NDData(data << unit, uncertainty=uncertainty)
    phot3 = psfphot(nddata, init_params=init_params)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot3[col].unit == unit
        assert_allclose(phot3[col].value, phot2[col].value)
    resid_nddata = psfphot.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.unit == unit

    # test return None if no stars are found on first iteration
    finder = DAOStarFinder(1000.0, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     mode='new',
                                     localbkg_estimator=localbkg_estimator,
                                     aperture_radius=4)
    match = 'No sources were found'
    with pytest.warns(NoDetectionsWarning, match=match):
        phot = psfphot(data, error=error)
    assert phot is None


def test_iterative_psf_photometry_mode_all():
    sources = QTable()
    sources['x_0'] = [50, 45, 55, 27, 22, 77, 82]
    sources['y_0'] = [50, 52, 48, 27, 30, 77, 79]
    sources['flux'] = [1000, 100, 50, 1000, 100, 1000, 100]

    shape = (101, 101)
    psf_model = CircularGaussianPRF(flux=500, fwhm=9.4)
    psf_shape = (41, 41)
    data = make_model_image(shape, psf_model, sources, model_shape=psf_shape)

    fit_shape = (5, 5)
    finder = DAOStarFinder(0.2, 6.0)
    sub_shape = psf_shape
    grouper = SourceGrouper(10)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     grouper=grouper, aperture_radius=4,
                                     sub_shape=sub_shape, mode='all',
                                     maxiters=3)
    phot = psfphot(data)
    cols = ['id', 'group_id', 'group_size', 'iter_detected', 'local_bkg']
    assert phot.colnames[:5] == cols

    assert len(phot) == 7
    assert_equal(phot['group_id'], [1, 2, 3, 1, 2, 2, 3])
    assert_equal(phot['iter_detected'], [1, 1, 1, 2, 2, 2, 2])
    assert_allclose(phot['flux_fit'], [1000, 1000, 1000, 100, 50, 100, 100])

    resid = psfphot.make_residual_image(data, psf_shape=sub_shape)
    assert_allclose(resid, 0, atol=1e-6)

    match = 'mode must be "new" or "all"'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=grouper, aperture_radius=4,
                                         sub_shape=sub_shape, mode='invalid')

    match = 'grouper must be input for the "all" mode'
    with pytest.raises(ValueError, match=match):
        psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                         grouper=None, aperture_radius=4,
                                         sub_shape=sub_shape, mode='all')

    # test with units and mode='all'
    unit = u.Jy
    finderu = DAOStarFinder(0.2 * unit, 6.0)
    psfphotu = IterativePSFPhotometry(psf_model, fit_shape, finder=finderu,
                                      grouper=grouper, aperture_radius=4,
                                      sub_shape=sub_shape, mode='all',
                                      maxiters=3)
    phot2 = psfphotu(data << unit)
    assert len(phot2) == 7
    assert_equal(phot2['group_id'], [1, 2, 3, 1, 2, 2, 3])
    assert_equal(phot2['iter_detected'], [1, 1, 1, 2, 2, 2, 2])
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot2[col].unit == unit
        assert_allclose(phot2[col].value, phot[col])

    # test NDData with units
    nddata = NDData(data * unit)
    phot3 = psfphotu(nddata)
    colnames = ('flux_init', 'flux_fit', 'flux_err', 'local_bkg')
    for col in colnames:
        assert phot3[col].unit == unit
        assert_allclose(phot3[col].value, phot[col])
    resid_nddata = psfphotu.make_residual_image(nddata, psf_shape=fit_shape)
    assert isinstance(resid_nddata, NDData)
    assert resid_nddata.unit == unit


def test_iterative_psf_photometry_overlap():
    """
    Regression test for #1769.

    A ValueError should not be raised for no overlap.
    """
    fwhm = 3.5
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    data, _ = make_psf_model_image((150, 150), psf_model, n_sources=300,
                                   model_shape=(11, 11), flux=(50, 100),
                                   min_separation=1, seed=0)
    noise = make_noise_image(data.shape, mean=0, stddev=0.01, seed=0)
    data += noise
    error = np.abs(noise)
    slc = (slice(0, 50), slice(0, 50))
    data = data[slc]
    error = error[slc]

    daofinder = DAOStarFinder(threshold=0.5, fwhm=fwhm)
    grouper = SourceGrouper(min_separation=1.3 * fwhm)
    fitter = TRFLSQFitter()
    fit_shape = (5, 5)
    sub_shape = fit_shape
    psfphot = IterativePSFPhotometry(psf_model, fit_shape=fit_shape,
                                     finder=daofinder, mode='all',
                                     grouper=grouper, maxiters=2,
                                     sub_shape=sub_shape,
                                     aperture_radius=3, fitter=fitter)
    match = r'One or more .* may not have converged'
    with pytest.warns(AstropyUserWarning, match=match):
        phot = psfphot(data, error=error)
    assert len(phot) == 38


def test_iterative_psf_photometry_subshape():
    """
    A ValueError should not be raised if sub_shape=None and the model
    does not have a bounding box.
    """
    fwhm = 3.5
    psf_model = CircularGaussianPRF(flux=1, fwhm=fwhm)
    data, _ = make_psf_model_image((150, 150), psf_model, n_sources=30,
                                   model_shape=(11, 11), flux=(50, 100),
                                   min_separation=1, seed=0)

    daofinder = DAOStarFinder(threshold=0.5, fwhm=fwhm)
    grouper = SourceGrouper(min_separation=1.3 * fwhm)
    fitter = TRFLSQFitter()
    fit_shape = (5, 5)
    sub_shape = None
    psf_model.bounding_box = None
    psfphot = IterativePSFPhotometry(psf_model, fit_shape=fit_shape,
                                     finder=daofinder, mode='all',
                                     grouper=grouper, maxiters=2,
                                     sub_shape=sub_shape,
                                     aperture_radius=3, fitter=fitter)
    match = r'model_shape must be specified .* does not have a bounding_box'
    with pytest.raises(ValueError, match=match):
        psfphot(data)


def test_iterative_psf_photometry_inputs():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(10.0, 2.0)

    match = 'finder cannot be None for IterativePSFPhotometry'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=None,
                                   aperture_radius=4)

    match = 'aperture_radius cannot be None for IterativePSFPhotometry'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=None)

    match = 'maxiters must be a strictly-positive scalar'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=-1)
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=[1, 2])

    match = 'maxiters must be an integer'
    with pytest.raises(ValueError, match=match):
        _ = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                   aperture_radius=4, maxiters=3.14)


def test_negative_xy():
    sources = Table()
    sources['id'] = np.arange(3) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 15.2, -0.7]
    sources['y_0'] = [-0.3, -0.4, 18.7]
    sources['sigma'] = 3.1
    shape = (31, 31)
    psf_model = CircularGaussianPRF(flux=1, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=10)
    phot = psfphot(data, init_params=sources)
    assert_equal(phot['x_init'], sources['x_0'])
    assert_equal(phot['y_init'], sources['y_0'])


def test_out_of_bounds_centroids():
    sources = Table()
    sources['id'] = np.arange(8) + 1
    sources['flux'] = 1
    sources['x_0'] = [-1.4, 34.5, 14.2, -0.7, 34.5, 14.2, 51.3, 52.0]
    sources['y_0'] = [13, -0.2, -1.6, 40, 51.1, 50.9, 12.2, 42.3]
    sources['sigma'] = 3.1

    shape = (51, 51)
    psf_model = CircularGaussianPRF(flux=1, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)
    fit_shape = (11, 11)
    psfphot = PSFPhotometry(psf_model, fit_shape, aperture_radius=10)

    phot = psfphot(data, init_params=sources)

    # at least one of the best-fit centroids should be
    # out of the bounds of the dataset, producing a
    # masked value in the `cfit` column:
    assert np.any(np.isnan(phot['cfit']))


def test_make_psf_model():
    normalize = False
    sigma = 3.0
    amplitude = 1.0 / (2 * np.pi * sigma**2)
    xcen = ycen = 0.0
    psf0 = Gaussian2D(amplitude, xcen, ycen, sigma, sigma)
    psf1 = make_psf_model(psf0, x_name='x_mean', y_name='y_mean',
                          normalize=normalize)
    psf2 = make_psf_model(psf0, normalize=normalize)
    psf3 = make_psf_model(psf0, x_name='x_mean', normalize=normalize)
    psf4 = make_psf_model(psf0, y_name='y_mean', normalize=normalize)

    yy, xx = np.mgrid[0:101, 0:101]
    psf = psf1.copy()
    xval = 48
    yval = 52
    flux = 14.51
    psf.x_mean_2 = xval
    psf.y_mean_2 = yval
    data = psf(xx, yy) * flux

    fit_shape = 7
    init_params = Table([[46.1], [57.3], [7.1]],
                        names=['x_0', 'y_0', 'flux_0'])
    phot1 = PSFPhotometry(psf1, fit_shape, aperture_radius=None)
    tbl1 = phot1(data, init_params=init_params)

    phot2 = PSFPhotometry(psf2, fit_shape, aperture_radius=None)
    tbl2 = phot2(data, init_params=init_params)

    phot3 = PSFPhotometry(psf3, fit_shape, aperture_radius=None)
    tbl3 = phot3(data, init_params=init_params)

    phot4 = PSFPhotometry(psf4, fit_shape, aperture_radius=None)
    tbl4 = phot4(data, init_params=init_params)

    assert_allclose((tbl1['x_fit'][0], tbl1['y_fit'][0],
                     tbl1['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl2['x_fit'][0], tbl2['y_fit'][0],
                     tbl2['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl3['x_fit'][0], tbl3['y_fit'][0],
                     tbl3['flux_fit'][0]), (xval, yval, flux))
    assert_allclose((tbl4['x_fit'][0], tbl4['y_fit'][0],
                     tbl4['flux_fit'][0]), (xval, yval, flux))


def make_mock_finder(x_col, y_col):
    def finder(data, mask=None):  # noqa: ARG001
        source_table = Table()
        source_table[x_col] = [25.1]
        source_table[y_col] = [24.9]
        return source_table
    return finder


FINDER_COLUMN_NAMES = [
    ('x', 'y'),
    ('x_init', 'y_init'),
    ('xcentroid', 'ycentroid'),
    ('x_centroid', 'y_centroid'),
    ('xpos', 'ypos'),
    ('x_peak', 'y_peak'),
    ('xcen', 'ycen'),
    ('x_fit', 'y_fit'),
    ('x_invalid', 'y_invalid'),
]


@pytest.mark.parametrize(('x_col', 'y_col'), FINDER_COLUMN_NAMES)
def test_finder_column_names(x_col, y_col):
    """
    Test that PSFPhotometry works correctly with a finder that returns
    different column names for source positions.
    """
    finder = make_mock_finder(x_col, y_col)

    sources = Table()
    sources['id'] = [1]
    sources['flux'] = 7.0
    sources['x_0'] = 25.0
    sources['y_0'] = 25.0
    shape = (31, 31)
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)

    fit_shape = (9, 9)
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=10)

    # invalid column names should raise an error
    if x_col == 'x_invalid' or y_col == 'y_invalid':
        match = 'must contain columns for x and y coordinates'
        with pytest.raises(ValueError, match=match):
            psfphot(data)
        return

    phot_tbl = psfphot(data)

    assert len(phot_tbl) == 1
    assert_allclose(phot_tbl['x_init'][0], 25.1)
    assert_allclose(phot_tbl['y_init'][0], 24.9)
    assert_allclose(phot_tbl['x_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['y_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['flux_fit'][0], 7.0, rtol=1e-6)


@pytest.mark.parametrize(('x_col', 'y_col'), FINDER_COLUMN_NAMES)
def test_iterative_finder_column_names(x_col, y_col):
    """
    Test that IterativePSFPhotometry works correctly with a finder that
    returns different column names for source positions.
    """
    finder = make_mock_finder(x_col, y_col)

    sources = Table()
    sources['id'] = [1]
    sources['flux'] = 7.0
    sources['x_0'] = 25.0
    sources['y_0'] = 25.0
    shape = (31, 31)
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    data = make_model_image(shape, psf_model, sources)

    fit_shape = (9, 9)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=10, maxiters=3)

    # invalid column names should raise an error
    if x_col == 'x_invalid' or y_col == 'y_invalid':
        match = 'must contain columns for x and y coordinates'
        with pytest.raises(ValueError, match=match):
            psfphot(data, init_params=sources)
        return

    phot_tbl = psfphot(data)

    assert_allclose(phot_tbl['x_init'][0], 25.1)
    assert_allclose(phot_tbl['y_init'][0], 24.9)
    assert_allclose(phot_tbl['x_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['y_fit'][0], 25.0, atol=1e-6)
    assert_allclose(phot_tbl['flux_fit'][0], 7.0, rtol=1e-6)


def test_repr():
    psf_model = CircularGaussianPRF(flux=1.0, fwhm=3.1)
    fit_shape = (9, 9)
    finder = DAOStarFinder(6.0, 2.0)

    psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                            aperture_radius=10)
    cls_repr = repr(psfphot)
    assert cls_repr.startswith(f'{psfphot.__class__.__name__}(')

    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=10)
    cls_repr = repr(psfphot)
    assert cls_repr.startswith(f'{psfphot.__class__.__name__}(')


def test_move_column():
    psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
    fit_shape = (5, 5)
    finder = DAOStarFinder(6.0, 2.0)
    psfphot = IterativePSFPhotometry(psf_model, fit_shape, finder=finder,
                                     aperture_radius=4, maxiters=1)
    tbl = QTable()
    tbl['a'] = [1, 2, 3]
    tbl['b'] = [4, 5, 6]
    tbl['c'] = [7, 8, 9]

    tbl1 = psfphot._move_column(tbl, 'a', 'c')
    assert tbl1.colnames == ['b', 'c', 'a']
    tbl2 = psfphot._move_column(tbl, 'd', 'b')
    assert tbl2.colnames == ['a', 'b', 'c']
    tbl3 = psfphot._move_column(tbl, 'b', 'b')
    assert tbl3.colnames == ['a', 'b', 'c']
