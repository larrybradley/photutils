.. _build-epsf:

Building an effective Point Spread Function (ePSF)
==================================================

The ePSF
--------

The instrumental PSF is a combination of many factors that are
generally difficult to model. `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
showed that accurate stellar photometry and astrometry can be derived
by modeling the net PSF, which they call the effective PSF (ePSF). The
ePSF is an empirical model describing what fraction of a star's light
will land in a particular pixel. The constructed ePSF may be oversampled
with respect to the detector pixels.

Oversampling matters when the PSF is undersampled by the detector, e.g.,
a FWHM of only one or two pixels. Since stars can land at fractional
pixel positions on the detector, the appearance of such a PSF varies
with the star's position within a pixel, and an oversampled ePSF
captures this pixel-phase variation so that the PSF can be interpolated
to the exact position of any star. When the PSF is well sampled (a FWHM
of about 4 pixels or more), an ePSF with no oversampling already captures
its shape, and a larger oversampling factor improves it only slightly
(see :ref:`epsf-guidelines`).


Building an ePSF
----------------

Photutils provides tools for building an ePSF that are based
on the method of `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
and `Anderson 2016 (WFC3 ISR 2016-12)
<https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_. The
implementation differs from that method in several steps, so that it
can be used for other instruments, samplings, and oversampling factors
(see :ref:`epsf-anderson-differences`). The process iteratively refines
the ePSF model and star positions: the current ePSF is fitted to the
stars to improve their centers, and then the ePSF is rebuilt using the
improved star positions.

To begin, we must first define a sample of stars used to build the
ePSF. Ideally these stars should be bright (high S/N) and isolated to
prevent contamination from nearby stars. One may use the star-finding
tools in Photutils (e.g., :class:`~photutils.detection.DAOStarFinder`
or :class:`~photutils.detection.IRAFStarFinder`) to identify an initial
sample of stars. However, the step of creating a good sample of stars
generally requires visual inspection and manual selection to ensure
stars are sufficiently isolated and of good quality (e.g., no cosmic
rays, detector artifacts, etc.). To produce a good ePSF, one should
have a reasonably large sample of stars (e.g., a few hundred) in order
to sample the PSF at all subpixel phases and to help reduce the effects
of noise. Otherwise, the resulting ePSF may be noisy or biased. See
:ref:`epsf-guidelines` for guidance on choosing the oversampling factor
and the star sample.

Simulating a Star Field
^^^^^^^^^^^^^^^^^^^^^^^

For this example we simulate an image of stars from a known ePSF, so
that the ePSF that we build can be compared with the true one at the
end (see :ref:`epsf-example-comparison`). The input is a simulated
ePSF of the JWST NIRCam F115W filter with an oversampling factor
of 4 (see :ref:`epsf-example-input` for how it was made). It has a
FWHM of about 1.5 pixels, so it is undersampled. Let's load it as an
`~photutils.psf.ImagePSF` model::

    >>> from astropy.io import fits
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> from photutils.psf import ImagePSF
    >>> filename = get_pkg_data_filename('data/jwst_nircam_f115w_epsf.fits',
    ...                                  package='photutils.datasets')
    >>> true_epsf = ImagePSF(fits.getdata(filename), oversampling=4)

Next we define 500 stars at random positions in a 1000x1000 pixel image.
The stars are at least 30 pixels from each other, so that they are
isolated, and at least 20 pixels from the image borders. Their fluxes
(in electrons) are uniformly distributed in magnitude over a range of
3.75 magnitudes::

    >>> import numpy as np
    >>> from photutils.datasets import make_model_params
    >>> shape = (1000, 1000)
    >>> params = make_model_params(shape, n_sources=500, min_separation=30,
    ...                            border_size=20, seed=0)
    >>> rng = np.random.default_rng(1)
    >>> params['flux'] = 10**rng.uniform(3.5, 5.0, len(params))

Now we make the image. Every star has the same ePSF. We add a constant
background of 20 electrons per pixel and then noise, which is the
Poisson noise of the stars and the background (approximated as Gaussian)
plus a read noise of 5 electrons::

    >>> from photutils.datasets import make_model_image
    >>> data = make_model_image(shape, true_epsf, params,
    ...                         model_shape=(25, 25))
    >>> data += 20.0
    >>> data += rng.normal(scale=np.sqrt(data + 5.0**2))

Let's show the image:

.. plot::
    :context: reset

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table
    from astropy.utils.data import get_pkg_data_filename
    from astropy.visualization import simple_norm
    from photutils.datasets import make_model_image, make_model_params
    from photutils.detection import DAOStarFinder
    from photutils.psf import EPSFBuilder, ImagePSF, extract_stars
    from photutils.utils import circular_footprint
    from scipy.ndimage import binary_dilation

    filename = get_pkg_data_filename('data/jwst_nircam_f115w_epsf.fits',
                                     package='photutils.datasets')
    true_epsf = ImagePSF(fits.getdata(filename), oversampling=4)

    shape = (1000, 1000)
    params = make_model_params(shape, n_sources=500, min_separation=30,
                               border_size=20, seed=0)
    rng = np.random.default_rng(1)
    params['flux'] = 10**rng.uniform(3.5, 5.0, len(params))

    data = make_model_image(shape, true_epsf, params, model_shape=(25, 25))
    data += 20.0
    data += rng.normal(scale=np.sqrt(data + 5.0**2))

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(data, 'sqrt', percent=99.0)
    ax.imshow(data, norm=norm, origin='lower')

Finding the Stars
^^^^^^^^^^^^^^^^^

We'll use the :class:`~photutils.detection.DAOStarFinder` class to
identify the stars and their initial positions. The diffraction
features around the bright stars are also detected as sources. The
``min_separation`` keyword removes them by keeping only the brightest
source within 15 pixels::

    >>> from photutils.detection import DAOStarFinder
    >>> finder = DAOStarFinder(threshold=100.0, fwhm=1.5, min_separation=15)
    >>> sources = finder(data)
    >>> len(sources)
    500

Let's show the detected stars overlaid on the image:

.. plot::
    :context: close-figs

    finder = DAOStarFinder(threshold=100.0, fwhm=1.5, min_separation=15)
    sources = finder(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(data, 'sqrt', percent=99.0)
    ax.imshow(data, norm=norm, origin='lower')
    ax.scatter(sources['x_centroid'], sources['y_centroid'],
               s=80, edgecolor='red', facecolor='none', lw=1.5)

All of the stars in the simulated image are isolated and far from the
image borders, so we do not need to exclude any of them. In practice
this step will require some manual inspection and selection, e.g., to
remove stars that have close neighbors or that are too close to the
image borders for a complete cutout.


Extracting Star Cutouts
-----------------------

Next, we need to extract cutouts of the stars using the
:func:`~photutils.psf.extract_stars` function. This function requires
a table of star positions either in pixel or sky coordinates. For this
example we are using pixel coordinates, which need to be in table
columns called ``x`` and ``y``::

    >>> from astropy.table import Table
    >>> stars_tbl = Table()
    >>> stars_tbl['x'] = sources['x_centroid']
    >>> stars_tbl['y'] = sources['y_centroid']

The star cutouts from which we build the ePSF must have the
background subtracted. Here we'll use the sigma-clipped median value
as the background level. If the background in the image varies
across the image, one should use more sophisticated methods (e.g.,
`~photutils.background.Background2D`).

The background level must be measured from pixels that are free of star
light. The extended wings of the stars cover a large fraction of this
image, and sigma clipping does not remove them. The sigma-clipped median
of the whole image is therefore 20.7, which is 0.7 electrons higher
than the true background of 20. That is a small fraction of the noise,
but summed over a 25x25 pixel cutout it is about 3% of the flux of a
typical star in this image, and subtracting it would make the ePSF too
concentrated. To avoid this bias, we first mask the pixels within 18
pixels of each detected star::

    >>> from photutils.utils import circular_footprint
    >>> from scipy.ndimage import binary_dilation
    >>> star_mask = np.zeros(data.shape, dtype=bool)
    >>> yidx = np.round(sources['y_centroid']).astype(int)
    >>> xidx = np.round(sources['x_centroid']).astype(int)
    >>> star_mask[yidx, xidx] = True
    >>> star_mask = binary_dilation(
    ...     star_mask, structure=circular_footprint(18))

Now let's subtract the background, measured from the unmasked pixels,
from the image::

    >>> from astropy.stats import sigma_clipped_stats
    >>> mean_val, median_val, std_val = sigma_clipped_stats(
    ...     data, sigma=2.0, mask=star_mask)
    >>> print(f'{median_val:.1f}')
    20.0
    >>> data -= median_val

We are now ready to create our star cutouts using the
:func:`~photutils.psf.extract_stars` function. The input image can be
a plain 2D array, as in this example, or an `~astropy.nddata.NDData`
object. An `~astropy.nddata.NDData` object is needed to input an
uncertainty, a mask, or a WCS along with the image.

For this simple example we are extracting stars from a single image
using a single catalog. The :func:`~photutils.psf.extract_stars`
function can also extract stars from multiple images using a separate
catalog for each image or a single catalog. When using a single
catalog with multiple images, the star positions must be in sky
coordinates (as `~astropy.coordinates.SkyCoord` objects) and the
images must be `~astropy.nddata.NDData` objects that contain valid
`~astropy.wcs.WCS` objects. In the case of using multiple images (i.e.,
dithered images) and a single catalog, the same physical star will be
"linked" across images, meaning it will be constrained to have the same
sky coordinate and, by default, the same flux in each input image (see
:ref:`epsf-linked-stars`).

Let's extract 25x25 pixel cutouts of our selected stars::

    >>> from photutils.psf import extract_stars
    >>> stars = extract_stars(data, stars_tbl, size=25)

The function returns an `~photutils.psf.EPSFStars` object containing the
cutouts of our selected stars that will be used to build the ePSF. Let's
show the first 25 of them:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> nrows = 5
    >>> ncols = 5
    >>> fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
    ...                        squeeze=True)
    >>> ax = ax.ravel()
    >>> for i in range(nrows * ncols):
    ...     norm = simple_norm(stars[i], 'log', percent=99.0)
    ...     ax[i].imshow(stars[i], norm=norm, origin='lower')

.. plot::
    :context: close-figs

    stars_tbl = Table()
    stars_tbl['x'] = sources['x_centroid']
    stars_tbl['y'] = sources['y_centroid']

    star_mask = np.zeros(data.shape, dtype=bool)
    yidx = np.round(sources['y_centroid']).astype(int)
    xidx = np.round(sources['x_centroid']).astype(int)
    star_mask[yidx, xidx] = True
    star_mask = binary_dilation(star_mask, structure=circular_footprint(18))
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0,
                                                        mask=star_mask)
    data -= median_val

    stars = extract_stars(data, stars_tbl, size=25)

    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()
    for i in range(nrows * ncols):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower')


Constructing the ePSF
---------------------

With the star cutouts, we are ready to construct the ePSF with the
:class:`~photutils.psf.EPSFBuilder` class. We'll create an ePSF with an
oversampling factor of 4, which is appropriate for these undersampled
stars (a FWHM of about 1.5 pixels). We use the default maximum of
10 iterations (``maxiters=10``). The build stops early once the
star centers have converged. Do not stop the build after only a few
iterations. An ePSF that has not converged can differ from the true ePSF
by a few percent of its peak, and the fitted star positions are less
accurate. The :class:`~photutils.psf.EPSFBuilder` class has many options
to control the ePSF build process, including the smoothing kernel, the
fitting box, the recentering function, and the convergence criterion.
Please see the :class:`~photutils.psf.EPSFBuilder` documentation for
further details.

We first initialize an :class:`~photutils.psf.EPSFBuilder` instance with
our desired parameters and then input the cutouts of our selected stars
to the instance::

    >>> from photutils.psf import EPSFBuilder
    >>> epsf_builder = EPSFBuilder(oversampling=4, progress_bar=False)
    >>> result = epsf_builder(stars)

The :class:`~photutils.psf.EPSFBuilder` returns an
`~photutils.psf.EPSFBuildResults` object containing the constructed ePSF,
the fitted stars, and detailed information about the build process. This
result object supports tuple unpacking, so both of the following work::

    >>> # Access result attributes
    >>> epsf = result.epsf
    >>> fitted_stars = result.fitted_stars

    >>> # Tuple unpacking also works
    >>> epsf, fitted_stars = result

The `~photutils.psf.EPSFBuildResults` object provides useful diagnostic
information about the build process::

    >>> result.converged
    True
    >>> result.iterations  # doctest: +SKIP
    9
    >>> result.n_excluded_stars
    0

The results also report the fraction of stars whose centers converged
(``final_converged_fraction``), the largest center movement in the final
iteration (``final_center_accuracy``), and the smoothing kernel and
fitting box that were used (``smoothing_kernel`` and ``fit_shape``). See
`~photutils.psf.EPSFBuildResults` for the full list.

The results also keep the ePSF image after each iteration in the
``iteration_epsfs`` attribute, and the ``iteration_info`` attribute
is a table with the convergence statistics of each iteration. The
``max_epsf_change`` column is the largest change of the ePSF from the
previous iteration as a fraction of its peak. If it is still large in
the last building iteration, increase ``maxiters``. The ``converged``
column tells whether the star centers had converged in each iteration,
so the last ``'build'`` row tells whether the building iterations
converged. The center changes are measured relative to their median
change, and ``max_center_shift`` is the largest of them. The build
has converged when the ``converged_fraction`` of the builder (95%
by default) of these relative changes and the median change itself
(``common_center_shift``) are less than ``center_accuracy``. The
``plot_iterations`` method plots the ePSF after each iteration and its
change from the previous iteration::

    >>> result.iteration_info['iteration', 'stage', 'max_epsf_change'].pprint(max_lines=6)  # doctest: +SKIP
    >>> fig = result.plot_iterations()  # doctest: +SKIP

The returned ``epsf`` is an `~photutils.psf.ImagePSF` object, and
``fitted_stars`` is a new `~photutils.psf.EPSFStars` object with the
updated star positions and fluxes from fitting the final ePSF model.

Finally, let's show the constructed ePSF:

.. doctest-skip::

    >>> import matplotlib.pyplot as plt
    >>> from astropy.visualization import simple_norm
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> norm = simple_norm(epsf.data, 'log', percent=99.0)
    >>> axim = ax.imshow(epsf.data, norm=norm, origin='lower')
    >>> fig.colorbar(axim)

.. plot::
    :context: close-figs

    epsf_builder = EPSFBuilder(oversampling=4, progress_bar=False)
    result = epsf_builder(stars)
    epsf, fitted_stars = result

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = simple_norm(epsf.data, 'log', percent=99.0)
    axim = ax.imshow(epsf.data, norm=norm, origin='lower')
    fig.colorbar(axim)

The `~photutils.psf.ImagePSF` object can be
used as a PSF model for :ref:`PSF Photometry
<psf-photometry>` (i.e., `~photutils.psf.PSFPhotometry` or
`~photutils.psf.IterativePSFPhotometry`).


.. _epsf-example-comparison:

Comparing with the Input ePSF
-----------------------------

Because the image was simulated, the results can be compared with the
true values. This cannot be done with real data, but it shows what
accuracy to expect in the best case. Here the stars are isolated, they
all have the same PSF, the background is constant, and the image was
made with the same interpolation of the ePSF that is used to fit it.

We first match each fitted star to the nearest simulated star and
compare the positions and the fluxes::

    >>> xy_fit = fitted_stars.center_flat
    >>> flux_fit = np.array([star.flux for star in fitted_stars.all_stars])
    >>> x_true = np.array(params['x_0'])
    >>> y_true = np.array(params['y_0'])
    >>> dist = np.hypot(xy_fit[:, :1] - x_true, xy_fit[:, 1:] - y_true)
    >>> idx = np.argmin(dist, axis=1)
    >>> dx = xy_fit[:, 0] - x_true[idx]
    >>> dy = xy_fit[:, 1] - y_true[idx]
    >>> flux_ratio = flux_fit / np.array(params['flux'])[idx]
    >>> print(f'{np.median(dx):.3f} {np.median(dy):.3f}')  # doctest: +SKIP
    0.017 0.003
    >>> print(f'{np.std(dx):.3f} {np.std(dy):.3f}')  # doctest: +SKIP
    0.009 0.009
    >>> bool(max(np.std(dx), np.std(dy)) < 0.02)
    True
    >>> print(f'{np.median(flux_ratio):.3f} {np.std(flux_ratio):.3f}')  # doctest: +SKIP
    1.000 0.013
    >>> bool(abs(np.median(flux_ratio) - 1.0) < 0.005)
    True

The fitted fluxes are unbiased, with a scatter of 1.3% that is set
by the noise of the stars. The scatter of the fitted positions is
0.009 pixels along each axis. The x positions, however, are all
offset by 0.017 pixels from the input positions. This is not an
error. The center of an ePSF is a convention, and the two ePSFs
use different ones. The input ePSF is centered where ``stpsf``
placed the star. :class:`~photutils.psf.EPSFBuilder` centers
the ePSF on the point about which its core is most symmetric
(`~photutils.centroids.centroid_symmetry`), and the core of this
PSF is not symmetric about the position where ``stpsf`` placed the
star. No definition of the center that uses only the image of the
PSF removes this offset. The offset is the same for every star,
so it has no effect on relative positions, and positions that are
measured with the built ePSF are consistent with each other (see
:ref:`epsf-anderson-differences`).

The offset matters in three cases:

* When positions are compared with positions that were measured with
  another ePSF or another centering method, e.g., in another filter.
  The offset is different for each ePSF.

* When sky coordinates are calculated from the fitted positions with
  the WCS of the image. That WCS was calibrated with star positions that
  were measured with some definition of the center, e.g., the centroids
  of a star finder or the fit of a PSF model. The sky coordinates are
  offset by the difference between that definition and the center of the
  ePSF. Aligning the fitted positions themselves to a reference catalog
  removes a constant offset.

* When the asymmetry of the PSF varies across the detector. The offset
  then varies with position, which looks like a distortion of the image.
  A single ePSF built for the whole image does not show it.

The size of the offset depends on the definition of the center.
The symmetry center, which is the definition of Anderson 2016,
depends only on the core of the ePSF. Before version 3.1,
:class:`~photutils.psf.EPSFBuilder` centered the ePSF on its center of
mass in a 5x5 pixel box, which is pulled toward the asymmetric structure
around the core::

    >>> from photutils.centroids import centroid_com
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            recentering_func=centroid_com,
    ...                            progress_bar=False)

With this builder the x offset is 0.071 pixels and the y offset is
-0.001 pixels. The scatter of the positions and of the fitted fluxes is
essentially unchanged.

With real data the true positions are not known, but the difference
between the two definitions can be measured from the ePSF itself. The
``center_asymmetry`` attribute of the results is the ``(x, y)`` offset
of the center of mass of the built ePSF from its symmetry center, in
detector pixels. It does not depend on which of the two definitions was
used to build the ePSF::

    >>> print(np.round(result.center_asymmetry, 3))  # doctest: +SKIP
    [ 0.055 -0.004]

The x value of 0.055 pixels agrees with the difference between the
two offsets from the input positions given above, 0.071 pixels for the
center of mass and 0.017 pixels for the symmetry center. It is not the
offset from the input positions, which cannot be measured with real
data. It is zero for a symmetric ePSF. A value that is not small compared with the
astrometric accuracy that is needed means that the measured positions
depend on the definition of the ePSF center by that amount.

To compare the two ePSFs, we evaluate the input ePSF on the grid of the
built ePSF, shifted by that offset, and normalize it in the same way::

    >>> n = epsf.data.shape[0]
    >>> offsets = (np.arange(n) - n // 2) / 4
    >>> xx, yy = np.meshgrid(offsets, offsets)
    >>> truth = true_epsf.evaluate(xx, yy, 1.0, -np.median(dx),
    ...                            -np.median(dy))
    >>> truth *= 16 / truth.sum()
    >>> residual = (epsf.data - truth) / truth.max()
    >>> print(f'{epsf.data.max() / truth.max():.3f}')  # doctest: +SKIP
    0.994
    >>> print(f'{np.abs(residual).max():.3f}')  # doctest: +SKIP
    0.006
    >>> bool(np.abs(residual).max() < 0.01)
    True

The peak of the built ePSF is 0.6% lower than the peak of the input
ePSF, and the largest difference between the two is 0.6% of the peak.
For the ePSF built with `~photutils.centroids.centroid_com`, the largest
difference is also 0.6% of the peak. Let's show the input ePSF, the built
ePSF, and their difference as a fraction of the peak:

.. doctest-skip::

    >>> fig, ax = plt.subplots(ncols=3, figsize=(15, 4.5))
    >>> norm = simple_norm(truth, 'log', percent=99.0)
    >>> ax[0].imshow(truth, norm=norm, origin='lower')
    >>> ax[0].set_title('Input ePSF')
    >>> ax[1].imshow(epsf.data, norm=norm, origin='lower')
    >>> ax[1].set_title('Built ePSF')
    >>> axim = ax[2].imshow(residual, vmin=-0.005, vmax=0.005, cmap='RdBu_r',
    ...                     origin='lower')
    >>> ax[2].set_title('(Built - Input) / peak')
    >>> cax = ax[2].inset_axes([1.04, 0.0, 0.05, 1.0])
    >>> fig.colorbar(axim, cax=cax)

.. plot::
    :context: close-figs

    xy_fit = fitted_stars.center_flat
    x_true = np.array(params['x_0'])
    y_true = np.array(params['y_0'])
    dist = np.hypot(xy_fit[:, :1] - x_true, xy_fit[:, 1:] - y_true)
    idx = np.argmin(dist, axis=1)
    dx = xy_fit[:, 0] - x_true[idx]
    dy = xy_fit[:, 1] - y_true[idx]

    n = epsf.data.shape[0]
    offsets = (np.arange(n) - n // 2) / 4
    xx, yy = np.meshgrid(offsets, offsets)
    truth = true_epsf.evaluate(xx, yy, 1.0, -np.median(dx), -np.median(dy))
    truth *= 16 / truth.sum()
    residual = (epsf.data - truth) / truth.max()

    fig, ax = plt.subplots(ncols=3, figsize=(15, 4.5))
    norm = simple_norm(truth, 'log', percent=99.0)
    ax[0].imshow(truth, norm=norm, origin='lower')
    ax[0].set_title('Input ePSF')
    ax[1].imshow(epsf.data, norm=norm, origin='lower')
    ax[1].set_title('Built ePSF')
    axim = ax[2].imshow(residual, vmin=-0.005, vmax=0.005, cmap='RdBu_r',
                        origin='lower')
    ax[2].set_title('(Built - Input) / peak')
    cax = ax[2].inset_axes([1.04, 0.0, 0.05, 1.0])
    fig.colorbar(axim, cax=cax)

With real data the true ePSF is not known. The checks that remain are
the residuals of the stars after the fitted ePSF is subtracted, the
distribution of the subpixel phases of the fitted centers, which should
be uniform, and the absence of any trend of the fitted fluxes and
positions with subpixel phase (see :ref:`epsf-guidelines`).

.. _epsf-example-input:

The Input ePSF
^^^^^^^^^^^^^^

The input ePSF of this example was made with version 2.2.0 of the `stpsf
<https://stpsf.readthedocs.io/>`_ package (with version 2.2.0 of its
data files) for the center of the NIRCam NRCA1 detector::

    import stpsf
    from photutils.psf import make_epsf_from_psf

    nrc = stpsf.NIRCam()
    nrc.filter = 'F115W'
    nrc.detector = 'NRCA1'
    hdulist = nrc.calc_psf(fov_pixels=27, oversample=4)
    psf_data = hdulist['OVERDIST'].data

    input_epsf = make_epsf_from_psf(psf_data, oversampling=4,
                                    midpoints=True)
    input_epsf = input_epsf[3:-3, 3:-3]
    input_epsf *= 16.0 / input_epsf.sum()

The ``OVERDIST`` extension is the PSF sampled at the points of
a grid that is 4 times finer than the detector pixels, with the
geometric distortion and the detector effects (charge diffusion and
interpixel capacitance) applied. That is not yet an ePSF. An ePSF
value is the flux in a whole detector pixel, so the second step
integrates the sampled PSF over the area of a detector pixel with
`~photutils.psf.make_epsf_from_psf`. The ``stpsf`` image has 108x108
points with the PSF centered between its four central points. The
``midpoints=True`` option makes the ePSF at the points midway between
them, which gives a 107x107 ePSF with a grid point at the PSF center.

The third step keeps the central 101x101 points, which cover 25x25
detector pixels. The PSF was computed over 27x27 pixels so that every
pixel of the kept region lies inside the ``stpsf`` image. The values
that `~photutils.psf.make_epsf_from_psf` returns within half a pixel of
the edges of its input image are integrals over only part of a pixel,
and they are outside of the kept region. The ePSF is then normalized to
a sum of 16 (the square of the oversampling factor), so that the flux
of the model is the flux within the 25x25 pixel region. The result was
saved as a 32-bit floating-point FITS image.


Customizing the ePSF Builder
----------------------------

The :class:`~photutils.psf.EPSFBuilder` class provides several options
to customize the ePSF build process.

Smoothing Kernel
^^^^^^^^^^^^^^^^

The ``smoothing_kernel`` parameter controls the smoothing applied to
the ePSF during each iteration. The smoothing helps to reduce noise
in the ePSF, especially when the star sample is small or noisy. The
smoothing kernels are least-squares polynomial smoothers. Each grid
value is replaced by the value at the center of a polynomial fit to
the surrounding grid values, which removes noise while preserving the
polynomial shape of the ePSF within the kernel window.

The default is ``'auto'``, which uses a quartic (fourth-degree)
polynomial kernel whose width is 0.7 times the FWHM of the ePSF in
oversampled grid points, measured in each iteration along its narrowest
axis. The width is rounded down to an odd number of grid points, and no
smoothing is applied when it would be smaller than 5 grid points, i.e.,
for undersampled ePSFs with fewer than about 7 grid points per FWHM,
where a fixed 5x5 kernel would lower the peak of the ePSF. The kernel is
square, so with anisotropic oversampling the axis with the fewer grid
points per FWHM sets its size. The chosen kernel is reported in the
``smoothing_kernel`` attribute of the results, and it can be input as a
fixed ``smoothing_kernel`` to reproduce the build. If the FWHM cannot be
measured, the ``'quartic'`` kernel is used and a warning is emitted.

You can also use ``'quartic'`` for the fixed 5x5 fourth-degree
polynomial kernel of `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_,
``'quadratic'`` for its second-degree counterpart, provide a custom 2D
array, or set it to `None` for no smoothing::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            smoothing_kernel='quadratic',
    ...                            progress_bar=False)

The fixed kernels are applied on the oversampled grid, so their physical
width is ``5 / oversampling`` detector pixels. The 5x5 quartic kernel
was developed for HST data with an oversampling factor of 4, where it
is about 0.7 FWHM wide. When using a fixed kernel for an undersampled
ePSF with fewer than about seven grid points per FWHM, the kernel lowers
the peak of the ePSF, and ``smoothing_kernel=None`` is a better choice,
especially when the stars have high signal-to-noise. Smoothing is most
useful for well-sampled ePSFs built from noisy or few stars.

The wings of the ePSF are smoothed separately. Far from its center the
ePSF is faint and varies slowly, so its noise can be averaged over a
larger area than in the core. After the last iteration, each value of
the ePSF beyond 3.5 FWHM from the center is blended into a least-squares
quadratic fit to the values in a box 1.25 FWHM wide around it, and
beyond 5 FWHM into the fit in a box 1.75 FWHM wide. The ePSF within 3.5
FWHM of the center changes only by the renormalization of the smoothed
ePSF (less than 0.03 percent in tests). The fluxes of the returned stars
were fit before that renormalization. This lowers the noise of the
wings, which matters when they are used, e.g., to subtract bright stars,
to make model images, or to measure encircled energies.

The smoothing also removes real structure in the wings that is finer
than about two FWHM, such as diffraction rings and spikes. Applied to
noise-free ePSFs, it changed the wings by 2 to 8 percent of their mean
value for JWST and Roman ePSFs and by 11 to 14 percent for HST WFC3/IR
ePSFs. With a few hundred stars the noise that it removes is larger
than this in every case that was tested (the residuals of the wings
were up to about 50 percent lower). For a large star sample of high
signal-to-noise (thousands of stars), the noise in the wings can be
smaller than this change, and the wings are then more accurate without
the smoothing.

The boxes are at least 5 oversampled grid points wide. For an ePSF with
a FWHM of less than 4 grid points they are therefore wider than given
above, and they remove more of the real structure. This matters most
for an oversampling factor of 1, where the boxes of an undersampled
ePSF (a FWHM of about 1.3 pixels) are nearly 4 FWHM wide. In tests with
an oversampling factor of 1, the smoothing made the wings of the most
undersampled HST and JWST ePSFs less accurate, by up to a factor of
about 2, and those of most other ePSFs slightly more accurate. With an
oversampling factor of 2 or larger it made the wings more accurate or
left them unchanged in every case. For an oversampling factor of 1,
compare the ePSFs built with and without the smoothing.

The smoothing is applied after the last iteration, so it is not part
of ``iteration_epsfs`` or ``iteration_info``. The ``plot_iterations``
method shows the smoothed ePSF and the change made by the smoothing
in a last row of its figure when all the iterations are plotted. Set
``show_smoothed=True`` to also plot that row with a subset of the
iterations, or ``show_smoothed=False`` to never plot it.

Set ``wing_smoothing=False`` to keep the wings as built::

    >>> epsf_builder = EPSFBuilder(oversampling=4, wing_smoothing=False,
    ...                            progress_bar=False)

.. _epsf-alias-passband:

Alias Filter Passband
^^^^^^^^^^^^^^^^^^^^^

Independently of the smoothing kernel, when the oversampling factor is
greater than one the builder applies a low-pass filter to the ePSF in
every iteration. The filter has unit gain up to a passband frequency,
a smooth transition, and zero gain at and above one cycle per detector
pixel. Here a spatial frequency of one cycle per pixel describes
structure that repeats with a period of one detector pixel, and a
frequency of 0.5 cycles per pixel describes structure that repeats every
two pixels. A pixel-integrated PSF has essentially no signal at one
cycle per pixel, but that is the frequency at which the pixel sampling
of the stars aliases onto the oversampled grid. Together with depositing
each star pixel residual on the oversampled grid points within 0.375
pixel of the pixel center along each axis, the filter prevents noise
from heterogeneous, contaminated, or low signal-to-noise stars from
growing into a checkerboard pattern in the ePSF.

The ``alias_passband`` parameter sets the end of the passband in cycles
per detector pixel. The default (``'auto'``) is 0.8 cycles per pixel, or
0.7 for an oversampling factor of 2. The default is the best choice for
most data. A different value can help in two cases, which depend on how
much real signal the ePSF has just below one cycle per pixel. That is
set by the optical cutoff frequency of the telescope expressed in cycles
per pixel:

.. math::

    \nu_c = \frac{D \, p}{\lambda}

where :math:`D` is the telescope diameter, :math:`\lambda` is the
mean wavelength of the bandpass (in the same units as :math:`D`), and
:math:`p` is the pixel scale in radians per pixel. A telescope transmits
no signal above this frequency. For example, for HST (:math:`D = 2.4`
m) WFC3/IR (0.13 arcsec per pixel) at 1.1 microns, :math:`\nu_c = 2.4
\times 6.3 \times 10^{-7} / 1.1 \times 10^{-6} = 1.4` cycles per pixel.

.. list-table::
    :header-rows: 1
    :widths: 22 33 45

    * - :math:`\nu_c` (cycles/pixel)
      - Examples
      - Guidance for ``alias_passband``
    * - greater than about 1
      - HST WFC3/IR F110W, JWST NIRCam F070W, JWST NIRISS F090W, Roman
        WFI F062 and F106
      - The default, or 0.85 with more iterations. The ePSF has real
        signal up to nearly one cycle per pixel. In tests the default
        recovered the peak of these ePSFs to within about 1 percent,
        except for HST WFC3/IR F110W (3 percent low). A value of 0.85
        with a ``maxiters`` of 15 to 20 recovered the peaks to within
        0.3 percent and lowered the largest residual in the core by
        10 to 35 percent. For F110W it left the peak 1.5 percent low,
        and 0.9 recovered it to within 0.3 percent. For the others 0.9
        gave larger residuals than 0.85.
    * - about 0.9 to 1
      - HST WFC3/IR F160W
      - The default (0.8)
    * - less than about 0.9
      - JWST NIRCam F115W and redder, JWST MIRI, Roman WFI F158 and
        F213, most ground-based data
      - The default, or 0.7. There is no signal to preserve near one
        cycle per pixel. In tests 0.7 lowered the residuals in the core
        by 10 to 40 percent and converged in fewer iterations.

Try 0.85 with more iterations for a strongly undersampled detector whose
cutoff frequency is well above one cycle per pixel::

    >>> epsf_builder = EPSFBuilder(oversampling=4, alias_passband=0.85,
    ...                            maxiters=20,
    ...                            progress_bar=False)

Try 0.9 only if that ePSF is still too broad, i.e., if the stars have
positive residuals at their centers after the fitted ePSF is subtracted.

Do not use 0.7 unless the cutoff frequency is known to be low. It leaves
the peak of a strongly undersampled ePSF low by 2 to 6 percent.

A passband that is wider than needed has a cost. A star that is sampled
once per pixel constrains the frequencies near one cycle per pixel only
weakly, because a small shift of the star center has nearly the same
effect on its pixel values. A wider passband therefore makes the build
converge more slowly and makes it more sensitive to noise. With the
default ``maxiters`` of 10, the peak of an ePSF built with a passband of
0.85 or 0.9 is too high by up to about 1 percent. Use a passband above
the default only with a large star sample (a few hundred stars) and an
oversampling factor of 4 or more, allow more iterations (``maxiters``
of 15 to 20), and check that the build converged. Do not use it with an
oversampling factor of 2 or 3. In tests with stars of different shapes
at those factors, the additional iterations biased the fitted star
centers toward particular subpixel phases, and at a factor of 3 some
builds grew a checkerboard pattern.

The filter acts separately along the x and y axes. The signal that it
removes from an undersampled ePSF therefore shows as a faint ripple
pattern, with a period of about one pixel, along the row and the
column through the center of the ePSF. To remove this pattern, the
builder refines the ePSF after the building iterations. Each refinement
iteration (``refinement_iters``, 5 by default) updates the ePSF five
times with the star centers and fluxes held fixed and then refits the
stars with the updated ePSF. The ePSF is recentered in each update,
as in the building iterations. These updates use a wider and smoother
low-pass filter, with unit gain up to 1.1 cycles per pixel and zero
gain at and above 1.33 cycles per pixel. It does not remove signal near
one cycle per pixel, so it leaves no ripple pattern. It removes only
the frequencies that the star residuals do not constrain. Such a filter
cannot be used from the start of the build, because the build then
converges slowly and is more sensitive to the initial star centers.

The refinement is applied only for an oversampling factor of 4 or
larger, and it roughly doubles the run time of the build. For a
well-sampled ePSF it has little to restore and it adds a small amount
of noise (up to about 15 percent of the residual of the ePSF). Set
``refinement_iters=0`` to skip it. More refinement iterations than the
default improve the most strongly undersampled ePSFs only slightly and
add more noise to the well-sampled ones.

The ``converged``, ``final_center_accuracy``, and
``final_converged_fraction`` attributes of the results describe the last
refinement iteration when the ePSF was refined, so that they match the
returned stars. The ``iterations`` attribute counts only the building
iterations.

Setting ``alias_passband=None`` turns the filter off. This is rarely
appropriate. Without the filter, noise at the alias frequencies
accumulates over the iterations, the build can stall before it
converges, and heterogeneous or contaminated star samples can grow a
checkerboard pattern. In tests with simulated HST, JWST, and Roman star
fields, the unfiltered ePSF was less accurate than the filtered one in
nearly every case, even for large, clean, and homogeneous star samples.
The option is provided for experimentation, e.g., to check how much
the filter changes a particular ePSF. Always compare the result with
a filtered build. The filter is never applied along an axis with an
oversampling factor of 1.

If the subpixel phases of the fitted star centers are strongly
non-uniform at the end of the build, which indicates biased star
centers, a warning is emitted. In that case the star sample should be
inspected for stars with different PSFs, saturated or contaminated
cutouts, or spurious detections.

.. _epsf-linked-stars:

Linked Stars from Dithered Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the same star is observed in several dithered images, the cutouts
can be linked as a `~photutils.psf.LinkedEPSFStar` (this happens
automatically when :func:`~photutils.psf.extract_stars` is given
multiple images and a single catalog of sky coordinates). After each
fitting iteration, the builder constrains the centers of the linked
stars to a single sky coordinate and, by default, their fluxes to
their mean value. Averaging the positions and the fluxes across
dithers is the key step of `Anderson and King 2000 (PASP 112, 1360)
<https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
that breaks the degeneracy between the shape of the ePSF and the
positions of the stars. `Godden and Blundell 2026 (RASTI 5, 1)
<https://doi.org/10.1093/rasti/rzaf063>`_ confirmed that constraining
the positions alone is not enough. The fluxes must also be constrained
to break the degeneracy between the flux of a star and its subpixel
position caused by intra-pixel sensitivity variations. Without it, the
pixel-phase dependence of the individual flux measurements is absorbed
into the ePSF. The flux constraint assumes that the linked images have
the same flux scale (e.g., the same exposure time and throughput). If
they do not, set ``constrain_fluxes=False``::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            constrain_fluxes=False,
    ...                            progress_bar=False)

To link stars across images, provide a single catalog with sky
coordinates and multiple `~astropy.nddata.NDData` objects, each with a
valid WCS:

.. doctest-skip::

    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> catalog = Table()
    >>> catalog['skycoord'] = SkyCoord(ra=[...]*u.deg, dec=[...]*u.deg)
    >>> stars = extract_stars([nddata1, nddata2], catalog, size=25)

Customizing the ePSF Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~photutils.psf.EPSFBuilder` class allows you to customize
the fitting process using the ``fit_shape`` parameter. This parameter
specifies the size of the box (in detector pixels) centered on each
star used for fitting. The default is ``'auto'``, which uses a square
box of twice the FWHM of the ePSF in detector pixels (measured in
each iteration along its narrowest axis), with a minimum of 5 pixels
and a maximum of the star cutout size. The chosen box is reported in
the ``fit_shape`` attribute of the results. A fixed box can be given
instead. A smaller box speeds up the fitting, but it should still cover
the core of the star. A box that is much smaller than the star uses only
its flat core, which biases the fitted centers and can prevent the build
from converging. The 5-pixel box of Anderson 2016 is about 2.5 FWHM wide
for HST data but only about 1 FWHM wide for a star with a FWHM of 5
pixels::

    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            fit_shape=7,
    ...                            progress_bar=False)

You can also customize the fitter itself by passing a
`~astropy.modeling.fitting.Fitter` instance::

    >>> from astropy.modeling.fitting import LMLSQFitter
    >>> fitter = LMLSQFitter()
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            fitter=fitter, fit_shape=7,
    ...                            progress_bar=False)

Sigma Clipping
^^^^^^^^^^^^^^

The ``sigma_clip`` parameter controls the sigma clipping applied when
stacking the ePSF residuals in each iteration. The default uses sigma
clipping with ``sigma=3.0`` and ``maxiters=10``. You can provide your
own `~astropy.stats.SigmaClip` instance to customize this behavior::

    >>> from astropy.stats import SigmaClip
    >>> sigclip = SigmaClip(sigma=2.5, maxiters=5)
    >>> epsf_builder = EPSFBuilder(oversampling=4,
    ...                            sigma_clip=sigclip,
    ...                            progress_bar=False)

Setting ``sigma_clip=None`` disables sigma clipping entirely.


Including Weights
-----------------

If your input `~astropy.nddata.NDData` object contains uncertainty
information, the :func:`~photutils.psf.extract_stars` function will
automatically create weights for each star cutout. These weights are
used during the ePSF fitting process to give more weight to pixels with
lower uncertainties.

To include weights, provide an ``uncertainty`` attribute in
your `~astropy.nddata.NDData` object. The uncertainty can be
any of the `~astropy.nddata.NDUncertainty` subclasses (e.g.,
`~astropy.nddata.StdDevUncertainty`)::

    >>> import numpy as np
    >>> from astropy.nddata import StdDevUncertainty
    >>> uncertainty = StdDevUncertainty(np.sqrt(np.abs(data)))  # doctest: +SKIP
    >>> nddata = NDData(data=data, uncertainty=uncertainty)  # doctest: +SKIP



.. _epsf-guidelines:

Guidelines for Building a Good ePSF
-----------------------------------

The quality of an ePSF depends more on the input stars and on a sensible
choice of the oversampling factor than on the other builder parameters.
The following guidelines are based on `Anderson and King 2000 (PASP 112,
1360) <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
and on the systematic tests of `Godden and Blundell 2026 (RASTI
5, 1) <https://doi.org/10.1093/rasti/rzaf063>`_, with the
oversampling and star-count advice updated from tests of the current
:class:`~photutils.psf.EPSFBuilder` implementation.

Choosing the oversampling factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ePSF is tabulated on a grid with a spacing of ``1 / oversampling``
detector pixels and is evaluated between grid points by cubic spline
interpolation. The interpolation is accurate when there are at least
about four grid points per FWHM of the ePSF, so the oversampling factor
should be at least ``4 / FWHM`` with the FWHM in pixels (measured along
the narrowest direction of an elongated PSF). For example, the smallest
adequate oversampling factor is 3 for a FWHM of about 1.5 pixels, 2 for
a FWHM of about 2 pixels, and 1 for a FWHM of 4 pixels or more.

:class:`~photutils.psf.EPSFBuilder` estimates each grid point from
the star pixels within 0.375 pixel of it along each axis, or within one
grid spacing if that is larger. This box does not shrink as the
oversampling factor grows. A larger factor therefore neither makes the
ePSF noisier nor requires more stars. The grid points are not
independent, however, because neighboring points share most of their
star pixels, so a finer grid reduces the interpolation error but does
not resolve finer structure.

This behavior was confirmed with seven simulated ePSFs (Gaussian,
Moffat, JWST, and Roman models with FWHMs of 1.3 to 4.1 pixels). Each
was built from 40, 150, and 450 stars with oversampling factors of 1
to 8. Too small a factor clearly degraded the result. For the ePSFs
with a FWHM of 1.3 to 1.5 pixels, the errors of the fitted star
positions were about twice as large with a factor of 2 as with a factor
of 4, and four to six times as large with a factor of 1. Increasing
the factor beyond 4 did not help, because factors of 6 and 8 gave about
the same accuracy as 4. It also did no harm, even with only 40 stars,
apart from a run time that grew by a factor of 2 to 3 for each doubling
of the oversampling factor. The tests used clean simulated stars with
random subpixel phases.

The default factor of 4 is therefore a good choice for a FWHM of about
1 pixel or more, and a larger factor is needed only for a smaller FWHM.
A smaller factor that still satisfies ``oversampling >= 4 / FWHM`` saves
run time and memory, and for well-sampled data (a FWHM of 4 pixels or
more) it gives the same fitted positions and fluxes.

Choosing the ePSF size
^^^^^^^^^^^^^^^^^^^^^^

The size of the ePSF is set by the size of the star cutouts, or by
the ``shape`` keyword of :class:`~photutils.psf.EPSFBuilder`. An
image-based PSF model is zero outside of its image, so the flux of the
PSF wings beyond the ePSF image is not in the model. Stars subtracted
with the model leave their outer wings in the residual image, and the
sum of the model over the detector pixels changes with the subpixel
position of the star, because the number of detector pixels inside the
ePSF image changes (see :ref:`psf-image-models` for example values).

The ePSF should therefore extend to where the PSF is small compared
with the accuracy that is needed. A larger ePSF has a cost. The stars
must be isolated over the larger cutout, and the outer part of the
ePSF is built from pixels that are dominated by the background noise.
The fitted positions and fluxes of the stars do not depend on the ePSF
size, because only the central pixels of each star are fitted.

Choosing the star sample
^^^^^^^^^^^^^^^^^^^^^^^^

The noise of the ePSF falls as more stars are used. In the tests
described above, the residuals of the ePSF built from 450 stars were
about three times smaller than those of the ePSF built from 40 stars.
A few hundred stars is a good target, and the number that is needed
does not depend on the oversampling factor.

The subpixel phases of the star centers must still be spread over the
whole pixel, because structure in the ePSF that is finer than a
pixel is constrained only by stars at different phases. A set of
exposures dithered by fractions of a pixel that uniformly cover the
subpixel phases is far more effective than random placement and also
allows the star fluxes and positions to be constrained across images
(see :ref:`epsf-linked-stars`).

The stars should be bright but unsaturated, isolated (no neighbors
within the cutout), free of cosmic rays and detector artifacts, and have
a clean background subtraction so that the total flux of each cutout
is a reliable normalization. Just as important, all of the stars must
share the same PSF. Do not combine exposures with different seeing or
focus, and do not mix regions of the field where the PSF differs unless
the variation is small compared to the accuracy you need. Heterogeneous
stars produce pixel-to-pixel noise in the oversampled grid that biases
the fitted star centers toward particular subpixel phases, and the
builder emits a warning if the subpixel phases of the fitted centers are
strongly non-uniform at the end of the build. In that case, inspect the
star sample rather than increasing the number of iterations.

Finally, check the result. The subpixel phases of the fitted star
centers should be uniformly distributed, and the fitted fluxes and
positions of the stars (or of an independent set of stars) should not
depend on their subpixel phase.


.. _epsf-anderson-differences:

Differences from the Anderson and King Algorithm
------------------------------------------------

:class:`~photutils.psf.EPSFBuilder` implements the ePSF concept and
the iterative building procedure of `Anderson and King 2000 (PASP 112,
1360) <https://ui.adsabs.harvard.edu/abs/2000PASP..112.1360A/abstract>`_
(hereafter AK2000) and `Anderson 2016 (WFC3 ISR 2016-12)
<https://ui.adsabs.harvard.edu/abs/2016wfc..rept...12A/abstract>`_
(hereafter ISR 2016-12). The implementation differs from Anderson's
approach in several steps. This section describes these differences to
facilitate comparisons between ePSFs built with Photutils and those
built with Anderson's method, such as the library ePSFs distributed for
HST and JWST.

The basic procedure is the same. For each star, after subtracting the
background and normalizing by the star's flux, the value of each pixel
provides a sample of the ePSF at the pixel's offset from the star
center. The ePSF is tabulated on a grid that is finer than the detector
pixels. In each iteration the differences between the samples and the
current ePSF are combined with a robust average at each grid point and
added to the ePSF, which is then smoothed and recentered. The stars are
then fit again with the improved ePSF. A star is also placed on an image
in the same way in both. The ePSF is evaluated once per pixel, at the
offset of the pixel center from the star center, and multiplied by the
flux. The ePSF already includes the integration over the pixel, so no
further integration is performed.

The Anderson and King method was developed for HST images. Its
constants are given in detector pixels or in grid points of an ePSF
with an oversampling factor of 4, and they suit a PSF with a FWHM of
about 1.5 to 2 pixels. :class:`~photutils.psf.EPSFBuilder` is meant
to work for any instrument, sampling, and oversampling factor, and for
a single exposure. Most of the differences follow from that. The
others make the build robust for star samples that are heterogeneous,
contaminated, or of low signal-to-noise.

The following table compares the steps. The entries for Anderson come
from the two publications.

.. list-table::
    :header-rows: 1
    :widths: 16 42 42

    * - Step
      - Anderson
      - ``EPSFBuilder``
    * - ePSF grid
      - Oversampling factor of 4. The grid covers 5x5 pixels (21x21
        points) in AK2000 and 25x25 pixels (101x101 points) in ISR
        2016-12.
      - Any integer oversampling factor, which can differ along the two
        axes. The grid covers the star cutouts unless ``shape`` is
        given.
    * - Variation over the detector
      - A 3x3 array of ePSFs across each detector, interpolated
        bilinearly to the position of each star.
      - A single ePSF. Build one ePSF per detector region to make a
        `~photutils.psf.GriddedPSFModel`.
    * - Background
      - Measured for each star in an annulus around it. AK2000 uses
        the mode of the pixels 4 to 7 pixels from the star.
      - Not measured. The star cutouts must be background subtracted
        by the user.
    * - Residual sampling
      - Each grid point uses the samples within 0.25 pixel of it along
        each axis.
      - Each grid point uses the samples within 0.375 pixel of it
        along each axis, and within at least one grid spacing.
    * - Combining the residuals
      - Mean with iterative rejection of the samples more than 2.5
        sigma from it.
      - Median after sigma clipping at 3 sigma (``sigma_clip``).
    * - Smoothing of the core
      - A 5x5 least-squares quartic kernel in grid points, in every
        iteration.
      - A least-squares quartic kernel whose width is 0.7 FWHM, in
        every iteration. No smoothing if that is less than 5 grid
        points (``smoothing_kernel``).
    * - Smoothing of the wings
      - Stronger smoothing at fixed radii in every iteration. ISR
        2016-12 allows quadratic variations beyond 3 pixels and uses a
        3x3 boxcar beyond 5 pixels.
      - Quadratic fits in boxes of 1.25 and 1.75 FWHM beyond 3.5 and 5
        FWHM, applied once to the final ePSF (``wing_smoothing``).
    * - Fourier filter
      - None.
      - A low-pass filter that removes the frequencies at and above one
        cycle per pixel, in every iteration, for oversampling factors
        greater than 1 (``alias_passband``).
    * - Centering
      - AK2000 requires equal values half a pixel on either side of
        the center. ISR 2016-12 shifts the ePSF to the position where
        it is most symmetric about its center within a radius of 1.5
        pixels.
      - The ISR 2016-12 definition
        (`~photutils.centroids.centroid_symmetry`), measured in a 5x5
        pixel box. The offsets are spaced by half a grid point for
        oversampling factors less than 4 (``recentering_func`` and
        ``recentering_boxsize``).
    * - Normalization
      - The pixel values of a star of unit flux sum to 1 over its
        central 5x5 pixels (AK2000) or within a radius of 5.5 pixels
        (ISR 2016-12).
      - The ePSF sums to the product of the oversampling factors over
        the whole grid, so the pixel values of a star of unit flux sum
        to 1 over an area the size of its cutout.
    * - Iteration scheme
      - An inner loop of 5 ePSF updates (adjust, smooth, and recenter)
        with the stars held fixed, inside an outer loop that fits the
        stars again. The outer loop continues until the fitted
        positions and fluxes show no trend with pixel phase (12
        iterations in AK2000, 9 in ISR 2016-12).
      - One ePSF update per fit of the stars until the star centers
        converge (``center_accuracy``, ``converged_fraction``, and
        ``maxiters``). Then refinement iterations of 5 ePSF updates per
        fit of the stars, with a wider filter for oversampling factors
        of 4 or more (``refinement_iters``).
    * - Fitting the stars
      - Weighted by the expected Poisson noise. AK2000 fits the pixels
        within 1.5 pixels of the center, with a taper to 2 pixels, and
        solves for the position with Newton-Raphson steps. ISR 2016-12
        fits the central 5x5 pixels with a grid search for the
        position.
      - A box 2 FWHM wide and at least 5 pixels (``fit_shape``),
        weighted only if the input data have uncertainties, with a
        nonlinear least-squares fitter from Astropy (``fitter``).
    * - Dithered exposures
      - Central to the method. The positions of each star are
        transformed to a common frame, and the positions and fluxes
        are averaged over the exposures after every fit.
      - Optional. The centers and fluxes of linked stars are
        constrained across the images using their WCS (see
        :ref:`epsf-linked-stars`).
    * - Interpolation of the ePSF
      - A bicubic spline within 4 pixels of the center and bilinear
        interpolation farther out (ISR 2016-12).
      - A single bicubic spline over the whole grid.

The differences that change the ePSF the most are explained below.

**Residual sampling and the Fourier filter.** The pixels of a star
sample the ePSF on a lattice with a spacing of one pixel. With a
narrow sampling box, each star contributes to only some of the grid
points, and noise with a period of one pixel in the ePSF and biases
in the fitted star centers can reinforce each other from one iteration
to the next. For heterogeneous or contaminated stars this grows into
a checkerboard pattern. The wider box and the low-pass filter of
:class:`~photutils.psf.EPSFBuilder` prevent it. In tests with
heterogeneous stars, a box of 0.25 pixel regrew the pattern. The cost
is that the filter and the wider box remove some real signal of an
undersampled ePSF just below one cycle per pixel, which the refinement
iterations restore (see :ref:`epsf-alias-passband`).

**Smoothing.** A 5x5 quartic kernel is about 0.7 FWHM wide for
HST data with an oversampling factor of 4. Applied to an ePSF with
fewer grid points per FWHM it lowers the peak, and applied to one
with many more it removes little noise. The same holds for wing
smoothing at fixed radii in pixels. :class:`~photutils.psf.EPSFBuilder`
therefore scales both with the measured FWHM. It smooths the wings
only once, after the last iteration, because smoothing them in every
iteration changed the core of the ePSF through the normalization.

**Iteration scheme.** Anderson's inner loop lets the ePSF settle for
fixed star positions before the stars are fit again, and the average
over the dithered exposures breaks the degeneracy between the shape of
the ePSF and the star positions. :class:`~photutils.psf.EPSFBuilder`
must also work for a single exposure, where each star has only one fit.
It fits the stars after every ePSF update, and it uses Anderson's scheme
of several updates per fit only in the refinement iterations. Several
updates per fit from the start of the build did not improve the ePSF in
tests.

**Centering.** The definition of the center of an ePSF is arbitrary
as long as the same ePSF is used to build and to fit. The centering
definitions agree for a symmetric ePSF. For an asymmetric ePSF
they differ by a small constant offset. The offset has no effect
on photometry or on relative astrometry made with the same ePSF,
because the fitted star positions shift with it. Positions measured
with ePSFs that were centered differently differ by that offset.
:class:`~photutils.psf.EPSFBuilder` uses the symmetry definition of
ISR 2016-12 (`~photutils.centroids.centroid_symmetry`) by default.
The center of mass (`~photutils.centroids.centroid_com`), which was
the default before version 3.1, is more sensitive to asymmetric
structure around the core. In tests with simulated JWST and Roman
PSFs it was offset from the position of the source in the optical
model by up to 0.1 pixel, compared with 0.05 pixel or less for the
symmetry center. For the asymmetric ePSF of the example above, the
two centers differ by 0.05 pixels (see :ref:`epsf-example-comparison`).
The ``center_asymmetry`` attribute of the results gives this
difference for any built ePSF. The recentering function and box can
be changed with ``recentering_func`` and ``recentering_boxsize``.

**Normalization and background.** These two conventions set the flux
scale and must be kept in mind when ePSFs from the two sources are
mixed:

* Fluxes fit with an ePSF from :class:`~photutils.psf.EPSFBuilder`
  are the fluxes within the area of the ePSF grid. Fluxes fit with one
  of Anderson's library ePSFs are the fluxes within a radius of 5.5
  pixels (for the ISR 2016-12 models). The values of such a library
  ePSF sum to more than the product of the oversampling factors over
  the whole grid (5 to 8 percent more for the HST WFC3/IR F110W and
  F160W models). `~photutils.psf.ImagePSF` and
  `~photutils.psf.GriddedPSFModel` do not renormalize their input,
  so they return fluxes in the convention of the ePSF that they are
  given.

* An annulus close to a star contains light from the wings of the
  star. An ePSF built with such a local background has a small
  constant subtracted from it compared with one built with the
  background far from the stars (0.05 to 0.5 percent of the peak in
  tests). Both are valid. Use the same background convention when
  building an ePSF and when fitting stars with it.

**Interpolation.** Interpolation schemes differ most where the ePSF
curves strongly between grid points. In the cores of undersampled
ePSFs with an oversampling factor of 4 they can differ by several
tenths of a percent of the peak. An ePSF is built to reproduce the star
pixels with the interpolation that was used to build it, so a library
ePSF that is evaluated with a different interpolation can show
residuals of that size in the cores of bright stars.

**Alternatives proposed by Godden and Blundell.** `Godden and Blundell
2026 (RASTI 5, 1) <https://doi.org/10.1093/rasti/rzaf063>`_ tested the
steps of the method with simulated stars and proposed two changes that
apply to both Anderson's method and :class:`~photutils.psf.EPSFBuilder`.
Both were tested with the current builder and neither is implemented:

* A clipped mean or median of the samples around a grid point
  underestimates the ePSF where it has a local maximum, i.e., at its
  peak. They found much smaller errors by fitting a two-dimensional
  polynomial surface to the samples and evaluating it at the grid
  point. Their tests used known star positions and took the samples of
  a single pass from the grid cell nearest each grid point. That result
  was reproduced under the same conditions. It does not carry over to
  :class:`~photutils.psf.EPSFBuilder`, which combines the residuals of
  the samples from the current ePSF evaluated at the position of each
  sample. The curvature of the ePSF within the sampling box is then
  removed by the model, and the bias of the median vanishes as the
  iteration converges. In tests, the median and a quadratic surface fit
  converged to nearly the same ePSF. The surface fit was less accurate
  for undersampled ePSFs (FWHM below about 1.5 pixels) when the build
  stopped at its convergence criterion, and noisier for well-sampled
  ones. A least-squares fit is also not robust. For a sample that
  included cutouts without a star and stars with different shapes, the
  surface fit grew a checkerboard pattern in most of the builds, while
  the median did not in any of them.

* Interpolating the ePSF grid with a radial basis function with a cubic
  kernel was more accurate than a bicubic spline. In tests with analytic
  pixel-integrated profiles, its interpolation errors were 2 to 4 times
  smaller for oversampling factors of 3 or larger and not better for
  factors of 1 and 2. For an oversampling factor of 4 the interpolation
  error of the bicubic spline is already small. For an ePSF with a
  FWHM of 1 pixel it was at most 0.08 percent of the peak in the core,
  and the typical (RMS) error was about ten times smaller, 0.007
  percent. The interpolation is therefore not what limits the accuracy
  of an ePSF built from stars. In tests with a few hundred stars, the
  difference between the built ePSF and the true one was about ten times
  larger than the interpolation error (a maximum of 0.2 to 1 percent
  of the peak and an RMS of about 0.1 percent), so a more accurate
  interpolation would not noticeably improve the result. The radial
  basis function is also about two hundred times slower to evaluate.
  If the interpolation error of the bicubic spline is too large for an
  application, building the ePSF with a larger oversampling factor also
  reduces it, without slowing the evaluation of the ePSF.
