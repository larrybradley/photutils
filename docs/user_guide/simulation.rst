.. _simulation:

Simulation (`photutils.simulation`)
===================================

Introduction
------------

`photutils.simulation` provides tools for making simulated images with
known sources and for scoring source-detection algorithms on them. A
typical workflow is:

1. Make tables of source parameters, for example with
   :func:`~photutils.simulation.make_model_params`.
2. Render the sources into an image with
   :func:`~photutils.simulation.make_model_image` (point sources and
   other models that are already integrated over the pixels) or
   :func:`~photutils.simulation.make_convolved_model_image` (extended
   sources convolved with a PSF).
3. Add noise with :func:`~photutils.simulation.make_noise_image`.
4. Run a detection algorithm and score it against the input sources
   with :func:`~photutils.simulation.match_segments` or
   :func:`~photutils.simulation.match_sources`,
   :func:`~photutils.simulation.completeness`, and
   :func:`~photutils.simulation.false_detection_rate`.

The functions return plain arrays and tables, so they can be combined
freely in tests, documentation examples, and benchmarks. The optional
arguments of all of the functions are keyword-only.


Source Parameter Tables
-----------------------

:func:`~photutils.simulation.make_model_params` makes a table of random
source positions and parameters. Each keyword parameter is given as a
``(low, high)`` range and is sampled uniformly, or uniformly in log10
if it is named in ``log_params``. Flux and size distributions in a
completeness test usually need log sampling. The optional ``kind``
argument adds a column that labels the population::

    >>> from photutils.simulation import make_model_params
    >>> stars = make_model_params((512, 512), 5, flux=(5, 300),
    ...                           log_params=('flux',), kind='star', seed=1)
    >>> stars['x_0'].info.format = '.2f'  # optional format
    >>> stars['y_0'].info.format = '.2f'
    >>> stars['flux'].info.format = '.2f'
    >>> print(stars)
     id  x_0    y_0    flux  kind
    --- ------ ------ ------ ----
      1 262.05 349.84  72.72 star
      2 486.64 402.99  29.23 star
      3  73.81  98.11 174.26 star
      4 485.71 410.81  66.53 star
      5 159.66  97.96 137.96 star

All tables use ``x_0``, ``y_0``, and ``flux`` columns by default, so the
tables of different populations can be stacked with
`~astropy.table.vstack` and scored together. Populations that will be
stacked need different ``seed`` values, or their positions are
correlated.

Two functions make controlled scenes for the cases where source
detection is hardest:

* :func:`~photutils.simulation.make_pair_params` makes close pairs of
  sources at set separations and flux ratios. Each pair has two rows
  that share a ``pair_id`` value, and pairs are spaced so that they
  never interleave.
* :func:`~photutils.simulation.make_satellite_params` places sources
  around host sources, for example faint stars on bright galaxies. The
  distance from each host can be given in pixels or in units of the
  host's effective radius::

    >>> from photutils.simulation import (make_pair_params,
    ...                                   make_satellite_params)
    >>> pairs = make_pair_params((512, 512), 20, (2, 6), flux=(50, 200),
    ...                          flux_ratio=(0.1, 1), kind='pair', seed=2)
    >>> hosts = make_model_params((512, 512), 5, flux=(1e4, 1e5),
    ...                           r_eff=(10, 20), min_separation=100,
    ...                           border_size=60, seed=3)
    >>> on_hosts = make_satellite_params(hosts, 10, (0.5, 2.5),
    ...                                  radius_unit='r_eff',
    ...                                  flux=(5, 300), kind='star_on_gal',
    ...                                  seed=4)
    >>> print(len(pairs), len(on_hosts))
    40 50

The positions from :func:`~photutils.simulation.make_satellite_params`
are not limited to any image, so sources around a host near an edge can
fall outside the image.


Rendering Images
----------------

Every PSF in this subpackage must be integrated over the detector
pixels: an ePSF (`~photutils.psf.ImagePSF` or
`~photutils.psf.GriddedPSFModel`, for example made with
`~photutils.psf.make_epsf_from_psf`) or an analytic model whose name
ends in ``PRF``, such as `~photutils.psf.CircularGaussianPRF`. Models
whose names end in ``PSF`` are evaluated at a point and would miss the
pixel response, so the functions that take a ``psf_model`` reject them
(see :ref:`psf-terminology`).

Point Sources and Pixel-Integrated Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~photutils.simulation.make_model_image` renders one model per
source. By default, the model is evaluated at the pixel centers, which
is correct for ePSFs and PRFs. For a model that is not integrated over
the pixels, such as `~photutils.psf.GaussianPSF` used as a Gaussian
galaxy, set ``n_nodes`` to integrate each pixel with ``n_nodes`` x
``n_nodes`` Gauss-Legendre nodes. The 3 x 3 pixels around each source
center are refined further, because most of the integration error of a
cuspy profile comes from its center. The default of 4 nodes gives about
0.1% for typical galaxies. Use 8 or more for Sersic profiles with n >= 6
or an effective radius below 0.5 pixels:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    from photutils.psf import CircularGaussianPRF
    from photutils.simulation import (make_model_image, make_model_params,
                                      make_noise_image)

    shape = (200, 300)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    params = make_model_params(shape, 60, flux=(100, 2000),
                               log_params=('flux',), min_separation=8,
                               border_size=5, seed=1)
    data = make_model_image(shape, psf_model, params)
    data += make_noise_image(shape, mean=5.0, stddev=2.0, seed=2)

    fig, ax = plt.subplots()
    norm = simple_norm(data, 'sqrt', percent=99.5)
    ax.imshow(data, norm=norm, origin='lower')
    ax.set_title('Point sources')

Extended Sources Convolved With a PSF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~photutils.simulation.make_convolved_model_image` renders sources
such as galaxies convolved with an ePSF or PRF. Each source is
integrated over a grid of cells ``1 / oversample`` pixels wide, with the
same ``n_nodes`` rule and core refinement as above, then convolved with
the PSF model evaluated at the same grid offsets, and sampled at the
pixel centers. Because the PSF model already includes the pixel
response, no further averaging over pixels is needed.

`~photutils.simulation.SersicSource` is a Sersic profile set by its total
flux instead of its surface brightness. Its default bounding box is a
square of half-width ``10 * r_eff``, and flux beyond the cutout around
each source is lost:

======== ========================================
Sersic n Flux lost at the default bounding box
======== ========================================
1        0.0%
2        0.3%
4        3.9%
6        8.0%
======== ========================================

Set ``model_shape`` or ``bbox_factor`` to render a larger cutout. For a
Gaussian galaxy, use `~photutils.psf.GaussianPSF`, which is also a
flux-normalized model evaluated at a point:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import simple_norm
    from photutils.psf import CircularGaussianPRF
    from photutils.simulation import (SersicSource,
                                      make_convolved_model_image,
                                      make_model_params, make_noise_image)

    shape = (200, 300)
    psf_model = CircularGaussianPRF(fwhm=2.5)
    params = make_model_params(shape, 25, flux=(500, 5000), r_eff=(1, 6),
                               ellip=(0, 0.6), theta=(0, np.pi),
                               log_params=('flux', 'r_eff'),
                               min_separation=30, border_size=20, seed=1)
    data = make_convolved_model_image(shape, SersicSource(n=1), params,
                                      psf_model)
    data += make_noise_image(shape, mean=5.0, stddev=2.0, seed=2)

    fig, ax = plt.subplots()
    norm = simple_norm(data, 'sqrt', percent=99.5)
    ax.imshow(data, norm=norm, origin='lower')
    ax.set_title('Exponential galaxies convolved with a PRF')

The result is exact up to the integration over the cells, with one
caveat. A source much smaller than a cell lands in a single cell and
renders as the PSF model at that cell's center, so it is displaced by
up to half a cell. Averaged over positions, this broadens such a source
by a box of width ``1 / oversample``. Render point sources with
:func:`~photutils.simulation.make_model_image` instead. A point source
centered on a grid point renders identically with both functions.

Example Images
^^^^^^^^^^^^^^

:func:`~photutils.simulation.make_4gaussians_image` and
:func:`~photutils.simulation.make_100gaussians_image` make two fixed
example images that are used throughout the Photutils documentation.
:func:`~photutils.simulation.make_psf_model_image` makes an image of
randomly placed PSF sources and returns their parameter table.


Noise
-----

:func:`~photutils.simulation.make_noise_image` makes Gaussian or Poisson
noise images, and :func:`~photutils.simulation.apply_poisson_noise`
applies Poisson noise to an image of expected counts::

    >>> from photutils.simulation import make_noise_image
    >>> noise = make_noise_image((512, 512), mean=5.0, stddev=2.0, seed=0)

Resampled images, such as mosaics, have correlated noise: neighboring
pixels share noise, so the per-pixel standard deviation understates
the noise on larger scales. The ``kernel`` keyword makes correlated
Gaussian noise by convolving white noise with a kernel, while keeping
the requested per-pixel standard deviation.
:func:`~photutils.simulation.noise_correlation_factor` returns the
factor by which the standard deviation of a sum over a region much
larger than the kernel exceeds the value for uncorrelated noise with
the same per-pixel standard deviation::

    >>> from astropy.convolution import Gaussian2DKernel
    >>> from photutils.simulation import noise_correlation_factor
    >>> kernel = Gaussian2DKernel(1.5)
    >>> noise = make_noise_image((512, 512), mean=0.0, stddev=1.0,
    ...                          kernel=kernel, seed=0)
    >>> print(f'{noise_correlation_factor(kernel):.2f}')
    5.32


World Coordinate Systems
------------------------

:func:`~photutils.simulation.make_wcs` makes a simple celestial WCS for
an image::

    >>> from photutils.simulation import make_wcs
    >>> wcs = make_wcs((500, 500))
    >>> wcs.pixel_to_world(0, 0)
    <SkyCoord (ICRS): (ra, dec) in deg
    (197.8899676, -1.3750039)>

and :func:`~photutils.simulation.make_gwcs` makes the equivalent
generalized WCS:

.. doctest-requires:: gwcs

    >>> from photutils.simulation import make_gwcs
    >>> gwcs = make_gwcs((500, 500))
    >>> gwcs.pixel_to_world(0, 0)
    <SkyCoord (ICRS): (ra, dec) in deg
    (197.8899676, -1.3750039)>


.. _simulation-completeness:

Measuring Detection Completeness
--------------------------------

This example measures the completeness of a simple detection algorithm
on a scene of stars, exponential galaxies, and close pairs of stars.

First, make the source tables and render the image. The stars and pairs
are rendered with :func:`~photutils.simulation.make_model_image`, and the
galaxies with :func:`~photutils.simulation.make_convolved_model_image`::

    >>> import numpy as np
    >>> from astropy.table import vstack
    >>> from photutils.psf import CircularGaussianPRF
    >>> from photutils.simulation import (SersicSource,
    ...                                   make_convolved_model_image,
    ...                                   make_model_image,
    ...                                   make_model_params,
    ...                                   make_noise_image,
    ...                                   make_pair_params)
    >>> shape = (512, 512)
    >>> psf_model = CircularGaussianPRF(fwhm=2.0)
    >>> stars = make_model_params(shape, 200, flux=(5, 300),
    ...                           log_params=('flux',), min_separation=6,
    ...                           border_size=10, kind='star', seed=1)
    >>> galaxies = make_model_params(shape, 100, flux=(10, 1000),
    ...                              r_eff=(0.7, 4), ellip=(0, 0.6),
    ...                              theta=(0, np.pi),
    ...                              log_params=('flux', 'r_eff'),
    ...                              min_separation=20, border_size=20,
    ...                              kind='galaxy', seed=2)
    >>> pairs = make_pair_params(shape, 40, (2, 6), flux=(20, 200),
    ...                          flux_ratio=(0.3, 1), border_size=10,
    ...                          kind='pair', seed=3)
    >>> galaxy_model = SersicSource(n=1)
    >>> data = (make_model_image(shape, psf_model, stars)
    ...         + make_model_image(shape, psf_model, pairs)
    ...         + make_convolved_model_image(shape, galaxy_model, galaxies,
    ...                                      psf_model)
    ...         + make_noise_image(shape, mean=0.0, stddev=1.0, seed=4))
    >>> error = np.ones(shape)

Next, assign each source its optimal signal-to-noise ratio with
:func:`~photutils.simulation.optimal_snr`. This is the matched-filter
signal-to-noise ratio of the source rendered alone, the best that any
linear detector can reach for white noise, and the right quantity to
bin completeness by. The galaxies are rendered with the same PSF as in
the image. The tables are then stacked into one table of input sources::

    >>> from photutils.simulation import optimal_snr
    >>> stars['snr'] = optimal_snr(psf_model, stars, error)
    >>> pairs['snr'] = optimal_snr(psf_model, pairs, error)
    >>> galaxies['snr'] = optimal_snr(galaxy_model, galaxies, error,
    ...                               psf_model=psf_model)
    >>> truth = vstack([stars, galaxies, pairs])

The detection algorithm smooths the image with a Gaussian kernel
matched to the PSF and finds segments of at least 5 pixels above 3 times
the noise of the smoothed image::

    >>> from astropy.convolution import Gaussian2DKernel, convolve
    >>> from photutils.segmentation import detect_sources
    >>> kernel = Gaussian2DKernel(2.0 / 2.3548)
    >>> kernel.normalize()
    >>> smoothed_noise = np.sqrt(np.sum(kernel.array**2))
    >>> def detect(image):
    ...     smoothed = convolve(image, kernel)
    ...     return detect_sources(smoothed, threshold=3 * smoothed_noise,
    ...                           n_pixels=5)
    >>> segm = detect(data)

:func:`~photutils.simulation.match_segments` reads the segment label
under each input position and credits each segment to one source only,
the brightest one it covers. A source that lies in a segment credited to
another source is marked as blended::

    >>> from photutils.simulation import match_segments
    >>> result = match_segments(segm, truth)
    >>> for kind in ('star', 'galaxy', 'pair'):
    ...     sel = result['kind'] == kind
    ...     print(f'{kind}: {sel.sum()} sources, '
    ...           f'{result["detected"][sel].sum()} detected, '
    ...           f'{result["blended"][sel].sum()} blended')
    star: 200 sources, 143 detected, 16 blended
    galaxy: 100 sources, 81 detected, 4 blended
    pair: 80 sources, 36 detected, 44 blended

Then :func:`~photutils.simulation.completeness` gives the detected
fraction in bins of the optimal signal-to-noise ratio, with Wilson score
intervals that stay honest for small bins::

    >>> from photutils.simulation import completeness
    >>> sel = result['kind'] == 'star'
    >>> tbl = completeness(result['detected'][sel], result['snr'][sel],
    ...                    [2, 4, 6, 8, 12, 20, 50, 1000])
    >>> for col in ('completeness', 'lower', 'upper'):
    ...     tbl[col].info.format = '.2f'  # optional format
    >>> print(tbl)
    bin_min bin_max n_sources n_detected completeness lower upper
    ------- ------- --------- ---------- ------------ ----- -----
        2.0     4.0        34          4         0.12  0.07  0.18
        4.0     6.0        13          8         0.62  0.48  0.74
        6.0     8.0         7          5         0.71  0.53  0.85
        8.0    12.0        27         24         0.89  0.81  0.94
       12.0    20.0        29         28         0.97  0.91  0.99
       20.0    50.0        42         38         0.90  0.85  0.94
       50.0  1000.0        37         36         0.97  0.93  0.99

Finally, :func:`~photutils.simulation.false_detection_rate` runs the
same algorithm on pure-noise images, where every detection is false::

    >>> from photutils.simulation import false_detection_rate
    >>> fdr = false_detection_rate(detect, shape, n_trials=5, seed=5)
    >>> print(fdr['n_detections'].tolist())
    [3, 5, 5, 4, 6]

The completeness curves show that close pairs are detected far less
often than isolated stars of the same signal-to-noise ratio, because a
pair that is not deblended is detected as one source:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.convolution import Gaussian2DKernel, convolve
    from astropy.table import vstack
    from photutils.psf import CircularGaussianPRF
    from photutils.segmentation import detect_sources
    from photutils.simulation import (SersicSource, completeness,
                                      make_convolved_model_image,
                                      make_model_image, make_model_params,
                                      make_noise_image, make_pair_params,
                                      match_segments, optimal_snr)

    shape = (512, 512)
    psf_model = CircularGaussianPRF(fwhm=2.0)
    stars = make_model_params(shape, 200, flux=(5, 300),
                              log_params=('flux',), min_separation=6,
                              border_size=10, kind='star', seed=1)
    galaxies = make_model_params(shape, 100, flux=(10, 1000),
                                 r_eff=(0.7, 4), ellip=(0, 0.6),
                                 theta=(0, np.pi),
                                 log_params=('flux', 'r_eff'),
                                 min_separation=20, border_size=20,
                                 kind='galaxy', seed=2)
    pairs = make_pair_params(shape, 40, (2, 6), flux=(20, 200),
                             flux_ratio=(0.3, 1), border_size=10,
                             kind='pair', seed=3)
    galaxy_model = SersicSource(n=1)
    data = (make_model_image(shape, psf_model, stars)
            + make_model_image(shape, psf_model, pairs)
            + make_convolved_model_image(shape, galaxy_model, galaxies,
                                         psf_model)
            + make_noise_image(shape, mean=0.0, stddev=1.0, seed=4))
    error = np.ones(shape)

    stars['snr'] = optimal_snr(psf_model, stars, error)
    pairs['snr'] = optimal_snr(psf_model, pairs, error)
    galaxies['snr'] = optimal_snr(galaxy_model, galaxies, error,
                                  psf_model=psf_model)
    truth = vstack([stars, galaxies, pairs])

    kernel = Gaussian2DKernel(2.0 / 2.3548)
    kernel.normalize()
    smoothed = convolve(data, kernel)
    segm = detect_sources(smoothed,
                          threshold=3 * np.sqrt(np.sum(kernel.array**2)),
                          n_pixels=5)
    result = match_segments(segm, truth)

    bins = [2, 4, 6, 8, 12, 20, 50, 1000]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (('star', 'o'), ('galaxy', 's'), ('pair', '^')):
        sel = result['kind'] == kind
        tbl = completeness(result['detected'][sel], result['snr'][sel],
                           bins)
        tbl = tbl[tbl['n_sources'] > 0]  # skip empty bins
        center = np.sqrt(tbl['bin_min'] * tbl['bin_max'])
        yerr = np.clip([tbl['completeness'] - tbl['lower'],
                        tbl['upper'] - tbl['completeness']], 0, None)
        ax.errorbar(center, tbl['completeness'], yerr=yerr, marker=marker,
                    capsize=3, label=kind)
    ax.set_xscale('log')
    ax.set_xlabel('Optimal signal-to-noise ratio')
    ax.set_ylabel('Completeness')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()


Migrating from photutils.datasets
---------------------------------

The functions of the deprecated ``photutils.datasets`` subpackage, and
``photutils.psf.make_psf_model_image``, are deprecated in version 3.2
and will be removed in version 4.0. The functions of the same name in
`photutils.simulation` differ in three ways:

* All optional arguments are keyword-only.
* :func:`~photutils.simulation.make_model_image` has an ``n_nodes``
  keyword in place of ``discretize_method`` and
  ``discretize_oversample``.
* :func:`~photutils.simulation.make_psf_model_image` raises an error for
  a PSF model whose name ends in ``PSF`` (except
  `~photutils.psf.ImagePSF`), since such a model is not integrated over
  the pixels.

The ``load_*`` functions and ``get_path`` have no replacement.


API Reference
-------------

:doc:`../reference/simulation_api`
