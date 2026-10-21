.. _source_detection:

Point-like Source Detection (`photutils.detection`)
===================================================

Introduction
------------

One generally needs to identify astronomical sources in the data before
performing photometry or other measurements. The `photutils.detection`
subpackage provides tools to detect point-like (stellar) sources in an
image. This subpackage also provides tools to find local peaks in an
image that are above a specified threshold value.

For general-use source detection and extraction of both point-like
and extended sources, please see :ref:`Image Segmentation
<image_segmentation>`.


Detecting Stars
---------------

Photutils includes two widely used tools for detecting stars in an
image, `DAOFIND`_ and IRAF's `starfind`_, plus a third tool that allows
input of a custom user-defined kernel.

:class:`~photutils.detection.DAOStarFinder` implements
the `DAOFIND`_ algorithm (`Stetson 1987, PASP 99, 191
<https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_).
It searches images for local density maxima that have a peak amplitude
above a specified threshold (applied to a convolved image) and with
size and shape similar to a defined 2D Gaussian kernel. To match the
original DAOFIND algorithm, the input ``threshold`` is internally scaled
by a factor derived from the convolution kernel. To apply the threshold
exactly as given (e.g., when supplying a spatial background-RMS map),
set ``scale_threshold=False``. The class also computes roundness and
sharpness statistics for detected sources, with configurable lower and
upper bounds.

:class:`~photutils.detection.IRAFStarFinder` is a class that
implements IRAF's `starfind`_ algorithm. It is fundamentally
similar to :class:`~photutils.detection.DAOStarFinder`, but
:class:`~photutils.detection.IRAFStarFinder` always uses a circular
Gaussian kernel whereas :class:`~photutils.detection.DAOStarFinder`
can use an elliptical Gaussian kernel. Another difference is that
:class:`~photutils.detection.IRAFStarFinder` calculates the objects'
centroid, roundness, and sharpness using image moments.

:class:`~photutils.detection.StarFinder` is a class similar to
:class:`~photutils.detection.IRAFStarFinder`, but which allows input
of a custom user-defined kernel as a 2D array. This allows for more
generalization beyond simple Gaussian kernels.

:class:`~photutils.detection.MatchedFilterStarFinder` takes a
different approach. It fits the flux of a point source at every pixel
and detects the peaks of the resulting signal-to-noise ratio image.
See :ref:`matched_filter_star_finder` below.

The usage of :class:`~photutils.detection.IRAFStarFinder` and
:class:`~photutils.detection.StarFinder` follows the same pattern
as :class:`~photutils.detection.DAOStarFinder` shown below. Replace
the class name and adjust the parameters (e.g., ``fwhm`` and
``kernel``) as needed. Note that the ``scale_threshold`` parameter
is specific to :class:`~photutils.detection.DAOStarFinder`.
Note also that each class returns different output columns. For
example, :class:`~photutils.detection.DAOStarFinder` includes
``roundness1``, ``roundness2``, and ``daofind_mag`` columns, while
:class:`~photutils.detection.IRAFStarFinder` includes ``fwhm`` and
``orientation`` columns. See each class's API documentation for the full
list of output columns.

As an example, let's load a simulated HST star image and add Gaussian
noise. We will then estimate the background and background noise using
sigma-clipped statistics::

    >>> import numpy as np
    >>> from astropy.io import fits
    >>> from astropy.stats import sigma_clipped_stats
    >>> from photutils.datasets import make_noise_image
    >>> url = ('https://github.com/astropy/photutils-datasets/raw/main/'
    ...        'data/hst_wfc3ir_f160w_simulated_starfield.fits')
    >>> data = fits.getdata(url)  # doctest: +REMOTE_DATA
    >>> data += make_noise_image(data.shape, distribution='gaussian',
    ...                          mean=10.0, stddev=5.0, seed=0)  # doctest: +REMOTE_DATA
    >>> mean, median, std = sigma_clipped_stats(data, sigma=3.0)  # doctest: +REMOTE_DATA
    >>> print(np.array((mean, median, std)))  # doctest: +REMOTE_DATA
    [10.44410657 10.39699777  5.09141794]

Now we will subtract the background and use an instance of
:class:`~photutils.detection.DAOStarFinder` to find the stars in the
image, using a Gaussian kernel with a FWHM of 2.5 pixels, that have
peaks approximately 5 times the background standard deviation above the
background (i.e., the threshold is ``5 * std``). The stars in the image
are undersampled, so we will slightly relax the ``sharpness_range`` to
allow for a wider range of values.

First, we create the star finder. Calling it on the data returns an
astropy `~astropy.table.QTable` containing the results of the star
finder::

    >>> from photutils.detection import DAOStarFinder
    >>> threshold = 5.0 * std  # doctest: +REMOTE_DATA
    >>> daofind = DAOStarFinder(threshold, fwhm=2.5,
    ...                         sharpness_range=(0.2, 1.5))  # doctest: +REMOTE_DATA

By default, :class:`~photutils.detection.DAOStarFinder` internally
scales the input threshold by a factor derived from the convolution
kernel to match the original `DAOFIND`_ algorithm. To apply
the threshold exactly as given (e.g., when supplying a spatial
background-RMS map), set ``scale_threshold=False``::

    >>> daofind_unscaled = DAOStarFinder(threshold, fwhm=2.5,
    ...                                  sharpness_range=(0.2, 1.5),
    ...                                  scale_threshold=False)  # doctest: +REMOTE_DATA

Running the finder on the background-subtracted data::

    >>> sources = daofind(data - median)  # doctest: +REMOTE_DATA
    >>> for col in sources.colnames:  # doctest: +REMOTE_DATA
    ...     if col not in ('id', 'n_pixels'):
    ...         sources[col].info.format = '%.2f'  # for consistent table output
    >>> sources.pprint(max_lines=12, max_width=76)  # doctest: +REMOTE_DATA
     id x_centroid y_centroid sharpness ...   peak    flux   mag  daofind_mag
    --- ---------- ---------- --------- ... ------- ------- ----- -----------
      1     848.57       2.15      0.89 ... 1051.78 3391.76 -8.83       -3.80
      2     181.85       3.75      0.97 ... 1711.87 4823.64 -9.21       -4.28
      3     323.88       3.70      0.96 ... 3005.97 8627.30 -9.84       -4.90
      4      99.89       8.95      1.07 ... 1134.12 2789.54 -8.61       -3.77
    ...        ...        ...       ... ...     ...     ...   ...         ...
    497     114.16     993.47      0.84 ... 1577.91 5697.73 -9.39       -4.26
    498     298.44     993.87      0.83 ...  644.97 2282.34 -8.40       -3.31
    499     207.21     998.15      0.97 ... 2800.62 7693.48 -9.72       -4.83
    500     691.03     998.77      1.15 ... 2600.83 5132.75 -9.28       -4.64
    Length = 500 rows

Let's plot the image and mark the location of detected sources:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_noise_image
    from photutils.detection import DAOStarFinder

    url = ('https://github.com/astropy/photutils-datasets/raw/main/'
           'data/hst_wfc3ir_f160w_simulated_starfield.fits')
    data = fits.getdata(url)
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = 5.0 * std
    daofind = DAOStarFinder(threshold, fwhm=2.5, sharpness_range=(0.2, 1.5))
    sources = daofind(data - median)
    positions = np.transpose((sources['x_centroid'], sources['y_centroid']))
    apertures = CircularAperture(positions, r=10.0)
    norm = simple_norm(data, 'sqrt', percent=99)
    fig, ax = plt.subplots()
    axim = ax.imshow(data, norm=norm, origin='lower')
    patches = apertures.plot(ax=ax, color='red')


Masking Regions
^^^^^^^^^^^^^^^

Regions of the input image can be masked by using the ``mask``
keyword with the :class:`~photutils.detection.DAOStarFinder`,
:class:`~photutils.detection.IRAFStarFinder`, or
:class:`~photutils.detection.StarFinder` instance. This simple example
uses :class:`~photutils.detection.DAOStarFinder` and masks two
rectangular regions. No sources will be detected in the masked regions:

.. doctest-skip::

   >>> from photutils.detection import DAOStarFinder
   >>> daofind = DAOStarFinder(threshold, fwhm=2.5, sharpness_range=(0.2, 1.5))
   >>> mask = np.zeros(data.shape, dtype=bool)
   >>> mask[650:851, 600:851] = True
   >>> mask[250:451, 150:551] = True
   >>> sources = daofind(data - median, mask=mask)

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    from photutils.aperture import CircularAperture, RectangularAperture
    from photutils.datasets import make_noise_image
    from photutils.detection import DAOStarFinder

    url = ('https://github.com/astropy/photutils-datasets/raw/main/'
           'data/hst_wfc3ir_f160w_simulated_starfield.fits')
    data = fits.getdata(url)
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = 5.0 * std
    daofind = DAOStarFinder(threshold, fwhm=2.5, sharpness_range=(0.2, 1.5))
    mask = np.zeros(data.shape, dtype=bool)
    mask[650:851, 600:851] = True
    mask[250:451, 150:551] = True
    sources = daofind(data - median, mask=mask)
    positions = np.transpose((sources['x_centroid'], sources['y_centroid']))
    apertures = CircularAperture(positions, r=10.0)
    fig, ax = plt.subplots()
    norm = simple_norm(data, 'sqrt', percent=99)
    axim = ax.imshow(data, norm=norm, origin='lower')
    ax.set_title('Star finder with a mask to exclude regions')
    p1 = apertures.plot(ax=ax, color='red')
    rect1 = RectangularAperture((725, 750), 250, 200, theta=0)
    rect2 = RectangularAperture((350, 350), 400, 200, theta=0)
    p2 = rect1.plot(ax=ax, color='white', ls='dashed')
    p3 = rect2.plot(ax=ax, color='white', ls='dashed')


.. _matched_filter_star_finder:

Matched-Filter Star Finder
^^^^^^^^^^^^^^^^^^^^^^^^^^

The star finders above convolve the image with a kernel that sums
to zero, which is equivalent to fitting the height of the kernel
plus a constant local background at every pixel. That fit uses only
the pixels within a small kernel footprint, so the fitted height is
much noisier than the data allow. The ``threshold`` of those finders is
also in the units of the data, and the significance it corresponds to
depends on the kernel.

:class:`~photutils.detection.MatchedFilterStarFinder` instead fits
the flux of a point source at every pixel by weighted least squares,
using the input ``kernel`` as the point spread function (PSF) and the
inverse variance of each pixel as its weight. Stars are the local
maxima of the resulting signal-to-noise ratio (S/N) image. The main
differences from the other star finders are:

* The ``threshold`` is the minimum S/N of the fitted flux. Its meaning
  does not depend on the width of the kernel or on the noise level of
  the image.

* It can detect significantly fainter stars at the same rate of false
  detections, because the fit uses all of the pixels within the
  ``kernel`` array.

* It accepts an ``error`` array. Pixels with a larger uncertainty have
  a smaller weight, and the threshold is relative to the local noise.

* Masked pixels, nonfinite pixels, and pixels beyond the image borders
  are excluded from the fit. No values are assumed for the pixels
  beyond the image borders.

* The output table includes the fitted ``flux`` of each source, its
  uncertainty, and its S/N.

The ``kernel`` should be the PSF of the image as a 2D array, and it
should be large enough to include most of the PSF flux. By default,
a constant local background is fitted along with the flux at every
pixel (``local_bkg=True``), so that the detection does not depend on the
background level.

As an example, let's create a simulated image of 30 stars with a
Gaussian PSF that has a FWHM of 3 pixels, and add Gaussian noise with a
standard deviation of 2::

    >>> from photutils.datasets import make_noise_image
    >>> from photutils.psf import CircularGaussianPRF, make_psf_model_image
    >>> psf_model = CircularGaussianPRF(fwhm=3.0)
    >>> data, true_params = make_psf_model_image(
    ...     (201, 201), psf_model, 30, flux=(40, 600), min_separation=15,
    ...     border_size=15, seed=0)
    >>> data += make_noise_image(data.shape, mean=0.0, stddev=2.0, seed=0)

We define the kernel as a Gaussian with the same FWHM and find the stars
with a fitted flux that has a S/N of at least 5. The ``error`` input is
the background noise of the image. It can be either a scalar or an array
with the same shape as the data. If it is not input, a constant noise
level is estimated from the data::

    >>> from photutils.detection import MatchedFilterStarFinder
    >>> from photutils.segmentation import make_2dgaussian_kernel
    >>> kernel = make_2dgaussian_kernel(3.0, size=11)
    >>> finder = MatchedFilterStarFinder(5.0, kernel)
    >>> sources = finder(data, error=2.0)
    >>> for col in sources.colnames:
    ...     if col != 'id':
    ...         sources[col].info.format = '%.2f'  # for consistent table output
    >>> sources.pprint(max_lines=12, max_width=76)
     id x_centroid y_centroid  flux  ... local_bkg reduced_chi2 peak_excess
    --- ---------- ---------- ------ ... --------- ------------ -----------
      1     138.33      19.11 432.46 ...      0.08         1.15        1.71
      2      21.96      21.86 420.62 ...     -0.04         0.92        2.44
      3     125.61      29.69 458.88 ...      0.06         1.10        1.29
      4      45.00      36.87 280.88 ...     -0.03         1.11        1.68
    ...        ...        ...    ... ...       ...          ...         ...
     25     139.78     154.11 428.01 ...     -0.41         1.13        1.53
     26     171.13     156.27 356.66 ...     -0.07         0.91        0.76
     27      87.32     162.53 441.82 ...     -0.10         1.11        1.65
     28      68.10     166.10 548.58 ...     -0.02         1.28        1.10
    Length = 28 rows

The two stars that were not found have a S/N less than 5. For
comparison, :class:`~photutils.detection.DAOStarFinder` with the
equivalent ``threshold`` of 5 times the noise finds 23 of the 30
stars::

    >>> from photutils.detection import DAOStarFinder
    >>> daofind = DAOStarFinder(5.0 * 2.0, fwhm=3.0)
    >>> len(daofind(data))
    23

Single-pixel artifacts such as cosmic rays and hot pixels can be
rejected with the ``max_peak_excess`` keyword. The ``peak_excess``
of a source is its largest positive residual from the fit within one
pixel of its peak, in units of the pixel uncertainty. For this test,
the pixel uncertainty should include the Poisson noise of the sources.
Otherwise, bright stars can also be rejected.

Please note the following limitations:

* If ``local_bkg=False``, the image must be accurately
  background-subtracted. A background residual that is a small fraction
  of the pixel noise can then produce spurious sources.

* The S/N is overestimated for images with correlated noise (e.g.,
  resampled images), unless it is corrected as described below.

* The positions and fluxes of stars that are truncated by the image
  border are less accurate. Set ``exclude_border=True`` to exclude them.

When the noise of neighboring pixels is correlated, the uncertainty of
the fitted flux is underestimated. The ``error_scale`` keyword
multiplies the uncertainty by a factor to correct it, and the S/N
image is divided by the same factor before the ``threshold`` is
applied. The factor can be input if it is known (e.g., from the
parameters used to resample the image). With
``error_scale='empirical'``, the factor is estimated from the S/N
image itself as its sigma-clipped standard deviation, so that the S/N
of the noise has a standard deviation of 1. The estimate assumes that
most pixels contain only noise. Pixels that contain sources can be
left out of it with the ``source_mask`` keyword of ``find_stars``::

    >>> finder = MatchedFilterStarFinder(5.0, kernel,
    ...                                  error_scale='empirical')
    >>> sources = finder(data, error=2.0)

A single factor is appropriate if the noise correlation is the same
over the image. If it varies over the image, the ``error`` array can
instead be multiplied by a map of the correction factor.


Local Peak Detection
--------------------

Photutils also includes a :func:`~photutils.detection.find_peaks`
function to find local peaks in an image that are above a specified
threshold value. Peaks are the local maxima above a specified threshold
that are separated by a specified minimum number of pixels, defined by a
box size or a local footprint.

The returned pixel coordinates for the peaks are always integer-valued
(i.e., no centroiding is performed, only the peak pixel is identified).
However, a centroiding function can be input via the ``centroid_func``
keyword to :func:`~photutils.detection.find_peaks` to also compute
centroid coordinates with subpixel precision.

The ``box_size`` parameter also effectively imposes a minimum separation
between detected peaks, since only one peak can be found within each box
of that size. Specifically, two peaks must be separated by at least
``box_size // 2 + 1`` pixels along at least one axis. For example, with
a ``box_size`` of 11, two peaks must be at least 6 pixels apart along
the x- or y-axis.

As a simple example, let's find the local peaks in the image above that
are 5 sigma above the background using a box size of 11 pixels::

    >>> from photutils.detection import find_peaks
    >>> threshold = median + (5.0 * std)  # doctest: +REMOTE_DATA
    >>> tbl = find_peaks(data, threshold, box_size=11)  # doctest: +REMOTE_DATA
    >>> tbl['peak_value'].info.format = '%.8g'  # doctest: +REMOTE_DATA
    >>> print(tbl)  # doctest: +REMOTE_DATA
     id x_peak y_peak peak_value
    --- ------ ------ ----------
      1    849      2  1062.1752
      2    182      4  1722.2687
      3    324      4  3016.3684
      4    100      9  1144.5217
      5    824      9  1311.2049
    ...    ...    ...        ...
    497    889    992  194.27323
    498    114    994  1588.3073
    499    299    994  655.36699
    500    207    998  2811.0195
    501    691    999  2611.2233
    Length = 501 rows

Let's plot the location of the detected peaks in the image:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import simple_norm
    from photutils.aperture import CircularAperture
    from photutils.datasets import make_noise_image
    from photutils.detection import find_peaks

    url = ('https://github.com/astropy/photutils-datasets/raw/main/'
           'data/hst_wfc3ir_f160w_simulated_starfield.fits')
    data = fits.getdata(url)
    data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                             stddev=5.0, seed=0)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    threshold = median + (5.0 * std)
    tbl = find_peaks(data, threshold, box_size=11)
    positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    apertures = CircularAperture(positions, r=10.0)
    fig, ax = plt.subplots()
    norm = simple_norm(data, 'sqrt', percent=99)
    axim = ax.imshow(data, norm=norm, origin='lower')
    patches = apertures.plot(color='red')


API Reference
-------------

:doc:`../reference/detection_api`


.. _DAOFIND: https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/daofind.html
.. _starfind: https://iraf.readthedocs.io/en/latest/tasks/images/imcoords/starfind.html
