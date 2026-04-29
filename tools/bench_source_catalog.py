"""
Cold-cache benchmark for every public ``SourceCatalog`` property and
several heavy methods.

Builds a synthetic 4096x4096 image with ~20,000 sources and times
each property on a *fresh* catalog so cached intermediate results
do not contaminate the timing.

Run as::

    python tools/bench_source_catalog.py

This is a developer tool used to guide performance work on
``photutils.segmentation.SourceCatalog``; it is not part of the
test suite.
"""
from __future__ import annotations

import gc
import time
import warnings

import numpy as np
from scipy.ndimage import label as scipy_label

from photutils.segmentation import SegmentationImage, SourceCatalog


def make_catalog_inputs(n_targets=20_000, shape=(4096, 4096), seed=0):
    """
    Build a synthetic image, error map and segmentation.

    Parameters
    ----------
    n_targets : int
        Approximate number of synthetic Gaussian sources to inject.
    shape : tuple of int
        Shape of the output image.
    seed : int
        Seed for the random-number generator.

    Returns
    -------
    data : `~numpy.ndarray`
        The simulated image.
    error : `~numpy.ndarray`
        Per-pixel 1-sigma error map.
    segm : `~photutils.segmentation.SegmentationImage`
        Segmentation image of the detections.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=float)
    n = int(np.sqrt(n_targets)) + 1
    xs = np.linspace(20, shape[1] - 20, n)
    ys = np.linspace(20, shape[0] - 20, n)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    centers = np.column_stack([yy.ravel(), xx.ravel()])[:n_targets]
    bright = rng.uniform(50, 500, size=len(centers))
    sigma = rng.uniform(1.5, 3.5, size=len(centers))
    yy_g, xx_g = np.indices((11, 11))
    for (y, x), b, s in zip(centers, bright, sigma, strict=True):
        yi, xi = round(y), round(x)
        ys0, ys1 = yi - 5, yi + 6
        xs0, xs1 = xi - 5, xi + 6
        g = b * np.exp(-((yy_g - 5) ** 2 + (xx_g - 5) ** 2) / (2 * s * s))
        data[ys0:ys1, xs0:xs1] += g
    data += rng.normal(0, 1.0, size=shape)
    error = np.full_like(data, 1.0)
    threshold = 5.0
    labels, _ = scipy_label(data > threshold)
    return data, error, SegmentationImage(labels)


PROPERTIES = (
    'segment_area', 'area', 'equivalent_radius', 'perimeter',
    'data_cutout', 'data_cutout_masked', 'segment_cutout',
    'segment_cutout_masked', 'error_cutout', 'error_cutout_masked',
    'conv_data_cutout', 'conv_data_cutout_masked',
    'min_value', 'max_value',
    'cutout_min_value_index', 'cutout_max_value_index',
    'min_value_index', 'max_value_index',
    'min_value_xindex', 'min_value_yindex',
    'max_value_xindex', 'max_value_yindex',
    'segment_flux', 'segment_flux_err',
    'moments', 'moments_central',
    'cutout_centroid', 'centroid', 'x_centroid', 'y_centroid',
    'centroid_quad', 'cutout_centroid_quad',
    'x_centroid_quad', 'y_centroid_quad',
    'centroid_win', 'cutout_centroid_win',
    'x_centroid_win', 'y_centroid_win',
    'inertia_tensor', 'covariance', 'covariance_eigvals',
    'semimajor_axis', 'semiminor_axis', 'fwhm',
    'orientation', 'eccentricity', 'elongation', 'ellipticity',
    'covariance_xx', 'covariance_yy', 'covariance_xy',
    'ellipse_cxx', 'ellipse_cyy', 'ellipse_cxy',
    'gini',
    'kron_radius', 'kron_aperture', 'kron_flux', 'kron_flux_err',
)

METHODS = (
    ('flux_radius_0.5', lambda c: c.flux_radius(0.5)),
    ('flux_radius_0.9', lambda c: c.flux_radius(0.9)),
    ('make_circular_apertures_5', lambda c: c.make_circular_apertures(5.0)),
    ('circular_photometry_5',
     lambda c: c.circular_photometry(5.0, name='c5')),
)


def _time(data, error, segm, fn, repeats):
    """
    Time ``fn(cat)`` on a fresh catalog.

    Parameters
    ----------
    data, error : `~numpy.ndarray`
        Image and error inputs for the catalog.
    segm : `~photutils.segmentation.SegmentationImage`
        Segmentation image.
    fn : callable
        Callable taking a fresh ``SourceCatalog`` and triggering the
        property/method to be measured.
    repeats : int
        Number of timing repeats; the minimum is returned.

    Returns
    -------
    float
        Best (minimum) runtime in seconds.
    """
    times = []
    for _ in range(repeats):
        gc.collect()
        cat = SourceCatalog(data, segm, error=error)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn(cat)
        times.append(time.perf_counter() - t0)
    return min(times)


def main(repeats=1):
    """
    Run the cold-cache benchmark and print the results.

    Parameters
    ----------
    repeats : int
        Number of fresh-catalog timings per property; the minimum
        runtime across repeats is reported.
    """
    print('Building 4096 x 4096 synthetic image ...')
    t0 = time.perf_counter()
    data, error, segm = make_catalog_inputs()
    print(f'  build={time.perf_counter() - t0:.2f}s, '
          f'n_labels={segm.n_labels}\n')

    rows = []
    for name in PROPERTIES:
        t = _time(data, error, segm,
                  lambda c, n=name: getattr(c, n), repeats)
        rows.append((name, t))
        print(f'  {name:35s} {t * 1000:8.1f} ms')

    print()
    for name, fn in METHODS:
        t = _time(data, error, segm, fn, repeats)
        rows.append((name, t))
        print(f'  {name:35s} {t * 1000:8.1f} ms')

    rows.sort(key=lambda r: r[1], reverse=True)
    print('\nTop 15 slowest:\n')
    for name, t in rows[:15]:
        print(f'  {name:35s} {t * 1000:8.1f} ms')

    print(f'\nTotal of measured properties+methods: '
          f'{sum(t for _, t in rows):.2f} s')


if __name__ == '__main__':
    main()
