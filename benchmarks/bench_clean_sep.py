#!/usr/bin/env python3
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Validation and benchmark of get_spurious_labels against SEP cleaning.

For each scene, ``sep.extract`` is run twice on the same image,
without and with cleaning (``clean_param=1.0``). The segmentation map
of the uncleaned run defines the sources, and the labels whose pixels
vanish from the cleaned map are the SEP cleaning victims.
``get_spurious_labels`` is then run on that same segmentation map with
the same threshold, minimum area, and convolved image, and its labels
are compared with SEP's. Because both use SEP's segments, the
comparison isolates the cleaning test from any difference in detection
or deblending. Each scene is run with SEP deblending disabled (a single
threshold level) and enabled.

The scenes are:

* the blended Gaussian-pair field of ``bench_segmentation.py``
* a halo field: a few very bright stars with faint sources scattered
  in their wings, plus noise, which triggers many absorptions

Both scenes are run with the ``'moffat'`` wing model, which SEP
implements, and the ``'measured'`` wing model, which has no SEP
counterpart and is timed and counted alongside. The halo field's
stars are pure Gaussians, so a correct measured wing absorbs nothing
there, and its measured count records the crowding limit of the
estimator.

A third scene has ground truth: a noisy field with a de Vaucouleurs
galaxy and an exponential disk, real faint companions injected around
both, and spurious fragments produced by the noise on the bulge's
halo. It scores each wing model by the real companions and the other
detections it absorbs near each galaxy.

The per-source measurements agree with SEP to float32 precision, so
the two codes remove the same labels apart from two documented
differences. Near the decision boundary, SEP occasionally keeps a
source that photutils flags, because the SourceExtractor heap that
selects the minarea-th brightest pixel descends into the wrong node
during its sift-down and can leave a larger value at the root. When
three or more sources interact, SEP tests them in label order and
lets an already-absorbed source absorb later ones, whereas photutils
resolves the sources in order of decreasing flux and only lets
surviving sources absorb. The SEP cleaning cost is the difference
between its two runs.

Requires the optional ``sep`` package. Run ``python
benchmarks/bench_clean_sep.py --help`` to see the available options.
"""

import argparse
import sys
from functools import partial

import numpy as np
from astropy.convolution import convolve
from astropy.modeling.models import Gaussian2D, Sersic2D
from bench_helpers import print_environment, time_best
from bench_segmentation import N_PIXELS, THRESHOLD, make_inputs

from photutils.segmentation import (SegmentationImage, detect_sources,
                                    get_spurious_labels,
                                    make_2dgaussian_kernel)

try:
    import sep
    HAS_SEP = True
    SEP_IMPORT_ERROR = None
except ImportError as exc:
    HAS_SEP = False
    SEP_IMPORT_ERROR = str(exc)

CLEAN_PARAM = 1.0
HALO_THRESHOLD = 5.0
HALO_N_PIXELS = 5
WING_MODELS = ('moffat', 'measured')
GALAXY_N_PIXELS = 5


def make_halo_image(size, *, n_bright=5, n_wing=300, n_field=200, seed=0):
    """
    Return a noisy image of bright stars with faint sources in their
    wings.

    Parameters
    ----------
    size : int
        The image size. The image is ``(size, size)``.

    n_bright : int, optional
        The number of bright stars.

    n_wing : int, optional
        The number of faint sources placed within 40 pixels of a
        bright star.

    n_field : int, optional
        The number of faint sources placed anywhere in the image.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image, with unit Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    data = rng.normal(0.0, 1.0, (size, size))

    margin = 60
    bright_xy = rng.uniform(margin, size - margin, (n_bright, 2))
    for x, y in bright_xy:
        amplitude = rng.uniform(5000.0, 20000.0)
        data += Gaussian2D(amplitude, x, y, 3.0, 3.0)(xx, yy)

    def add_faint(x, y):
        """
        Add a faint Gaussian source to the image.

        Parameters
        ----------
        x, y : float
            The source position.
        """
        amplitude = rng.uniform(4.0, 15.0)
        sigma = rng.uniform(1.5, 2.5)
        data[:] += Gaussian2D(amplitude, x, y, sigma, sigma)(xx, yy)

    for _ in range(n_wing):
        x0, y0 = bright_xy[rng.integers(n_bright)]
        radius = rng.uniform(15.0, 40.0)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        add_faint(x0 + radius * np.cos(angle), y0 + radius * np.sin(angle))

    for _ in range(n_field):
        add_faint(*rng.uniform(margin, size - margin, 2))

    return data


def run_sep(data, threshold, n_pixels, kernel, *, deblend, clean):
    """
    Run sep.extract and return the segmentation map.

    Parameters
    ----------
    data : 2D `~numpy.ndarray`
        The unfiltered image.

    threshold : float
        The detection threshold, applied to the filtered image.

    n_pixels : int
        The minimum object area.

    kernel : 2D `~numpy.ndarray`
        The convolution kernel.

    deblend : bool
        Whether to deblend (32 levels) or use a single level.

    clean : bool
        Whether to clean.

    Returns
    -------
    segmap : 2D `~numpy.ndarray`
        The SEP segmentation map.
    """
    _, segmap = sep.extract(data, threshold, minarea=n_pixels,
                            filter_kernel=kernel, filter_type='conv',
                            deblend_nthresh=32 if deblend else 1,
                            deblend_cont=0.005, clean=clean,
                            clean_param=CLEAN_PARAM, segmentation_map=True)
    return segmap


def sep_removed_labels(segmap, segmap_clean):
    """
    Return the labels of ``segmap`` whose pixels are absent from the
    cleaned map.

    Parameters
    ----------
    segmap : 2D `~numpy.ndarray`
        The uncleaned SEP segmentation map.

    segmap_clean : 2D `~numpy.ndarray`
        The cleaned SEP segmentation map.

    Returns
    -------
    removed : 1D `~numpy.ndarray`
        The removed labels.
    """
    removed = np.unique(segmap[(segmap > 0) & (segmap_clean == 0)])
    # SEP drops whole objects, so every pixel of a removed label must
    # be absent from the cleaned map
    if np.any(segmap_clean[np.isin(segmap, removed)] > 0):
        msg = 'a SEP object was only partially removed'
        raise ValueError(msg)
    return removed


def compare_scene(name, data, threshold, n_pixels, kernel, *, repeats=3):
    """
    Validate and benchmark get_spurious_labels against SEP cleaning on
    one scene.

    Parameters
    ----------
    name : str
        The scene description.

    data : 2D `~numpy.ndarray`
        The unfiltered image.

    threshold : float
        The detection threshold.

    n_pixels : int
        The minimum object area.

    kernel : 2D `~numpy.ndarray`
        The convolution kernel.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).
    """
    kernel32 = np.asarray(kernel, dtype=np.float32)
    convolved_data = convolve(data, kernel)

    print(f'\n-- {name} --')
    print(f'{"variant":>12}{"objects":>9}{"SEP":>7}{"moffat":>8}'
          f'{"both":>7}{"SEP only":>10}{"moffat only":>13}{"measured":>10}'
          f'{"t_moffat":>10}{"t_measured":>12}{"t_SEP":>10}')
    for deblend in (False, True):
        sep_run = partial(run_sep, data, threshold, n_pixels, kernel32,
                          deblend=deblend)
        segmap = sep_run(clean=False)
        segmap_clean = sep_run(clean=True)
        removed_sep = sep_removed_labels(segmap, segmap_clean)

        segm = SegmentationImage(segmap.astype(int))
        removed = {}
        times = {}
        for wing_model in WING_MODELS:
            bench = partial(get_spurious_labels, data, segm, threshold,
                            n_pixels, convolved_data=convolved_data,
                            clean_param=CLEAN_PARAM, wing_model=wing_model)
            removed[wing_model] = np.asarray(bench()['label'])
            times[wing_model] = time_best(bench, repeats=repeats)

        both = np.intersect1d(removed_sep, removed['moffat'])
        sep_only = np.setdiff1d(removed_sep, removed['moffat'])
        moffat_only = np.setdiff1d(removed['moffat'], removed_sep)
        t_sep = (time_best(partial(sep_run, clean=True), repeats=repeats)
                 - time_best(partial(sep_run, clean=False),
                             repeats=repeats))

        variant = 'deblend' if deblend else 'no deblend'
        print(f'{variant:>12}{segm.n_labels:>9}{len(removed_sep):>7}'
              f'{len(removed["moffat"]):>8}{len(both):>7}{len(sep_only):>10}'
              f'{len(moffat_only):>13}{len(removed["measured"]):>10}'
              f'{f"{times["moffat"]:.4f}s":>10}'
              f'{f"{times["measured"]:.4f}s":>12}{f"{t_sep:.4f}s":>10}')


def make_galaxy_scene(size, *, kernel, seed=3):
    """
    Return a noisy field with two galaxies and injected companions.

    A de Vaucouleurs galaxy and an exponential disk are placed in unit
    Gaussian noise, and eight real faint companions are injected
    around each at 1.5 to 2.5 isophotal radii. The noise riding on the
    bulge's halo produces spurious detections on its own.

    Parameters
    ----------
    size : int
        The image size. The image is ``(size, size)``.

    kernel : 2D `~numpy.ndarray`
        The convolution kernel used for detection.

    seed : int, optional
        The random number generator seed.

    Returns
    -------
    data : 2D `~numpy.ndarray`
        The image.

    convolved_data : 2D `~numpy.ndarray`
        The convolved image.

    threshold : float
        The detection threshold, five times the convolved noise.

    centers : 2D `~numpy.ndarray`
        The ``(x, y)`` centers of the bulge and the disk.

    companions : 2D `~numpy.ndarray`
        The ``(x, y)`` positions of the injected companions.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    data = rng.normal(0.0, 1.0, (size, size))
    bulge = Sersic2D(amplitude=60, r_eff=12, n=4, x_0=0.28 * size,
                     y_0=0.5 * size, ellip=0.2, theta=0.5)
    disk = Sersic2D(amplitude=60, r_eff=18, n=1, x_0=0.72 * size,
                    y_0=0.5 * size, ellip=0.5, theta=1.0)
    data += bulge(xx, yy) + disk(xx, yy)

    companions = []
    for galaxy, r_iso in ((bulge, 40.0), (disk, 45.0)):
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            radius = r_iso * rng.uniform(1.5, 2.5)
            x = galaxy.x_0.value + radius * np.cos(angle)
            y = galaxy.y_0.value + radius * np.sin(angle)
            companions.append((x, y))
            data += Gaussian2D(rng.uniform(2.0, 2.6), x, y, 2.0, 2.0)(xx, yy)

    convolved_data = convolve(data, kernel)
    noise = np.std(convolve(rng.normal(0.0, 1.0, (size, size)), kernel))
    centers = np.array([(bulge.x_0.value, bulge.y_0.value),
                        (disk.x_0.value, disk.y_0.value)])
    return data, convolved_data, 5.0 * noise, centers, np.array(companions)


def compare_galaxy_scene(size, *, kernel, repeats=3, seed=3):
    """
    Score each wing model on the galaxy field.

    A detection within three pixels of an injected companion counts
    as real, the two largest segments are the galaxies, and every
    other detection within 120 pixels of a galaxy center counts as
    spurious. Each model is scored by the real and spurious detections
    it absorbs near each galaxy, and timed.

    Parameters
    ----------
    size : int
        The image size.

    kernel : 2D `~numpy.ndarray`
        The convolution kernel used for detection.

    repeats : int, optional
        The number of repeats for each timing (best time is kept).

    seed : int, optional
        The random number generator seed.
    """
    data, convolved_data, threshold, centers, companions = (
        make_galaxy_scene(size, kernel=kernel, seed=seed))
    segm = detect_sources(convolved_data, threshold, GALAXY_N_PIXELS)

    x = np.array([0.5 * (slc[1].start + slc[1].stop) for slc in segm.slices])
    y = np.array([0.5 * (slc[0].start + slc[0].stop) for slc in segm.slices])
    d_companion = np.hypot(x[:, None] - companions[None, :, 0],
                           y[:, None] - companions[None, :, 1]).min(axis=1)
    is_galaxy = np.zeros(segm.n_labels, dtype=bool)
    is_galaxy[np.argsort(np.asarray(segm.areas))[-2:]] = True
    is_real = (d_companion < 3) & ~is_galaxy
    is_other = ~is_real & ~is_galaxy
    near = [np.hypot(x - cx, y - cy) < 120 for cx, cy in centers]

    print(f'\n-- galaxy field, {size}x{size} image, {segm.n_labels} '
          f'detections, {is_real.sum()} companions recovered --')
    print(f'{"wing model":>12}{"bulge real":>12}{"bulge other":>13}'
          f'{"disk real":>11}{"disk other":>12}{"time":>10}')
    for wing_model in WING_MODELS:
        bench = partial(get_spurious_labels, data, segm, threshold,
                        GALAXY_N_PIXELS, convolved_data=convolved_data,
                        clean_param=CLEAN_PARAM, wing_model=wing_model)
        absorbed = np.isin(segm.labels, bench()['label'])
        t_best = time_best(bench, repeats=repeats)
        cells = []
        for near_galaxy in near:
            for kind in (is_real, is_other):
                selected = near_galaxy & kind
                cells.append(f'{(absorbed & selected).sum()}/'
                             f'{selected.sum()}')
        print(f'{wing_model:>12}{cells[0]:>12}{cells[1]:>13}{cells[2]:>11}'
              f'{cells[3]:>12}{f"{t_best:.4f}s":>10}')


def main():
    """
    Run the SEP cleaning comparison.
    """
    parser = argparse.ArgumentParser(
        description='Validate and benchmark get_spurious_labels against '
                    'SEP cleaning.')
    parser.add_argument('--n-sources', type=int, default=4000,
                        help='number of sources in the Gaussian-pair '
                             'field (default: %(default)s)')
    parser.add_argument('--halo-size', type=int, default=1500,
                        help='image size of the halo field '
                             '(default: %(default)s)')
    parser.add_argument('--galaxy-size', type=int, default=600,
                        help='image size of the galaxy field '
                             '(default: %(default)s)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='number of repeats per timing. The best '
                             'time is reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random number generator seed '
                             '(default: %(default)s)')
    args = parser.parse_args()

    print_environment()
    if not HAS_SEP:
        print(f'sep is not available ({SEP_IMPORT_ERROR}). Nothing to '
              'compare')
        sys.exit(1)
    print(f'sep {sep.__version__}')
    print('\n== get_spurious_labels versus SEP cleaning '
          f'(clean_param={CLEAN_PARAM}) ==')
    print('SEP/moffat/measured = labels removed by each, both = removed '
          'by SEP and moffat, t_SEP = SEP clean minus no-clean time')

    kernel = make_2dgaussian_kernel(3.0, size=5).array  # as in make_inputs

    data, _, segm = make_inputs(args.n_sources, seed=args.seed)
    sep.set_extract_pixstack(max(sep.get_extract_pixstack(), data.size))
    compare_scene(f'{args.n_sources} sources, {segm.n_labels} segments, '
                  f'{data.shape[0]}x{data.shape[1]} image',
                  data, THRESHOLD, N_PIXELS, kernel, repeats=args.repeats)

    data = make_halo_image(args.halo_size, seed=args.seed)
    sep.set_extract_pixstack(max(sep.get_extract_pixstack(), data.size))
    compare_scene(f'halo field, {data.shape[0]}x{data.shape[1]} image',
                  data, HALO_THRESHOLD, HALO_N_PIXELS, kernel,
                  repeats=args.repeats)

    print('\n== wing models on a galaxy field (absorbed / total) ==')
    compare_galaxy_scene(args.galaxy_size, kernel=kernel,
                         repeats=args.repeats)


if __name__ == '__main__':
    main()
