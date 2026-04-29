"""
Cold-cache benchmark for `~photutils.aperture.PixelAperture.do_photometry`
across every concrete pixel-aperture type.

Builds a synthetic 4096x4096 image and times ``do_photometry`` for
each aperture type with several methods (``exact``, ``center``,
``subpixel``) and with/without an ``error`` array.

Run as::

    python tools/bench_aperture.py

This is a developer tool used to guide performance work on
``photutils.aperture``; it is not part of the test suite.
"""
from __future__ import annotations

import gc
import time

import numpy as np

from photutils.aperture import (CircularAnnulus, CircularAperture,
                                EllipticalAnnulus, EllipticalAperture,
                                RectangularAnnulus, RectangularAperture)


def make_inputs(n_positions=10_000, shape=(4096, 4096), seed=0):
    """
    Build a synthetic image, error map, and aperture-position array.

    Parameters
    ----------
    n_positions : int
        Number of aperture positions.
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
    positions : `~numpy.ndarray`
        ``(n_positions, 2)`` array of (x, y) aperture positions.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(np.float64)
    error = np.abs(rng.standard_normal(shape)).astype(np.float64)
    positions = rng.uniform(50, shape[0] - 50, size=(n_positions, 2))
    return data, error, positions


# Each entry is (name, factory, method).  ``factory`` takes the
# positions array and returns a fresh aperture instance.
APERTURE_CASES = (
    ('CircularAperture(r=3)',
     lambda p: CircularAperture(p, r=3.0), 'exact'),
    ('CircularAperture(r=5)',
     lambda p: CircularAperture(p, r=5.0), 'exact'),
    ('CircularAperture(r=10)',
     lambda p: CircularAperture(p, r=10.0), 'exact'),
    ('CircularAperture(r=5) center',
     lambda p: CircularAperture(p, r=5.0), 'center'),
    ('CircularAperture(r=5) subpix5',
     lambda p: CircularAperture(p, r=5.0), 'subpixel'),
    ('CircularAnnulus(5,8)',
     lambda p: CircularAnnulus(p, r_in=5.0, r_out=8.0), 'exact'),
    ('EllipticalAperture(a=8,b=5)',
     lambda p: EllipticalAperture(p, a=8.0, b=5.0, theta=0.3), 'exact'),
    ('EllipticalAnnulus(5,8,5)',
     lambda p: EllipticalAnnulus(p, a_in=5.0, a_out=8.0, b_out=5.0,
                                 theta=0.3), 'exact'),
    ('RectangularAperture(10,10)',
     lambda p: RectangularAperture(p, w=10.0, h=10.0, theta=0.3), 'exact'),
    ('RectangularAnnulus',
     lambda p: RectangularAnnulus(p, w_in=5.0, w_out=10.0, h_out=10.0,
                                  theta=0.3), 'exact'),
    ('RectangularAperture(10,10) center',
     lambda p: RectangularAperture(p, w=10.0, h=10.0, theta=0.3), 'center'),
)


def _time(fn, repeats):
    """
    Time ``fn()`` ``repeats`` times and return the best wall-clock
    runtime in seconds.
    """
    fn()  # warm any one-shot lazy state on the aperture instance
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main(repeats=3):
    """
    Run the aperture-photometry benchmark and print results.

    Parameters
    ----------
    repeats : int
        Number of timing repeats; the minimum runtime across repeats
        is reported.
    """
    print('Building 4096 x 4096 synthetic image ...')
    t0 = time.perf_counter()
    data, error, positions = make_inputs()
    print(f'  build={time.perf_counter() - t0:.2f}s, '
          f'n_positions={len(positions)}\n')

    rows = []
    print(f'{"":42s}  {"no error":>10s}  {"+ error":>10s}')
    for name, factory, method in APERTURE_CASES:
        ap = factory(positions)
        t_clean = _time(
            lambda ap=ap, m=method: ap.do_photometry(data, method=m),
            repeats)
        ap_e = factory(positions)
        t_err = _time(
            lambda ap_e=ap_e, m=method: ap_e.do_photometry(
                data, error=error, method=m), repeats)
        rows.append((name, t_clean, t_err))
        print(f'  {name:40s}  {t_clean * 1000:8.1f} ms  '
              f'{t_err * 1000:8.1f} ms')

    rows.sort(key=lambda r: r[1], reverse=True)
    print('\nTop 5 slowest (no error):\n')
    for name, t_clean, _t_err in rows[:5]:
        print(f'  {name:40s}  {t_clean * 1000:8.1f} ms')

    print(f'\nTotal across all cases (no error): '
          f'{sum(t for _, t, _ in rows):.2f} s')


if __name__ == '__main__':
    main()
