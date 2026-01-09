# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide tools for making simulated example images for documentation
examples and tests.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import line_aa


def make_cosmic_ray_image1(shape=(512, 512), num_cr=15,
                           intensity_range=(500, 2000),
                           length_range=(5, 30),
                           width=0.5,
                           seed=None):
    """
    Generates a 2D image containing synthetic trailed cosmic rays.

    Parameters:
        shape (tuple): The (height, width) of the output image.
        num_cr (int): Number of cosmic rays to generate.
        intensity_range (tuple): Min and Max brightness of the CRs.
        length_range (tuple): Min and Max length of the CR trails in pixels.
        width (float): The 'blur' or diffusion of the CR (typical range 0.4 - 1.0).
        seed (int/None): Seed for the random number generator for reproducibility.

    Returns:
        numpy.ndarray: A 2D array (float32) containing only the cosmic rays.
    """
    # Initialize the modern NumPy random generator
    rng = np.random.default_rng(seed=seed)

    h, w = shape
    cr_image = np.zeros(shape)

    for _ in range(num_cr):
        # 1. Randomize properties using the generator
        length = rng.uniform(*length_range)
        angle = rng.uniform(0, 2 * np.pi)
        intensity = rng.uniform(*intensity_range)

        # 2. Randomize starting position
        start_r = rng.integers(0, h)
        start_c = rng.integers(0, w)

        # 3. Calculate end position
        end_r = start_r + length * np.sin(angle)
        end_c = start_c + length * np.cos(angle)

        # 4. Draw anti-aliased line
        # Use line_aa for smooth, sub-pixel accurate tracks
        rr, cc, val = line_aa(int(start_r), int(start_c), int(end_r), int(end_c))

        # Filter coordinates that fall outside the image frame
        mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc, val = rr[mask], cc[mask], val[mask]

        # Add to the image
        cr_image[rr, cc] += val * intensity

    # 5. Simulate charge diffusion (blurring)
    if width > 0:
        cr_image = gaussian_filter(cr_image, sigma=width)

    return cr_image


def draw_cosmic_ray(
    shape,
    start,
    angle_rad,
    length,
    total_flux,
    sigma=0.5,
    oversample=10
):
    """
    Draw an anti-aliased cosmic ray track into a 2D image.

    Parameters
    ----------
    img : 2D numpy array (modified in place)
    start : (row, col)
    angle_rad : float
        Angle CCW from +x axis
    length : float
        Track length in pixels
    total_flux : float
        Total deposited signal
    sigma : float
        Gaussian width in pixels (controls thickness)
    oversample : int
        Sub-pixel sampling factor
    """
    img = np.zeros(shape)
    h, w = shape
    r0, c0 = start

    # Direction vector
    dr = np.sin(angle_rad)
    dc = np.cos(angle_rad)

    # Subpixel samples along the track
    n = int(length * oversample)
    t = np.linspace(0, length, n, endpoint=True)

    rows = r0 + dr * t
    cols = c0 + dc * t

    # Flux per sample
    flux = total_flux / n

    # Kernel half-size (3-sigma)
    k = int(np.ceil(3 * sigma))

    for r, c in zip(rows, cols):
        rr = int(np.floor(r))
        cc = int(np.floor(c))

        rmin = max(rr - k, 0)
        rmax = min(rr + k + 1, h)
        cmin = max(cc - k, 0)
        cmax = min(cc + k + 1, w)

        y, x = np.mgrid[rmin:rmax, cmin:cmax]
        d2 = (y - r)**2 + (x - c)**2

        weights = np.exp(-0.5 * d2 / sigma**2)
        weights /= weights.sum()

        img[rmin:rmax, cmin:cmax] += flux * weights

    return img

