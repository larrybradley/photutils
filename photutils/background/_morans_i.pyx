# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: freethreading_compatible=True
"""
Cython kernel that computes the sliding-patch Moran's I statistic.

The statistic at each pixel is the global Moran's I of the square patch
centered on that pixel, using the weighted mean of the 3x3 neighbors
(excluding the pixel itself) as the spatial lag. It reduces to the
weighted covariance between the pixel values and their neighbor means
divided by the weighted variance of the pixel values, where both are
taken over the patch.

The image is processed in a single pass. A few sums over the patch are
needed (the sums of the weights and of the weighted values, squared
values, neighbor means, and value times neighbor mean products, plus
the pixel count if the pixels have weights). Each is kept as a running
sum per column that is updated as rows enter and leave the patch, and
the column sums are then combined with a running sum along the row.
The cost per pixel is therefore independent of the patch size, and the
working memory is a few row buffers plus the last ``patch_size`` rows
of the per-pixel terms.

Repeated box passes chain several of these box-sum stages. The rows of
box sums that a stage produces are the input rows of the next stage,
which lag the input rows of the stage by half of the patch size.

The module also has the two binary morphology kernels that clean up
the thresholded map. They use the default structuring element of
`scipy.ndimage.binary_fill_holes` and `scipy.ndimage.binary_opening`,
but work in place on one byte per pixel. The opening differs from the
SciPy function at the image edges, where it does not erode the mask.

The kernels run without the GIL and use no global mutable state, so
this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

__all__ = ['fill_holes', 'morans_i_rows', 'opening']

cdef extern from "math.h" nogil:
    bint isfinite(double x)
    const float NAN

ctypedef fused real_t:
    float
    double

# Patches whose variance is below this fraction of their mean squared
# value are treated as constant. This rejects the rounding residuals of
# the running sums in regions of constant data.
cdef double VARIANCE_FLOOR = 1.0e-10

# The number of sums over the patch (the sums of the weights and of the
# weighted values, squared values, neighbor means, and products, plus
# the pixel count). The count is used only if the pixels have weights.
# Otherwise it is equal to the sum of the weights.
cdef enum:
    N_SUMS = 6


cdef inline void load_row(const real_t[:, ::1] data,
                          const unsigned char[:, ::1] mask, bint has_mask,
                          const real_t[:, ::1] weights, bint has_weights,
                          double pedestal, Py_ssize_t row,
                          double[::1] values, double[::1] wts,
                          double[::1] wvalues, double[::1] wts3,
                          double[::1] wvalues3) noexcept nogil:
    """
    Load the values, the weights, and the weighted values of the valid
    pixels of an image row, and the 3-pixel sums of the last two.

    ``values``, ``wts``, and ``wvalues`` are padded by one element at
    each end. Invalid pixels and rows outside the image have zero
    weight.
    """
    cdef Py_ssize_t nx = data.shape[1]
    cdef Py_ssize_t j
    cdef double value
    cdef double weight = 1.0

    if row < 0 or row >= data.shape[0]:
        for j in range(nx):
            wts[j + 1] = 0.0
            wvalues[j + 1] = 0.0
            wts3[j] = 0.0
            wvalues3[j] = 0.0
        return

    for j in range(nx):
        value = data[row, j]
        if has_weights:
            weight = weights[row, j]
        if (isfinite(value) and not (has_mask and mask[row, j])
                and isfinite(weight) and weight > 0.0):
            value -= pedestal
            values[j + 1] = value
            wts[j + 1] = weight
            wvalues[j + 1] = weight * value
        else:
            wts[j + 1] = 0.0
            wvalues[j + 1] = 0.0

    for j in range(nx):
        wts3[j] = wts[j] + wts[j + 1] + wts[j + 2]
        wvalues3[j] = wvalues[j] + wvalues[j + 1] + wvalues[j + 2]


cdef inline void add_terms(double* sums, double weight, double value,
                           double neighbor, bint has_weights) noexcept nogil:
    """
    Add the terms of a pixel to a set of sums.
    """
    cdef double wvalue = weight * value

    sums[0] += weight
    sums[1] += wvalue
    sums[2] += wvalue * value
    sums[3] += weight * neighbor
    sums[4] += wvalue * neighbor
    if has_weights:
        sums[5] += 1.0


cdef inline void subtract_terms(double* sums, double weight, double value,
                                double neighbor,
                                bint has_weights) noexcept nogil:
    """
    Subtract the terms of a pixel from a set of sums.
    """
    cdef double wvalue = weight * value

    sums[0] -= weight
    sums[1] -= wvalue
    sums[2] -= wvalue * value
    sums[3] -= weight * neighbor
    sums[4] -= wvalue * neighbor
    if has_weights:
        sums[5] -= 1.0


cdef inline void add_sums(double* sums, const double* other,
                          bint has_weights) noexcept nogil:
    """
    Add a set of sums to another set of sums.
    """
    sums[0] += other[0]
    sums[1] += other[1]
    sums[2] += other[2]
    sums[3] += other[3]
    sums[4] += other[4]
    if has_weights:
        sums[5] += other[5]


cdef inline void subtract_sums(double* sums, const double* other,
                               bint has_weights) noexcept nogil:
    """
    Subtract a set of sums from another set of sums.
    """
    sums[0] -= other[0]
    sums[1] -= other[1]
    sums[2] -= other[2]
    sums[3] -= other[3]
    sums[4] -= other[4]
    if has_weights:
        sums[5] -= other[5]


cdef inline void start_slide(double* total, const double* col,
                             Py_ssize_t half, Py_ssize_t nx,
                             bint has_weights) noexcept nogil:
    """
    Start the running sums along a row with the columns to the left
    of the center of the first patch.
    """
    cdef Py_ssize_t j, k

    for k in range(N_SUMS):
        total[k] = 0.0
    for j in range(min(half, nx)):
        add_sums(total, col + N_SUMS * j, has_weights)


cdef inline void advance_slide(double* total, const double* col,
                               Py_ssize_t j, Py_ssize_t half,
                               Py_ssize_t nx,
                               bint has_weights) noexcept nogil:
    """
    Move the running sums along a row to the patch centered on
    column ``j``.
    """
    if j + half < nx:
        add_sums(total, col + N_SUMS * (j + half), has_weights)
    if j - half - 1 >= 0:
        subtract_sums(total, col + N_SUMS * (j - half - 1), has_weights)


def morans_i_rows(const real_t[:, ::1] data,
                  const unsigned char[:, ::1] mask,
                  const real_t[:, ::1] weights, double pedestal,
                  Py_ssize_t patch_size, Py_ssize_t n_passes,
                  double min_count, float[:, ::1] out,
                  Py_ssize_t row_start, Py_ssize_t row_stop):
    """
    Calculate the sliding-patch Moran's I statistic for a range of
    image rows.

    Parameters
    ----------
    data : 2D float32 or float64 `~numpy.ndarray`
        The C-contiguous image. Non-finite values are excluded.

    mask : 2D uint8 `~numpy.ndarray` or `None`
        The C-contiguous mask with the same shape as ``data``. Pixels
        with a non-zero value are excluded.

    weights : 2D `~numpy.ndarray` or `None`
        The C-contiguous pixel weights with the same shape and data
        type as ``data``. Pixels with a non-finite or non-positive
        weight are excluded.

    pedestal : float
        A constant subtracted from ``data`` to limit the rounding error
        of the running sums. The statistic does not depend on it.

    patch_size : int
        The odd side length of the square patch.

    n_passes : int
        The number of box-sum passes.

    min_count : float
        The minimum sum of the patch kernel over the usable pixels.
        Pixels with a smaller sum are set to NaN.

    out : 2D float32 `~numpy.ndarray`
        The C-contiguous output array with the same shape as ``data``.
        Only the rows from ``row_start`` to ``row_stop`` are written.

    row_start, row_stop : int
        The half-open range of output rows to compute.
    """
    cdef Py_ssize_t nx = data.shape[1]
    cdef Py_ssize_t ny = data.shape[0]
    cdef Py_ssize_t half = patch_size // 2
    cdef Py_ssize_t total_half = n_passes * half
    cdef Py_ssize_t row_first = row_start - total_half
    cdef Py_ssize_t ok_size = total_half + 1
    cdef Py_ssize_t row_in, row_out, row, step, stage, stage_step, slot
    cdef Py_ssize_t j, k
    cdef Py_ssize_t prev, cur, nxt
    cdef bint has_mask = mask is not None
    cdef bint has_weights = weights is not None
    cdef Py_ssize_t count_idx = 5 if has_weights else 0
    cdef double weight, neighbor_wt, neighbor, mean, mean_sq, var
    cdef double[N_SUMS] total
    cdef double* terms
    cdef double* sums
    cdef double* box
    cdef const double* prev_sums
    cdef unsigned char* ok

    # Three rolling rows of the valid pixels and their 3-pixel sums
    cdef double[:, ::1] values = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts = np.zeros((3, nx + 2))
    cdef double[:, ::1] wvalues = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts3 = np.zeros((3, nx))
    cdef double[:, ::1] wvalues3 = np.zeros((3, nx))

    # The weight, value, and neighbor mean of the pixels in the rows
    # that are inside the patch of the first stage. Unusable pixels
    # have zero weight.
    cdef double[:, :, ::1] pixels = np.zeros((patch_size, nx, 3))

    # The box sums made by each stage but the last, for the rows that
    # are inside the patch of the next stage
    cdef double[:, :, :, ::1] boxes = np.zeros(
        (n_passes - 1, patch_size, nx, N_SUMS))

    # The running sums over the patch rows for each stage and column.
    # The sums of a column are adjacent so that they share cache lines.
    cdef double[:, :, ::1] col = np.zeros((n_passes, nx, N_SUMS))

    # Whether the pixels of the rows not yet output are usable
    cdef unsigned char[:, ::1] ok_ring = np.zeros((ok_size, nx),
                                                  dtype=np.uint8)

    if nx == 0 or ny == 0:
        return

    with nogil:
        # The rolling row buffers are indexed by the number of rows
        # loaded, so row ``row_first + step`` is in slot
        # ``(step + 1) % 3``
        load_row(data, mask, has_mask, weights, has_weights, pedestal,
                 row_first - 1, values[0], wts[0], wvalues[0], wts3[0],
                 wvalues3[0])
        load_row(data, mask, has_mask, weights, has_weights, pedestal,
                 row_first, values[1], wts[1], wvalues[1], wts3[1],
                 wvalues3[1])

        for row_in in range(row_first, row_stop + total_half):
            step = row_in - row_first
            prev = step % 3
            cur = (step + 1) % 3
            nxt = (step + 2) % 3
            load_row(data, mask, has_mask, weights, has_weights, pedestal,
                     row_in + 1, values[nxt], wts[nxt], wvalues[nxt],
                     wts3[nxt], wvalues3[nxt])

            # The first stage sums the terms of the pixels
            slot = step % patch_size
            terms = &pixels[slot, 0, 0]
            sums = &col[0, 0, 0]
            if step >= patch_size:
                # Remove the row that leaves the patch
                for j in range(nx):
                    if terms[3 * j] != 0.0:
                        subtract_terms(sums + N_SUMS * j, terms[3 * j],
                                       terms[3 * j + 1], terms[3 * j + 2],
                                       has_weights)
            if row_in < 0 or row_in >= ny:
                for j in range(nx):
                    terms[3 * j] = 0.0
            else:
                # Add the row that enters the patch
                ok = &ok_ring[step % ok_size, 0]
                for j in range(nx):
                    weight = wts[cur, j + 1]
                    # The center pixel is left out of the neighbor
                    # sums instead of being subtracted from them, so
                    # that a sum of zero weights is exactly zero
                    neighbor_wt = (wts3[prev, j] + wts3[nxt, j]
                                   + wts[cur, j] + wts[cur, j + 2])
                    if weight == 0.0 or neighbor_wt <= 0.0:
                        terms[3 * j] = 0.0
                        ok[j] = 0
                        continue
                    neighbor = (wvalues3[prev, j] + wvalues3[nxt, j]
                                + wvalues[cur, j]
                                + wvalues[cur, j + 2]) / neighbor_wt
                    terms[3 * j] = weight
                    terms[3 * j + 1] = values[cur, j + 1]
                    terms[3 * j + 2] = neighbor
                    ok[j] = 1
                    add_terms(sums + N_SUMS * j, weight, values[cur, j + 1],
                              neighbor, has_weights)

            # Each later stage sums the box sums of the stage before
            # it. It starts a full patch later and its rows lag by
            # half of a patch.
            for stage in range(1, n_passes):
                stage_step = step - 2 * stage * half
                if stage_step < 0:
                    break
                row = row_in - stage * half
                slot = stage_step % patch_size
                box = &boxes[stage - 1, slot, 0, 0]
                sums = &col[stage, 0, 0]
                prev_sums = &col[stage - 1, 0, 0]
                if stage_step >= patch_size:
                    for j in range(nx):
                        subtract_sums(sums + N_SUMS * j, box + N_SUMS * j,
                                      has_weights)
                if row < 0 or row >= ny:
                    for j in range(N_SUMS * nx):
                        box[j] = 0.0
                else:
                    start_slide(total, prev_sums, half, nx, has_weights)
                    for j in range(nx):
                        advance_slide(total, prev_sums, j, half, nx,
                                      has_weights)
                        for k in range(N_SUMS):
                            box[N_SUMS * j + k] = total[k]
                        add_sums(sums + N_SUMS * j, total, has_weights)

            row_out = row_in - total_half
            if row_out < row_start:
                continue

            # Slide the patch of the last stage along the row
            ok = &ok_ring[(row_out - row_first) % ok_size, 0]
            prev_sums = &col[n_passes - 1, 0, 0]
            start_slide(total, prev_sums, half, nx, has_weights)
            for j in range(nx):
                advance_slide(total, prev_sums, j, half, nx, has_weights)
                if not ok[j] or total[count_idx] < min_count:
                    out[row_out, j] = NAN
                    continue
                mean = total[1] / total[0]
                mean_sq = total[2] / total[0]
                var = mean_sq - mean * mean
                if var <= VARIANCE_FLOOR * mean_sq:
                    out[row_out, j] = NAN
                    continue
                out[row_out, j] = <float>(
                    (total[4] / total[0] - mean * total[3] / total[0]) / var)


def fill_holes(unsigned char[:, ::1] mask):
    """
    Fill the holes of a binary mask in place.

    A hole is a region of background pixels that is not connected to
    the image border through horizontal and vertical steps.

    Parameters
    ----------
    mask : 2D uint8 `~numpy.ndarray`
        The C-contiguous mask with values of 0 (background) and 1. It
        is modified in place.
    """
    cdef Py_ssize_t ny = mask.shape[0]
    cdef Py_ssize_t nx = mask.shape[1]
    cdef Py_ssize_t y, x
    cdef bint changed = True

    if ny == 0 or nx == 0:
        return

    with nogil:
        # Background pixels that are connected to the border are
        # marked with a value of 2, starting from the border itself
        for x in range(nx):
            if mask[0, x] == 0:
                mask[0, x] = 2
            if mask[ny - 1, x] == 0:
                mask[ny - 1, x] = 2
        for y in range(ny):
            if mask[y, 0] == 0:
                mask[y, 0] = 2
            if mask[y, nx - 1] == 0:
                mask[y, nx - 1] = 2

        # Propagate the marks with alternating forward and backward
        # raster scans until nothing changes. Compact regions need
        # only a few scans.
        while changed:
            changed = False
            for y in range(ny):
                for x in range(nx):
                    if mask[y, x] == 0 and (
                            (y > 0 and mask[y - 1, x] == 2)
                            or (x > 0 and mask[y, x - 1] == 2)):
                        mask[y, x] = 2
                        changed = True
            for y in range(ny - 1, -1, -1):
                for x in range(nx - 1, -1, -1):
                    if mask[y, x] == 0 and (
                            (y < ny - 1 and mask[y + 1, x] == 2)
                            or (x < nx - 1 and mask[y, x + 1] == 2)):
                        mask[y, x] = 2
                        changed = True

        for y in range(ny):
            for x in range(nx):
                mask[y, x] = mask[y, x] != 2


cdef void erode_or_dilate(const unsigned char[:, ::1] src,
                          unsigned char[:, ::1] dst,
                          const unsigned char[::1] outside_row,
                          bint erode) noexcept nogil:
    """
    Erode or dilate a binary mask by one pixel in the horizontal and
    vertical directions.

    The pixels outside of the image have the value of ``outside_row``,
    which must be 1 for an erosion (so that the mask is not eroded from
    the image edges) and 0 for a dilation.
    """
    cdef Py_ssize_t ny = src.shape[0]
    cdef Py_ssize_t nx = src.shape[1]
    cdef Py_ssize_t y, x, k
    cdef const unsigned char* cur
    cdef const unsigned char* up
    cdef const unsigned char* down
    cdef unsigned char* out
    cdef unsigned char left, right
    cdef unsigned char outside = outside_row[0]

    for y in range(ny):
        cur = &src[y, 0]
        up = &src[y - 1, 0] if y > 0 else &outside_row[0]
        down = &src[y + 1, 0] if y < ny - 1 else &outside_row[0]
        out = &dst[y, 0]
        if erode:
            for x in range(1, nx - 1):
                out[x] = (cur[x] & cur[x - 1] & cur[x + 1] & up[x]
                          & down[x])
        else:
            for x in range(1, nx - 1):
                out[x] = (cur[x] | cur[x - 1] | cur[x + 1] | up[x]
                          | down[x])

        # The first and last columns have a neighbor outside the image
        for k in range(min(nx, 2)):
            x = k * (nx - 1)
            left = cur[x - 1] if x > 0 else outside
            right = cur[x + 1] if x < nx - 1 else outside
            if erode:
                out[x] = cur[x] & left & right & up[x] & down[x]
            else:
                out[x] = cur[x] | left | right | up[x] | down[x]


def opening(unsigned char[:, ::1] mask, Py_ssize_t iterations):
    """
    Apply a binary opening to a mask in place.

    The mask is eroded ``iterations`` times and then dilated
    ``iterations`` times, which removes the regions that do not survive
    the erosions. The pixels outside of the image are taken as part of
    the mask for the erosions, so the mask is not eroded from the image
    edges.

    Parameters
    ----------
    mask : 2D uint8 `~numpy.ndarray`
        The C-contiguous mask with values of 0 (background) and 1. It
        is modified in place.

    iterations : int
        The number of erosions and of dilations.
    """
    cdef Py_ssize_t i
    cdef unsigned char[:, ::1] work = np.empty_like(mask)
    cdef const unsigned char[::1] zeros = np.zeros(mask.shape[1],
                                                   dtype=np.uint8)
    cdef const unsigned char[::1] ones = np.ones(mask.shape[1],
                                                 dtype=np.uint8)

    if mask.shape[0] == 0 or mask.shape[1] == 0:
        return

    # Each pair of passes goes to the work array and back
    with nogil:
        for i in range(iterations):
            if i % 2 == 0:
                erode_or_dilate(mask, work, ones, True)
            else:
                erode_or_dilate(work, mask, ones, True)
        for i in range(iterations, 2 * iterations):
            if i % 2 == 0:
                erode_or_dilate(mask, work, zeros, False)
            else:
                erode_or_dilate(work, mask, zeros, False)
