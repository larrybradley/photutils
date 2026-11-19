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

The module also has the two binary morphology kernels that clean up
the thresholded map. They use the default structuring element of
`scipy.ndimage.binary_fill_holes` and `scipy.ndimage.binary_opening`,
but work in place on one byte per pixel. The opening differs from the
SciPy function at the image edges and next to excluded pixels, where it
does not erode the mask.

The kernels run without the GIL and use no global mutable state, so
this module is safe to use from multiple threads, including on
free-threaded Python builds.
"""

import numpy as np

from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memset

__all__ = ['fill_holes', 'morans_i_rows', 'morans_i_rows_rects', 'opening']

cdef extern from "math.h" nogil:
    bint isfinite(double x)
    const float NAN

ctypedef fused real_t:
    float
    double

# Patches whose variance is below this fraction of the sum of their
# mean squared value and the mean squared value of the image are
# treated as constant. This rejects the rounding residuals of the
# running sums in regions of constant data. The image term is needed
# where the constant is equal or close to the value subtracted from the
# data, because the mean squared value of the patch is then itself only
# a rounding residual.
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
                  double image_mean_sq, Py_ssize_t patch_size,
                  double min_count,
                  float[:, ::1] out,
                  Py_ssize_t row_start, Py_ssize_t row_stop):
    """
    Calculate the sliding-patch Moran's I statistic for a range of
    image rows.

    Parameters
    ----------
    data : 2D float32 or float64 `~numpy.ndarray`
        The C-contiguous image. Nonfinite values are excluded.

    mask : 2D uint8 `~numpy.ndarray` or `None`
        The C-contiguous mask with the same shape as ``data``. Pixels
        with a nonzero value are excluded.

    weights : 2D `~numpy.ndarray` or `None`
        The C-contiguous pixel weights with the same shape and data
        type as ``data``. Pixels with a nonfinite or nonpositive
        weight are excluded.

    pedestal : float
        A constant subtracted from ``data`` to limit the rounding error
        of the running sums. The statistic does not depend on it.

    image_mean_sq : float
        The mean squared value of the whole image after ``pedestal`` is
        subtracted. It sets the scale below which the variance of a
        patch is a rounding residual.

    patch_size : int
        The odd side length of the square patch.

    min_count : float
        The minimum number of usable pixels in the patch. Pixels with
        fewer usable pixels in their patch are set to NaN.

    out : 2D float32 `~numpy.ndarray`
        The C-contiguous output array with the same shape as ``data``.
        Only the rows from ``row_start`` to ``row_stop`` are written.

    row_start, row_stop : int
        The half-open range of output rows to compute.
    """
    cdef Py_ssize_t nx = data.shape[1]
    cdef Py_ssize_t ny = data.shape[0]
    cdef Py_ssize_t half = patch_size // 2
    cdef Py_ssize_t row_first = row_start - half
    cdef Py_ssize_t ok_size = half + 1
    cdef Py_ssize_t row_in, row_out, step, slot
    cdef Py_ssize_t j
    cdef Py_ssize_t prev, cur, nxt
    cdef bint has_mask = mask is not None
    cdef bint has_weights = weights is not None
    cdef Py_ssize_t count_idx = 5 if has_weights else 0
    cdef double weight, neighbor_wt, neighbor, mean, mean_sq, var
    cdef double[N_SUMS] total
    cdef double* terms
    cdef double* sums
    cdef unsigned char* ok

    # Three rolling rows of the valid pixels and their 3-pixel sums
    cdef double[:, ::1] values = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts = np.zeros((3, nx + 2))
    cdef double[:, ::1] wvalues = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts3 = np.zeros((3, nx))
    cdef double[:, ::1] wvalues3 = np.zeros((3, nx))

    # The weight, value, and neighbor mean of the pixels in the rows
    # that are inside the patch. Unusable pixels have zero weight.
    cdef double[:, :, ::1] pixels = np.zeros((patch_size, nx, 3))

    # The running sums over the patch rows for each column. The sums of
    # a column are adjacent so that they share cache lines.
    cdef double[:, ::1] col = np.zeros((nx, N_SUMS))

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

        for row_in in range(row_first, row_stop + half):
            step = row_in - row_first
            prev = step % 3
            cur = (step + 1) % 3
            nxt = (step + 2) % 3
            load_row(data, mask, has_mask, weights, has_weights, pedestal,
                     row_in + 1, values[nxt], wts[nxt], wvalues[nxt],
                     wts3[nxt], wvalues3[nxt])

            # Update the running column sums with the rows that leave
            # and enter the patch
            slot = step % patch_size
            terms = &pixels[slot, 0, 0]
            sums = &col[0, 0]
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

            row_out = row_in - half
            if row_out < row_start:
                continue

            # Slide the patch along the row
            ok = &ok_ring[(row_out - row_first) % ok_size, 0]
            start_slide(total, sums, half, nx, has_weights)
            for j in range(nx):
                advance_slide(total, sums, j, half, nx, has_weights)
                if not ok[j] or total[count_idx] < min_count:
                    out[row_out, j] = NAN
                    continue
                mean = total[1] / total[0]
                mean_sq = total[2] / total[0]
                var = mean_sq - mean * mean
                if var <= VARIANCE_FLOOR * (mean_sq + image_mean_sq):
                    out[row_out, j] = NAN
                    continue
                out[row_out, j] = <float>(
                    (total[4] / total[0] - mean * total[3] / total[0]) / var)


cdef struct PixelStack:
    Py_ssize_t* data
    Py_ssize_t size
    Py_ssize_t capacity


cdef inline bint push_runs(PixelStack* stack, const unsigned char* pixels,
                           Py_ssize_t start, Py_ssize_t length) noexcept nogil:
    """
    Push the index of the first pixel of each run of unmarked background
    pixels (with a value of 0) in a segment of a row.

    The segment starts at the flat index ``start`` and has ``length``
    pixels. Return `False` if the stack cannot be enlarged.
    """
    cdef Py_ssize_t i = start
    cdef Py_ssize_t stop = start + length
    cdef Py_ssize_t* data

    while i < stop:
        if pixels[i] != 0:
            i += 1
            continue
        if stack.size == stack.capacity:
            data = <Py_ssize_t*>realloc(
                stack.data, 2 * stack.capacity * sizeof(Py_ssize_t))
            if data == NULL:
                return False
            stack.data = data
            stack.capacity *= 2
        stack.data[stack.size] = i
        stack.size += 1
        i += 1
        while i < stop and pixels[i] == 0:
            i += 1
    return True


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
    cdef Py_ssize_t y, left, right, i
    cdef unsigned char* pixels
    cdef unsigned char* row
    cdef PixelStack stack
    cdef bint ok = True

    if ny == 0 or nx == 0:
        return

    stack.size = 0
    stack.capacity = 4096
    stack.data = <Py_ssize_t*>malloc(stack.capacity * sizeof(Py_ssize_t))
    if stack.data == NULL:
        raise MemoryError

    pixels = &mask[0, 0]
    with nogil:
        # Background pixels that are connected to the border are
        # marked with a value of 2. The fill starts from the
        # background runs on the border.
        ok = (push_runs(&stack, pixels, 0, nx)
              and push_runs(&stack, pixels, (ny - 1) * nx, nx))
        y = 1
        while ok and y < ny - 1:
            ok = (push_runs(&stack, pixels, y * nx, 1)
                  and push_runs(&stack, pixels, y * nx + nx - 1, 1))
            y += 1

        # Each popped pixel is extended to the whole horizontal run of
        # unmarked background pixels that contains it. The run is
        # marked, and the first pixel of each unmarked background run
        # that touches it in the rows above and below is pushed.
        while ok and stack.size > 0:
            stack.size -= 1
            i = stack.data[stack.size]
            if pixels[i] != 0:
                continue
            y = i // nx
            row = pixels + y * nx
            left = i - y * nx
            right = left + 1
            while left > 0 and row[left - 1] == 0:
                left -= 1
            while right < nx and row[right] == 0:
                right += 1
            memset(row + left, 2, right - left)
            if y > 0:
                ok = push_runs(&stack, pixels, (y - 1) * nx + left,
                               right - left)
            if ok and y < ny - 1:
                ok = push_runs(&stack, pixels, (y + 1) * nx + left,
                               right - left)

        if ok:
            for i in range(ny * nx):
                pixels[i] = pixels[i] != 2

    free(stack.data)
    if not ok:
        raise MemoryError


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


def opening(unsigned char[:, ::1] mask, unsigned char[:, ::1] excluded,
            Py_ssize_t iterations):
    """
    Apply a binary opening to a mask in place.

    The mask is eroded ``iterations`` times and then dilated
    ``iterations`` times, which removes the regions that do not survive
    the erosions. The pixels outside of the image and the excluded
    pixels are taken as part of the mask for the erosions, so the mask
    is not eroded from the image edges or from the excluded pixels.
    The excluded pixels are removed again before the dilations, and
    the result is limited to the pixels of the input mask.

    Parameters
    ----------
    mask : 2D uint8 `~numpy.ndarray`
        The C-contiguous mask with values of 0 (background) and 1. It
        is modified in place.

    excluded : 2D uint8 `~numpy.ndarray`
        The C-contiguous mask of the excluded pixels, with values of 0
        and 1 and the same shape as ``mask``. It is overwritten, because
        its second bit is used to keep the input mask.

    iterations : int
        The number of erosions and of dilations.
    """
    cdef Py_ssize_t ny = mask.shape[0]
    cdef Py_ssize_t nx = mask.shape[1]
    cdef Py_ssize_t i, y, x
    cdef unsigned char[:, ::1] work = np.empty_like(mask)
    cdef unsigned char[:, ::1] eroded
    cdef const unsigned char[::1] zeros = np.zeros(nx, dtype=np.uint8)
    cdef const unsigned char[::1] ones = np.ones(nx, dtype=np.uint8)

    if ny == 0 or nx == 0:
        return

    # Each pair of passes goes to the work array and back
    eroded = mask if iterations % 2 == 0 else work
    with nogil:
        for y in range(ny):
            for x in range(nx):
                excluded[y, x] |= mask[y, x] << 1
                mask[y, x] |= excluded[y, x] & 1

        for i in range(iterations):
            if i % 2 == 0:
                erode_or_dilate(mask, work, ones, True)
            else:
                erode_or_dilate(work, mask, ones, True)

        for y in range(ny):
            for x in range(nx):
                if excluded[y, x] & 1:
                    eroded[y, x] = 0

        for i in range(iterations, 2 * iterations):
            if i % 2 == 0:
                erode_or_dilate(mask, work, zeros, False)
            else:
                erode_or_dilate(work, mask, zeros, False)

        for y in range(ny):
            for x in range(nx):
                mask[y, x] &= excluded[y, x] >> 1


cdef inline void start_band(double* total, const double* col,
                            Py_ssize_t outer, Py_ssize_t inner,
                            Py_ssize_t nx, bint has_weights) noexcept nogil:
    """
    Add to the running sums along a row the columns of a band that are
    to the left of the center of the first patch.

    The band covers the column offsets from the patch center that are
    larger than ``inner`` and at most ``outer`` in absolute value. An
    ``inner`` value of -1 gives a full box.
    """
    cdef Py_ssize_t j

    for j in range(min(outer, nx)):
        add_sums(total, col + N_SUMS * j, has_weights)
    for j in range(min(inner, nx)):
        subtract_sums(total, col + N_SUMS * j, has_weights)


cdef inline void slide_band(double* total, const double* col,
                            Py_ssize_t j, Py_ssize_t outer,
                            Py_ssize_t inner, Py_ssize_t nx,
                            bint has_weights) noexcept nogil:
    """
    Move the running sums of a band along a row to the band centered
    on column ``j``.
    """
    advance_slide(total, col, j, outer, nx, has_weights)
    if inner >= 0:
        if j + inner < nx:
            subtract_sums(total, col + N_SUMS * (j + inner), has_weights)
        if j - inner - 1 >= 0:
            add_sums(total, col + N_SUMS * (j - inner - 1), has_weights)


cdef inline void update_rect(double* sums, const double* terms,
                             Py_ssize_t nx, bint add,
                             bint has_weights) noexcept nogil:
    """
    Add the terms of an image row to, or subtract them from, the
    running column sums of a rectangle.
    """
    cdef Py_ssize_t j

    for j in range(nx):
        if terms[3 * j] == 0.0:
            continue
        if add:
            add_terms(sums + N_SUMS * j, terms[3 * j], terms[3 * j + 1],
                      terms[3 * j + 2], has_weights)
        else:
            subtract_terms(sums + N_SUMS * j, terms[3 * j],
                           terms[3 * j + 1], terms[3 * j + 2], has_weights)


def morans_i_rows_rects(const real_t[:, ::1] data,
                        const unsigned char[:, ::1] mask,
                        const real_t[:, ::1] weights, double pedestal,
                        double image_mean_sq,
                        const Py_ssize_t[::1] heights,
                        const Py_ssize_t[::1] widths, double min_count,
                        float[:, ::1] out, Py_ssize_t row_start,
                        Py_ssize_t row_stop):
    """
    Calculate the sliding-patch Moran's I statistic for a range of
    image rows with a patch that is a union of centered rectangles.

    A digital disk is the union of one rectangle for each distinct
    half-width of its rows. Each rectangle has its own running column
    sums over its rows. The sums of the patch along a row are the sums
    over the band of columns that each rectangle adds to the narrower
    rectangles before it.

    Parameters
    ----------
    data, mask, weights, pedestal, image_mean_sq
        See `morans_i_rows`.

    heights : 1D intp `~numpy.ndarray`
        The half-heights of the rectangles, in decreasing order.

    widths : 1D intp `~numpy.ndarray`
        The half-widths of the rectangles, in increasing order.

    min_count, out, row_start, row_stop
        See `morans_i_rows`.
    """
    cdef Py_ssize_t nx = data.shape[1]
    cdef Py_ssize_t ny = data.shape[0]
    cdef Py_ssize_t n_rects = heights.shape[0]
    cdef Py_ssize_t half = heights[0]
    cdef Py_ssize_t n_rows = 2 * half + 1
    cdef Py_ssize_t row_first = row_start - half
    cdef Py_ssize_t ok_size = half + 1
    cdef Py_ssize_t row_in, row_out, step, slot, j, k, inner
    cdef Py_ssize_t prev, cur, nxt
    cdef bint has_mask = mask is not None
    cdef bint has_weights = weights is not None
    cdef Py_ssize_t count_idx = 5 if has_weights else 0
    cdef double weight, neighbor_wt, neighbor, mean, mean_sq, var
    cdef double[N_SUMS] total
    cdef double* terms
    cdef unsigned char* ok

    cdef double[:, ::1] values = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts = np.zeros((3, nx + 2))
    cdef double[:, ::1] wvalues = np.zeros((3, nx + 2))
    cdef double[:, ::1] wts3 = np.zeros((3, nx))
    cdef double[:, ::1] wvalues3 = np.zeros((3, nx))

    # The weight, value, and neighbor mean of the pixels in the rows of
    # the tallest rectangle. Unusable pixels have zero weight.
    cdef double[:, :, ::1] pixels = np.zeros((n_rows, nx, 3))

    # The running sums over the rows of each rectangle for each column
    cdef double[:, :, ::1] col = np.zeros((n_rects, nx, N_SUMS))

    cdef unsigned char[:, ::1] ok_ring = np.zeros((ok_size, nx),
                                                  dtype=np.uint8)

    if nx == 0 or ny == 0:
        return

    with nogil:
        load_row(data, mask, has_mask, weights, has_weights, pedestal,
                 row_first - 1, values[0], wts[0], wvalues[0], wts3[0],
                 wvalues3[0])
        load_row(data, mask, has_mask, weights, has_weights, pedestal,
                 row_first, values[1], wts[1], wvalues[1], wts3[1],
                 wvalues3[1])

        for row_in in range(row_first, row_stop + half):
            step = row_in - row_first
            prev = step % 3
            cur = (step + 1) % 3
            nxt = (step + 2) % 3
            load_row(data, mask, has_mask, weights, has_weights, pedestal,
                     row_in + 1, values[nxt], wts[nxt], wvalues[nxt],
                     wts3[nxt], wvalues3[nxt])

            # Remove the rows that leave each rectangle. The row that
            # leaves the tallest rectangle is in the slot of the new row,
            # so it is removed first.
            for k in range(n_rects):
                if step >= half + heights[k] + 1:
                    slot = (step - half - heights[k] - 1) % n_rows
                    update_rect(&col[k, 0, 0], &pixels[slot, 0, 0], nx,
                                False, has_weights)

            slot = step % n_rows
            terms = &pixels[slot, 0, 0]
            if row_in < 0 or row_in >= ny:
                for j in range(nx):
                    terms[3 * j] = 0.0
            else:
                ok = &ok_ring[step % ok_size, 0]
                for j in range(nx):
                    weight = wts[cur, j + 1]
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

            # Add the rows that enter each rectangle. Shorter rectangles
            # are centered on the same row, so their rows enter later.
            for k in range(n_rects):
                if step >= half - heights[k]:
                    slot = (step - half + heights[k]) % n_rows
                    update_rect(&col[k, 0, 0], &pixels[slot, 0, 0], nx,
                                True, has_weights)

            row_out = row_in - half
            if row_out < row_start:
                continue

            ok = &ok_ring[(row_out - row_first) % ok_size, 0]
            for k in range(N_SUMS):
                total[k] = 0.0
            for k in range(n_rects):
                inner = widths[k - 1] if k > 0 else -1
                start_band(total, &col[k, 0, 0], widths[k], inner, nx,
                           has_weights)
            for j in range(nx):
                for k in range(n_rects):
                    inner = widths[k - 1] if k > 0 else -1
                    slide_band(total, &col[k, 0, 0], j, widths[k], inner,
                               nx, has_weights)
                if not ok[j] or total[count_idx] < min_count:
                    out[row_out, j] = NAN
                    continue
                mean = total[1] / total[0]
                mean_sq = total[2] / total[0]
                var = mean_sq - mean * mean
                if var <= VARIANCE_FLOOR * (mean_sq + image_mean_sq):
                    out[row_out, j] = NAN
                    continue
                out[row_out, j] = <float>(
                    (total[4] / total[0] - mean * total[3] / total[0]) / var)
