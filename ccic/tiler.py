from math import ceil
import numpy as np
import torch


def get_start_and_clips(n, tile_size, overlap, soft_end: bool = False):
    """Calculate start indices and numbers of clipped pixels for a given
    side length, tile size and overlap.

    Args:
        n: The image size to tile in pixels.
        tile_size: The size of each tile
        overlap: The number of pixels of overlap.
        soft_end: Allow the last tile to go beyond ``n``, see notes for details

    Return:
        A tuple ``(start, clip)`` containing the start indices of each tile
        and the number of pixels to clip between each neighboring tiles.

    Notes:
        ``soft_end`` is intended to use for cylindrical wrapping of the tiles
        along the horizontal dimension, for example, to have a tile that
        covers the antimeridian. The handling of this special tile should be
        done outside this function.
    """
    start = []
    clip = []
    j = 0
    while j + tile_size < n:
        start.append(j)
        if j > 0:
            clip.append(overlap // 2)
        j = j + tile_size - overlap
    if not soft_end:
        start.append(max(n - tile_size, 0))
    else:
        start.append(j)
    if len(start) > 1:
        clip.append((start[-2] + tile_size - start[-1]) // 2)
    start = start
    clip = clip
    return start, clip


class Tiler:
    """
    Helper class that performs two-dimensional tiling of retrieval inputs and
    calculates clipping ranges for the reassembly of tiled predictions.

    Attributes:
        M: The number of tiles along the first image dimension (rows).
        N: The number of tiles along the second image dimension (columns).
    """

    def __init__(self, x, tile_size=512, overlap=32, wrap_columns=False):
        """
        Args:
            x: List of input tensors for the retrieval.
            tile_size: The size of a single tile.
            overlap: The overlap between two subsequent tiles.
            wrap_columns: Apply a circular tiling along the horizontal
                dimension, intended to overcome issues at the antimeridian
        """
        self.x = x
        self.wrap_columns = wrap_columns
        *_, m, n = x.shape
        self.m = m
        self.n = n

        if isinstance(tile_size, int):
            tile_size = (tile_size, tile_size)
        if len(tile_size) == 1:
            tile_size = tile_size * 2
        self.tile_size = (min(m, tile_size[0]), min(n, tile_size[1]))
        self.overlap = overlap

        min_len = min(self.tile_size[0], self.tile_size[1])
        if overlap > min_len // 2:
            raise ValueError("Overlap must not exceed the half of the tile size.")

        i_start, i_clip = get_start_and_clips(self.m, tile_size[0], overlap)
        self.i_start = i_start
        self.i_clip = i_clip

        # `soft_end` should be True if circular tiling is expected
        j_start, j_clip = get_start_and_clips(
            self.n, tile_size[1], overlap, soft_end=self.wrap_columns
        )
        self.j_start = j_start
        self.j_clip = j_clip

        self.M = len(i_start)
        self.N = len(j_start)

    def get_tile(self, i, j):
        """
        Get tile in the 'i'th row and 'j'th column of the two
        dimensional tiling.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            List containing the tile extracted from the list
            of input tensors.
        """
        i_start = self.i_start[i]
        i_end = i_start + self.tile_size[0]
        j_start = self.j_start[j]
        j_end = j_start + self.tile_size[1]
        if self.wrap_columns:
            if j_end > self.n:
                # Wrap the tile around the last column
                if isinstance(self.x, np.ndarray):
                    return np.concatenate(
                        (
                            self.x[..., i_start:i_end, j_start:],
                            self.x[
                                ...,
                                i_start:i_end,
                                : (self.tile_size[1] - (self.n - j_start)),
                            ],
                        ),
                        axis=-1,
                    )
                elif isinstance(self.x, torch.Tensor):
                    return torch.cat(
                        (
                            self.x[..., i_start:i_end, j_start:],
                            self.x[
                                ...,
                                i_start:i_end,
                                : (self.tile_size[1] - (self.n - j_start)),
                            ],
                        ),
                        axis=-1,
                    )
                else:
                    raise TypeError(
                        "Only numpy.ndarray and torch.Tensor types are supported"
                    )
        # Ordinary slicing in all other cases
        return self.x[..., i_start:i_end, j_start:j_end]

    def get_slices(self, i, j):
        """
        Return slices for the clipping of the result tensors.

        Args:
            i: The 0-based row index of the tile.
            j: The 0-based column index of the tile.

        Return:
            Tuple of slices that can be used to clip the retrieval
            results to obtain non-overlapping tiles.
        """
        if i == 0:
            i_clip_l = 0
        else:
            i_clip_l = self.i_clip[i - 1]
        if i >= self.M - 1:
            i_clip_r = self.tile_size[0]
        else:
            i_clip_r = self.tile_size[0] - self.i_clip[i]
        slice_i = slice(i_clip_l, i_clip_r)

        if j == 0:
            j_clip_l = 0
        else:
            j_clip_l = self.j_clip[j - 1]
        if j >= self.N - 1:
            j_clip_r = self.tile_size[1]
        else:
            j_clip_r = self.tile_size[1] - self.j_clip[j]
        slice_j = slice(j_clip_l, j_clip_r)

        return (slice_i, slice_j)

    def get_weights(self, i, j):
        """
        Get weights to reassemble results.

        Args:
            i: Row-index of the tile.
            j: Column-index of the tile.

        Return:
            Numpy array containing weights for the corresponding tile.
        """
        m, n = self.tile_size
        w_i = np.ones((m, n))
        if i > 0:
            trans_start = self.i_start[i]
            # Shift start to right if transition overlaps with
            # antepenultimate tile.
            if i > 1:
                trans_end_prev = self.i_start[i - 2] + self.tile_size[0]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.i_start[i]
            trans_end = self.i_start[i - 1] + self.tile_size[0]
            # Limit transition zone to overlap.
            l_trans = min(trans_end - trans_start, self.overlap)
            w_i[:zeros] = 0.0
            w_i[zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[..., np.newaxis]

        if i < self.M - 1:
            trans_start = self.i_start[i + 1]
            if i > 0:
                trans_end_prev = self.i_start[i - 1] + self.tile_size[0]
                trans_start = max(trans_start, trans_end_prev)
            trans_end = self.i_start[i] + self.tile_size[0]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.i_start[i]
            w_i[start : start + l_trans] = np.linspace(1, 0, l_trans)[..., np.newaxis]
            w_i[start + l_trans :] = 0.0

        w_j = np.ones((m, n))
        if j > 0:
            trans_start = self.j_start[j]
            # Shift start to right if transition overlaps with
            # antepenultimate tile and wrapping is not desired
            if j > 1 and not self.wrap_columns:
                trans_end_prev = self.j_start[j - 2] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            zeros = trans_start - self.j_start[j]
            trans_end = self.j_start[j - 1] + self.tile_size[1]
            # Limit transition zone to overlap.
            l_trans = min(trans_end - trans_start, self.overlap)
            w_j[:, :zeros] = 0.0
            w_j[:, zeros : zeros + l_trans] = np.linspace(0, 1, l_trans)[np.newaxis]
        elif self.wrap_columns:
            trans_end = (self.j_start[-1] + self.tile_size[1]) - self.n
            l_trans = min(trans_end, self.overlap)
            w_j[:, :l_trans] = np.linspace(0, 1, l_trans)[np.newaxis]

        if j < self.N - 1:
            trans_start = self.j_start[j + 1]
            if j > 0:
                trans_end_prev = self.j_start[j - 1] + self.tile_size[1]
                trans_start = max(trans_start, trans_end_prev)
            trans_end = self.j_start[j] + self.tile_size[1]
            l_trans = min(trans_end - trans_start, self.overlap)

            start = trans_start - self.j_start[j]
            w_j[:, start : start + l_trans] = np.linspace(1, 0, l_trans)[np.newaxis]
            w_j[:, start + l_trans :] = 0.0
        elif self.wrap_columns:
            trans_end = self.j_start[-1] + self.tile_size[1]
            l_trans = min(trans_end % self.n, self.overlap)
            start = self.n - self.j_start[-1]
            w_j[:, start : start + l_trans] = np.linspace(1, 0, l_trans)[np.newaxis]
            w_j[:, start + l_trans :] = 0.0

        return w_i * w_j

    def assemble(self, slices):
        """
        Assemble slices back to original shape using linear interpolation in
        overlap regions.

        Args:
            slices: List of lists of slices.

        Return:
            ``numpy.ndarray`` containing the data from the slices reconstructed
            to the original shape.
        """
        slice_0 = slices[0][0]

        shape = slice_0.shape[:-2] + (self.m, self.n)
        results = np.zeros(shape, dtype=slice_0.dtype)

        for i, row in enumerate(slices):
            for j, slc in enumerate(row):
                i_start = self.i_start[i]
                i_end = i_start + self.tile_size[0]
                row_slice = slice(i_start, i_end)
                j_start = self.j_start[j]
                j_end = j_start + self.tile_size[1]
                # modulo self.n in case self.wrap_columns is True
                col_slice = np.arange(j_start, j_end) % self.n

                results[..., row_slice, col_slice] += self.get_weights(i, j) * slc

        return results

    def __repr__(self):
        return f"Tiler(tile_size={self.tile_size}, overlap={self.overlap})"


def calculate_padding(tensor, multiple_of=32):
    """
    Calculate torch padding dimensions required to pad the input tensor
    to a multiple of 32.

    Args:
        tensor: The tensor to pad.
        multiple_of: Integer of which the spatial dimensions of 'tensor'
            should be a multiple of.

    Return
        A tuple ``(p_l_n, p_r_n, p_l_m, p_r_m)`` containing the
        left and right padding  for the second to last dimension
        (``p_l_m, p_r_m``) and for the last dimension (``p_l_n, p_r_n``).
    """
    shape = tensor.shape

    n = shape[-1]
    d_n = ceil(n / multiple_of) * multiple_of - n
    p_l_n = d_n // 2
    p_r_n = d_n - p_l_n

    m = shape[-2]
    d_m = ceil(m / multiple_of) * multiple_of - m
    p_l_m = d_m // 2
    p_r_m = d_m - p_l_m
    return (p_l_n, p_r_n, p_l_m, p_r_m)
