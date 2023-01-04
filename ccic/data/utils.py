"""
ccic.data.utils
===============

Utility functions for the data processing.
"""

def included_pixel_mask(indices, c_i, c_j, extent, n_roll=0):
    """
    Calculate mask of pixels included in a rectangular scene.

    Args:
        indices: Tuple containing the row- and column-indices of the pixels
            with CloudSat measurements.
        c_i: The row-coordinate of the center of the rectangular
            domain to extract.
        c_i: The column-coordinate of the center of the rectangular
            domain to extract.
        extent: The extent of the domain in pixels.
        n_roll: The roll applied along the zonal direction.

    Return:
        A boolean array identifying the indices withing the give rectangular
        domain.
    """
    i_start = c_i - extent // 2
    i_end = c_i + (extent - extent // 2)
    j_start = c_j - extent // 2 + n_roll
    j_end = c_j + (extent - extent // 2) + n_roll

    mask = (
        (i_start <= indices[0]) * (indices[0] < i_end) *
        (j_start <= indices[1] + n_roll) * (indices[1] + n_roll < j_end)
    )
    return mask
