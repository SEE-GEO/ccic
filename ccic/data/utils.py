"""
ccic.data.utils
===============

Utility functions for the data processing.
"""
import numpy as np


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


def extract_roi(dataset, roi, min_size=None):
    """
    Extract region of interest (ROI) from dataset.

    Args:
        dataset: The 'xarray.Dataset' from which to extract data in a
            given ROI. Expected to have longitude and latitude coordinates
            in ``lon`` and ``lat`` dimensions, respectively.
        roi: The region of interest given as a coordinates
             ``(lon_min, lat_min, lon_max, lat_max)`` so that
             ``(lon_min, lat_min)`` specify the longitude and latitude
             coordinates of the lower left point and ``(lon_max, lat_max)``
             those of the upper right point.
        min_size: The minimum size of the region in both dimensions.
    """
    lat_mask = (
        (dataset.lat.data >= roi[1]) *
        (dataset.lat.data <= roi[3])
    )

    lon_mask = (
        (dataset.lon.data >= roi[0]) *
        (dataset.lon.data <= roi[2])
    )

    if min_size is not None:
        try:
            first, last = np.where(lat_mask)[0][[0, -1]]
        except IndexError:
            raise ValueError(
                "No latitude points in bounding box."
            )

        if last - first + 1 < min_size:
            diff = min_size - (last - first + 1)
            first = first - diff // 2
            last = first + min_size
            lat_mask[first:last] = True

        if lat_mask.astype(np.int).sum() < min_size:
            lat_mask[last - min_size: last] = True

        try:
            first, last = np.where(lon_mask)[0][[0, -1]]
        except IndexError:
            raise ValueError(
                "No longitude points in bounding box."
            )
        if last - first + 1 < min_size:
            diff = min_size - (last - first + 1)
            first = first - diff // 2
            last = first + min_size
            lon_mask[first:last] = True

        if lat_mask.astype(np.int).sum() < min_size:
            lat_mask[last - min_size: last] = True

    return dataset[{"lat": lat_mask, "lon": lon_mask}]
