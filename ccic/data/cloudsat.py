"""
ccic.data.cloudsat
==================

This module provides functionality to read and resample  CloudSat 2C-Ice
files.
"""
import dask.array as da
import numpy as np
from pansat.download.providers.cloudsat_dpc import CloudSatDPCProvider
from pansat.products.satellite.cloud_sat import l2c_ice
from pansat.time import to_datetime
from pyresample.bucket import BucketResampler
from scipy.signal import convolve

PROVIDER = CloudSatDPCProvider(l2c_ice)


def get_sample_indices(resampler):
    """
    Return target and source indices for random-sample resampling.

    Args:
        resampler: A bucket resampler object with precomputed
            indices.

    Return:
        A tuple ``(target_inds, source_inds)`` containing the flattened
        indices of the random samples w.r.t. output and input grids,
        respectively.
    """
    # The indices of all samples w.r.t to the target grid.
    indices = resampler.idxs.compute().ravel()

    # Emulate sampling by shuffling indices and selecting each
    # first unique index.
    shuffle = np.random.permutation(indices.size)
    indices = indices[shuffle]
    unique_inds, unique_inds_data = np.unique(indices, return_index=True)
    unique_inds_data = shuffle[unique_inds_data]

    valid = unique_inds >= 0
    return unique_inds[valid], unique_inds_data[valid]


def subsample_iwc_and_height(iwc, height):
    """
    Smoothes and subsamples IWC and height fields to an approximate
    resolution of 1km.

    Args:
        iwc: The ice water content field from a CloudSat2CIce file.
        height: The corresponding height field.

    Return:
        A tuple ``(iwc, height)`` containing the subsampled IWC and
        height fields.
    """
    k = np.linspace(-4 * 240, 4 * 240, 9)
    k = np.exp(np.log(0.5) * (k / 500) ** 2).reshape(1, -1)
    k /= k.sum()
    iwc = convolve(iwc, k, mode="valid", method="direct")[:, ::4]
    height = convolve(height, k, mode="valid", method="direct")[:, ::4]
    return iwc, height


def remap_iwc(
        iwc,
        altitude,
        surface_altitude,
        target_altitudes,
):
    """
    Remap IWC to new altitude grid relative to surface.

    Args:
        iwc: 2D array containing IWC with altitude levels along last
            dimension.
        altitude: The altitudes corresponding to ``iwc``
        surface_altitude: The surface altitude for each IWC profile.
        target_altitude: 1D array specifying the altitudes to remap the
            IWC data to.

    Return:
        2D array containing the remapped IWC.
    """
    iwc_r = np.zeros(iwc.shape[:-1] + (target_altitudes.size,), dtype=iwc.dtype)

    for index in range(iwc.shape[0]):
        z_new = target_altitudes
        z_old = altitude[index] - surface_altitude[index]
        iwc_r[index] = np.interp(z_new, z_old, iwc[index])

    return iwc_r

class CloudSat2CIce:
    """
    Interface class to read CloudSat 2C-Ice files.
    """

    @staticmethod
    def get_available_files(date):
        """
        Return list of times at which this data is available.
        """
        date = to_datetime(date)
        day = int(date.strftime("%j"))
        files = PROVIDER.get_files_by_day(date.year, day)
        return files

    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def download(filename, destination):
        """
        Download file.

        Args:
            filename: The filename of the 2CIce file to download.
            destination: Path to store the result.
        """
        PROVIDER.download_file(filename, destination)

    def to_xarray_dataset(self):
        """
        Load data from file into an ``xarray.Dataset``.
        """
        data = l2c_ice.open(self.filename)
        time = (
            np.datetime64("1993-01-01T00:00:00", "s")
            + data.attrs["start_time"][0, 0].astype("timedelta64[s]")
            + data.time_since_start.data.astype("timedelta64[s]")
        )
        data["time"] = (("rays"), time)
        return data


def resample_data(target_dataset, target_grid, cloudsat_data):
    """
    Resample cloudsat data and include in dataset.

    This function adds IWC and IWP variables in-place to the provided
    dataset.

    Args:
        target_dataset: The ``xarray.Dataset`` to which to add resampled
            IWC and IWP.
        target_grid: pyresample area definition defining the grid of the
            target dataset.
    """
    resampler = BucketResampler(
        target_grid,
        source_lons=da.from_array(cloudsat_data.longitude.data),
        source_lats=da.from_array(cloudsat_data.latitude.data)
    )
    # Indices of random samples.
    indices_target, indices_source = get_sample_indices(resampler)

    # Resample IWP.
    iwp_r = resampler.get_average(cloudsat_data.iwp.data).compute()[::-1]
    iwp_r_rand = np.zeros_like(iwp_r)
    iwp_r_rand.ravel()[indices_target] = cloudsat_data.iwp.data[indices_source]

    # Resample CloudSat time.
    time_r = resampler.get_average(cloudsat_data.time.data.astype(np.int64)).compute()[
        ::-1
    ]
    time_r = time_r.astype(np.int64).astype(cloudsat_data.time.data.dtype)

    # Smooth, resample and remap IWC to fixed altitude relative
    # to surface.
    iwc = cloudsat_data.iwc.data
    height = cloudsat_data.height.data
    iwc, height = subsample_iwc_and_height(iwc, height)
    surface_altitude = np.maximum(cloudsat_data.surface_elevation.data, 0.0)

    # Pick random samples from iwc, height and surface altitude.
    iwc = iwc[indices_source]
    height = height[indices_source]
    surface_altitude = surface_altitude[indices_source]
    altitude_levels = (np.arange(0, 20) + 0.5) * 1e3
    iwc = remap_iwc(iwc, height, surface_altitude, altitude_levels)

    iwc_r = np.zeros(iwp_r.shape + (20,), dtype=np.float32) * np.nan
    iwc_r.reshape(-1, 20)[indices_target] = iwc

    target_dataset["levels"] = (("levels",), altitude_levels)
    target_dataset["iwc"] = (("latitude", "longitude", "levels"), iwc_r)
    target_dataset["iwp"] = (("latitude", "longitude"), iwp_r)
    target_dataset["iwp_rand"] = (("latitude", "longitude"), iwp_r_rand)
    target_dataset["time_cloudsat"] = (("latitude", "longitude"), time_r)
