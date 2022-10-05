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


def get_sample_indices(resampler, data):
    """
    Return indices for random bucket resampling.
    """
    indices = resampler.idxs.compute().ravel()
    data = data.ravel()
    shuffle = np.random.permute(indices.size)
    indices = indices[shuffle]
    data = data[shuffle]

    unique_inds, unique_inds_data = np.unique(indices, return_indices=True)
    data_unique = data[unique_inds_data]

    result = np.zeros((resample.target_area.size), dtype=np.float32)
    valid = unique_inds >= 0

    return unique_inds[valid], unique_inds_data[valid]


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
    source_lons = da.from_array(cloudsat_data.longitude.data)
    source_lats = da.from_array(cloudsat_data.latitude.data)
    resampler = BucketResampler(
        target_grid, source_lons=source_lons, source_lats=source_lats
    )

    iwp_r = resampler.get_average(cloudsat_data.iwp.data).compute()[::-1]

    t_time = cloudsat_data.time.data.dtype
    time_r = resampler.get_average(cloudsat_data.time.data.astype(np.int64)).compute()[
        ::-1
    ]
    time_r = time_r.astype(np.int64).astype(t_time)

    surface = np.maximum(cloudsat_data.surface_elevation, 0.0)
    surface_r = resampler.get_average(surface).compute()[::-1]

    iwc = cloudsat_data.iwc.data.T
    height = cloudsat_data.height.data.T

    k = np.linspace(-4 * 240, 4 * 240, 9)
    k = np.exp(np.log(0.5) * (k / 500) ** 2).reshape(-1, 1)
    k /= k.sum()
    iwc = convolve(iwc, k, mode="valid", method="direct")[::4]
    height = convolve(height, k, mode="valid", method="direct")[::4]

    iwc_r_n = np.zeros(iwc.shape[:1] + iwp_r.shape, dtype=np.float32)
    height_r_n = np.zeros(iwc.shape[:1] + iwp_r.shape, dtype=np.float32)

    for i in range(iwc_r_n.shape[0]):
        iwc_r_n[i] = resampler.get_average(iwc[i]).compute()[::-1]
        height_r_n[i] = resampler.get_average(height[i]).compute()[::-1]

    z = (np.arange(0, 20) + 0.5) * 1e3
    iwc_r = np.zeros(iwp_r.shape + (20,), dtype=np.float32) * np.nan

    indices = np.where(np.isfinite(iwp_r))
    sort = np.argsort(time_r[indices[0], indices[1]])
    indices = (indices[0][sort], indices[1][sort])

    for ind_i, ind_j in zip(*indices):
        z_n = height_r_n[..., ind_i, ind_j][::-1]
        s = surface_r[..., ind_i, ind_j]
        iwc_n = iwc_r_n[..., ind_i, ind_j][::-1]
        iwc_r[ind_i, ind_j] = np.interp(z, z_n - s, iwc_n)

    target_dataset["levels"] = (("levels",), z)
    target_dataset["iwc"] = (("latitude", "longitude", "levels"), iwc_r)
    target_dataset["iwp"] = (("latitude", "longitude"), iwp_r)
    target_dataset["time_cloudsat"] = (("latitude", "longitude"), time_r)
