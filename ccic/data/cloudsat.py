"""
ccic.data.cloudsat
==================

This module provides functionality to read and resample  CloudSat 2C-Ice
files.
"""
from pathlib import Path
import warnings

import dask.array as da
import numpy as np
from pansat.download.providers.cloudsat_dpc import CloudSatDPCProvider
from pansat.products.satellite.cloud_sat import l2c_ice, l2b_cldclass
from pansat.time import to_datetime, to_datetime64
from pyresample.bucket import BucketResampler
from scipy.signal import convolve
from scipy.interpolate import interp1d


PROVIDER_2CICE = CloudSatDPCProvider(l2c_ice)
PROVIDER_2BCLDCLASS = CloudSatDPCProvider(l2b_cldclass)
ALTITUDE_LEVELS = (np.arange(0, 20) + 0.5) * 1e3

CLOUD_CLASSES = [
    "No cloud",
    "Ci",
    "As",
    "Ac",
    "St",
    "Sc",
    "Cu",
    "Ns",
    "DC"
]


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
    k = np.linspace(-5 * 240, 5 * 240, 11)
    k = np.exp(np.log(0.5) * (k / 500) ** 2).reshape(1, -1)
    k /= k.sum()
    iwc = convolve(iwc, k, mode="valid", method="direct")
    height = convolve(height, k, mode="valid", method="direct")
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
    iwp = np.trapz(iwc, x=altitude, axis=-1)

    for index in range(iwc.shape[0]):
        z_new = target_altitudes
        z_old = altitude[index] - surface_altitude[index]
        iwc_r[index] = np.interp(z_new, z_old, iwc[index])
        if iwp[index] > 1e-4:
            iwp_r = np.trapz(iwc_r[index], x=target_altitudes)
            iwc_r[index] *= iwp[index] / iwp_r

    return iwc_r


def remap_cloud_classes(
    labels,
    altitude,
    surface_altitude,
    target_altitudes,
):
    """
    Remap cloud class fields to new altitude grid relative to surface.


    Args:
        labels: 2D array containing cloud-class labels with altitude levels
            along last dimension.
        altitude: The altitudes corresponding to ``classes``
        surface_altitude: The surface altitude for each profile.
        target_altitude: 1D array specifying the altitudes to remap the
            IWC data to.

    Return:
        2D array containing the cloud labels.
    """
    n_levels = labels.shape[-1]
    indices = np.arange(0, n_levels, 4)
    indices += np.random.randint(0, 3, size=indices.size)
    indices = np.minimum(indices, n_levels - 1)

    labels = labels[..., indices]
    altitude = altitude[..., indices].astype(np.float32)
    surface_altitude = np.maximum(surface_altitude, 0).astype(np.float32)

    labels_r = np.zeros(
        labels.shape[:-1] + (target_altitudes.size,), dtype=labels.dtype
    )

    for index in range(labels.shape[0]):
        z_old = altitude[index]
        z_new = target_altitudes
        z_old = z_old - surface_altitude[index]
        interpolator = interp1d(
            z_old,
            labels[index],
            kind="nearest",
            bounds_error=False,
            fill_value=-99
        )
        labels_r[index] = interpolator(z_new)

    return labels_r


class CloudsatFile:
    product = None
    provider = None
    """
    Generic interface class to read CloudSat files.
    """

    @classmethod
    def get_available_files(cls, date):
        """
        Return list of times at which this data is available.
        """
        date = to_datetime(date)
        day = int(date.strftime("%j"))
        files = cls.provider.get_files_by_day(date.year, day)
        return files

    @classmethod
    def download(cls, filename, destination):
        """
        Download file.

        Args:
            filename: The filename of the 2CIce file to download.
            destination: Path to store the result.
        """
        cls.provider.download_file(filename, destination)


    @staticmethod
    def download_files(date, destination):
        """
        Download all files for a given day and return a dictionary
        mapping start time to CloudSat files.
        """
        destination = Path(destination)
        available_files = cls.get_available_files(date)
        files = []
        for filename in available_files:
            cls.download(filename, destination / filename)
            files.append(cls(destination_filename))
        return {start_time: f for f in files}


    def __init__(self, filename):
        """
        Args:
            filename: Path to the CloudSat product file.
        """
        self.filename = Path(filename)
        self.start_time = to_datetime64(
            self.product.filename_to_date(self.filename)
        )
        self.granule = int(self.filename.name.split("_")[1])

    def __repr__(self):
        return f"{type(self).__name__}({self.filename})"

    def to_xarray_dataset(self, start_time=None, end_time=None):
        """
        Load data from file into an ``xarray.Dataset``.

        Args:
            start_time: Optional start time to limit the source profiles that
                are loaded.
            end_time: Optional end time to limit the source profiles that
                are loaded.
        """
        data = self.product.open(self.filename)
        time = (
            np.datetime64("1993-01-01T00:00:00", "s")
            + data.attrs["start_time"][0, 0].astype("timedelta64[s]")
            + data.time_since_start.data.astype("timedelta64[s]")
        )
        data["time"] = (("rays"), time)

        time_mask = np.ones(data.time.size, dtype=bool)
        if start_time is not None:
            time_mask *= data.time >= start_time
        if end_time is not None:
            time_mask *= data.time < end_time
        data = data[{"rays": time_mask}]

        return data

    def add_latitude_and_longitude(
        self,
        target_dataset,
        resampler,
        target_indices,
        source_indices,
        start_time=None,
        end_time=None,
    ):
        """
        Adds latitude and longitude from CloudSat data.

        Args:
            target_dataset: The ``xarray.Dataset`` to add the resampled
                retrieval targets to.
            resampler: The ``pyresample.BucketResampler`` to use for
                resampling.
            target_indices: Indices of the flattened target grids for
                probabilistic resampling of profiles.
            source_indices: Corresponding indices of the 2CIce data for
                the probabilistic resampling of profiles.
            start_time: Optional start time to limit the source profiles that
                are resampled.
            end_time: Optional end time to limit the source profiles that
                are resampled.
        """
        data = self.to_xarray_dataset(start_time=start_time, end_time=end_time)
        latitude_r = resampler.get_average(data.latitude.data).compute()
        longitude_r = resampler.get_average(data.longitude.data).compute()

        target_dataset["latitude_cloudsat"] = (("latitude", "longitude"), latitude_r)
        target_dataset["longitude_cloudsat"] = (("latitude", "longitude"), longitude_r)


class CloudSat2CIce(CloudsatFile):
    """
    Interface class to read CloudSat 2C-Ice files.
    """
    provider = PROVIDER_2CICE
    product = l2c_ice

    def add_retrieval_targets(
        self,
        target_dataset,
        resampler,
        target_indices,
        source_indices,
        start_time=None,
        end_time=None,
    ):
        """
        Add retrieval targets from the CloudSat2CIce file (source) to
        a target dataset (target).

        Args:
            target_dataset: The ``xarray.Dataset`` to add the resampled
                retrieval targets to.
            resampler: The ``pyresample.BucketResampler`` to use for
                resampling.
            target_indices: Indices of the flattened target grids for
                probabilistic resampling of profiles.
            source_indices: Corresponding indices of the 2CIce data for
                the probabilistic resampling of profiles.
            start_time: Optional start time to limit the source profiles that
                are resampled.
            end_time: Optional end time to limit the source profiles that
                are resampled.
        """
        data = self.to_xarray_dataset(start_time=start_time, end_time=end_time)

        # Resample IWP.
        iwp_r = resampler.get_average(data.iwp.data).compute()
        iwp_r_rand = np.nan * np.zeros_like(iwp_r)
        iwp_r_rand.ravel()[target_indices] = data.iwp.data[source_indices]

        # Resample CloudSat time.
        time_r = np.zeros(
            iwp_r.shape,
            dtype="datetime64[s]"
        )
        time_r[:] = np.datetime64("nat", "s")
        time_r.ravel()[target_indices] = data.time.data[source_indices]

        # Smooth, resample and remap IWC to fixed altitude relative
        # to surface.
        # NOTE: Height dimension is flipped in CloudSat data.
        iwc = data.iwc.data[..., ::-1]
        height = data.height.data[..., ::-1]
        iwc, height = subsample_iwc_and_height(iwc, height)
        surface_altitude = np.maximum(data.surface_elevation.data, 0.0)

        # Pick random samples from iwc, height and surface altitude.
        iwc = iwc[source_indices]
        height = height[source_indices]
        surface_altitude = surface_altitude[source_indices]
        iwc = remap_iwc(iwc, height, surface_altitude, ALTITUDE_LEVELS)

        iwc_r = np.zeros(iwp_r.shape + (20,), dtype=np.float32) * np.nan
        iwc_r.reshape(-1, 20)[target_indices] = iwc

        target_dataset["altitude"] = (("altitude",), ALTITUDE_LEVELS)
        target_dataset["altitude"].attrs = {
            "units": "meters",
            "positive": "up"
        }

        target_dataset["tiwc"] = (("latitude", "longitude", "altitude"), iwc_r)
        target_dataset["tiwc"].attrs["long_name"] = "Total ice water content"
        target_dataset["tiwc"].attrs["unit"] = "g m-3"

        target_dataset["tiwp_fpavg"] = (("latitude", "longitude"), iwp_r)
        target_dataset["tiwp_fpavg"].attrs["long_name"] = "Footprint-averaged total ice water path"
        target_dataset["tiwp_fpavg"].attrs["unit"] = "g m-3"

        target_dataset["tiwp"] = (("latitude", "longitude"), iwp_r_rand)
        target_dataset["tiwp"].attrs["long_name"] = "Total ice water path"
        target_dataset["tiwp"].attrs["unit"] = "g m-3"

        target_dataset["time_cloudsat"] = (("latitude", "longitude"), time_r)


class CloudSat2BCLDCLASS(CloudsatFile):
    """
    Interface class to read CloudSat 2B-CLDCLASS files.
    """

    provider = PROVIDER_2BCLDCLASS
    product = l2b_cldclass

    def add_retrieval_targets(
        self,
        target_dataset,
        resampler,
        target_indices,
        source_indices,
        start_time=None,
        end_time=None,
    ):
        """
        Add retrieval targets from the CloudSat2BCldClass file (source) to
        a target dataset (target).

        Args:
            target_dataset: The ``xarray.Dataset`` to add the resampled
                retrieval targets to.
            resampler: The ``pyresample.BucketResampler`` to use for
                resampling.
            target_indices: Indices of the flattened target grids for
                probabilistic resampling of profiles.
            source_indices: Corresponding indices of the 2CIce data for
                the probabilistic resampling of profiles.
            start_time: Optional start time to limit the source profiles that
                are resampled.
            end_time: Optional end time to limit the source profiles that
                are resampled.
        """
        data = self.to_xarray_dataset(start_time=start_time, end_time=end_time)
        output_shape = resampler.target_area.shape
        labels = data.cloud_class.data[..., ::-1]
        valid = data.cloud_class_flag.data[..., ::-1] > 0
        labels[~valid] = -1

        cloud_mask = labels.max(axis=-1) > 0
        cloud_mask[np.any(~valid, -1)] = -1

        height = data.height.data[..., ::-1]
        surface_altitude = data.surface_elevation.data
        labels = remap_cloud_classes(labels, height, surface_altitude, ALTITUDE_LEVELS)

        cloud_mask_r = -1 *  np.ones(output_shape, dtype=np.int8)
        cloud_mask_r.ravel()[target_indices] = cloud_mask[source_indices]
        labels_r = -1 * np.ones(output_shape + (20,), dtype=np.int8)
        labels_r.reshape(-1, 20)[target_indices] = labels[source_indices]

        target_dataset["altitude"] = (("altitude",), ALTITUDE_LEVELS)
        target_dataset["altitude"].attrs = {
            "units": "meters",
            "positive": "up"
        }

        target_dataset["cloud_mask"] = (("latitude", "longitude"), cloud_mask_r)
        target_dataset["cloud_mask"].attrs = {
            "long_name": "Cloud presence in atmospheric column",
            "flag_values": "0, 1",
            "flag_meaning": "no cloud, cloud present",
            "_FillValue": -1
        }
        target_dataset["cloud_class"] = (("latitude", "longitude", "altitude"), labels_r)
        target_dataset["cloud_class"].attrs = {
            "long_name": "3D cloud classificiation",
            "flag_values": "0, 1, 2, 3, 4, 5, 6, 7, 8",
            "flag_meanings": ("no cloud, cirrus, altostratus, altocumulus, "
                              "stratus, stratocumulus, cumulus, nimbostratuc, "
                              "deep convection"),
            "_FillValue": -1,
        }


def get_available_granules(date):
    """
    Collects the names of available CloudSat files for a given day
    and groups them by the granule number.

    Returns:
        A list of CloudSat file objects.

    NOTE: The files are not downloaded so the filename attribute of
         the returned objects points to non-existing files.
    """
    cloudsat_files = []
    cloudsat_classes = [CloudSat2CIce, CloudSat2BCLDCLASS]
    for cls in cloudsat_classes:
        cloudsat_files += [
            cls(filename) for filename in cls.get_available_files(date)
        ]
    granules = {}
    for cs_file in cloudsat_files:
        filename = cs_file.filename
        granule = int(filename.name.split("_")[1])
        granules.setdefault(granule, []).append(cs_file)
    return granules


def resample_data(
    target_dataset,
    target_grid,
    cloudsat_files,
    start_time=None,
    end_time=None,
):
    """
    Resample cloudsat data and include in dataset.

    This function adds retrieval target variables from CloudSat
    2CIce and 2BCLDCLASS files to a target dataset.

    Args:
        target_dataset: The ``xarray.Dataset`` to which the retrieval
            targets will be added.
        target_grid: pyresample area definition defining the grid of the
            target dataset.
        cloudsat_files: List of CloudSat files from which to add retrieval
            targets to the target dataset.
        cloudsat_2bcldclass_file: Path to the CloudSat 2BCLDCLASS file from
            which to read the 2CIce data.

    Return:
        The target_dataset or ``None`` if no matches can be found within
        the given time constraints.
    """

    cloudsat_data = cloudsat_files[0].to_xarray_dataset(
        start_time=start_time, end_time=end_time
    )

    if cloudsat_data.rays.size == 0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resampler = BucketResampler(
            target_grid,
            source_lons=da.from_array(cloudsat_data.longitude.data),
            source_lats=da.from_array(cloudsat_data.latitude.data),
        )
    # Indices of random samples.
    target_indices, source_indices = get_sample_indices(resampler)

    for cloudsat_file in cloudsat_files:
        cloudsat_file.add_retrieval_targets(
            target_dataset,
            resampler,
            target_indices,
            source_indices,
            start_time=start_time,
            end_time=end_time,
        )
    cloudsat_files[0].add_latitude_and_longitude(
        target_dataset,
        resampler,
        target_indices,
        source_indices,
        start_time=start_time,
        end_time=end_time
    )
    return target_dataset
