"""
ccic.data.dardar
==================

This module provides functionality to read and resample 
DARDAR 3.00/3.10 files, based on ccic.data.cloudsat
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import convolve
import xarray as xr

from .cloudsat import ALTITUDE_LEVELS, remap_iwc, remap_cloud_classes


def get_iwp(dataset: xr.Dataset, above_ground: bool = True) -> npt.NDArray:
    """Returns the IWP in g/m2

    Args:
        dataset: the DARDAR dataset
        above_ground: compute the IWP from the ground height based on
            the DARDAR mask (DARMASK)

    Note: IWC profiles can contain or consist only of NaNs, resulting in NaN IWP
    """
    # Get IWC values
    iwc = dataset.iwc

    # Transform them from kg/m3 to g/m3
    iwc = iwc * 1_000

    # Get a mask for IWC to zero them if they correspond to the (sub)surface
    if above_ground:
        above_ground_mask = get_surface_mask(dataset) == False
    else:
        above_ground_mask = np.ones_like(iwc.shape, dtype=bool)

    # Compute the integral, zeroing IWC if needed, and return
    # Note that the height dimension is in decreasing order, hence the flip
    return np.trapz((iwc * above_ground_mask)[..., ::-1], dataset.height[::-1], axis=1)


def get_surface_mask(dataset: xr.Dataset) -> npt.NDArray:
    """Returns a binary mask indicating if the bin is at a (sub)surface
    height based on the DARDAR mask (DARMASK)

    Args:
        dataset: The DARDAR dataset
    """
    # Sanity check for the algorithm to work:
    # height is sorted decreasingly, i.e.
    height = dataset.height.values
    assert (np.flip(np.sort(height)) == height).all()

    # Find indices where it is surface and subsurface
    surface_mask = dataset.DARMASK_Simplified_Categorization == -1

    # Make sure all heights below this maximum height are set to True
    surface_mask = np.cumsum(surface_mask, axis=1)
    surface_mask = np.where(surface_mask > 1, True, False)

    return surface_mask


def subsample_iwc_and_height(iwc, height):
    """
    Smoothes and subsamples IWC and height fields to an approximate
    resolution of 1km.

    Based on .cloudsat.subsample_iwc_and_height

    Args:
        iwc: The ice water content field from a DARDAR file.
        height: The corresponding height field.

    Return:
        A tuple ``(iwc, height)`` containing the subsampled IWC and
        height fields.
    """
    # Line adapted from .cloudsat.subsample_iwc_and_height
    # to account for the different resolutions
    k = np.linspace(-9 * 60, 9 * 60, 19)
    k = np.exp(np.log(0.05) * (k / 500) ** 2).reshape(1, -1)
    k /= k.sum()
    iwc = convolve(iwc, k, mode="valid", method="direct")
    height = convolve(height, k, mode="valid", method="direct")
    return iwc, height


class DardarFile:
    """
    Generic interface class to read DARDAR files.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Path to the DARDAR product file.
        """
        self.filename = Path(filename)
        timestamp_str = self.filename.name.split("_")[1]
        self.start_time = np.datetime64(datetime.strptime(timestamp_str, "%Y%j%H%M%S"))
        self.granule = int(self.filename.name.split("_")[2])

    def __repr__(self):
        return f"{type(self).__name__}({self.filename})"

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
        Adds latitude and longitude from DARDAR data.

        Args:
            target_dataset: The ``xarray.Dataset`` to add the resampled
                retrieval targets to.
            resampler: The ``pyresample.BucketResampler`` to use for
                resampling.
            target_indices: Indices of the flattened target grids for
                probabilistic resampling of profiles.
            source_indices: Corresponding indices of the DARDAR data for
                the probabilistic resampling of profiles.
            start_time: Optional start time to limit the source profiles that
                are resampled.
            end_time: Optional end time to limit the source profiles that
                are resampled.
        """
        data = self.to_xarray_dataset(start_time=start_time, end_time=end_time)
        latitude_r = resampler.get_average(data.latitude.data).compute()
        longitude_r = resampler.get_average(data.longitude.data).compute()

        target_dataset["latitude_dardar"] = (("latitude", "longitude"), latitude_r)
        target_dataset["longitude_dardar"] = (("latitude", "longitude"), longitude_r)

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
        Add retrieval targets from the DARDAR file (source) to
        a target dataset (target).

        Args:
            target_dataset: The ``xarray.Dataset`` to add the resampled
                retrieval targets to.
            resampler: The ``pyresample.BucketResampler`` to use for
                resampling.
            target_indices: Indices of the flattened target grids for
                probabilistic resampling of profiles.
            source_indices: Corresponding indices of the DARDAR data for
                the probabilistic resampling of profiles.
            start_time: Optional start time to limit the source profiles that
                are resampled.
            end_time: Optional end time to limit the source profiles that
                are resampled.
        """
        source_dataset = self.to_xarray_dataset(
            start_time=start_time, end_time=end_time
        )

        # Resample IWP
        iwp = get_iwp(source_dataset)
        iwp_r = resampler.get_average(iwp).compute()
        iwp_r_rand = np.nan * np.zeros_like(iwp_r)
        iwp_r_rand.ravel()[target_indices] = iwp[source_indices]

        # Resample DARDAR time
        time_r = np.zeros(iwp_r.shape, dtype="datetime64[s]")
        time_r[:] = np.datetime64("nat", "s")
        time_r.ravel()[target_indices] = source_dataset.time.data[source_indices]

        # Smooth, resample and remap IWC to fixed altitude relative
        # to surface
        # NOTE 1: Height dimension is in decreasing order in DARDAR data
        # NOTE 2: `height` is 1D, duplicate it to have the same shape as iwc
        iwc = source_dataset.iwc.data[..., ::-1]
        source_height = np.tile(source_dataset.height.data[::-1], (iwp.shape[0], 1))
        # Convert IWC from kg/m3 to g/m3 for consistency with code prepared for 2C-ICE
        iwc = iwc * 1_000
        iwc, height = subsample_iwc_and_height(iwc, source_height)

        # Compute the surface altitude for each profile and flip it for consistency
        surface_mask = get_surface_mask(source_dataset)[..., ::-1]
        # Compute the surface elevation for each profile
        surface_altitude_source = np.max(surface_mask * source_height, axis=1)

        # Pick random samples from IWC, height and surface altitude
        iwc = iwc[source_indices]
        height = height[source_indices]
        surface_altitude = surface_altitude_source[source_indices]
        iwc = remap_iwc(iwc, height, surface_altitude, ALTITUDE_LEVELS)

        iwc_r = np.zeros(iwp_r.shape + (20,), dtype=np.float32) * np.nan
        iwc_r.reshape(-1, 20)[target_indices] = iwc

        # Get the DARDAR mask labels and flip for consistency
        labels = source_dataset.DARMASK_Simplified_Categorization.data[..., ::-1]

        # Here we take as a synonym a not clear sky flag which is
        # not ground or unknown as a cloud
        cloud_mask = labels.max(axis=-1) > 0

        # Remap cloud classes
        labels = remap_cloud_classes(
            labels, source_height, surface_altitude_source, ALTITUDE_LEVELS
        )

        # Pick the labels for the corresponding profiles
        output_shape = resampler.target_area.shape
        cloud_mask_r = -1 * np.ones(output_shape, dtype=np.int8)
        cloud_mask_r.ravel()[target_indices] = cloud_mask[source_indices]
        labels_r = -1 * np.ones(output_shape + (20,), dtype=np.int8)
        labels_r.reshape(-1, 20)[target_indices] = labels[source_indices]

        target_dataset["altitude"] = (("altitude",), ALTITUDE_LEVELS)
        target_dataset["altitude"].attrs = {"units": "meters", "positive": "up"}

        target_dataset["tiwc"] = (("latitude", "longitude", "altitude"), iwc_r)
        target_dataset["tiwc"].attrs["long_name"] = "Total ice water content"
        target_dataset["tiwc"].attrs["unit"] = "g m-3"

        target_dataset["tiwp_fpavg"] = (("latitude", "longitude"), iwp_r)
        target_dataset["tiwp_fpavg"].attrs["long_name"] = "Footprint-averaged total ice water path"
        target_dataset["tiwp_fpavg"].attrs["unit"] = "g m-3"

        target_dataset["tiwp"] = (("latitude", "longitude"), iwp_r_rand)
        target_dataset["tiwp"].attrs["long_name"] = "Total ice water path"
        target_dataset["tiwp"].attrs["unit"] = "g m-3"

        target_dataset["time_dardar"] = (("latitude", "longitude"), time_r)

        target_dataset["cloud_mask"] = (("latitude", "longitude"), cloud_mask_r)
        target_dataset["cloud_mask"].attrs = {
            "long_name": "Cloud presence in atmospheric column",
            "flag_values": "0, 1",
            "flag_meaning": "no cloud, cloud present",
            "_FillValue": -1,
        }
        target_dataset["cloud_class"] = (
            ("latitude", "longitude", "altitude"),
            labels_r,
        )
        target_dataset["cloud_class"].attrs = {
            "long_name": "Resampled DARMASK Categorization Flags",
            "flag_values": "-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15",
            "flag_meanings": (
                "presence of liquid unknow",
                "surface and subsurface",
                "clear sky",
                "ice clouds",
                "spherical or 2D ice",
                "supercooled water",
                "supercooled and ice",
                "cold rain",
                "aerosol",
                "warm rain",
                "stratospheric clouds",
                "highly concentrated ice",
                "top of convective towers",
                "liquid cloud",
                "warm rain and liquid clouds",
                "cold rain and liquid clouds",
                "rain may be mixed with liquid",
                "Multiple scattering due to supercooled water",
            ),
            "_FillValue": -1,
        }

    def to_xarray_dataset(self, start_time=None, end_time=None):
        """
        Load data from file into an ``xarray.Dataset``.

        Args:
            start_time: Optional start time to limit the source profiles that
                are loaded.
            end_time: Optional end time to limit the source profiles that
                are loaded.
        """
        data = xr.open_dataset(self.filename)
        time_mask = np.ones(data.time.size, dtype=bool)
        if start_time is not None:
            time_mask *= data.time >= start_time
        if end_time is not None:
            time_mask *= data.time < end_time
        data = data.sel({"time": time_mask})
        # Apply renaming for consistency and return
        return data.rename_dims({"time": "rays"})
