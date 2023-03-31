"""
ccic.data.dardar
==================

This module provides functionality to read and resample 
DARDAR 3.00/3.10 files, based on ccic.data.cloudsat
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .cloudsat import (
    get_sample_indices,
    remap_iwc,
    remap_cloud_classes,
    subsample_iwc_and_height,
    ALTITUDE_LEVELS
)

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
        timestamp_str = self.filename.name.split('_')[1]
        self.start_time = np.datetime64(datetime.strptime(timestamp_str, '%Y%j%H%M%S'))
        self.granule = int(self.filename.name.split('_')[2])
    
    def __repr__(self):
        return f"{type(self).__name__}({self.filename})"
    
    def get_surface_mask(self, dataset):
        """Returns a binary mask indicating if the bin is at a (sub)surface
        height based on the DARDAR mask (DARMASK)"""
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
    
    def get_iwp(self, dataset, from_ground=True):

        # Get IWC values
        iwc = dataset.iwc
        
        # Get a mask for IWC to zero them if they correspond to the (sub)surface
        if from_ground:
            above_ground_mask = (self.get_surface_mask(dataset) == False)
        else:
            above_ground_mask = np.ones_like(iwc.shape, dtype=bool)

        # Compute the integral, zeroing IWC if needed, and return
        return np.trapz(iwc * above_ground_mask, dataset.height, axis=1)

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
        data = data.sel({'time': time_mask})
        return data
    
    def add_retrieval_targets(
            self,
            target_dataset,
            resampler,
            target_indices,
            source_indices,
            start_time=None,
            end_time=None
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
        source_dataset = self.to_xarray_dataset(start_time=start_time, end_time=end_time)

        # Resample IWP
        iwp = self.get_iwp(source_dataset)
        iwp_r = resampler.get_average(iwp).compute()
        iwp_r_rand = np.nan * np.zeros_like(iwp_r)
        iwp_r_rand.ravel()[target_indices] = iwp[source_indices]

        # Resample DARDAR time
        time_r = np.zeros(
            iwp_r.shape,
            dtype="datetime64[s]"
        )
        time_r[:] = np.datetime64("nat", "s")
        time_r.ravel()[target_indices] = source_dataset.time.data[source_indices]

        # Smooth, resample and remap IWC to fixed altitude relative
        # to surface
        # NOTE 1: Height dimension is in decreasing order in DARDAR data
        # NOTE 2: `height` is 1D
        iwc = source_dataset.iwc.data[..., ::-1]
        height = source_dataset.height.data[::-1]
        iwc, height = subsample_iwc_and_height(iwc, height)

        # Compute the surface altitude for each profile and flip it for consistency
        surface_mask = self.get_surface_mask(source_dataset)[..., ::-1]
        # Compute the surface elevation for each profile
        surface_elevation = np.max(surface_mask * height, axis=1)


        # Pick random samples from IWC, height and surface altitude
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