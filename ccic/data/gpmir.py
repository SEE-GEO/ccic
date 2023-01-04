"""
ccic.data.gpmir
===============

This module provides classes to read the GPM merged IR observations.
"""
import logging
from pathlib import Path

from pansat.download.providers.ges_disc import Disc2Provider
from pansat.products.satellite.gpm import gpm_mergeir
from pansat.time import to_datetime
from pyresample import create_area_def
import numpy as np
import torch
import xarray as xr

from ccic.data import cloudsat
from ccic.data.utils import included_pixel_mask


PROVIDER = Disc2Provider(gpm_mergeir)
GPMIR_GRID = create_area_def(
    "gpmir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution= (0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="GPMIR grid",
)


class GPMIR:
    """
    Interface class to access GPM IR data.
    """

    provider = PROVIDER

    @classmethod
    def get_available_files(cls, date):
        """
        Return list of times at which this data is available.
        """
        date = to_datetime(date)
        day = int(date.strftime("%j"))
        files = PROVIDER.get_files_by_day(date.year, day)
        return files

    @classmethod
    def download(cls, filename, destination):
        """
        Download file to given destination.

        Args:
            filename: Name of the file to download.
            destination: Destination to store the file.
        """
        PROVIDER.download_file(filename, destination)

    @classmethod
    def download_files(cls, date, destination):
        """
        Download all files for a given day and return a dictionary
        mapping start time to CloudSat files.
        """
        destination = Path(destination)
        available_files = cls.get_available_files(date)
        files = []
        for filename in available_files:
            output_file = destination / filename
            cls.download(filename, output_file)
            files.append(cls(output_file))
        return files

    def __init__(self, filename):
        """
        Args:
            filename: The local filename of the file.
        """
        self.filename = filename
        with xr.open_dataset(self.filename) as data:
            self.start_time = data.time[0].data
            self.end_time = data.time[-1].data

    def to_xarray_dataset(self):
        """Load data into ``xarray.Dataset``"""
        data = xr.load_dataset(self.filename)
        # Flip latitudes to be consistent with pyresample.
        return data.isel(lat=slice(None, None, -1))

    def get_retrieval_input(self):
        """
        Load and normalize retrieval input from file.
        """
        from ccic.data.training_data import NORMALIZER

        input_data = xr.load_dataset(self.filename)
        m = input_data.lat.size
        n = input_data.lon.size
        tbs = input_data.Tb.data

        xs = []
        for i in range(2):
            x_i = np.nan * np.ones((3, m, n))
            x_i[-1] = tbs[i]
            x_i = NORMALIZER(x_i)
            xs.append(x_i)
        x = np.stack(xs)

        return torch.tensor(x).to(torch.float32)

    def get_matches(self, cloudsat_files, size=128, timedelta=15):
        """
        Extract matches of given cloudsat data with observations.

        Args:
            cloudsat_files: List of paths to CloudSat product files
                with which to match the data.
            size: The size of the windows to extract.
            timedelta: The time range for which to extract matches.

        Return:
            List of ``xarray.Dataset`` object with the extracted matches.
        """
        logger = logging.getLogger(__file__)

        data = self.to_xarray_dataset()
        new_names = {"Tb": "ir_win", "lat": "latitude", "lon": "longitude"}
        data = data[["Tb", "time"]].rename(new_names)

        scenes = []

        for i in range(2):
            data_t = data[{"time": i}].copy()
            start_time = data_t.time - np.array(60 * timedelta, dtype="timedelta64[s]")
            end_time = data_t.time + np.array(60 * timedelta, dtype="timedelta64[s]")
            data_t = cloudsat.resample_data(
                data_t, GPMIR_GRID, cloudsat_files, start_time, end_time
            )
            # No matches found
            if data_t is None:
                continue

            # Extract single scenes.
            indices = np.where(np.isfinite(data_t.tiwp.data))
            rnd = np.random.permutation(indices[0].size)
            indices = (indices[0][rnd], indices[1][rnd])

            while len(indices[0]) > 0:

                # Choose random pixel and use a center of window.
                # Add zonal offset.
                c_i, c_j = indices[0][0], indices[1][0]
                offset = np.random.randint(
                    -int(0.3 * size), int(0.3 * size)
                )
                c_j += offset

                i_start = c_i - size // 2
                i_end = c_i + (size - size // 2)
                j_start = c_j - size // 2
                j_end = c_j + (size - size // 2)

                # Shift window if it exceeds boundaries
                if j_start < 0:
                    shift = np.abs(j_start)
                    j_start += shift
                    j_end += shift
                if j_end >= data_t.longitude.size:
                    shift = j_end - data_t.longitude.size - 1
                    j_start += shift
                    j_end += shift
                if i_start < 0:
                    shift = np.abs(i_start)
                    i_start += shift
                    i_end += shift
                if i_end >= data_t.latitude.size:
                    shift = i_end - data_t.latitude.size - 1
                    i_start += shift
                    i_end += shift

                # Determine pixels included in scene.
                included = included_pixel_mask(
                    indices,
                    c_i,
                    c_j,
                    size
                )

                # If no pixels are in scene we need to abort to avoid
                # livelock.
                if not np.any(included):
                    logger.warning(
                        "Found an empty sample when extracting scenes from "
                        "GPM IR file '%s'.", self
                    )
                    break
                indices = (indices[0][~included], indices[1][~included])

                coords = {
                    "latitude": slice(i_start, i_end),
                    "longitude": slice(j_start, j_end),
                }
                scene = data_t[coords]
                if (scene.latitude.size == size) and (scene.longitude.size == size):
                    scene.attrs = {}
                    scene.attrs["input_source"] = "GPMIR"
                    scenes.append(scene.copy())

        return scenes
