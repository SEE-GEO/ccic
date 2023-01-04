"""
ccic.data.gridsat
=================

This module provides classes to represent and handle the NOAA
GridSat-B1 files.
"""
import logging
from pathlib import Path

import numpy as np
from pansat.download.providers.noaa_ncei import NOAANCEIProvider
from pansat.products.satellite.gridsat import gridsat_b1
from pansat.time import to_datetime
from pyresample import create_area_def
import xarray as xr

from ccic.data import cloudsat
from ccic.data.utils import included_pixel_mask

PROVIDER = NOAANCEIProvider(gridsat_b1)
GRIDSAT_GRID = create_area_def(
    "gridsat_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.035, -70.035, 179.975, 69.965],
    resolution=0.07,
    units="degrees",
    description="GridSat-B1 grid.",
)


class GridSatB1:
    """
    Interface to download an read GridSat B1 files.
    """

    provider = PROVIDER

    @classmethod
    def get_available_files(cls, date):
        """
        Return list of available files for a given day.

        Args:
            date: The desired date
        Return:
            List of filename that are available on the given day.
        """
        date = to_datetime(date)
        day = int(date.strftime("%j"))
        files = PROVIDER.get_files_by_day(date.year, day)
        return files

    @classmethod
    def download(cls, filename, destination):
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
        self.filename = filename
        with xr.open_dataset(self.filename) as data:
            self.start_time = data.time[0].data
            self.end_time = data.time[0].data

    def get_matches(self, cloudsat_data, size=128, timedelta=15):
        """
        Extract matches of given cloudsat data with observations.

        Args:
            cloudsat_data: ``xarray.Dataset`` containing the CloudSat data
                to match.
            size: The size of the windows to extract.
            timedelta: The maximum time different for collocations.

        Return:
            List of ``xarray.Dataset`` object with the extracted matches.
        """
        logger = logging.getLogger(__file__)

        data = self.to_xarray_dataset()[{"time": 0}]
        new_names = {
            "vschn": "vis",
            "irwvp": "ir_wv",
            "irwin_cdr": "ir_win",
            "lat": "latitude",
            "lon": "longitude",
        }
        data = data[["vschn", "irwvp", "irwin_cdr"]].rename(new_names)

        start_time = data.time - np.array(60 * timedelta, dtype="timedelta64[s]")
        end_time = data.time + np.array(60 * timedelta, dtype="timedelta64[s]")

        data = cloudsat.resample_data(
            data, GRIDSAT_GRID, cloudsat_data, start_time=start_time, end_time=end_time
        )
        if data is None:
            return []

        #
        # Scene extraction
        #

        scenes = []

        indices = np.where(np.isfinite(data.tiwp.data))
        sort = np.argsort(data.time_cloudsat.data[indices[0], indices[1]])
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
            if j_end >= data.longitude.size:
                shift = j_end - data.longitude.size - 1
                j_start += shift
                j_end += shift
            if i_start < 0:
                shift = np.abs(i_start)
                i_start += shift
                i_end += shift
            if i_end >= data.latitude.size:
                shift = i_end - data.latitude.size - 1
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
            scene = data[coords]
            if (scene.latitude.size == size) and (scene.longitude.size == size):
                scene.attrs = {}
                scene.attrs["input_source"] = "GRIDSAT"
                scenes.append(scene.copy())

        return scenes

    def to_xarray_dataset(self):
        """Return data in file as ``xarray.Dataset``"""
        data = xr.load_dataset(self.filename)
        # Flip latitudes to be consistent with pyresample.
        return data.isel(lat=slice(None, None, -1))
