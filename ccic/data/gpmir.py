"""
ccic.data.gpmir
===============

This module provides classes to read the GPM merged IR observations.
"""
from pansat.download.providers.ges_disc import Disc2Provider
from pansat.products.satellite.gpm import gpm_mergeir
from pansat.time import to_datetime
from pyresample import create_area_def
import numpy as np
import torch
import xarray as xr

from ccic.data import cloudsat

PROVIDER = Disc2Provider(gpm_mergeir)
GPMIR_GRID = create_area_def(
    "gpmir_area",
    {"proj": 'longlat', 'datum': 'WGS84'},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution=(0.036378335, 0.03638569),
    units='degrees',
    description='GPMIR grid'
)


class GPMIR:
    """
    Interface class to access GPM IR data.
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

    @staticmethod
    def download(filename, destination):
        """
        Download file to given destination.

        Args:
            filename: Name of the file to download.
            destination: Destination to store the file.
        """
        PROVIDER.download_file(filename, destination)

    def __init__(self, filename):
        """
        Args:
            filename: The local filename of the file.
        """
        self.filename = filename
        with xr.open_dataset(self.filename) as data:
            self.start_time = data.time[0].data - np.timedelta64(15 * 60, "s")
            self.end_time = data.time[-1].data + np.timedelta64(15 * 60, "s")

    def to_xarray_dataset(self):
        """Load data into ``xarray.Dataset``"""
        return xr.load_dataset(self.filename)

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

    def get_matches(self,
                    cloudsat_files,
                    size=128,
                    timedelta=15*60):
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
        data = xr.load_dataset(self.filename)
        new_names = {
            "Tb": "ir_window",
            "lat": "latitude",
            "lon": "longitude"
        }
        data = data[["Tb", "time"]].rename(new_names)

        scenes = []

        for i in range(2):
            data_t = data[{"time": i}].copy()

            start_time = data_t.time - np.array(timedelta, dtype="timedelta64[s]")
            end_time = data_t.time + np.array(timedelta, dtype="timedelta64[s]")

            data_t = cloudsat.resample_data(
                data_t,
                GPMIR_GRID,
                cloudsat_files
            )
            # No matches found
            if data_t is None:
                continue

            # Extract single scenes.
            indices = np.where(np.isfinite(data_t.iwp.data))
            sort = np.argsort(data_t.time_cloudsat.data[indices[0], indices[1]])
            indices = (indices[0][sort], indices[1][sort])

            n_pixels = int(np.sqrt(2.0) * size)
            while len(indices[0]) > 0:
                inds_i = indices[0][:n_pixels]
                inds_j = indices[1][:n_pixels]
                indices = (indices[0][n_pixels:], indices[1][n_pixels:])

                c_i = inds_i[len(inds_i) // 2]
                c_j = inds_j[len(inds_j) // 2]
                i_start = max(c_i - size // 2, 0)
                i_end = i_start + size
                j_start = max(c_j - size // 2, 0)
                j_end = j_start + size
                coords = {
                    "latitude": slice(i_start, i_end),
                    "longitude": slice(j_start, j_end),
                }
                scene = data_t[coords]
                if (scene.latitude.size == size) and (scene.longitude.size == size):
                    scenes.append(scene.copy())
        return scenes
