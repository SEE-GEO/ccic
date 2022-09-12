"""
ccic.data.gridsat
=================

This module provides classes to represent and handle the NOAA
GridSat-B1 files.
"""
import numpy as np
from pansat.download.providers.noaa_ncei import NOAANCEIProvider
from pansat.products.satellite.gridsat import gridsat_b1
from pansat.time import to_datetime
from pyresample import create_area_def
import xarray as xr

from ccic.data import cloudsat

PROVIDER = NOAANCEIProvider(gridsat_b1)


class GridSatB1:

    @staticmethod
    def get_available_files(date):
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

    @staticmethod
    def download(filename, destination):
        PROVIDER.download_file(filename, destination)

    def __init__(self, filename):
        self.filename = filename
        with xr.open_dataset(self.filename) as data:
            self.start_time = data.time[0].data - np.timedelta64(15 * 60, "s")
            self.end_time = data.time[0].data + np.timedelta64(15 * 60, "s")

    def get_matches(self,
                    cloudsat_data,
                    size=128,
                    timedelta=15*60):
        """
        Extract matches of given cloudsat data with observations.

        Args:
            cloudsat_data: ``xarray.Dataset`` containing the CloudSat data
                to match.
            size: The size of the windows to extract.
            timedelta: The time range for which to extract matches.

        Return:
            List of ``xarray.Dataset`` object with the extracted matches.
        """
        data = xr.load_dataset(self.filename)[{"time": 0}]
        new_names = {
            "vschn": "vis",
            "irwvp": "ir_wv",
            "irwin_cdr": "ir_window",
            "lat": "latitude",
            "lon": "longitude"
        }
        data = data[["vschn", "irwvp", "irwin_cdr"]].rename(new_names)

        start_time = data.time - np.array(timedelta, dtype="timedelta64[s]")
        end_time = data.time + np.array(timedelta, dtype="timedelta64[s]")

        # Determine rays within timedelta
        indices = (cloudsat_data.time >= start_time) * (cloudsat_data.time <= end_time)
        if indices.sum() == 0:
            return []

        cloudsat_data = cloudsat_data[{"rays": indices}]

        #
        # Resampling
        #

        target_grid = create_area_def(
            'gridsat_area',
            {'proj': 'longlat', 'datum': 'WGS84'},
            area_extent=[-180.035, -70.035, 179.975, 69.965],
            resolution=0.07,
            units='degrees',
            description='GridSat-B1 grid.'
        )
        cloudsat.resample_data(data, target_grid, cloudsat_data)

        #
        # Scene extraction
        #

        scenes = []

        indices = np.where(np.isfinite(data.iwp.data))
        sort = np.argsort(data.time_cloudsat.data[indices[0], indices[1]])
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
            scene = data[coords]
            if (scene.latitude.size == size) and (scene.longitude.size == size):
                scene.attrs = {}
                scene.attrs["source"] = "GridSat"
                scenes.append(scene)

        return scenes

    def to_xarray_dataset(self):
        """Return data in file as ``xarray.Dataset``"""
        return xr.load_dataset(self.filename)
