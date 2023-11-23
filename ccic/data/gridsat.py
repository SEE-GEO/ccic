"""
ccic.data.gridsat
=================

This module provides classes to represent and handle the NOAA
GridSat files.
"""
from datetime import datetime
import logging
from pathlib import Path

import numpy as np
from pansat.download.providers.noaa_ncei import NOAANCEIProvider
from pansat.products.satellite.gridsat import gridsat_b1
from pansat.time import to_datetime
from pyresample import create_area_def
import torch
import xarray as xr

from ccic.data import cloudsat
from ccic.data.utils import included_pixel_mask, extract_roi

PROVIDER = NOAANCEIProvider(gridsat_b1)
GRIDSAT_GRID = create_area_def(
    "gridsat_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.035, -70.035, 179.975, 69.965],
    resolution=0.07,
    units="degrees",
    description="GridSat grid.",
)


class GridSat:
    """
    Interface to download an read GridSat B1 files.
    """

    provider = PROVIDER

    @classmethod
    def find_files(cls, path, start_time=None, end_time=None):
        """
        Find GridSat files in folder.

        Args:
            path: Path to the folder in which to look for GridSat files.
            start_time: Optional start time to filter returned files.
            end_time: Optional end time to filter returned files.

        Return:
            A list containing the found GridSat files that match the
            time constraints given by 'start_time' and 'end_time'

        """
        pattern = r"**/GRIDSAT-B1.????.??.??.??.v02r01.nc"
        files = sorted(list(Path(path).glob(pattern)))

        def get_date(path):
            return datetime.strptime(
                path.name,
                "GRIDSAT-B1.%Y.%m.%d.%H.v02r01.nc"
            )

        if start_time is not None:
            start_time = to_datetime(start_time)
            files = [
                fil for fil in files
                if get_date(fil) >= start_time
            ]

        if end_time is not None:
            end_time = to_datetime(end_time)
            files = [
                fil for fil in files
                if get_date(fil) <= end_time
            ]
        return files

    @classmethod
    def get_available_files(cls, start_time, end_time=None):
        """
        Return list of times at which this data is available.
        """
        start_time = to_datetime(start_time)
        day = int(start_time.strftime("%j"))

        if end_time is None:
            files = PROVIDER.get_files_by_day(start_time.year, day)
        else:
            end_time = to_datetime(end_time)
            files = PROVIDER.get_files_in_range(start_time, end_time)
        return files

    @classmethod
    def download(cls, filename, destination):
        logger = logging.getLogger(__name__)
        logger.info(
            "Starting download of file '%s' to '%s'.",
            filename,
            destination
        )
        return PROVIDER.download_file(filename, destination)

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

    def get_input_file_attributes(self):
        """Get attributes from input file to include in retrieval output."""
        return {
            "input_filename": self.filename.name,
            "processing_time": datetime.now().isoformat()
        }

    def get_retrieval_input(self, roi=None):
        """
        Return observations as retrieval input.

        NOTE: Even if a bounding box is given, a minimum size of 256 x 256
            pixels is enforced.

        Args:
            roi: Coordinates ``(lon_min, lat_min, lon_max, lat_max)`` of a
                bounding box from which to extract the training data. If given,
                only data from the given bounding box is extracted.

        Return:
            A torch tensor containing the observations as a torch.tensor that
            can be fed into the CCIC retrieval model.
        """
        from ccic.data.training_data import NORMALIZER
        input_data = self.to_xarray_dataset()
        if roi is not None:
            input_data = extract_roi(input_data, roi, min_size=256)
        xs = []
        for name in ["irwin_cdr"]:
            xs.append(input_data[name].data[0])
        x = NORMALIZER(np.stack(xs))
        return torch.tensor(x[None]).to(torch.float32)


    def get_matches(self, rng, cloudsat_files, size=128, timedelta=15):
        """
        Extract matches of given cloudsat data with observations.

        Args:
            rng: Numpy random generator to use for randomizing the scene
                extraction.
            cloudsat_files: A list of the CloudSat file objects to match
                with the GridSat data.
            size: The size of the windows to extract.
            timedelta: The maximum time different for collocations.

        Return:
            List of ``xarray.Dataset`` object with the extracted matches.
        """
        logger = logging.getLogger(__name__)

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
            data, GRIDSAT_GRID, cloudsat_files, start_time=start_time, end_time=end_time
        )
        if data is None:
            return []

        #
        # Scene extraction
        #

        scenes = []

        indices = np.where(np.isfinite(data.tiwp.data))
        sort = np.argsort(data.time_cloudsat.data[indices[0], indices[1]])
        rnd = rng.permutation(indices[0].size)
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
                    "CPC IR file '%s'.", self
                )
                break
            indices = (indices[0][~included], indices[1][~included])

            coords = {
                "latitude": slice(i_start, i_end),
                "longitude": slice(j_start, j_end),
            }
            scene = data[coords]
            if (scene.latitude.size == size) and (scene.longitude.size == size):
                granule = cloudsat_files[0].granule
                scene.attrs["granule"] = f"{granule:06}"
                scene.attrs["input_source"] = "GRIDSAT"
                scenes.append(scene.copy())

        del data
        return scenes

    def to_xarray_dataset(self):
        """Return data in file as ``xarray.Dataset``"""
        data = xr.load_dataset(self.filename)
        # Flip latitudes to be consistent with pyresample.
        return data.isel(lat=slice(None, None, -1))
