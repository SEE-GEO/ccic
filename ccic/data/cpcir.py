"""
ccic.data.cpcir
===============

This module provides classes to read the NCEP/CPC 4-km IR data.
"""
from datetime import datetime
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
from ccic.data.utils import included_pixel_mask, extract_roi

PROVIDER = Disc2Provider(gpm_mergeir)
CPCIR_GRID = create_area_def(
    "cpcir_area",
    {"proj": "longlat", "datum": "WGS84"},
    area_extent=[-180.0, -60.0, 180.0, 60.0],
    resolution= (0.03637833468067906, 0.036385688295936934),
    units="degrees",
    description="CPCIR grid",
)


def extract_scenes(rng, collocation, size):
    """
    Extract scenes from collocations.

    Args:
        rng: Numpy random generator to use to randomize the scene
            extraction.
        collocation: 'xarray.Dataset' containing the collocated
            observations and CloudSat retrievals.
        size: The size of the scenes to extract.

    Return:
        A list of the extracted scenes.
    """
    # Extract single scenes.
    indices = np.where(np.isfinite(collocation.tiwp.data))
    rnd = rng.permutation(indices[0].size)
    indices = (indices[0][rnd], indices[1][rnd])

    scenes = []

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
        if j_end >= collocation.longitude.size:
            shift = j_end - collocation.longitude.size - 1
            j_start += shift
            j_end += shift
        if i_start < 0:
            shift = np.abs(i_start)
            i_start += shift
            i_end += shift
        if i_end >= collocation.latitude.size:
            shift = i_end - collocation.latitude.size - 1
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
        scene = collocation[coords]
        if (scene.latitude.size == size) and (scene.longitude.size == size):
            scenes.append(scene.copy())
    return scenes

def subsample_dataset(dataset):
    """
    Subsamples CPCIR dataset by a factor of two.

    Args:
        dataset: The content of the CPCIR file as xarray.Dataset

    Return:
        The subsampled dataset.
    """
    dataset_new = dataset[{
        "longitude": slice(0, None, 2),
        "latitude": slice(0, None, 2),
    }].copy()

    dataset_new["ir_win"].data += dataset.ir_win.data[0::2, 1::2]
    dataset_new["ir_win"].data += dataset.ir_win.data[1::2, 0::2]
    dataset_new["ir_win"].data += dataset.ir_win.data[1::2, 1::2]

    dataset_new["longitude"] = 0.5 * (
        dataset.longitude.data[::2] +
        dataset.longitude.data[1::2]
    )
    dataset_new["latitude"] = 0.5 * (
        dataset.latitude.data[::2] +
        dataset.latitude.data[1::2]
    )

    return dataset_new


class CPCIR:
    """
    Interface class to access NCEP/CPC 4-km IR data.
    """
    provider = PROVIDER

    @classmethod
    def find_files(cls, path, start_time=None, end_time=None):
        """
        Find CPCIR files in folder.

        Args:
            path: Path to the folder in which to look for CPCIR files.
            start_time: Optional start time to filter returned files.
            end_time: Optional end time to filter returned files.

        Return:
            A list containing the found CPCIR files that match the
            time constraints given by 'start_time' and 'end_time'

        """
        pattern = r"**/merg_??????????_4km-pixel.nc4"
        files = sorted(list(Path(path).glob(pattern)))

        def get_date(path):
            return datetime.strptime(
                path.name,
                "merg_%Y%m%d%H_4km-pixel.nc4"
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
        """
        Download file to given destination.

        Args:
            filename: Name of the file to download.
            destination: Destination to store the file.
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "Starting download of file '%s' to '%s'.",
            filename,
            destination
        )
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

    def get_input_file_attributes(self):
        """Get attributes from input file to include in retrieval output."""
        return {
            "input_filename": self.filename.name,
            "processing_time": datetime.now().isoformat()
        }

    def get_retrieval_input(self, roi=None):
        """
        Load and normalize retrieval input from file.

        NOTE: Even if a bounding box is given, a minimum size of 256 x 256
            pixels is enforced.

        Args:
            roi: Coordinates ``(lon_min, lat_min, lon_max, lat_max)`` of a
                bounding box from which to extract the training data. If given,
                only data from the given bounding box is extracted.

        Return:

            A torch tensor containing the observations from this file
            as input.
        """
        from ccic.data.training_data import NORMALIZER

        input_data = self.to_xarray_dataset()
        if roi is not None:
            input_data = extract_roi(input_data, roi, min_size=256)

        m = input_data.lat.size
        n = input_data.lon.size
        tbs = input_data.Tb.data

        xs = []
        for i in range(2):
            x_i = np.nan * np.ones((1, m, n))
            x_i[-1] = tbs[i]
            xs.append(NORMALIZER(x_i))
        return torch.tensor(np.stack(xs)).to(torch.float32)

    def get_matches(
            self,
            rng,
            cloudsat_files,
            size=128,
            timedelta=15,
            subsample=False
    ):
        """
        Extract matches of given cloudsat data with observations.

        Args:
            rng: Numpy random generator to use for randomizing the scene
                extraction.
            cloudsat_files: List of paths to CloudSat product files
                with which to match the data.
            size: The size of the windows to extract.
            timedelta: The time range in minutes for which to extract matches.
            subsample: If set to 'True' samples will be extract at resolution
                reduced by a factor of two to roughly match that of the
                GridSat B1 dataset.

        Return:
            List of ``xarray.Dataset`` object with the extracted matches.
        """
        logger = logging.getLogger(__name__)
        data = self.to_xarray_dataset()

        new_names = {"Tb": "ir_win", "lat": "latitude", "lon": "longitude"}
        data = data[["Tb", "time"]].rename(new_names)

        scenes = []

        for t_index in range(2):
            data_t = data[{
                "time": t_index,
            }].copy()

            if subsample:
                data_t = subsample_dataset(data_t)
                source = "CPCIR2"
                grid = CPCIR_GRID[::2, ::2]
            else:
                source = "CPCIR"
                grid = CPCIR_GRID

            start_time = (
                data_t.time -
                np.array(60 * timedelta, dtype="timedelta64[s]")
            )
            end_time = (
                data_t.time +
                np.array(60 * timedelta, dtype="timedelta64[s]")
            )
            data_t = cloudsat.resample_data(
                data_t, grid, cloudsat_files, start_time, end_time
            )
            # No matches found
            if data_t is None:
                continue

            new_scenes = extract_scenes(rng, data_t, size)
            for scene in new_scenes:
                granule = cloudsat_files[0].granule
                scene.attrs["granule"] = f"{granule:06}"
                scene.attrs["input_source"] = source
                scenes.append(scene.copy())
            scenes += new_scenes

            del data_t

        del data
        return scenes
