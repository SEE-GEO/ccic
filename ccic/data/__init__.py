from concurrent.futures import ThreadPoolExecutor
from copy import copy
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
from pansat.time import to_datetime
import xarray as xr

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.gpmir import GPMIR
from ccic.data.gridsat import GridSatB1


class DownloadCache:
    """
    Asynchronous download cache..
    """
    def __init__(self, n_threads=2):
        self.path = TemporaryDirectory()
        self.files = {}
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self.lock = Lock()

    def get(self, product, filename):
        """
        Get file from the cache.
        """
        provider = product.provider
        self.lock.acquire()
        if filename not in self.files:
            def return_file(filename):
                local_file = Path(self.path.name) / filename
                provider.download_file(filename, local_file)
                new_file = product(local_file)
                return new_file
            self.files[filename] = self.pool.submit(return_file, filename)
        self.lock.release()
        return self.files[filename]


def process_cloudsat_files(
        cloudsat_files,
        cache,
        size=256,
        timedelta=15,
):
    """
    Match CloudSat product files for a given granule with GPMIR and
    GridSat B1 observations.

    Args:
        cloudsat_files: A list of the CloudSat file objects to match
            with the GPMIR and GridSat data.
        cache: A download cache to use for the GPMIR and GridSat files.
        size: The size of the match-up scenes to extract.
        timedelta: The maximum time difference to allow between CloudSat
            and geostationary observations.

    Return:
        A list of match-up scenes.
    """
    seed = hash("".join([cs_file.filename for cs_file in cloudsat_files]))
    rng = np.random.default_rng(seed)
    cloudsat_files = [
        cache.get(type(cs_file), cs_file.filename) for cs_file in cloudsat_files
    ]

    data = cloudsat_files[0].result().to_xarray_dataset()
    d_t = np.array(timedelta * 60, dtype="timedelta64[s]")
    start_time = data.time.data[0] - d_t
    end_time = data.time.data[-1]  + d_t

    scenes = []

    gpmir_files = GPMIR.provider.get_files_in_range(
        to_datetime(start_time),
        to_datetime(end_time),
        start_inclusive=True
    )
    cloudsat_files = [cs_file.result() for cs_file in cloudsat_files]
    for filename in gpmir_files:
        gpmir_file = cache.get(GPMIR, filename).result()
        scenes += gpmir_file.get_matches(
            rng,
            cloudsat_files,
            size=size,
            timedelta=timedelta
        )
        scenes += gpmir_file.get_matches(
            rng,
            cloudsat_files,
            size=size,
            timedelta=timedelta,
            subsample=True
        )
    gridsat_files = GridSatB1.provider.get_files_in_range(
        to_datetime(start_time),
        to_datetime(end_time),
        start_inclusive=True
    )
    for filename in gridsat_files:
        gs_file = cache.get(GridSatB1, filename).result()
        scenes += gs_file.get_matches(
            rng,
            cloudsat_files,
            size=size,
            timedelta=timedelta
        )
    return scenes


def write_scenes(
        scenes,
        destination,
        valid_input=0.2
):
    """
    Write extracted match-up scenes to training files.

    Args:
       scenes: A list of xarray.Datasets each containing a single
           matchup file.
       valid_input: A minimum fraction of valid input in the IR
           window channel for a sample to be saved.
    """
    destination = Path(destination)

    for scene in scenes:

        # Check if scene has sufficient valid inputs.
        if np.isfinite(scene.ir_win.data).mean() < valid_input:
            continue

        # Calculate median time of cloudsat overpass.
        valid = np.isfinite(scene.time_cloudsat.data)
        times = scene.time_cloudsat.data[valid]
        dtype = times.dtype
        time = np.median(times.astype("int64")).astype(dtype)
        time_s = xr.DataArray(time).dt.strftime("%Y%m%d_%H%M%S").data.item()

        # Use sparse storage for CloudSat output data.
        profile_row_inds, profile_column_inds = np.where(np.isfinite(scene.tiwp_fpavg.data))
        time_cloudsat = scene.time_cloudsat.data.astype("datetime64[s]")
        scene["profile_row_inds"] = (("profiles"), profile_row_inds)
        scene["profile_column_inds"] = (("profiles"), profile_column_inds)
        dims = ["profiles", "levels"]
        vars = [
            "time_cloudsat",
            "latitude_cloudsat",
            "longitude_cloudsat",
            "tiwp",
            "tiwp_fpavg",
            "tiwc",
            "cloud_mask",
            "cloud_class",
        ]
        for var in vars:
            data = scene[var].data
            scene[var] = (dims[:data.ndim - 1], data[valid])

        comp = {
            "dtype": "int16",
            "scale_factor": 0.1,
            "zlib": True,
            "_FillValue": -99
        }
        encoding = {var: copy(comp) for var in scene.variables.keys()}
        encoding["profile_row_inds"] = {"dtype": "int16", "zlib": True}
        encoding["profile_column_inds"] = {"dtype": "int16", "zlib": True}
        encoding["longitude"] = {"dtype": "float32", "zlib": True}
        encoding["levels"]["scale_factor"] = 10
        encoding["latitude"] = {"dtype": "float32", "zlib": True}
        encoding["latitude_cloudsat"] = {"dtype": "float32", "zlib": True}
        encoding["longitude_cloudsat"] = {"dtype": "float32", "zlib": True}
        encoding["tiwc"] = {"dtype": "float32", "zlib": True}
        encoding["tiwp"] = {"dtype": "float32", "zlib": True}
        encoding["tiwp_fpavg"] = {"dtype": "float32", "zlib": True}
        encoding["cloud_class"] = {"dtype": "int8", "zlib": True}
        encoding["cloud_mask"] = {"dtype": "int8", "zlib": True}
        encoding["time_cloudsat"] = {"dtype": "int64", "zlib": True}

        source = scene.attrs["input_source"].lower()
        output = f"cloudsat_match_{source}_{time_s}.nc"
        scene.to_netcdf(destination / output, encoding=encoding)
