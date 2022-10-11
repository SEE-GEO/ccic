from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np
from pansat.time import to_datetime

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
    cloudsat_files = [
        cache.get(type(cs_file), cs_file.filename) for cs_file in cloudsat_files
    ]

    data = cloudsat_files[0].result().to_xarray_dataset()
    d_t = np.array(timedelta * 60, dtype="timedelta64[s]")
    start_time = data.time.data[0] - d_t
    end_time = data.time.data[1]  + d_t

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
            cloudsat_files,
            size=size,
            timedelta=timedelta
        )
    gridsat_files = GridSatB1.provider.get_files_in_range(
        to_datetime(start_time),
        to_datetime(end_time),
        start_inclusive=True
    )
    for filename in gpmir_files:
        gs_file = cache.get(GridSatB1, filename).result()
        scenes += gs_file.get_matches(
            cloudsat_files,
            size=size,
            timedelta=timedelta
        )
    return scenes


def write_scenes(
        scenes,
        destination,
        valid_inputs=0.2
):
    """
    Write extracted match-up scenes to training files.

    Args:
       scenes: A list of xarray.Datasets each containing a single
           matchup file.
       valid_inputs: A minimum fraction of valid inputs in the IR
           window channel for a sample to be saved.
    """
    for scene in scenes:

        # Check if scene has sufficient valid inputs.
        if np.isfinite(scene.ir_win.data).mean() < valid_inputs:
            continue

        # Calculate median time of cloudsat overpass.
        valid = np.isfinite(scene.time_cloudsat.data)
        times = scene.time_cloudsat.data[valid]
        dtype = times.dtype
        time = np.median(times.astype("int64")).astype(dtype)
        time_s = xr.DataArray(time).dt.strftime("%Y%m%d_%H%M%S").data.item()

        comp = {
            "dtype": "int16",
            "scale_factor": 0.01,
            "zlib": True,
            "_FillValue": -99
        }
        encoding = {var: comp for var in scene.variables.keys()}
        encoding["iwc"] = {"dtype": "float32", "zlib": True}
        encoding["iwp"] = {"dtype": "float32", "zlib": True}
        scene.to_netcdf(destination / f"cloudsat_match_{time_s}.nc", encoding=encoding)
