from concurrent.futures import ThreadPoolExecutor
from copy import copy
import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep

import numpy as np
from pansat.time import to_datetime
import xarray as xr

from ccic.data.cloudsat import CloudSat2CIce, CloudSat2BCLDCLASS
from ccic.data.cpcir import CPCIR
from ccic.data.gridsat import GridSat


def get_file(provider, product, path, filename, retries):
    """
    Tries to download file from provider.

    Args:
        provider: The 'pansat' data provider to use to download the file.
        product: The product class that is being downloaded.
        path: The path to which to download the file.
        filename: The name of the file.
        retries: The number of retries to perform in case of failure.

    Return:
        A instance of the given product class pointing to the downloaded file.

    Raises:
        RuntimeError if the download fails even after the given number of
        retries.
    """
    local_file = Path(path) / filename

    failed = True
    n_tries = 0
    while failed and n_tries < retries:
        n_tries += 1
        try:
            provider.download_file(filename, local_file)
            failed = False
        except Exception as e:
            sleep(10)
            pass

    if failed:
        raise RuntimeError(
            "Downloading of file '%s' failed after three retries.", filename
        )

    new_file = product(local_file)
    return new_file


class DownloadCache:
    """
    Asynchronous download cache..
    """

    def __init__(self, n_threads=2, retries=3):
        self.path = TemporaryDirectory()
        self.files = {}
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        self.lock = Lock()
        self.retries = retries

    def get(self, product, filename):
        """
        Get file from the cache.
        """
        provider = product.provider
        # Synchronize lookup between threads to avoid inconsistent states.
        with self.lock:
            if filename not in self.files:
                self.files[filename] = self.pool.submit(
                    get_file,
                    provider,
                    product,
                    Path(self.path.name),
                    filename,
                    self.retries,
                )
        return self.files[filename]


def process_cloudsat_files(
    cloudsat_files,
    cache,
    size=256,
    timedelta=15,
):
    """
    Match CloudSat product files for a given granule with CPCIR and
    GridSat B1 observations.

    Args:
        cloudsat_files: A list of the CloudSat file objects to match
            with the CPCIR and GridSat data.
        cache: A download cache to use for the CPCIR and GridSat files.
        size: The size of the match-up scenes to extract.
        timedelta: The maximum time difference to allow between CloudSat
            and geostationary observations.

    Return:
        A list of match-up scenes.
    """
    logger = logging.getLogger(__name__)

    seed = hash("".join([cs_file.filename.name for cs_file in cloudsat_files]))
    rng = np.random.default_rng(abs(seed))
    cloudsat_files = [
        cache.get(type(cs_file), cs_file.filename) for cs_file in cloudsat_files
    ]

    data = cloudsat_files[0].result().to_xarray_dataset()
    d_t = np.array(timedelta * 60, dtype="timedelta64[s]")
    start_time = data.time.data[0] - d_t
    end_time = data.time.data[-1] + d_t

    scenes = []

    cpcir_files = CPCIR.provider.get_files_in_range(
        to_datetime(start_time), to_datetime(end_time), start_inclusive=True
    )
    gridsat_files = GridSat.provider.get_files_in_range(
        to_datetime(start_time), to_datetime(end_time), start_inclusive=True
    )

    cloudsat_files = [cs_file.result() for cs_file in cloudsat_files]

    for filename in cpcir_files:
        try:
            cpcir_file = cache.get(CPCIR, filename).result()
        except RuntimeError as err:
            logger.error(err)
            continue

        scenes += cpcir_file.get_matches(
            rng, cloudsat_files, size=size, timedelta=timedelta
        )
        scenes += cpcir_file.get_matches(
            rng, cloudsat_files, size=size, timedelta=timedelta, subsample=True
        )

    for filename in gridsat_files:
        try:
            gs_file = cache.get(GridSat, filename).result()
        except RuntimeError as err:
            logger.error(err)
            continue

        scenes += gs_file.get_matches(
            rng, cloudsat_files, size=size, timedelta=timedelta
        )
    return scenes


def write_scenes(
    scenes,
    destination,
    valid_input=0.2,
    product="cloudsat",
):
    """
    Write extracted match-up scenes to training files.

    Args:
        scenes: A list of xarray.Datasets each containing a single
            matchup file.
        destination: Path to save the scenes
        valid_input: A minimum fraction of valid input in the IR
            window channel for a sample to be saved.
        product: name prepended in the file and used in the variables
    """
    destination = Path(destination)

    for scene in scenes:
        # Check if scene has sufficient valid inputs.
        if np.isfinite(scene.ir_win.data).mean() < valid_input:
            continue

        # Calculate median time of cloudsat overpass.
        valid = np.isfinite(scene[f"time_{product}"].data)
        times = scene[f"time_{product}"].data[valid]
        dtype = times.dtype
        time = np.median(times.astype("int64")).astype(dtype)
        time_s = xr.DataArray(time).dt.strftime("%Y%m%d_%H%M%S").data.item()

        # Use sparse storage for CloudSat output data.
        # `valid` and `valid_tiwp_fpavg_mask` should be equal, but in DARDAR
        # data some tiwp_fpavg are NaN due to unfortunate collocations:
        # profiles consisting of NaN IWCs falling alone in a bin
        valid_tiwp_fpavg_mask = np.isfinite(scene.tiwp_fpavg.data)
        profile_row_inds, profile_column_inds = np.where(valid_tiwp_fpavg_mask)
        scene["profile_row_inds"] = (("profiles"), profile_row_inds)
        scene["profile_column_inds"] = (("profiles"), profile_column_inds)
        dims = ["profiles", "altitude"]
        vars = [
            f"time_{product}",
            f"latitude_{product}",
            f"longitude_{product}",
            "tiwp",
            "tiwp_fpavg",
            "tiwc",
            "cloud_mask",
            "cloud_class",
        ]
        for var in vars:
            data = scene[var].data
            scene[var] = (dims[: data.ndim - 1], data[valid_tiwp_fpavg_mask])

        comp = {"dtype": "int16", "scale_factor": 0.1, "zlib": True, "_FillValue": -99}
        encoding = {var: copy(comp) for var in scene.variables.keys()}
        encoding["profile_row_inds"] = {"dtype": "int16", "zlib": True}
        encoding["profile_column_inds"] = {"dtype": "int16", "zlib": True}
        encoding["longitude"] = {"dtype": "float32", "zlib": True}
        encoding["altitude"] = {"dtype": "float32", "zlib": True}
        encoding["latitude"] = {"dtype": "float32", "zlib": True}
        encoding[f"latitude_{product}"] = {"dtype": "float32", "zlib": True}
        encoding[f"longitude_{product}"] = {"dtype": "float32", "zlib": True}
        encoding["tiwc"] = {"dtype": "float32", "zlib": True}
        encoding["tiwp"] = {"dtype": "float32", "zlib": True}
        encoding["tiwp_fpavg"] = {"dtype": "float32", "zlib": True}
        encoding["cloud_class"] = {"dtype": "int8", "zlib": True}
        encoding["cloud_mask"] = {"dtype": "int8", "zlib": True}
        encoding[f"time_{product}"] = {"dtype": "int64", "zlib": True}

        source = scene.attrs["input_source"].lower()
        output = f"{product}_match_{source}_{time_s}.nc"
        scene.to_netcdf(destination / output, encoding=encoding)
