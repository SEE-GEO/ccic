"""
ccic.processing
===============

Implements functions for the operational processing of the CCIC
retrieval.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import StringIO
from logging import Handler
from pathlib import Path
import sqlite3

import numpy as np
from pansat.time import to_datetime
from scipy.ndimage.morphology import binary_closing
import torch
import xarray as xr
import zarr

from ccic import __version__
from ccic.tiler import Tiler
from ccic.data.cpcir import CPCIR
from ccic.data.gridsat import GridSat
from ccic.data.utils import extract_roi
from ccic.codecs import LogBins


@dataclass
class RemoteFile:
    """
    Simple wrapper class around a file that is not locally available
    but downloaded via pansat.
    """

    file_cls: type
    filename: str

    def __init__(self, file_cls, filename, working_dir, thread_pool=None):
        self.file_cls = file_cls
        self.filename = filename
        self.working_dir = working_dir
        self.prefetch_task = None
        if thread_pool is not None and working_dir is not None:
            self.prefetch(thread_pool)

    def prefetch(self, thread_pool):
        """
        Uses a thread pool to schedule prefetching of the remote file.

        Args:
            thread_pool: The thread pool to use for the prefetching.
        """
        output_path = Path(self.working_dir) / self.filename
        self.prefetch_task = thread_pool.submit(
            self.file_cls.download, self.filename, output_path
        )

    def get(self, working_dir=None):
        """
        Download the file.

        Args:
            working_dir: The local folder in which to store the file.

        Return:
            A ``file_cls`` object pointing to the downloaded file.
        """
        if working_dir is None:
            working_dir = self.working_dir

        if working_dir is None:
            raise ValueError(
                "A 'working_dir' must be provided either on creation of "
                "the RemoteFile object or when the 'get' method is called."
            )

        output_path = Path(working_dir) / self.filename

        # Check if file is pre-fetched.
        if self.prefetch_task is not None:
            self.prefetch_task.result()
            result = self.file_cls(output_path)
            return result

        self.file_cls.download(self.filename, output_path)
        return self.file_cls(output_path)


class OutputFormat(Enum):
    """
    Enum class to represent the available output formats.
    """

    NETCDF = 1
    ZARR = 2


@dataclass
class RetrievalSettings:
    """
    A record class to hold the retrieval settings.
    """

    tile_size: int = 512
    overlap: int = 128
    targets: list = None
    roi: list = None
    device: str = "cpu"
    precision: int = 32
    output_format: OutputFormat = OutputFormat["NETCDF"]
    database_path: str = "ccic_processing.db"


def get_input_files(
    input_cls, start_time, end_time=None, path=None, thread_pool=None, working_dir=None
):
    """
    Determine local or remote input files.

    Calculates a list of files that fall within the requested time interval.
    If the files are remote RemoteFile objects will be returned that can be
    used to download the files.

    Args:
        input_cls: The input file class determining the type of input files
            to obtain.
        start_time: The time for which to look for input files. If only
            start time is given, only files with the exact same time will
            be considered.
        end_time: If both 'start_time' and 'end_time' are given, all files
            falling within the specified time range will be considered.
        path: If given, will be used to look for local files.
        thread_pool: An optional thread pool to use for the prefetching
            of remote files.
        working_dir: A temporary directory to use for the prefetching of files.
    """
    start_time = to_datetime(start_time)
    if end_time is None:
        end_time = start_time

    # Return remote files if no path if given.
    if path is None:
        files = input_cls.get_available_files(start_time=start_time, end_time=end_time)
        return [
            RemoteFile(
                input_cls,
                filename,
                working_dir,
                thread_pool=thread_pool,
            )
            for filename in files
        ]

    # Return local files if path if given.
    files = input_cls.find_files(path=path, start_time=start_time, end_time=end_time)
    return [input_cls(filename) for filename in files]


###############################################################################
# Database logging
###############################################################################


class LogContext:
    """
    Context manager to handle logging to database.
    """

    def __init__(self, handler, logger):
        """
        Args:
            handler: The ProcessingLog handler to use for logging.
            logger: The current logger.
        """
        self.logger = logger
        self.handler = handler

    def __enter__(self):
        self.handler.start_logging(self.logger)

    def __exit__(self, type, value, traceback):
        self.handler.finish_logging(self.logger)


class ProcessingLog(Handler):
    """
    A logging handler that logs processing info to a database.
    """

    def __init__(self, database_path, input_file):
        super().__init__(level="DEBUG")
        if database_path is not None:
            self.database_path = Path(database_path)
        else:
            self.database_path = None

        self.input_file = input_file
        self.buffer = StringIO()

        if self.database_path is not None and not self.database_path.exists():
            self._init_db()
        self._init_entry()

    def _init_db(self):
        """
        Initializes DB.
        """
        if self.database_path is None:
            return None

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE files(
                    name TEXT PRIMARY KEY,
                    date date,
                    success INTEGER,
                    output_file TEXT,
                    tiwp_min FLOAT,
                    tiwp_max FLOAT,
                    tiwp_mean FLOAT,
                    n_missing INTEGER,
                    log BLOB
                )
                """
            )
        return None

    def _init_entry(self):
        """
        Initializes entry for current file.
        """
        if self.database_path is None:
            return None

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute("SELECT * FROM files WHERE name=?", (self.input_file,))
            if res.fetchone() is None:
                res = cursor.execute(
                    """
                    INSERT INTO files
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.input_file,
                        datetime.now(),
                        False,
                        "",
                        np.nan,
                        np.nan,
                        np.nan,
                        -1,
                        bytes(),
                    ),
                )

    def log(self, logger):
        """
        Return context manager to capture logging.

        Args:
            The currently active logger.
        """
        return LogContext(self, logger)

    def emit(self, record):
        """
        Custom emit function that stores log message in buffer.
        """
        message = self.format(record)
        self.buffer.write(message + "\n")

    def start_logging(self, logger):
        """
        Start capturing log messages.

        Should be call through context manager return by 'log' method.
        """
        logger.addHandler(self)
        self.buffer = StringIO()

    def finish_logging(self, logger):
        """
        Writes capture log messages to DB.
        """
        logger.removeHandler(self)

        if self.database_path is None:
            return None

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT log FROM files WHERE name=?", (self.input_file,)
            )
            self.buffer.seek(0)
            log = res.fetchone()[0] + self.buffer.read().encode()
            res = cursor.execute(
                "UPDATE files SET log=? WHERE name=?",
                (log, self.input_file),
            )
        return None

    def finalize(self, results, output_file):
        """
        Finalizes log for current files. Sets success flag and calculates
        tiwp statistics.
        """
        if self.database_path is None:
            return None

        tiwp_mean = np.nan
        tiwp_min = np.nan
        tiwp_max = np.nan
        n_missing = -1

        if "tiwp" in results:
            tiwp_mean = results.tiwp.mean().item()
            tiwp_min = results.tiwp.min().item()
            tiwp_max = results.tiwp.max().item()
            n_missing = int(np.isnan(results.tiwp.data).sum())

        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT log FROM files WHERE name=?", (self.input_file,)
            )

            if res.fetchone() is not None:
                cmd = """
            UPDATE files
            SET success=?,
                output_file=?,
                tiwp_min=?,
                tiwp_max=?,
                tiwp_mean=?,
                n_missing=?
            WHERE name=?
            """
                data = (
                    True,
                    output_file,
                    tiwp_min,
                    tiwp_max,
                    tiwp_mean,
                    n_missing,
                    self.input_file,
                )
                res = cursor.execute(cmd, data)
        return None


def get_output_filename(input_file, date, retrieval_settings):
    """
    Get filename for CCIC output file.

    Args:
        input_file: The input file object.
        date: Time stamp of the first observations in the input file.
        retrieval_settings: RetrievalSettings object specifying the output format.

    Return:
        A string containing the filename.
    """
    if isinstance(input_file, CPCIR):
        file_type = "cpcir"
    elif isinstance(input_file, GridSat):
        file_type = "gridsat"
    else:
        raise ValueError(
            "'input_file' should be an instance of 'CPCIR' or " "'GridSat' not '%s'.",
            type(input_file),
        )
    date_str = to_datetime(date).strftime("%Y%m%d%H%M")

    if retrieval_settings.output_format == OutputFormat["NETCDF"]:
        suffix = "nc"
    elif retrieval_settings.output_format == OutputFormat["ZARR"]:
        suffix = "zarr"
    return f"ccic_{file_type}_{date_str}.{suffix}"


REGRESSION_TARGETS = ["iwp", "iwp_rand", "iwc"]
SCALAR_TARGETS = ["iwp", "iwp_rand"]
THRESHOLDS = {"iwp": 1e-3, "iwp_rand": 1e-3, "iwc": 1e-3}

# Maps NN target names to output names.
OUTPUT_NAMES = {"iwp": "tiwp_fpavg", "iwp_rand": "tiwp", "iwc": "tiwc"}


def process_regression_target(
    mrnn,
    y_pred,
    invalid,
    target,
    means,
    log_std_devs,
    p_non_zeros,
):
    """
    Implements the processing logic for regression targets.

    The posterior mean is calculated for all regression targets. A random sample
    and posterior quantiles, however, are only calculated for scalar retrieval
    targets.
    Args:
        mrnn: The MRNN model used for the inference.
        y_pred: The dictionary containing all predictions from the model.
        target: The retrieval target to process.
        means: Result dict to which to store the calculated posterior means.
        log_std_devs: Result dict to which to store the calculate log standard
            deviations.
        p_non_zeros: Result dict to which to store the calculated probability that
            the target is larger than the corresponding minimum threshold.
    """
    mean = mrnn.posterior_mean(y_pred=y_pred[target], key=target).cpu().float().numpy()
    for ind in range(invalid.shape[0]):
        mean[ind, ..., invalid[ind]] = np.nan
    means[target][-1].append(mean)

    if target in SCALAR_TARGETS:
        log_std_dev = (
            mrnn.posterior_std_dev(y_pred=torch.log10(y_pred[target]), key=target)
            .cpu()
            .float()
            .numpy()
        )
        for ind in range(invalid.shape[0]):
            log_std_dev[ind, ..., invalid[ind]] = np.nan

        log_std_devs[target][-1].append(log_std_dev)
        p_non_zero = (
            mrnn.probability_larger_than(
                y_pred=y_pred[target], y=THRESHOLDS[target], key=target
            )[:]
            .cpu()
            .float()
            .numpy()
        )

        for ind in range(invalid.shape[0]):
            p_non_zero[ind, ..., invalid[ind]] = np.nan

        p_non_zeros[target][-1].append(p_non_zero)


def get_invalid_mask(x_in):
    """
    Get mask of invalid retrieval results.

    The function detects large consecutive regions of missing values
    in the input. It performs a binary closing operations thus allowing
    isolated missing values in the input. Missing inputs are identified
    as input pixels with a value below -1.4 assuming that a quantnn
    MinMax normalizer has been used that uses -1.5 for missing inputs.

    Args:
        x_in: The torch tensor that is fed into the network.

    Return:
        A numpy binary mask identifying regions with invalid inputs.
    """
    x_in = x_in.cpu().numpy()
    masks = []
    for ind in range(x_in.shape[0]):
        mask = np.any(x_in[ind] > -1.4, axis=0)
        mask = np.pad(mask, ((8, 8), (8, 8)), mode="reflect")
        mask = binary_closing(mask, iterations=8, border_value=0)[8:-8, 8:-8]
        masks.append(mask)
    return ~np.stack(masks)


def process_input(mrnn, x, retrieval_settings=None):
    """
    Process given retrieval input using tiling.

    Args:
        mrnn: The MRNN to use to perform the retrieval.
        x: A 'torch.Tensor' containing the retrieval input.
        retrieval_settings: A RetrievalSettings object defining the settings
            for the retrieval

    Return:
        An 'xarray.Dataset' containing the results of the retrieval.
    """
    if retrieval_settings is None:
        retrieval_settings = RetrievalSettings()

    tile_size = retrieval_settings.tile_size
    overlap = retrieval_settings.overlap
    targets = retrieval_settings.targets
    if targets is None:
        targets = [
            "iwp",
            "iwp_rand",
            "iwc",
            "cloud_prob_2d",
            "cloud_prob_3d",
            "cloud_type",
        ]

    tiler = Tiler(x, tile_size=tile_size, overlap=overlap)
    means = {}
    log_std_devs = {}
    p_non_zeros = {}
    cloud_prob_2d = []
    cloud_prob_3d = []
    cloud_type = []

    device = retrieval_settings.device
    precision = retrieval_settings.precision

    mrnn.model.to(device)

    with torch.no_grad():
        for i in range(tiler.M):

            # Insert empty list into list of row results.
            for target in targets:
                if target in REGRESSION_TARGETS:
                    means.setdefault(target, []).append([])
                    if target in SCALAR_TARGETS:
                        log_std_devs.setdefault(target, []).append([])
                        p_non_zeros.setdefault(target, []).append([])
                elif target == "cloud_prob_2d":
                    cloud_prob_2d.append([])
                elif target == "cloud_prob_3d":
                    cloud_prob_3d.append([])
                elif target == "cloud_type":
                    cloud_type.append([])

            for j in range(tiler.N):
                x_t = tiler.get_tile(i, j)

                # Use torch autocast for mixed precision.
                x_t = x_t.to(device)
                if precision == 16:
                    with torch.autocast(device_type=device):
                        y_pred = mrnn.predict(x_t)
                else:
                    y_pred = mrnn.predict(x_t)

                invalid = get_invalid_mask(x_t)

                for target in targets:
                    if target in REGRESSION_TARGETS:
                        process_regression_target(
                            mrnn,
                            y_pred,
                            invalid,
                            target,
                            means=means,
                            log_std_devs=log_std_devs,
                            p_non_zeros=p_non_zeros,
                        )
                    elif target == "cloud_prob_2d":
                        cloud_prob_2d[-1].append(
                            y_pred["cloud_mask"].cpu().float().numpy()[:, 0]
                        )
                        cloud_prob_2d[-1][-1][invalid] = np.nan
                    elif target == "cloud_prob_3d":
                        cp = 1.0 - y_pred["cloud_class"]
                        cloud_prob_3d[-1].append(cp[:, 0].cpu().float().numpy())
                        for ind in range(invalid.shape[0]):
                            cloud_prob_3d[-1][-1][ind, ..., invalid[ind]] = np.nan
                    elif target == "cloud_type":
                        ct = torch.softmax(y_pred["cloud_class"][:, 1:], 1)
                        cloud_type[-1].append(ct.cpu().float().numpy())
                        for ind in range(invalid.shape[0]):
                            cloud_type[-1][-1][ind, ..., invalid[ind]] = -1

    results = xr.Dataset()
    for target, mean in means.items():
        mean = tiler.assemble(mean)
        if mean.ndim == 3:
            dims = ("time", "latitude", "longitude")
        else:
            dims = ("time", "latitude", "longitude", "altitude")
            mean = np.transpose(mean, [0, 2, 3, 1])

        results[OUTPUT_NAMES[target]] = (dims, mean)

    dims = ("time", "latitude", "longitude")
    for target, p_non_zero in p_non_zeros.items():
        smpls = tiler.assemble(p_non_zero)
        results["p_" + OUTPUT_NAMES[target]] = (dims, smpls)

    for target, log_std_dev in log_std_devs.items():
        log_std_dev = tiler.assemble(log_std_dev)
        results[OUTPUT_NAMES[target] + "_log_std_dev"] = (dims, log_std_dev)

    dims = ("time", "latitude", "longitude")
    if len(cloud_prob_2d) > 0:
        cloud_prob_2d = tiler.assemble(cloud_prob_2d)
        results["cloud_prob_2d"] = (dims, cloud_prob_2d)

    dims = ("time", "latitude", "longitude", "altitude")
    if len(cloud_prob_3d) > 0:
        cloud_prob_3d = tiler.assemble(cloud_prob_3d)
        cloud_prob_3d = np.transpose(cloud_prob_3d, [0, 2, 3, 1])
        results["cloud_prob_3d"] = (dims, cloud_prob_3d)

    dims = ("time", "latitude", "longitude", "altitude", "type")
    if len(cloud_type) > 0:
        cloud_type = tiler.assemble(cloud_type)
        cloud_type = np.transpose(cloud_type, [0, 3, 4, 2, 1])
        results["cloud_type"] = (dims, cloud_type)
    results["altitude"] = (("altitude,",), np.arange(20) * 1e3 + 500.0)

    return results


def process_input_file(mrnn, input_file, retrieval_settings=None):
    """
    Processes an input file and returns the retrieval result together with
    meta data.

    Args:
        mrnn: The MRNN to use for the retrieval processing.
        input_file: The file containing the input data.
        retrieval_settings: A RetrievalSettings object specifying the settings for
            the retrieval.

    Return:
        A 'xarray.Dataset' containing the retrival results.
    """
    if retrieval_settings is None:
        retrieval_settings = RetrievalSettings()
    roi = retrieval_settings.roi

    retrieval_input = input_file.get_retrieval_input(roi=roi)
    results = process_input(
        mrnn, retrieval_input, retrieval_settings=retrieval_settings
    )

    # Copy values of dimension
    input_data = input_file.to_xarray_dataset()
    if roi is not None:
        input_data = extract_roi(input_data, roi, min_size=256)
    input_data = input_data.rename({"lat": "latitude", "lon": "longitude"})

    for dim in ["time", "latitude", "longitude"]:
        results[dim] = input_data[dim]

    results.attrs.update(input_file.get_input_file_attributes())
    add_static_cf_attributes(results)

    return results


def add_static_cf_attributes(dataset):
    """
    Adds static attributes required by CF convections.
    """
    dataset["latitude"].attrs["units"] = "degrees_north"
    dataset["latitude"].attrs["axis"] = "Y"
    dataset["latitude"].attrs["standard_name"] = "latitude"
    dataset["longitude"].attrs["units"] = "degrees_east"
    dataset["longitude"].attrs["axis"] = "X"
    dataset["longitude"].attrs["standard_name"] = "longitude"
    dataset["altitude"].attrs["axis"] = "Z"
    dataset["altitude"].attrs["standard_name"] = "altitude"
    dataset["altitude"].attrs["units"] = "meters"
    dataset["altitude"].attrs["positive"] = "up"

    dataset.attrs["title"] = "The Chalmers Cloud Ice Climatology"
    dataset.attrs["institution"] = "Chalmers University of Technology"
    dataset.attrs["source"] = f"framework: ccic-{__version__}"
    dataset.attrs["history"] = f"{datetime.now()}: Retrieval processing"

    if "tiwp" in dataset:
        dataset["tiwp"].attrs["standard_name"] = "atmosphere_mass_content_of_cloud_ice"
        dataset["tiwp"].attrs["units"] = "kg m-2"
        dataset["tiwp"].attrs[
            "long_name"
        ] = "Vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp"].attrs["ancillary_variables"] = "tiwp_log_std_dev p_tiwp"

        dataset["tiwp_log_std_dev"].attrs[
            "long_name"
        ] = "Standard deviation of log-transformed vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp_log_std_dev"].attrs["units"] = "log(kg m-2)"
        dataset["p_tiwp"].attrs[
            "long_name"
        ] = "Probability that 'tiwp' exceeds 1e-3 kg m-2"
        dataset["p_tiwp"].attrs["units"] = "1"

    if "tiwp_fpavg" in dataset:
        dataset["tiwp_fpavg"].attrs["units"] = "kg m-2"
        dataset["tiwp_fpavg"].attrs[
            "long_name"
        ] = "Vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp_fpavg"].attrs[
            "ancillary_variables"
        ] = "tiwp_fpavg_log_std_dev p_tiwp_fpavg"

        dataset["tiwp_fpavg_log_std_dev"].attrs[
            "long_name"
        ] = "Standard deviation of log-transformed vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp_fpavg_log_std_dev"].attrs["units"] = "log(kg m-2)"
        dataset["p_tiwp_fpavg"].attrs[
            "long_name"
        ] = "Probability that 'tiwp_fpavg' exceeds 1e-3 kg m-2"
        dataset["p_tiwp_fpavg"].attrs["units"] = "1"

    if "tiwc" in dataset:
        dataset["tiwc"].attrs["units"] = "kg m-3"
        dataset["tiwc"].attrs["long_name"] = "Concentration of frozen hydrometeors"

    if "cloud_prob_2d" in dataset:
        dataset["cloud_prob_2d"].attrs["units"] = "1"
        dataset["cloud_prob_2d"].attrs["long_name"] = "Probability of presence of a cloud anywhere in the atmosphere"

    if "cloud_prob_3d" in dataset:
        dataset["cloud_prob_3d"].attrs["units"] = "1"
        dataset["cloud_prob_3d"].attrs["long_name"] = "Probability of presence of a cloud"

    if "cloud_type" in dataset:
        dataset["cloud_type"].attrs["units"] = "1"
        dataset["cloud_type"].attrs["long_name"] = "Most likely cloud type"
        dataset["cloud_type"].attrs["flag_values"] = "0, 1, 2, 3, 4, 5, 6, 7, 8"
        dataset["cloud_type"].attrs["flag_meanings"] = "No cloud, Cirrus, Altostratus, Altocumulus, Stratus, Stratocumulus, Cumulus, Nimbostratus, Deep convection"




def get_encodings_zarr(variable_names):
    """
    Get variable encoding dict for storing the results for selected
    target variables in zarr format.
    """
    compressor = zarr.Blosc(cname="lz4", clevel=9, shuffle=2)
    filters = [LogBins(1e-3, 1e2)]
    all_encodings = {
        "tiwp": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "tiwc": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "p_tiwp": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
        },
        "tiwp_log_std_dev": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 12.5,
            "_FillValue": 255,
        },
        "tiwp_fpavg": {
            "compressor": compressor,
            "filters": filters,
            "dtype": "float32",
        },
        "p_tiwp_fpavg": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
        },
        "tiwp_fpavg_log_std_dev": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 12.5,
            "_FillValue": 255,
        },
        "cloud_prob_2d": {
            "compressor": compressor,
            "scale_factor": 250,
            "_FillValue": 255,
            "dtype": "uint8",
        },
        "cloud_prob_3d": {
            "compressor": compressor,
            "scale_factor": 250,
            "_FillValue": 255,
            "dtype": "uint8",
        },
        "cloud_type": {"compressor": compressor, "dtype": "uint8", "_FillValue": -1},
        "longitude": {
            "compressor": compressor,
            "dtype": "float32",
        },
        "latitude": {
            "compressor": compressor,
            "dtype": "float32",
        },
    }
    return {
        name: all_encodings[name] for name in variable_names if name in all_encodings
    }


def get_encodings_netcdf(variable_names):
    """
    Get variable encoding dict for storing the results for selected
    target variables in netcdf format.
    """
    all_encodings = {
        "tiwp": {"dtype": "float32", "zlib": True},
        "tiwc": {"dtype": "float32", "zlib": True},
        "p_tiwp": {
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "zlib": True,
        },
        "tiwp_log_std_dev": {
            "dtype": "uint8",
            "scale_factor": 1 / 12.5,
            "_FillValue": 255,
            "zlib": True,
        },
        "tiwp_fpavg": {"dtype": "float32", "zlib": True},
        "p_tiwp_fpavg": {
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "zlib": True,
        },
        "tiwp_fpavg_log_std_dev": {
            "dtype": "uint8",
            "scale_factor": 1 / 12.5,
            "_FillValue": 255,
            "zlib": True,
        },
        "cloud_prob_2d": {
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "dtype": "uint8",
            "zlib": True,
        },
        "cloud_prob_3d": {
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "dtype": "uint8",
            "zlib": True,
        },
        "cloud_type": {"dtype": "uint8", "zlib": True},
        "longitude": {"dtype": "float32", "zlib": True},
        "latitude": {"dtype": "float32", "zlib": True},
    }
    return {
        name: all_encodings[name] for name in variable_names if name in all_encodings
    }


def get_encodings(variable_names, retrieval_settings):
    """
    Get output encodings for given retrieval settings.

    Simple wrapper that uses 'retrieval_settings' to forward the call to
    the method corresponding to the 'output_format' in retrieval settings.

    Return:
        The encoding settings for 'NETCDF' or 'ZARR' output format.
    """
    if retrieval_settings.output_format == OutputFormat["ZARR"]:
        return get_encodings_zarr(variable_names)
    return get_encodings_netcdf(variable_names)
