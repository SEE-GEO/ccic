"""
ccic.processing
===============

Implements functions for the operational processing of the CCIC
retrieval.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import gc
from io import StringIO
import logging
from pathlib import Path
import sqlite3
from typing import List, Optional

import numpy as np
from pansat.time import to_datetime
from scipy.ndimage.morphology import binary_closing
import torch
from torch import nn
import xarray as xr
import zarr

from ccic import __version__
from ccic.tiler import Tiler, calculate_padding
from ccic.data.cpcir import CPCIR
from ccic.data.gridsat import GridSat
from ccic.data.utils import extract_roi
from ccic.codecs import LogBins


LOGGER = logging.getLogger(__name__)


TIMEOUT = 120


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

    def __eq__(self, other):
        """
        Two remote files are considered equal if the refer to files with the
        same filename.
        """
        if isinstance(other, RemoteFile):
            return self.filename == other.filename
        return NotImplemented

    def __hash__(self):
        """
        The hash of a RemoteFile object is the hash of its filename.
        """
        return hash(self.filename)


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
        result = None
        if output_path.exists():
            # If file is in disk
            try:
                result = self.file_cls(output_path)
            except:
                output_path.unlink(missing_ok=True)
        elif self.prefetch_task is not None:
            # Check if file is pre-fetched
            self.prefetch_task.result()
            try:
                result = self.file_cls(output_path)
            except:
                output_path.unlink(missing_ok=True)
        else:
            # Download file
            self.file_cls.download(self.filename, output_path)
            try:
                result = self.file_cls(output_path)
            except:
                output_path.unlink(missing_ok=True)
        
        return result, False


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
    inpainted_mask: bool = False
    credible_interval: float = 0.9
    transfer: Optional[str] = None


def get_input_files(
    input_cls,
    start_time,
    end_time=None,
    path=None,
    thread_pool=None,
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

    Return:
        A list
    """
    start_time = to_datetime(start_time)
    if end_time is None:
        end_time = start_time

    files = input_cls.get_available_files(start_time=start_time, end_time=end_time)
    return [
        RemoteFile(
            input_cls,
            filename,
            path,
            thread_pool=thread_pool,
        )
        for filename in files
    ]


def determine_cloud_class(class_probs, threshold=0.638, axis=1):
    """
    Determines cloud classes from a tensor of cloud-type probabilities.

    If the 'no cloud' probability of a tensor element is larger than a
    threshold, the class will be 'no cloud'. Otherwise the diagnosed
    cloud type will be the one that is most likely and is not 'no cloud'.


    Args:
        class_probs: A torch tensor containing cloud-type probabilities.
        threshold: The no-cloud probability threshold to use.

    Return:
        A tensor containing the class indices of the most likely cloud
        type.
    """
    shape = list(class_probs.shape)
    del shape[axis]
    types = np.zeros(shape, dtype="uint8")

    inds = [slice(0, None)] * class_probs.ndim
    inds[axis] = 0
    cloud_mask = class_probs[tuple(inds)] < threshold
    inds[axis] = slice(1, None)
    prob_types = np.argmax(class_probs[tuple(inds)], axis=axis).astype("uint8") + 1
    types[cloud_mask] = prob_types[cloud_mask]
    return types


def determine_column_cloud_class(cloud_classes):
    """
    Determine the the type of column cloud class from CCIC
    cloud class probabilities. The column cloud classes are given
    by the indices:

    * 0: clear-sky column
    * 1: cloudy column
    * 2: convective column
    * -1: column with all invalid cloud classes

    An atmospheric column is classified as convective if it contains
    at least one parcel that is classified as Cu, Ns or DC.

    Args:
        cloud_classes: Cloud class probabilities as predicted by CCIC.

    Return:
        The corresponding column cloud class indices
    """
    cc = np.zeros(cloud_classes.shape[:-1], dtype="int8")
    cloudy = np.any(cloud_classes > 0, -1)
    cc[cloudy] = 1
    conv = np.any(cloud_classes > 5, -1)
    cc[conv] = 2
    invalid = np.all(cloud_classes > 8, -1)
    cc[invalid] = -1
    return cc


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


class ProcessingLog(logging.Handler):
    """
    A logging handler that logs processing info to a database.
    """

    @staticmethod
    def get_input_file(database_path, success: bool=None) -> List[str]:
        with sqlite3.connect(
                f"file:{database_path}?mode=ro",
                timeout=TIMEOUT,
                uri=True
        ) as conn:
            cursor = conn.cursor()
            query = "SELECT input_file FROM files"
            if success is not None:
                query = f"{query} WHERE success={int(success)}"
            res = cursor.execute(query)
            return [tpl[0] for tpl in res.fetchall()]

    def __init__(self, database_path, input_file):
        super().__init__(level="DEBUG")
        if database_path is not None:
            self.database_path = Path(database_path)
        else:
            self.database_path = None

        if isinstance(input_file, RemoteFile):
            input_file = input_file.filename

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

        with sqlite3.connect(
                f"file:{self.database_path}?mode=rwc",
                timeout=TIMEOUT,
                uri=True
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE files(
                    input_file TEXT PRIMARY KEY,
                    output_file TEXT,
                    date date,
                    success INTEGER,
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

        with sqlite3.connect(
                    f"file:{self.database_path}?mode=rw",
                    timeout=TIMEOUT,
                    uri=True
        ) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT * FROM files WHERE input_file=?", (self.input_file,)
            )
            if res.fetchone() is None:
                res = cursor.execute(
                    """
                    INSERT INTO files
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.input_file,
                        "",
                        datetime.now(),
                        False,
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

        with sqlite3.connect(
                    f"file:{self.database_path}?mode=rw",
                    timeout=TIMEOUT,
                    uri=True
        ) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT log FROM files WHERE input_file=?", (self.input_file,)
            )
            self.buffer.seek(0)
            log = res.fetchone()[0] + self.buffer.read().encode()
            res = cursor.execute(
                "UPDATE files SET log=? WHERE input_file=?",
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

        with sqlite3.connect(
                    f"file:{self.database_path}?mode=rw",
                    timeout=TIMEOUT,
                    uri=True
        ) as conn:
            cursor = conn.cursor()
            res = cursor.execute(
                "SELECT log FROM files WHERE input_file=?", (self.input_file,)
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
            WHERE input_file=?
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


REGRESSION_TARGETS = ["tiwp", "tiwp_fpavg", "tiwc"]
SCALAR_TARGETS = ["tiwp", "tiwp_fpavg"]
THRESHOLDS = {"tiwp": 1e-3, "tiwp_fpavg": 1e-3, "tiwc": 1e-3}


def process_regression_target(
    retrieval_settings,
    mrnn,
    y_pred,
    invalid,
    target,
    means,
    cred_ints,
    p_non_zeros,
):
    """
    Implements the processing logic for regression targets.

    The posterior mean is calculated for all regression targets. A random sample
    and posterior quantiles, however, are only calculated for scalar retrieval
    targets.
    Args:
        retrieval_settings: The retrieval settings representing the retrieval
            configuration.
        mrnn: The MRNN model used for the inference.
        y_pred: The dictionary containing all predictions from the model.
        target: The retrieval target to process.
        means: Result dict to which to store the calculated posterior means.
        cred_ints: Result dict to which to store the lower and upper bounds of
            the calculated credible intervals.
        p_non_zeros: Result dict to which to store the calculated probability that
            the target is larger than the corresponding minimum threshold.
    """
    mean = mrnn.posterior_mean(y_pred=y_pred[target], key=target).cpu().float().numpy()
    for ind in range(invalid.shape[0]):
        mean[ind, ..., invalid[ind]] = np.nan
    means[target][-1].append(mean)

    if target in SCALAR_TARGETS:
        cred = retrieval_settings.credible_interval
        lower = 0.5 * (1.0 - cred)
        upper = 1.0 - lower
        cred_int = (
            mrnn.posterior_quantiles(
                y_pred=y_pred[target], quantiles=[lower, upper], key=target
            )
            .cpu()
            .float()
            .numpy()
        )
        for ind in range(invalid.shape[0]):
            cred_int[ind, ..., invalid[ind]] = np.nan

        cred_ints[target][-1].append(cred_int)
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


def process_input(mrnn, x, retrieval_settings=None, lock=None):
    """
    Process given retrieval input using tiling.

    Args:
        mrnn: The MRNN to use to perform the retrieval.
        x: A 'torch.Tensor' containing the retrieval input.
        retrieval_settings: A RetrievalSettings object defining the settings
            for the retrieval
        lock: Optional multiprocessing.Lock to synchronize device
            access.

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
            "tiwp",
            "tiwp_fpavg",
            "tiwc",
            "cloud_prob_2d",
            "cloud_prob_3d",
            "cloud_type",
        ]

    tiler = Tiler(
        x,
        tile_size=tile_size,
        overlap=overlap,
        wrap_columns=retrieval_settings.roi is None,
    )
    means = {}
    cred_ints = {}
    p_non_zeros = {}
    cloud_prob_2d = []
    cloud_prob_3d = []
    cloud_type = []

    device = retrieval_settings.device
    precision = retrieval_settings.precision

    if lock is not None:
        lock.acquire()

    try:
        mrnn.model.to(device)
        LOGGER.info(f"Starting processing tiles.")
        with torch.no_grad():
            for i in range(tiler.M):
                # Insert empty list into list of row results.
                for target in targets:
                    if target in REGRESSION_TARGETS:
                        means.setdefault(target, []).append([])
                        if target in SCALAR_TARGETS:
                            cred_ints.setdefault(target, []).append([])
                            p_non_zeros.setdefault(target, []).append([])
                    elif target == "cloud_prob_2d":
                        cloud_prob_2d.append([])
                    elif target == "cloud_prob_3d":
                        cloud_prob_3d.append([])
                    elif target == "cloud_type":
                        cloud_type.append([])

                for j in range(tiler.N):
                    x_t = tiler.get_tile(i, j)
                    x_t = x_t.to(device)

                    if (x_t.shape[-2] % 32 > 0) or (x_t.shape[-1] % 32 > 0):
                        padding = calculate_padding(x_t, 32)
                        x_t = nn.functional.pad(x_t, padding, mode="reflect")
                        slices = [
                            slice(padding[2], x_t.shape[-2] - padding[3]),
                            slice(padding[0], x_t.shape[-1] - padding[1]),
                        ]
                    else:
                        slices = None

                    if precision == 16:
                        with torch.autocast(device_type=device):
                            y_pred = mrnn.predict(x_t)
                    else:
                        y_pred = mrnn.predict(x_t)

                    # Remove padding if has been applied.
                    if slices is not None:
                        x_t = x_t[..., slices[0], slices[1]]
                        y_pred = {
                            key: val[..., slices[0], slices[1]]
                            for key, val in y_pred.items()
                        }

                    invalid = get_invalid_mask(x_t)

                    for target in targets:
                        if target in REGRESSION_TARGETS:
                            process_regression_target(
                                retrieval_settings,
                                mrnn,
                                y_pred,
                                invalid,
                                target,
                                means=means,
                                cred_ints=cred_ints,
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
                            del cp
                        elif target == "cloud_type":
                            ct = torch.softmax(y_pred["cloud_class"][:, 1:], 1)
                            cloud_type[-1].append(ct.cpu().float().numpy())
                            for ind in range(invalid.shape[0]):
                                cloud_type[-1][-1][ind, ..., invalid[ind]] = -1
                            del ct

                    # Explicitly delete tensors to make sure GPU memory is freed.
                    del x_t
                    del invalid
                    for key in y_pred:
                        y_pred[key] = y_pred[key].cpu()
                    del y_pred

        if device != "cpu":
            torch.cuda.synchronize()
            mrnn.model.cpu()
            mrnn.model = None
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    finally:
        if lock is not None:
            lock.release()

    LOGGER.info(f"Assembling results.")
    results = xr.Dataset()
    for target, mean in means.items():
        mean = tiler.assemble(mean)
        if mean.ndim == 3:
            dims = ("time", "latitude", "longitude")
        else:
            dims = ("time", "latitude", "longitude", "altitude")
            mean = np.transpose(mean, [0, 2, 3, 1])

        results[target] = (dims, mean)

    dims = ("time", "latitude", "longitude")
    for target, p_non_zero in p_non_zeros.items():
        smpls = tiler.assemble(p_non_zero)
        results["p_" + target] = (dims, smpls)

    dims = ("time", "latitude", "longitude", "ci_bounds")
    for target, cred_int in cred_ints.items():
        cred_int = tiler.assemble(cred_int)
        results[target + "_ci"] = (dims, np.transpose(cred_int, (0, 2, 3, 1)))

    dims = ("time", "latitude", "longitude")
    if len(cloud_prob_2d) > 0:
        cloud_prob_2d = tiler.assemble(cloud_prob_2d)
        results["cloud_prob_2d"] = (dims, cloud_prob_2d)

    dims = ("time", "latitude", "longitude", "altitude")
    if len(cloud_prob_3d) > 0:
        cloud_prob_3d = tiler.assemble(cloud_prob_3d)
        cloud_prob_3d = np.transpose(cloud_prob_3d, [0, 2, 3, 1])
        results["cloud_prob_3d"] = (dims, cloud_prob_3d)

    dims = ("time", "latitude", "longitude", "altitude")
    if len(cloud_type) > 0:
        cloud_type = determine_cloud_class(tiler.assemble(cloud_type))
        cloud_type = np.transpose(cloud_type, [0, 2, 3, 1])
        results["cloud_type"] = (dims, cloud_type)

    if "altitude" in results.dims:
        results["altitude"] = (("altitude",), np.arange(20) * 1e3 + 500.0)

    if len(cred_ints) > 0:
        lower_bound = 0.5 * (1.0 - retrieval_settings.credible_interval)
        upper_bound = 1.0 - lower_bound
        results["ci_bounds"] = (("ci_bounds",), [lower_bound, upper_bound])

    if retrieval_settings.inpainted_mask:
        # Assumes a quantnn.normalizer.MinMaxNormalizer is applied on x which
        # replaces NaNs with -1.5 and normalizes everything else between -1 and +1
        results["inpainted"] = (
            ("time", "latitude", "longitude"),
            x.reshape(-1, *x.shape[-2:]) < -1.4,
        )

    return results


def process_input_file(mrnn, input_file, retrieval_settings=None, lock=None):
    """
    Processes an input file and returns the retrieval result together with
    meta data.

    Args:
        mrnn: The MRNN to use for the retrieval processing.
        input_file: The file containing the input data.
        retrieval_settings: A RetrievalSettings object specifying the settings for
            the retrieval.
        lock: Optional multiprocessing.Lock to synchronize device access.

    Return:
        A 'xarray.Dataset' containing the retrival results.
    """
    if retrieval_settings is None:
        retrieval_settings = RetrievalSettings()
    roi = retrieval_settings.roi

    LOGGER.info(f"Loading retrieval input from file {input_file.filename}.")
    retrieval_input = input_file.get_retrieval_input(roi=roi)
    results = process_input(
        mrnn, retrieval_input, retrieval_settings=retrieval_settings, lock=lock
    )

    # Copy values of dimension
    input_data = input_file.to_xarray_dataset()
    if roi is not None:
        input_data = extract_roi(input_data, roi, min_size=256)
    input_data = input_data.rename({"lat": "latitude", "lon": "longitude"})

    for dim in ["time", "latitude", "longitude"]:
        results[dim] = input_data[dim]

    results.attrs.update(input_file.get_input_file_attributes())
    add_static_cf_attributes(retrieval_settings, results)

    return results


def add_static_cf_attributes(retrieval_settings, dataset):
    """
    Adds static attributes required by CF convections.
    """
    dataset["latitude"].attrs["units"] = "degrees_north"
    dataset["latitude"].attrs["axis"] = "Y"
    dataset["latitude"].attrs["standard_name"] = "latitude"
    dataset["longitude"].attrs["units"] = "degrees_east"
    dataset["longitude"].attrs["axis"] = "X"
    dataset["longitude"].attrs["standard_name"] = "longitude"

    if "altitude" in dataset:
        dataset["altitude"].attrs["axis"] = "Z"
        dataset["altitude"].attrs["standard_name"] = "altitude"
        dataset["altitude"].attrs["units"] = "meters"
        dataset["altitude"].attrs["positive"] = "up"

    dataset.attrs["title"] = "The Chalmers Cloud Ice Climatology"
    dataset.attrs["institution"] = "Chalmers University of Technology"
    dataset.attrs["source"] = f"framework: ccic-{__version__}"
    dataset.attrs["history"] = f"{datetime.now()}: Retrieval processing"
    dataset.attrs["references"] = f"https://doi.org/10.5281/zenodo.8278127"

    if "tiwp" in dataset:
        dataset["tiwp"].attrs["standard_name"] = "atmosphere_mass_content_of_cloud_ice"
        dataset["tiwp"].attrs["units"] = "kg m-2"
        dataset["tiwp"].attrs[
            "long_name"
        ] = "Vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp"].attrs["ancillary_variables"] = "tiwp_ci p_tiwp"

        dataset["tiwp_ci"].attrs[
            "long_name"
        ] = (
            f"{int(100 * retrieval_settings.credible_interval)}% equal-"
            "tailed credible interval for the retrieved TIWP"
        )
        dataset["tiwp_ci"].attrs["units"] = "kg m-2"
        
        dataset["p_tiwp"].attrs[
            "long_name"
        ] = "Probability that 'tiwp' exceeds 1e-3 kg m-2"
        dataset["p_tiwp"].attrs["units"] = "1"
        
        dataset["ci_bounds"].attrs["standard_name"] = "credible_interval_bounds"
        dataset["ci_bounds"].attrs["units"] = "1"
        dataset["ci_bounds"].attrs["long_name"] = (
            "Quantile levels of the credible interval"
        )

    if "tiwp_fpavg" in dataset:
        dataset["tiwp_fpavg"].attrs["units"] = "kg m-2"
        dataset["tiwp_fpavg"].attrs[
            "long_name"
        ] = "Vertically-integrated concentration of frozen hydrometeors"
        dataset["tiwp_fpavg"].attrs[
            "ancillary_variables"
        ] = "tiwp_fpavg_ci p_tiwp_fpavg"

        dataset["tiwp_fpavg_ci"].attrs[
            "long_name"
        ] = (
            f"{int(100 * retrieval_settings.credible_interval)}% equal-tailed"
            "credible interval for the retrieved footprint-averaged TIWP"
        )
        dataset["tiwp_fpavg_ci"].attrs["units"] = "kg m-2"

        dataset["p_tiwp_fpavg"].attrs[
            "long_name"
        ] = "Probability that 'tiwp_fpavg' exceeds 1e-3 kg m-2"
        dataset["p_tiwp_fpavg"].attrs["units"] = "1"

        dataset["ci_bounds"].attrs["standard_name"] = "credible_interval_bounds"
        dataset["ci_bounds"].attrs["units"] = "1"
        dataset["ci_bounds"].attrs["long_name"] = (
            "Quantile levels of the credible interval"
        )

    if "tiwc" in dataset:
        dataset["tiwc"].attrs["units"] = "g m-3"
        dataset["tiwc"].attrs["long_name"] = "Concentration of frozen hydrometeors"

    if "cloud_prob_2d" in dataset:
        dataset["cloud_prob_2d"].attrs["units"] = "1"
        dataset["cloud_prob_2d"].attrs[
            "long_name"
        ] = "Probability of presence of a cloud anywhere in the atmosphere"

    if "cloud_prob_3d" in dataset:
        dataset["cloud_prob_3d"].attrs["units"] = "1"
        dataset["cloud_prob_3d"].attrs[
            "long_name"
        ] = "Probability of presence of a cloud"

    if "cloud_type" in dataset:
        dataset["cloud_type"].attrs["units"] = "1"
        dataset["cloud_type"].attrs["long_name"] = "Most likely cloud type"
        dataset["cloud_type"].attrs["flag_values"] = "0, 1, 2, 3, 4, 5, 6, 7, 8"
        dataset["cloud_type"].attrs[
            "flag_meanings"
        ] = "No cloud, Cirrus, Altostratus, Altocumulus, Stratus, Stratocumulus, Cumulus, Nimbostratus, Deep convection"

    if "inpainted" in dataset:
        dataset["inpainted"].attrs["units"] = "1"
        dataset["inpainted"].attrs[
            "long_name"
        ] = "Inpainted pixel from input pixel with NaN"
        dataset["inpainted"].attrs["flag_values"] = "0, 1"
        dataset["inpainted"].attrs[
            "flag_meanings"
        ] = "Pixel not inpainted, pixel inpainted"


def get_encodings_zarr(variable_names):
    """
    Get variable encoding dict for storing the results for selected
    target variables in zarr format.
    """
    compressor = zarr.Blosc(cname="lz4", clevel=9, shuffle=2)
    filters_iwp = [LogBins(1e-3, 1e2)]
    filters_iwc = [LogBins(1e-4, 1e2)]
    all_encodings = {
        "tiwp": {"compressor": compressor, "filters": filters_iwp, "dtype": "float32"},
        "tiwc": {"compressor": compressor, "filters": filters_iwc, "dtype": "float32"},
        "p_tiwp": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
        },
        "tiwp_ci": {
            "compressor": compressor,
            "filters": filters_iwp,
            "dtype": "float32",
        },
        "tiwp_fpavg": {
            "compressor": compressor,
            "filters": filters_iwp,
            "dtype": "float32",
        },
        "p_tiwp_fpavg": {
            "compressor": compressor,
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
        },
        "tiwp_fpavg_ci": {
            "compressor": compressor,
            "filters": filters_iwp,
            "dtype": "float32",
        },
        "ci_bounds": {
            "compressor": compressor,
            "dtype": "float32"
        },
        "cloud_prob_2d": {
            "compressor": compressor,
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "dtype": "uint8",
        },
        "cloud_prob_3d": {
            "compressor": compressor,
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "dtype": "uint8",
        },
        "cloud_type": {"compressor": compressor, "dtype": "uint8", "_FillValue": 255},
        "longitude": {
            "compressor": compressor,
            "dtype": "float32",
        },
        "latitude": {
            "compressor": compressor,
            "dtype": "float32",
        },
        "inpainted": {"compressor": compressor, "dtype": "uint8", "_FillValue": 255},
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
        "tiwp_ci": {"dtype": "float32", "zlib": True},
        "tiwp_fpavg": {"dtype": "float32", "zlib": True},
        "tiwp_fpavg_ci": {"dtype": "float32", "zlib": True},
        "p_tiwp_fpavg": {
            "dtype": "uint8",
            "scale_factor": 1 / 250,
            "_FillValue": 255,
            "zlib": True,
        },
        "ci_bounds": {"dtype": "float32", "zlib": True},
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
        "inpainted": {"dtype": "uint8", "zlib": True},
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
