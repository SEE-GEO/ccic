"""
ccic.processing
===============

Implements functions for the operational processing of the CCIC
retrieval.
"""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pansat.time import to_datetime
import torch
import xarray as xr
import zarr

from ccic.tiler import Tiler
from ccic.data.gpmir import GPMIR
from ccic.data.gridsat import GridSatB1
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
    def __init__(
            self,
            file_cls,
            filename,
            working_dir,
            thread_pool=None
    ):
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
            self.file_cls.download,
            self.filename,
            output_path
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


def get_input_files(
        input_cls,
        start_time,
        end_time=None,
        path=None,
        thread_pool=None,
        working_dir=None
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
            ) for filename in files
        ]

    # Return local files if path if given.
    files = input_cls.find_files(path=path, start_time=start_time, end_time=end_time)
    return [input_cls(filename) for filename in files]


def get_output_filename(
        input_file,
        date
):
    """
    Get filename for CCIC output file.

    Args:
        input_file: The input file object.
        date: Time stamp of the first observations in the input file.
    """
    if isinstance(input_file, GPMIR):
        file_type = "gpmir"
    elif isinstance(input_file, GridSatB1):
        file_type = "gridsat"
    else:
        raise ValueError(
            "'input_file' should be an instance of 'GPMIR' or "
            "'GridSatB1' not '%s'.",
            type(input_file)
        )
    date_str = to_datetime(date).strftime("%Y%m%d%H%M")
    return f"ccic_{file_type}_{date_str}.nc"


REGRESSION_TARGETS = ["iwp", "iwp_rand", "iwc"]
SCALAR_TARGETS = ["iwp", "iwp_rand"]
THRESHOLDS = {
    "iwp": 1e-3,
    "iwp_rand": 1e-3,
    "iwc": 1e-3
}

# Maps NN target names to output names.
OUTPUT_NAMES = {
    "iwp": "tiwp_fpavg",
    "iwp_rand": "tiwp",
    "iwc": "tiwc"
}

def process_regression_target(
        mrnn,
        y_pred,
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
    mean = (
        mrnn.posterior_mean(y_pred=y_pred[target], key=target)
        .cpu()
        .float()
        .numpy()
    )
    means[target][-1].append(mean)
    if target in SCALAR_TARGETS:
        log_std_dev = (
            mrnn.posterior_std_dev(
                y_pred=torch.log10(y_pred[target]),
                key=target
            ).cpu().float().numpy()
        )
        log_std_devs[target][-1].append(log_std_dev)
        p_non_zero = mrnn.probability_larger_than(
                y_pred=y_pred[target], y=THRESHOLDS[target], key=target
            )[:].cpu().float().numpy()
        p_non_zeros[target][-1].append(p_non_zero)


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
            "cloud_type"
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

                for target in targets:
                    if target in REGRESSION_TARGETS:
                        process_regression_target(
                            mrnn,
                            y_pred,
                            target,
                            means=means,
                            log_std_devs=log_std_devs,
                            p_non_zeros=p_non_zeros
                        )
                    elif target == "cloud_prob_2d":
                        cloud_prob_2d[-1].append(
                            y_pred["cloud_mask"].cpu().float().numpy()
                        )
                    elif target == "cloud_prob_3d":
                        cp = 1.0 - y_pred["cloud_class"]
                        cloud_prob_3d[-1].append(
                            cp[:, 0].cpu().float().numpy()
                        )
                    elif target == "cloud_type":
                        ct = torch.softmax(y_pred["cloud_class"][:, 1:], 1)
                        cloud_type[-1].append(
                            ct.cpu().float().numpy()
                        )

    results = xr.Dataset()
    for target, mean in means.items():
        mean = tiler.assemble(mean)
        if mean.ndim == 3:
            dims = ("time", "latitude", "longitude")
        else:
            dims = ("time", "latitude", "longitude", "levels")
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
        cloud_prob_2d = tiler.assemble(cloud_prob_2d)[:, 0]
        results["cloud_prob_2d"] = (dims, cloud_prob_2d)

    dims = ("time", "latitude", "longitude", "levels")
    if len(cloud_prob_3d) > 0:
        cloud_prob_3d = tiler.assemble(cloud_prob_3d)
        cloud_prob_3d = np.transpose(cloud_prob_3d, [0, 2, 3, 1])
        results["cloud_prob_3d"] = (dims, cloud_prob_3d)

    dims = ("time", "latitude", "longitude", "levels", "type")
    if len(cloud_type) > 0:
        cloud_type = tiler.assemble(cloud_type)
        cloud_type = np.transpose(cloud_type, [0, 3, 4, 2, 1])
        results["cloud_type"] = (dims, cloud_type)

    return results


def process_input_file(
        mrnn,
        input_file,
        retrieval_settings=None
):
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
        mrnn,
        retrieval_input,
        retrieval_settings=retrieval_settings
    )

    # Copy values of dimension
    input_data = input_file.to_xarray_dataset()
    if roi is not None:
        input_data = extract_roi(input_data, roi, min_size=256)
    print(input_data)
    print(results)
    input_data = input_data.rename({"lat": "latitude", "lon": "longitude"})

    for dim in ["time", "latitude", "longitude"]:
        results[dim] = input_data[dim]
    results.attrs.update(input_file.get_input_file_attributes())

    return results


def get_encodings_zarr(variable_names):
    """
    Get variable encoding dict for storing the results for selected
    target variables.
    """
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    filters = [LogBins(1e-3, 1e2)]
    all_encodings = {
        "iwp_mean": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwp_quantiles": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwp_sample": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwp_rand_mean": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwp_rand_quantiles": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwp_rand_sample": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "iwc_mean": {"compressor": compressor, "filters": filters, "dtype": "float32"},
        "cloud_prob_2d": {
            "compressor": compressor,
            "scale_factor": 250,
            "_FillValue": 255,
            "dtype": "uint8"
        },
        "cloud_prob_3d": {
            "compressor": compressor,
            "scale_factor": 250,
            "_FillValue": 255,
            "dtype": "uint8"
        },
        "cloud_type": {
            "compressor": compressor,
            "dtype": "uint8"
        }
    }
    return {all_encodings[name] for name in variable_names}
