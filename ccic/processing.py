"""
ccic.processing
===============

Implements functions for the operational processing of the CCIC
retrieval.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pansat.time import to_datetime
import torch
import xarray as xr

from ccic.tiler import Tiler
from ccic.data.gpmir import GPMIR
from ccic.data.gridsat import GridSatB1
from ccic.data.utils import extract_roi


@dataclass
class RemoteFile:
    """
    Simple wrapper class around a file that is not locally available
    but downloaded via pansat.
    """
    file_cls: type
    filename: str

    def get(self, path):
        """
        Download the file.

        Args:
            path: The local folder in which to store the file.

        Return:
            A ``file_cls`` object pointing to the downloaded file.
        """
        output_path = path / self.filename
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


def get_input_files(input_cls, start_time, end_time=None, path=None):
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
    """
    start_time = to_datetime(start_time)
    if end_time is None:
        end_time = start_time

    # Return remote files if no path if given.
    if path is None:
        files = input_cls.get_available_files(start_time=start_time, end_time=end_time)
        return [RemoteFile(input_cls, filename) for filename in files]

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

def process_regression_target(
        mrnn,
        y_pred,
        target,
        target_quantiles,
        means,
        quantiles,
        samples
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
        target_quantiles: The quantiles to calculate for the output.
        means: Result dict to which to store the calculated posterior means.
        quantiles: Result dict to which to store the calculated posterior quantiles.
        samples: Result dict to which to store the calculated samples.
    """
    mean = (
        mrnn.posterior_mean(y_pred=y_pred[target], key=target)
        .cpu()
        .numpy()
    )
    means[target][-1].append(mean)
    if target in SCALAR_TARGETS:
        quants = (
            mrnn.posterior_quantiles(
                y_pred=y_pred[target],
                quantiles=target_quantiles,
                key=target
            ).cpu() .numpy()
        )
        quantiles[target][-1].append(quants)
        sample = (
            mrnn.sample_posterior(
                y_pred=y_pred[target], n_samples=1, key=target
            )[:, 0].cpu().numpy()
        )
        samples[target][-1].append(sample)


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
    quantiles = {}
    samples = {}
    cloud_prob_2d = []
    cloud_prob_3d = []
    cloud_type = []

    target_quantiles = [0.022, 0.156, 0.841, 0.977]

    device = retrieval_settings.device
    mrnn.model.to(device)

    with torch.no_grad():
        for i in range(tiler.M):

            # Insert empty list into list of row results.
            for target in targets:
                if target in REGRESSION_TARGETS:
                    means.setdefault(target, []).append([])
                    if target in SCALAR_TARGETS:
                        quantiles.setdefault(target, []).append([])
                        samples.setdefault(target, []).append([])
                elif target == "cloud_prob_2d":
                    cloud_prob_2d.append([])
                elif target == "cloud_prob_3d":
                    cloud_prob_3d.append([])
                elif target == "cloud_type":
                    cloud_type.append([])

            for j in range(tiler.N):
                x_t = tiler.get_tile(i, j).to(device)

                y_pred = mrnn.predict(x_t)

                for target in targets:
                    if target in REGRESSION_TARGETS:
                        process_regression_target(
                            mrnn,
                            y_pred,
                            target,
                            target_quantiles,
                            means=means,
                            quantiles=quantiles,
                            samples=samples
                        )
                    elif target == "cloud_prob_2d":
                        cloud_prob_2d[-1].append(y_pred["cloud_mask"].cpu().numpy())
                    elif target == "cloud_prob_3d":
                        cloud_prob_3d[-1].append(
                            1.0 - y_pred["cloud_class"][:, 0].cpu().numpy()
                        )
                    elif target == "cloud_type":
                        ct = torch.softmax(y_pred["cloud_class"][:, 1:], 1)
                        cloud_type[-1].append(
                            ct.cpu().numpy()
                        )


    results = xr.Dataset()

    for target, mean in means.items():
        mean = tiler.assemble(mean)
        if mean.ndim == 3:
            dims = ("time", "latitude", "longitude")
        else:
            dims = ("time", "latitude", "longitude", "levels")
            mean = np.transpose(mean, [0, 2, 3, 1])
        results[target + "_mean"] = (dims, mean)

    dims = ("time", "latitude", "longitude")
    for target, sample in samples.items():
        smpls = tiler.assemble(sample)
        results[target + "_sample"] = (dims, smpls)

    dims = ("time", "latitude", "longitude", "quantiles")
    for target, quants in quantiles.items():
        quants = tiler.assemble(quants)
        quants = np.transpose(quants, [0, 2, 3, 1])
        results[target + "_quantiles"] = (dims, quants)

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
    input_data = input_data.rename({"lat": "latitude", "lon": "longitude"})

    for dim in ["time", "latitude", "longitude"]:
        results[dim].data[:] = input_data[dim].data
    results.attrs.update(input_file.get_input_file_attributes())

    return results

