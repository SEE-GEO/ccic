"""
================
ccic.bin.process
================

This sub-module implements the CLI to run the CCIC retrievals.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from multiprocessing import Manager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'process' command to top-level parser. This function
    is called from the top-level parser defined in 'ccic.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "process",
        help="Run CCIC retrieval.",
        description=(
            """
            Run CCIC retrieval.
            """
        ),
    )
    parser.add_argument(
        "model",
        metavar="model",
        type=str,
        help="Path to the trained CCIC model.",
    )
    parser.add_argument(
        "input_type",
        metavar="GPMIR/GRIDSAT",
        type=str,
        help="For which type of input to run the retrieval.",
    )
    parser.add_argument(
        "output",
        metavar="ouput",
        type=str,
        help="Folder to which to write the output.",
    )
    parser.add_argument(
        "start_time",
        metavar="t1",
        type=str,
        help="The time for which to run the retrieval.",
    )
    parser.add_argument(
        "end_time",
        metavar="t2",
        type=str,
        help=(
            "If given, the retrieval will be run for all files in the time "
            "range t1 <= t <= t2"
        ),
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--input_path",
        metavar="path",
        type=str,
        default=None,
        help=(
            "Path to a local directory containing input files. If not given, "
            "input files will be downloaded using pansat."
        ),
    )
    parser.add_argument(
        "--targets", metavar="target", type=str, nargs="+", default=["iwp", "iwp_rand"]
    )
    parser.add_argument(
        "--tile_size",
        metavar="N",
        type=int,
        help="Tile size to use for processing.",
        default=512,
    )
    parser.add_argument(
        "--overlap",
        metavar="N",
        type=int,
        help="Tile size to use for processing.",
        default=128,
    )
    parser.add_argument(
        "--device",
        metavar="dev",
        type=str,
        help="The name of the torch device to use for processing.",
        default="cpu"
    )
    parser.add_argument(
        "--precision",
        metavar="16/32",
        type=int,
        help=(
            "The precision with which to run the retrieval. Only has an "
            "effect for GPU processing."
        ),
        default=32
    )
    parser.add_argument("--roi", metavar="x", type=float, nargs=4, default=None)
    parser.add_argument("--n_processes", metavar="n", type=int, default=1)
    parser.set_defaults(func=run)


ENCODINGS = {
    "iwp_mean": {"zlib": True},
    "iwp_quantiles": {"zlib": True},
    "iwp_sample": {"zlib": True},
    "iwp_rand_mean": {"zlib": True},
    "iwp_rand_quantiles": {"zlib": True},
    "iwp_rand_sample": {"zlib": True},
    "iwc": {"zlib": True},
    "cloud_prob_2d": {
        "zlib": True,
        "scale_factor": 250,
        "_FillValue": 255,
        "dtype": "uint8"
    },
    "cloud_prob_3d": {
        "zlib": True,
        "scale_factor": 250,
        "_FillValue": 255,
        "dtype": "uint8"
    },
    "cloud_type": {
        "zlib": True,
        "dtype": "uint8"
    }
}


def process_file(model, file_queue, output, retrieval_settings):
    """
    Processing of a single input file.
    """
    from quantnn.mrnn import MRNN
    from ccic.processing import process_input_file, get_output_filename, RemoteFile

    mrnn = MRNN.load(model)
    mrnn.model.eval()

    input_file = file_queue.get()

    results = process_input_file(
        mrnn, input_file, retrieval_settings=retrieval_settings
    )
    input_data = input_file.to_xarray_dataset()
    output_filename = get_output_filename(input_file, input_data.time[0].item())

    encodings = {key: ENCODINGS[key] for key in results.variables.keys()}
    results.to_netcdf(output / output_filename, encoding=encodings)


def download_file(input_file, file_queue):
    from ccic.processing import RemoteFile
    if isinstance(input_file, RemoteFile):
        input_file = input_file.get()
    file_queue.put(input_file)


def run(args):
    """
    Process input files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from ccic.data.gridsat import GridSatB1
    from ccic.data.gpmir import GPMIR
    from ccic.processing import process_input_file, get_input_files, RetrievalSettings

    # Load model.
    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The provides model '%s' does not exist.", model.name)
        return 1

    # Determine input data.
    input_type = args.input_type.lower()
    if not input_type in ["gpmir", "gridsatb1"]:
        LOGGER.error(
            "'input_type' must be one of ['gpmir', gridsatb1'] not '%s'.", input_type
        )
        return 1
    if input_type == "gpmir":
        input_cls = GPMIR
    else:
        input_cls = GridSatB1

    # Output path
    output = Path(args.output)

    # Start and end time.
    try:
        start_time = np.datetime64(args.start_time)
    except ValueError:
        LOGGER.error(
            "'start_time' argument must be a valid numpy.datetime64 " " time string."
        )
        return 1

    end_time = args.end_time
    if end_time is None:
        end_time = start_time
    else:
        try:
            end_time = np.datetime64(args.end_time)
        except ValueError:
            LOGGER.error(
                "'start_time' argument must be a valid numpy.datetime64 time " "string."
            )
            return 1

    download_pool = ThreadPoolExecutor(max_workers=1)
    with TemporaryDirectory() as tmp:
        input_files = get_input_files(
            input_cls,
            start_time,
            end_time=end_time,
            working_dir=tmp,
        )

        # Retrieval settings
        valid_targets = [
            "iwp",
            "iwp_rand",
            "iwc",
            "cloud_type",
            "cloud_prob_2d",
            "cloud_prob_3d",
        ]
        targets = args.targets
        if any([target not in valid_targets for target in targets]):
            LOGGER.error("Targets must be a subset of %s.", valid_targets)

        retrieval_settings = RetrievalSettings(
            tile_size=args.tile_size,
            overlap=args.overlap,
            targets=targets,
            roi=args.roi,
            device=args.device,
            precision=args.precision
        )

        pool = ProcessPoolExecutor(max_workers=args.n_processes)
        tasks = []

        # Use managed queue to pass files between download threads
        # and processing processes.
        manager = Manager()
        file_queue = manager.Queue(2)

        # Submit a download task for each file.
        for input_file in input_files:
            download_pool.submit(download_file, input_file, file_queue)
        # Submit a corresponding processing task for each file.
        for input_file in input_files:
            tasks.append(
                pool.submit(process_file, model, file_queue, output, retrieval_settings)
            )

        # Fetch results.
        for task in tasks:
            task.result()
