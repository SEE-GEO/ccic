"""
=================================
ccic.bin.run_validation_retrieval
=================================

This sub-module implements the CLI to run CCIC radar-only retrievals.
"""
from calendar import monthrange
from datetime import datetime
import logging
import importlib
from multiprocessing import Manager, Process
from pathlib import Path

import numpy as np
from pansat.time import to_datetime64

LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'run_validation_retrieval' command to top-level parser.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "run_radar_retrieval",
        help="Run radar-only retrieval",
        description=(
            """
            Run radar-only retrieval.
            """
        ),
    )
    parser.add_argument(
        "instrument",
        metavar="name",
        type=str,
        help=("The name of the radar instrument.")
    )
    parser.add_argument(
        "observations",
        metavar="observations",
        type=str,
        help=("Folder containing the input observations.")
    )
    parser.add_argument(
        "era5_data",
        metavar="era5_data",
        type=str,
        help=("Folder containing the ERA5 data.")
    )

    parser.add_argument(
        "year",
        metavar="year",
        type=int,
        help="The year for which to run the retrieval.",
    )
    parser.add_argument(
        "month",
        metavar="month",
        type=int,
        help="The month for which to run the retrieval.",
    )
    parser.add_argument(
        "days",
        metavar="day",
        type=int,
        nargs="*",
        help="The days for which to run the retrieval.",
    )
    parser.add_argument(
        "output_path",
        metavar="results",
        type=str,
        help="The path at which the retrieval results will be stored.",
    )
    parser.add_argument(
        "--static_data",
        metavar="path",
        type=str,
        default="data/",
        help="Path pointing to the directory holding static retrieval data."
    )
    parser.add_argument(
        "--ice_psd",
        metavar="PSD",
        type=str,
        default="f07",
        help=(
            "The PSD to use for ice hydrometoers. 'F07' or 'D14'."
        )
    )
    parser.add_argument(
        "--ice_shapes",
        metavar="shape",
        type=str,
        nargs="+",
        default=["EvansSnowAggregate"],
        help=(
            "The ARTS SSDB ice particle shapes to run the retrieval for."
        )
    )
    parser.add_argument(
        "--input_path",
        metavar="shape",
        type=str,
        default=None,
        help=(
            "By default a temporary directory is used to download the "
            "retrieval input. Setting this path will use a non-temporary "
            "directory and existing files will not be downloaded a second "
            "time."
        )
    )
    parser.add_argument(
        "--n_workers",
        metavar="n",
        type=int,
        default=1,
        help=(
            "The number of processes to use to parallelize the retrieval"
            "processing."
        )
    )
    parser.add_argument(
        "--time_step",
        metavar="s",
        type=int,
        default=300,
        help=(
            "Temporal sampling of the radar-only retrievals."
        )
    )
    parser.add_argument(
        "--retrieval_resolution",
        metavar="m",
        type=float,
        default=200.0,
        help=(
            "Vertical resolution of the retrieval."
        )
    )
    parser.add_argument(
        "--radar_resolution",
        metavar="m",
        type=float,
        default=133.0,
        help=(
            "Vertical resolution to which the input observations will be resampled."
        )
    )
    parser.set_defaults(func=run)

def download_data(
        download_queue,
        processing_queue,
        n_workers=1
):
    """
    Downloads data for a given day.

    Args:
        download_queue: A queue object containing names of files to
            download.
        processing_queue: A queue object on which the downloaded files
            will be put.
        input_data: An input data object providing an interface to download
            and read input data.
    """
    logger = logging.getLogger(__name__)
    while True:
        task = download_queue.get()
        if task is None:
            break
        input_data, date = task
        try:
            if not input_data.has_data():
                logger.info(
                    "Downloading input data for %s",
                    date
                )
                input_data.download_data()
            processing_queue.put((input_data, date))
        except Exception as exc:
            logger.exception(exc)
    for _ in range(n_workers):
        processing_queue.put(None)


def process_files(
        processing_queue,
        ice_psd,
        ice_shapes,
        output_path,
        time_step,
):
    """
    Processes input data from the processing queue

    Args:
        processing_queue: Queue on which the downloaded input files are put.
        results_queue: The queue to hold the results to store to disk.
        model: The neural network model to run the retrieval with.
        retrieval_settings: RetrievalSettings object specifying the retrieval
            settings.
        output_path: The path to which to write the retrieval results.
    """
    from ccic.validation.retrieval import process_day
    logger = logging.getLogger(__name__)

    while True:
        task = processing_queue.get()
        if task is None:
            break
        input_data, date = task

        try:
            logger.info(
                "Starting %s radar-only retrieval for '%s'.",
                type(input_data.radar).__name__,
                input_data.radar_file
            )
            process_day(
                date,
                input_data,
                ice_psd,
                ice_shapes,
                output_path,
                time_step,
            )
        except Exception as exc:
            logger.exception(exc)


def run(args):
    """
    Process input files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from ccic.validation.input_data import RetrievalInput

    instrument = args.instrument
    module = importlib.import_module("ccic.validation.radars")
    radar = getattr(module, instrument)

    # Input data paths.
    radar_data_path = Path(args.observations)
    if not radar_data_path.exists():
        radar_data_path.mkdir(parents=True, exist_ok=True)
    era5_data_path = Path(args.era5_data)
    if not era5_data_path.exists():
        era5_data_path.mkdir(parents=True, exist_ok=True)

    # Determine days for which to run the retrieval.
    year = args.year
    month = args.month
    days = args.days
    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = list(range(1, n_days + 1))

    start = to_datetime64(datetime(year, month, 1))
    dates = [start + np.timedelta64(24 * 60 * 60, "s") * (day - 1) for day in days]

    # Output path
    output_path = Path(args.output_path)

    static_data_path = Path(args.static_data)
    ice_shapes = args.ice_shapes
    ice_psd = args.ice_psd.lower()
    n_workers = args.n_workers
    time_step = np.timedelta64(args.time_step, "s")

    # Use managed queue to pass files between download threads
    # and processing processes.
    manager = Manager()
    download_queue = manager.Queue()
    processing_queue = manager.Queue(4)

    retrieval_resolution = args.retrieval_resolution
    radar_resolution = args.radar_resolution

    logging.basicConfig(level="INFO", force=True)

    args = (download_queue, processing_queue)
    download_process = Process(
        target=download_data,
        args=args,
        kwargs={"n_workers": n_workers}
    )
    args = (processing_queue, ice_psd, ice_shapes, output_path, time_step)
    processing_processes = [
        Process(target=process_files, args=args) for _ in range(n_workers)
    ]

    for date in dates:
        files = radar.get_files(radar_data_path, date)
        for input_file in files:
            input_data = RetrievalInput(
                radar,
                radar_data_path,
                input_file,
                era5_data_path,
                static_data_path,
                vertical_resolution=retrieval_resolution,
                radar_resolution=radar_resolution,
            )
            download_queue.put((input_data, date))
    download_queue.put(None)

    download_process.start()
    [proc.start() for proc in processing_processes]

    download_process.join()
    [proc.join() for proc in processing_processes]
