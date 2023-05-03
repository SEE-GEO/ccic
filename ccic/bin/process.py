"""
================
ccic.bin.process
================

This sub-module implements the CLI to run the CCIC retrievals.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging
from multiprocessing import Manager, Process
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

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
        metavar="cpcir/gridsat",
        type=str,
        help="For which type of input to run the retrieval.",
    )
    parser.add_argument(
        "output",
        metavar="output",
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
        "--targets", metavar="target", type=str, nargs="+", default=["tiwp", "tiwp_fpavg"]
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
    parser.add_argument(
        "--output_format",
        metavar="netcdf/zarr",
        type=str,
        help=(
            "The output format in which to store the output: 'zarr' or"
            " 'netcdf'. 'zarr' format applies a custom filter to allow "
            " storing 'tiwp' fields as 8-bit integer, which significantly"
            " reduces the size of the output files."
        ),
        default="netcdf"
    )
    parser.add_argument(
        "--database_path",
        metavar="path",
        type=str,
        help=(
            "Path to the database to use to log processing progress."
        ),
        default=None
    )
    parser.add_argument("--roi", metavar="x", type=float, nargs=4, default=None)
    parser.add_argument("--n_processes", metavar="n", type=int, default=1)
    parser.add_argument(
        "--inpainted_mask",
        action='store_true',
        help=(
            "Create a variable `inpainted` indicating if "
            "the retrieved pixel is inpainted (the input data was NaN)."
        )
    )
    parser.add_argument(
        "--confidence_interval",
        type=float,
        default=0.9,
        help=(
            "Width of the confidence interval to use to report retrieval "
            "uncertainty of scalar retrieval targets. Must be within [0, 1]."
        )
    )
    parser.set_defaults(func=run)


def process_files(processing_queue, result_queue, model, retrieval_settings):
    """
    Take a file from the queue, process it and write the output to
    the given folder.

    Args:
        processing_queue: Queue on which the downloaded input files are put.
        results_queue: The queue to hold the results to store to disk.
        model: The neural network model to run the retrieval with.
        retrieval_settings: RetrievalSettings object specifying the retrieval
            settings.
    """
    from quantnn.mrnn import MRNN
    from ccic.processing import (
        process_input_file,
        RemoteFile,
        ProcessingLog
    )
    logger = logging.getLogger(__file__)

    mrnn = MRNN.load(model)
    mrnn.model.eval()

    while True:
        input_file = processing_queue.get()
        if input_file is None:
            break

        log = ProcessingLog(
            retrieval_settings.database_path,
            Path(input_file.filename).name
        )

        with log.log(logger):
            try:
                logger.info("Starting processing input file '%s'.", input_file.filename)
                results = process_input_file(
                    mrnn, input_file, retrieval_settings=retrieval_settings
                )
                result_queue.put((input_file, results))
                logger.info("Finished processing file '%s'.", input_file.filename)
            except Exception as e:
                logger.exception(e)

    result_queue.put(None)


def download_files(download_queue, processing_queue, retrieval_settings):
    """
    Downloads file from download queue, if required.

    Args:
        download_queue: A queue object containing names of files to
            download.
        processing_queue: A queue object on which the downloaded files
            will be put.
    """
    from ccic.processing import (
        RemoteFile,
        ProcessingLog
    )
    logger = logging.getLogger(__file__)

    while True:
        input_file = download_queue.get()
        if input_file is None:
            break

        log = ProcessingLog(
            retrieval_settings.database_path,
            Path(input_file.filename).name
        )
        with log.log(logger):
            try:
                if isinstance(input_file, RemoteFile):
                    logger.info(
                        "Input file not locally available, download required."
                    )
                    input_file = input_file.get()
                else:
                    logger.info("Input file locally available.")
            except Exception as e:
                logger.exception(e)
        processing_queue.put(input_file)
    processing_queue.put(None)


def write_output(result_queue, retrieval_settings, output_path):
    """
    Write output.

    Args:
        result_queue: Queue on which retrieval results are put.
        retrieval_settings: The retrieval settings specifying the output
            format.
        output_path: The path to which to write the output.
    """
    from ccic.processing import (
        OutputFormat,
        get_encodings,
        get_output_filename,
        ProcessingLog
    )
    logger = logging.getLogger(__file__)

    while True:
        results = result_queue.get()
        if results is None:
            break

        input_file, data = results
        log = ProcessingLog(
            retrieval_settings.database_path,
            input_file.filename.name
        )
        output_file = get_output_filename(
            input_file,
            data.time.data[0],
            retrieval_settings
        )
        with log.log(logger):
            try:
                logger.info("Writing retrieval results to '%s'.", output_file)
                encodings = get_encodings(data.variables, retrieval_settings)
                if retrieval_settings.output_format == OutputFormat["NETCDF"]:
                    data.to_netcdf(output_path / output_file, encoding=encodings)
                else:
                    data.to_zarr(output_path / output_file, encoding=encodings)
                logger.info(
                    "Successfully processed input file '%s'.",
                    input_file.filename
                )
            except Exception as e:
                logger.exception(e)
        log.finalize(data, output_file)


def run(args):
    """
    Process input files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from ccic.data.gridsat import GridSat
    from ccic.data.cpcir import CPCIR
    from ccic.processing import (
        process_input_file,
        get_input_files,
        RetrievalSettings,
        OutputFormat
    )

    # Load model.
    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The provides model '%s' does not exist.", model.name)
        return 1

    # Determine input data.
    input_type = args.input_type.lower()
    if not input_type in ["cpcir", "gridsat"]:
        LOGGER.error(
            "'input_type' must be one of ['cpcir', gridsat'] not '%s'.", input_type
        )
        return 1
    if input_type == "cpcir":
        input_cls = CPCIR
    else:
        input_cls = GridSat

    # Output path
    output = Path(args.output)

    input_path = args.input_path
    if input_path is not None:
        input_path = Path(input_path)
        if not input_path.exists():
            LOGGER.error(
                "If 'input_path' argument is provided it must point to an "
                "existing local path."
            )
            return 1

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

    with TemporaryDirectory() as tmp:
        input_files = get_input_files(
            input_cls,
            start_time,
            end_time=end_time,
            working_dir=tmp,
            path=input_path
        )

        # Retrieval settings
        valid_targets = [
            "tiwp",
            "tiwp_fpavg",
            "tiwc",
            "cloud_type",
            "cloud_prob_2d",
            "cloud_prob_3d",
        ]
        targets = args.targets
        if any([target not in valid_targets for target in targets]):
            LOGGER.error("Targets must be a subset of %s.", valid_targets)

        output_format = None
        if not args.output_format.upper() in ["NETCDF", "ZARR"]:
            LOGGER.error(
                "'output_format' must be one of 'NETCDF' or 'ZARR'."
            )
            return 1
        output_format = args.output_format.upper()

        database_path = args.database_path
        if not database_path is None:
            database_path = Path(database_path)
            if not database_path.parent.exists():
                LOGGER.error(
                    "If provided, database path must point to a file in an "
                    " directory."
                )
                return 1

        if ((args.confidence_interval < 0.0) or
            (args.confidence_interval > 1.0)):
            LOGGER.error(
                "Width of confidence interval must be within [0, 1]."
            )
            return 1

        retrieval_settings = RetrievalSettings(
            tile_size=args.tile_size,
            overlap=args.overlap,
            targets=targets,
            roi=args.roi,
            device=args.device,
            precision=args.precision,
            output_format=OutputFormat[output_format],
            database_path=args.database_path,
            inpainted_mask=args.inpainted_mask,
            confidence_interval=args.confidence_interval
        )

        # Use managed queue to pass files between download threads
        # and processing processes.
        manager = Manager()
        download_queue = manager.Queue()
        processing_queue = manager.Queue(4)
        result_queue = manager.Queue(4)

        args = (download_queue, processing_queue, retrieval_settings)
        download_thread = Thread(target=download_files, args=args)
        args = (processing_queue, result_queue, model, retrieval_settings)
        processing_process = Process(target=process_files, args=args)
        args = (result_queue, retrieval_settings, output)
        output_process = Process(target=write_output, args=args)

        # Submit a download task for each file.
        for input_file in input_files:
            download_queue.put(input_file)
        download_queue.put(None)

        download_thread.start()
        processing_process.start()
        output_process.start()

        download_thread.join()
        processing_process.join()
        output_process.join()

