"""
================
ccic.bin.process
================

This sub-module implements the CLI to run the CCIC retrievals.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import logging
from multiprocessing import Manager, Process, Lock
from pathlib import Path
import shutil
import subprocess
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
            "Path to a local directory to store and search for input files. If not given, "
            "input files are downloaded to a temporary directory and discarded after the "
            "retrieval."
        ),
    )
    parser.add_argument(
        "--targets",
        metavar="target",
        type=str,
        nargs="+",
        default=["tiwp", "tiwp_fpavg"],
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
        default="cpu",
    )
    parser.add_argument(
        "--precision",
        metavar="16/32",
        type=int,
        help=(
            "The precision with which to run the retrieval. Only has an "
            "effect for GPU processing."
        ),
        default=32,
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
        default="netcdf",
    )
    parser.add_argument(
        "--database_path",
        metavar="path",
        type=str,
        help=("Path to the database to use to log processing progress."),
        default=None,
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        help=(
            "If this flag is set, the retrieval will be run only for input "
            "whose processing is flagged as failed in the processing log "
            " database."
        ),
    )
    parser.add_argument(
        "--roi",
        metavar=("lon_min", "lat_min", "lon_max", "lat_max"),
        type=float,
        nargs=4,
        default=None,
        help=(
            "Corner coordinates (lon_min, lat_min, lon_max, lat_max) of a "
            " rectangular bounding box for which to run the retrieval. If "
            " given, the retrieval will be run only for  a limited subset "
            " of the global input data that in guaranteed to include the "
            " given ROI. "
            "NOTE: that the minimum size of the output will be at "
            " leat 256 pixels."
        ),
    )
    parser.add_argument(
        "--inpainted_mask",
        action="store_true",
        help=(
            "Create a variable `inpainted` indicating if "
            "the retrieved pixel is inpainted (the input data was NaN)."
        ),
    )
    parser.add_argument(
        "--credible_interval",
        type=float,
        default=0.9,
        help=(
            "Probability of the equal-tailed credible interval used to "
            "report retrieval uncertainty of scalar retrieval targets. "
            "Must be within [0, 1]."
        ),
    )
    parser.add_argument(
        "--transfer",
        type=str,
        default=None,
        help=(
            "Optional ssh target to which the produced output file will be "
            " copied using scp. This requires ssh public-key authentication "
            " to be set up on the system."
        ),
    )
    parser.add_argument(
        "--n_processes",
        type=int,
        default=2,
        help=(
            "The number of concurrent processes to use for processing of the "
            "retrieval."
        ),
    )
    parser.set_defaults(func=run)


def process_files(
    processing_queue, model, retrieval_settings, output_path, device_lock
):
    """
    Take a file from the queue, process it and write the output to
    the given folder.

    Args:
        processing_queue: Queue on which the downloaded input files are put.
        model: The neural network model to run the retrieval with.
        retrieval_settings: RetrievalSettings object specifying the retrieval
            settings.
    """
    from quantnn.mrnn import MRNN
    from ccic.processing import (
        process_input_file,
        RemoteFile,
        ProcessingLog,
        get_encodings,
        get_output_filename,
        OutputFormat,
    )

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mrnn = MRNN.load(model)
    mrnn.model.eval()

    while True:
        args = processing_queue.get()
        if args is None:
            processing_queue.task_done()
            processing_queue.put(None)
            break
        input_file, clean_up = args

        log = ProcessingLog(
            retrieval_settings.database_path, Path(input_file.filename).name
        )

        with log.log(logger):
            try:
                logger.info(
                    "Starting processing of input file '%s'.", input_file.filename
                )
                results = process_input_file(
                    mrnn,
                    input_file,
                    retrieval_settings=retrieval_settings,
                    lock=device_lock,
                )
                output_file = get_output_filename(
                    input_file, results.time.data[0], retrieval_settings
                )
                logger.info("Finished processing file '%s'.", input_file.filename)
                logger.info("Writing retrieval results to '%s'.", output_path / output_file)
                encodings = get_encodings(results.variables, retrieval_settings)
                if retrieval_settings.output_format == OutputFormat["NETCDF"]:
                    results.to_netcdf(output_path / output_file, encoding=encodings)
                else:
                    results.to_zarr(output_path / output_file, encoding=encodings)

                if retrieval_settings.transfer is not None:
                    output = output_path / output_file
                    command = ["scp", "-r", str(output), retrieval_settings.transfer]
                    subprocess.run(command, check=True)
                    if output.is_dir():
                        shutil.rmtree(output)
                    else:
                        output.unlink()

                log.finalize(results, output_file)
                logger.info(
                    "Successfully processed input file '%s'.", input_file.filename
                )
            except Exception as e:
                logger.exception(e)
            finally:
                if clean_up:
                    Path(input_file.filename).unlink()
                processing_queue.task_done()


def download_files(download_queue, processing_queue, retrieval_settings):
    """
    This function implements a thread target that handles the download
    of the input function. The function waits for RemoteFile objects to
    be put on the 'download_queue' and will start downloading them. After
    the files are downloaded, they are put on the processing queue.

    Args:
        download_queue: A queue object containing names of files to
            download.
        processing_queue: A queue object on which the downloaded files
            will be put.

    """
    from ccic.processing import RemoteFile, ProcessingLog

    logger = logging.getLogger(__name__)

    while True:
        input_file = download_queue.get()
        if input_file is None:
            break

        log = ProcessingLog(
            retrieval_settings.database_path, Path(input_file.filename).name
        )
        with log.log(logger):
            try:
                input_file, clean_up = input_file.get()
            except Exception:
                logger.exception(
                    "Downloading of input file '%s' failed.",
                    input_file.filename
                )
                continue
            if input_file is None:
                # Something went wrong when opening the file
                continue
        processing_queue.put((input_file, clean_up))

    processing_queue.put(None)


def _get_database_name(args) -> str:
    """Determine database name based on arguments."""
    hsh = hashlib.md5()
    hsh.update(
        (
            f"{args.model}{args.input_type}{args.start_time}{args.end_time}"
            f"{args.roi}"
        ).encode()
    )
    database_name = f"ccic_processing_{hsh.hexdigest()[:32]}.db"
    return database_name


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
        RemoteFile,
        RetrievalSettings,
        OutputFormat,
        ProcessingLog,
    )

    # Load model.
    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The provided CCIC retrieval model '%s' does not exist.", model)
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
    if not output.exists() or not output.is_dir():
        LOGGER.error(
            "The given output path '%s' does not point to an existing directory.",
            output
        )
        return 1

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
        LOGGER.error("'output_format' must be one of 'NETCDF' or 'ZARR'.")
        return 1
    output_format = args.output_format.upper()

    database_path = args.database_path
    if database_path is not None:
        database_path = Path(database_path)
        if database_path.is_dir() and database_path.exists():
            command_hash = hash(
                f"{args.model}{args.input_type}{args.start_time}{args.end_time}"
                f"{args.roi}"
            )
            database_name = _get_database_name(args)
            database_path = database_path / database_name
        elif not database_path.parent.exists():
            LOGGER.error(
                "If provided, database path must point to a file in an "
                "existing directory."
            )
            return 1

    if input_path is None:
        tmp = TemporaryDirectory()
        working_dir = Path(tmp.name)
    else:
        working_dir = input_path

    input_files = get_input_files(
        input_cls,
        start_time,
        end_time=end_time,
        path=working_dir,
    )
    if args.failed:
        if database_path is None:
            LOGGER.error(
                "To reprocess failed files the path to a processing database must "
                " be provided using the '--database_path' argument."
            )
            return 1

        if not database_path.exists():
            LOGGER.warning(
                "The database path '%s' doesn't yet exist, so there are no "
                "failed files to process. ",
                database_path
            )
            failed_files = []
        else:
            processed_files = [
                RemoteFile(input_cls, name, working_dir=Path(working_dir))
                for name in ProcessingLog.get_input_file(database_path, success=True)
            ]
            failed_files = [
                RemoteFile(input_cls, name, working_dir=Path(working_dir))
                for name in ProcessingLog.get_input_file(database_path, success=False)
            ]
        
        processed_files = set(processed_files)
        failed_files = set(failed_files)
        input_files = set(input_files)

        processed = input_files.intersection(processed_files)
        failed = input_files.intersection(failed_files)
        new_files = input_files.difference(processed).difference(failed_files)

        # Initialize database with files that weren't in DB.
        for input_file in new_files:
            ProcessingLog(database_path, input_file)

        # Sort files, to process them chronologically
        input_files = sorted(
            list(failed) + list(new_files),
            key=lambda x: x.filename
        )
        message = (
            f"Found {len(failed)} failed input files "
            f"in logging database {database_path}."
        )
        if len(new_files) > 0:
            message = f"{message} {len(new_files)} were not yet in the database."
        
        LOGGER.info(message)
    else:
        # Initialize database with all found files.
        if database_path is not None:
            for input_file in input_files:
                ProcessingLog(database_path, input_file)

    if ((args.credible_interval < 0.0) or
        (args.credible_interval > 1.0)):
        LOGGER.error(
            "Probability of credible interval must be within [0, 1]."
        )
        return 1

    n_processes = args.n_processes

    retrieval_settings = RetrievalSettings(
        tile_size=args.tile_size,
        overlap=args.overlap,
        targets=targets,
        roi=args.roi,
        device=args.device,
        precision=args.precision,
        output_format=OutputFormat[output_format],
        database_path=database_path,
        inpainted_mask=args.inpainted_mask,
        credible_interval=args.credible_interval,
        transfer=args.transfer,
    )

    # Use managed queue to pass files between download threads
    # and processing processes.
    manager = Manager()
    download_queue = manager.Queue()
    processing_queue = manager.Queue(4)
    device_lock = manager.Lock()

    args = (download_queue, processing_queue, retrieval_settings)
    download_thread = Thread(target=download_files, args=args)
    args = (processing_queue, model, retrieval_settings, output, device_lock)
    processing_processes = [
        Process(target=process_files, args=args) for i in range(n_processes)
    ]

    # Submit a download task for each file.
    for input_file in input_files:
        download_queue.put(input_file)
    download_queue.put(None)

    download_thread.start()
    [proc.start() for proc in processing_processes]

    running = [download_thread] + processing_processes

    any_failed = False
    while True:
        running = [proc for proc in running if proc.is_alive()]
        if len(running) == 0:
            break
        for processing_process in processing_processes:
            if not processing_process.is_alive():
                if processing_process.exitcode != 0:
                    LOGGER.warning(
                        "One of the processing processes terminated with a "
                        " non-zero exit code. This indicates that the process "
                        " was killed. Potentially due to memory issues."
                    )
                any_failed = True
            processing_processes = [
                proc for proc in processing_processes if proc.is_alive()
            ]

    processing_queue.get()
    processing_queue.task_done()
    processing_queue.join()

    return not any_failed
