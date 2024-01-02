"""
===================
process_most_recent
===================

This command processes the most recent CPCIR data that is available.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np

def run(args):
    """
    Process input files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    start_time = datetime.now()
    end_time = start_time - timedelta(days=args.days)
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    output = args.output
    input_type = "cpcir"
    model = args.model

    command = f"ccic process {model} {input_type} {output} {start_time} {end_time}"

    input_path = args.input_path
    if input_path is not None:
        command += f" --input_path input_path"

    targets = args.targets
    command += f" --targets {' '.join(targets)}"
    tile_size = args.tile_size
    command += f" --tile_size {tile_size}"
    overlap = args.overlap
    command += f" --overlap {overlap}"
    device = args.device
    command += f" --device {device}"
    precision = args.precision
    command += f" --precision {precision}"
    output_format = args.output_format
    command += f" --output_format {output_format}"
    credible_interval = args.credible_interval
    command += f" --credible_interval {credible_interval}"
    n_processes = args.n_processes
    command += f" --n_processes {n_processes}"

    roi = args.roi
    if roi is not None:
        command += f" --roi {' '.join(roi)}"

    if args.inpainted_mask:
        command += " --inpainted mask"

    transfer = args.transfer
    if transfer is not None:
        command += " --transfer {transfer}"

    input_path = args.input_path
    if input_path is not None:
        command += f" --input_path {input_path}"

    slurm = args.slurm
    if slurm is not None:
        with open(args.slurm, "r") as inpt:
            script = inpt.read()
        script = script.format(command=command)
        with NamedTemporaryFile() as script_file:
            script_file = "process_most_recent.sh"
            with open(script_file, "w") as output:
                output.write(script)
            slurm_command = f"sbatch {script_file}"
            subprocess.run(slurm_command.split(" "))
    else:
        subprocess.run(command.split(" "))



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

    input_cls = CPCIR

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
    start_time = datetime.now()
    end_time = start_time - timedelta(days=args.days)
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)

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
        database_path.is_dir() and database_path.exists()
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
            LOGGER.error(
                "The database path '%s' doesn't yet exist, so there are no "
                "failed files to process. ",
                database_path
            )
            return 1
        failed_files = [
            RemoteFile(input_cls, name, working_dir=Path(working_dir))
            for name in ProcessingLog.get_failed(database_path)
        ]
        input_files = [
            input_file for input_file in input_files if input_file in failed_files
        ]
        LOGGER.info(
            f"Found {len(input_files)} failed input files in logging database "
            f" {database_path}."
        )
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

    return not any_failed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "process_most_recent",
        description=(
            """
            Run CCIC retrieval for most recent CPCIR files..
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
        "output",
        metavar="output",
        type=str,
        help="Folder to which to write the output.",
    )
    parser.add_argument(
        "--days",
        metavar="n",
        type=int,
        help="How many days to go back in time.",
        default=7
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
    parser.add_argument(
        "--slurm",
        type=str,
        help=(
            "Filename of a bash script that will be used to submit the processing "
            " to a slurm cluster."
        ),
        default=None
    )
    run(parser.parse_args())



