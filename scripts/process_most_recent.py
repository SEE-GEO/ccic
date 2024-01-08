"""
===================
process_most_recent
===================

This command processes the most recent CPCIR data that is available.
"""
import argparse
from datetime import datetime, timedelta
import subprocess
from tempfile import NamedTemporaryFile


def run(args):
    """
    Process most recent CPCIR files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)
    start_time = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_time = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    output = args.output
    input_type = "cpcir"
    model = args.model

    command = f"ccic process {model} {input_type} {output} {start_time} {end_time}"

    input_path = args.input_path
    if input_path is not None:
        command += f" --input_path {input_path}"

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
        command += " --inpainted_mask"

    transfer = args.transfer
    if transfer is not None:
        command += f" --transfer {transfer}"

    input_path = args.input_path
    if input_path is not None:
        command += f" --input_path {input_path}"

    command += " --failed"
    database_path = args.database_path
    command += f" --database_path {database_path}"

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "process_most_recent",
        description=(
            """
            Run CCIC retrieval for most recent CPCIR files.
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
        default=7,
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
        "--database_path",
        type=str,
        default="processing_cpcir.db",
        help=(
            "Name of the processing database to use."
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
        default=None,
    )
    run(parser.parse_args())
