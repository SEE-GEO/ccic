"""
================
ccic.bin.process
================

This sub-module implements the CLI to run the CCIC retrievals.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'train' command to top-level parser. This function
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
        "input",
        metavar="input",
        type=str,
        help="Input file or folder containing multiple input files.",
    )
    parser.add_argument(
        "output",
        metavar="ouput",
        type=str,
        help="Folder to which to write the output.",
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
    parser.set_defaults(func=run)


def process_file(model, input_file, output_file, tile_size, overlap):
    from ccic.data.gpm_ir import GPMIR
    from ccic.processing import process_input

    input_file = GPMIR(input_file)
    x = input_file.get_retrieval_input()
    results = process_input(model, x, tile_size=tile_size, overlap=overlap)
    results.to_netcdf(output_file)


def run(args):
    """
    Process input files.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from quantnn.qrnn import QRNN

    input = Path(args.input)
    if not input.exists():
        LOGGER.error("The provided input path '%s' does not exist.")
        return 1
    if input.is_dir():
        input_files = list(input.glob("*.nc4"))
    else:
        input_files = [input]

    output = Path(args.output)
    if len(input_files) > 1:
        if not output.exists():
            output.mkdir(exist_ok=True, parents=True)
        output_files = [output / f_in.name for f_in in input_files]
    else:
        output.parent.mkdir(exist_ok=True, parents=True)
        output_files = [output]

    model = Path(args.model)
    if not model.exists():
        LOGGER.error("The provides model '%s' does not exist.", model.name)
    qrnn = QRNN.load(model)
    qrnn.model.train(False)

    for input_file, output_file in zip(input_files, output_files):
        process_file(
            qrnn,
            input_file,
            output_file,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
