"""
==============================
ccic.bin.extract_training_data
==============================

This sub-module implements the CLI to extract the CCIC training
data.
"""
from calendar import monthrange
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from pansat.time import to_datetime64

from ccic.data.cloudsat import get_available_granules
from ccic.data import DownloadCache, process_cloudsat_files, write_scenes


LOGGER = logging.getLogger(__name__)


def add_parser(subparsers):
    """
    Add parser for 'extract_training_data' command to top-level parser. This
    function is called from the top-level parser defined in 'ccic.bin'.

    Args:
        subparsers: The subparsers object provided by the top-level parser.
    """
    parser = subparsers.add_parser(
        "extract_training_data",
        help="Extract training data for the CCIC retrieval.",
        description=(
            """
            Extract training data for the CCIC retrieval.
            """
        ),
    )
    parser.add_argument(
        "year",
        metavar="year",
        type=int,
        help="The year for which to extract training data.",
    )
    parser.add_argument(
        "month",
        metavar="month",
        type=int,
        help="The month for which to extract training data.",
    )
    parser.add_argument(
        "days",
        metavar="days",
        type=int,
        nargs="*",
        help="The days of the month for which to extract training data.",
    )
    parser.add_argument(
        "destination",
        metavar="destination",
        type=str,
        help="Where to write the extracted training scenes to.",
    )
    parser.add_argument(
        "--scene_size",
        metavar="n",
        type=int,
        help="The size of the extracted training scenes.",
        default=256,
    )
    parser.add_argument(
        "--max_time_difference",
        metavar="mins",
        type=int,
        help="The time difference in minutes to allow for collocations.",
        default=15,
    )
    parser.add_argument(
        "--min_valid_input",
        metavar="r",
        type=int,
        help=(
            "Minimum fraction of valid input observation for a scene to"
            " be retained in the training data."
        ),
        default=0.2,
    )
    parser.add_argument(
        "--n_workers",
        metavar="n",
        type=int,
        help="The number of concurrent processes to use for data extraction.",
        default=4,
    )
    parser.set_defaults(func=run)


def process_day(year, month, day, destination, size=256, timedelta=15, valid_input=0.2):
    """
    Extract collocations for a day.

    Args:
        year: The year.
        month: The month.
        day: The day.
        destination: Path to write the extracted scenes to.
        size: The size of the training scenes.
        timedelta: The allowed time difference between geostationary
             observations and CloudSat.
        valid_input: A minimum fraction of valid inputs for a scene to be
            included in the training data.
    """
    date = to_datetime64(datetime(year, month, day))
    granules = get_available_granules(date)
    LOGGER.info(
        "Found %s granules for %s-%s-%s.",
        len(granules),
        year,
        f"{month:02}",
        f"{day:02}",
    )

    for granule, cloudsat_files in granules.items():
        if not len(cloudsat_files) == 2:
            LOGGER.info(
                "Skipping granule %s because less than two CloudSat product"
                " files are available."
            )
            continue
        try:
            cache = DownloadCache(n_threads=4)
            scenes = process_cloudsat_files(
                cloudsat_files, cache, size=size, timedelta=timedelta
            )
            write_scenes(scenes, destination, valid_input=valid_input)
            LOGGER.info(
                "Extracted %s match-ups from CloudSat granule %s.",
                len(scenes),
                granule
            )
        except Exception as ex:
            LOGGER.exception(
                "The following error was encountered while processing CloudSat"
                " granule '%s' on %s-%s-%s.",
                granule,
                year,
                f"{month:02}",
                f"{day:02}",
            )


def run(args):
    """
    Extract training data.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    year = args.year
    month = args.month
    days = args.days
    if len(days) == 0:
        _, n_days = monthrange(year, month)
        days = range(1, n_days + 1)

    destination = Path(args.destination)
    if not destination.exists():
        LOGGER.error("The 'destination' argmument must be an existing directory.")
        return 1

    size = args.scene_size
    timedelta = args.max_time_difference
    valid_input = args.min_valid_input

    pool = ProcessPoolExecutor(max_workers=args.n_workers)
    tasks = [
        pool.submit(
            process_day,
            year,
            month,
            day,
            destination,
            size=size,
            timedelta=timedelta,
            valid_input=valid_input,
        )
        for day in days
    ]

    for day, task in zip(days, tasks):
        task.result()
        LOGGER.info("Finished processing day %s of %s-%s.", day, year, month)

    return 0
