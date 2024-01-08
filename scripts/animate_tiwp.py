"""
===============
animate_tiwp.py
===============

This command makes an animation of the CPCIR from the last N days.
"""
import argparse
from datetime import datetime, timedelta
import logging
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("CCIC")

logging.basicConfig(level="INFO")


def run(args):
    """
    Make animation of CCIC results.

    Args:
        args: The namespace object provided by the top-level parser.
    """
    from ccic.plotting import set_style, animate_tiwp
    now = datetime.now()

    result_path = Path(args.result_path)
    result_files = []
    for day in range(args.days, 0, -1):
        time = now - timedelta(days=day)
        year = time.year
        month = time.month
        day = time.day
        pattern = f"ccic_cpcir_{year:04}{month:02}{day:02}*.*"
        result_files += sorted(list(result_path.glob(pattern)))

    LOGGER.info(
        "Found %s files in the last %s days.",
        len(result_files),
        args.days
    )
    d_t = args.d_t
    if d_t is not None:
        d_t = np.timedelta64(d_t * 60, "s")

    output_path = Path(args.output_path)
    set_style()
    animate_tiwp(result_files, output_path / "tiwp.mp4", d_t)

    import xarray as xr
    with xr.open_zarr(result_files[-1]) as last_results:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import LogNorm
        from PIL import Image
        tiwp = last_results.tiwp[{
            "time": -1,
            "latitude": slice(0, None, 2),
            "longitude": slice(0, None, 2)
        }]
        norm = LogNorm(1e-2, 1e1)
        mappable = ScalarMappable(norm=norm)
        img = Image.fromarray(
            (mappable.to_rgba(np.maximum(tiwp.data, 1e-3) * 255)).astype(np.uint8)
        )
        img.save(output_path / "tiwp.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "animate_tiwp",
        description=(
            """
            Render animation of most recent TIWP.
            """
        ),
    )
    parser.add_argument(
        "result_path",
        type=str,
        help="Path to the folder containing the most recent files.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to which to write the resulting animation.",
    )
    parser.add_argument(
        "--days",
        metavar="n",
        type=int,
        help=(
            "The number of days prior to now from which results will be used in "
            "the animation."
        ),
        default=7
    )
    parser.add_argument(
        "--d_t",
        metavar="mins",
        type=int,
        help=(
            "The time step used for the animation in minutes."
        ),
        default=7
    )
    run(parser.parse_args())


