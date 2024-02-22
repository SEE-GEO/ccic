"""
This script is intended to compress input files
using the zlib option with maximum compression
when saving an xarray to a netCDF.

Using this compression not only saves a great amount of
disk space (can be about 25% per file) but also decreases
the time required to read the input file (by about 10%).

The script is somewhat complicated, but it is designed
to match the characteristics of the computer system on
which it is meant to be executed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import logging
import multiprocessing
from pathlib import Path
import tempfile

# `h5netcdf`is the engine used to read files remotely,
# by importing it we assert that it is installed, even
# if it is not explicitly used
import h5netcdf
from paramiko.config import SSHConfig
from sshfs import SSHFileSystem
import tqdm
import xarray as xr


def get_date_from_fname(fpath: Path, product: str) -> datetime.datetime:
    stem = fpath.stem
    return datetime.datetime.strptime(
        stem,
        "merg_%Y%m%d%H_4km-pixel"
        if product == "cpcir"
        else "GRIDSAT-B1.%Y.%m.%d.%H.v02r01.nc"
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--start", help="from which date process in the format YYYYmmdd"
)
parser.add_argument(
    "--end", help="until which date process in the format YYYYmmdd"
)
parser.add_argument(
    "--source",
    required=True,
    type=Path,
    help="path to the directory containing the source netCDF files",
)
parser.add_argument(
    "--destination",
    required=True,
    type=Path,
    help="path to the directory to place the compressed netCDF files",
)
parser.add_argument(
    "--remove_source_files",
    action="store_true",
    help=(
        "remove the source netCDF files after they are processed "
        "WARNING: this is destructive"
    ),
)
parser.add_argument(
    "--n",
    default=8,
    type=int,
    help=("number of parallel processes " "(files being compressed)"),
)
parser.add_argument(
    "--host",
    type=str,
    help=(
        "if provided, --source and --destination "
        "will refer to paths in this host as "
        "configured in ~/.ssh/config (requires "
        "passwordless ssh credential configured)"
    ),
)
parser.add_argument(
    "--product",
    default="cpcir",
    choices=["cpcir", "gridsat"],
    help="input product used",
)
parser.add_argument(
    "--tmpdir",
    type=Path,
    help=(
        "if provided, use this temporary directory "
        "to compress the files prior to moving the "
        "files to the final directory"
    )
)

args = parser.parse_args()

start = datetime.datetime.strptime(
    args.start if args.start is not None else "19700101", "%Y%m%d"
)

end = datetime.datetime.strptime(
    args.end if args.end is not None else "29700101", "%Y%m%d"
)

if args.host is not None:
    sshconfig = SSHConfig.from_path(Path("~/.ssh/config").expanduser()).lookup(
        args.host
    )
    fs_ssh = SSHFileSystem(
        sshconfig["hostname"],
        username=sshconfig["user"],
        client_keys=sshconfig["identityfile"],
    )

extension = "nc4" if args.product == "cpcir" else "nc"
if args.host:
    files = [
        Path(p) for p in fs_ssh.glob(str(args.source / f"**/*{extension}"))
    ]
else:
    files = args.source.rglob(f"*{extension}")

# Filter files outside [start, end)
files = sorted(
    [f for f in files if (start <= get_date_from_fname(f, args.product) < end)]
)


def wrapper(f: Path) -> None:
    try:
        ds = xr.load_dataset(
            fs_ssh.open(str(f)) if args.host else f, engine="h5netcdf"
        )
        dst_path = args.destination / f.name
        with tempfile.TemporaryDirectory(dir=args.tmpdir) as tmpdir:
            # Saving to netCDF with compression is what takes more time
            # Use a local directory (sshfs) or a (fast) temporary
            # directory (can be) specified with --tmpdir
            tmp_fpath = Path(tmpdir) / f.name
            ds.to_netcdf(
                tmp_fpath,
                encoding={
                    var: {"zlib": True, "complevel": 9} for var in ds
                },
            )
            if args.host:
                fs_ssh.put_file(str(tmp_fpath), str(dst_path))
            else:
                tmp_fpath.rename(dst_path)
    except Exception as e:
        # Something should be wrong with the netCDF
        logging.warning(f"File {f.name} -- {str(e)}")
    finally:
        if args.remove_source_files:
            f.unlink()


if __name__ == "__main__":
    with ProcessPoolExecutor(
        max_workers=args.n, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        list(
            tqdm.tqdm(
                executor.map(wrapper, files),
                total=len(files),
                dynamic_ncols=True,
            )
        )
