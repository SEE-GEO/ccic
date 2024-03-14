"""
Script to upload the data to an S3 bucket.

It requires the AWS credentials in ~/.aws/credentials

The approach implemented here is not the fastest
but the most robust at the time of writing.

The use of xarray.Dataset.to_zarr(s3_store), where
is slower than, e.g., using
```python
s3 = s3fs.S3FileSystem()
s3.put('source_dir/', 'bucket/', recursive=True)
```
or
```shell
$ aws s3 cp source_dir s3://bucket/ --recursive
```

The xarray approach takes 1.5 min per file, the 'non-xarray'
approach, where the CCIC file is existing in disk takes 0.5 min.

The reason why is the slower approach is taken is two-fold:

- With the xarray approach, the CCIC file has to first be read.
  This ensures that the file to be uploaded is not corrupted.

- With the xarray approach, it is easy to append variables
  to an exisiting CCIC file in the S3 bucket or upload only
  part of the file.

  Technically it is possible to append with the 'non-xarray'
  approach, but here it is chosen the simplest solution.
"""

import argparse
from datetime import datetime
import glob
from pathlib import Path

from ccic.processing import get_encodings_zarr
import fsspec
import tqdm
import xarray as xr

def datetime_from_filename(fname: str) -> datetime:
    """Get a datetime object corresponding to the filename"""
    fname = Path(fname).stem
    date = fname.split('_')[-1][:8]
    return datetime.strptime(date, '%Y%m%d')

def uploader(file: str, args: argparse) -> None:
    """
    Upload the data in `file` to the bucket given
    by the arguments `args`
    """
    year = datetime_from_filename(file).year
    target = fsspec.get_mapper(
        f's3://{args.bucket}/record/{args.product}/{year}/{Path(file).name}',
        default_block_size=args.block_size * 2**20,
    )
    ds = xr.open_zarr(file)[args.variables]
    encoding = get_encodings_zarr(ds)
    if not args.dry_run:
        ds.to_zarr(target, mode="a", encoding=encoding)


parser = argparse.ArgumentParser(
    description=(
        "Upload CCIC data to an S3 bucket as "
        "bucketname/record/{product}/{year}/."
    )
)

parser.add_argument(
    '--block_size',
    default=1024,
    type=int,
    help="Block size in MB to use with the s3fs mapper."
)
parser.add_argument(
    '--bucket',
    default='chalmerscloudiceclimatology',
    help='Name of the S3 bucket.'
)
parser.add_argument(
    '--date_start',
    type=lambda x: datetime.strptime(x, '%Y%m%d'),
    required=True,
    help="Upload from this date, in YYYYmmdd."
)
parser.add_argument(
    '--date_end',
    type=lambda x:datetime.strptime(x, '%Y%m%d'),
    help="Upload through this date. defaults to --date_start."
)
parser.add_argument(
    '--dry-run',
    action='store_true',
    help="Execute a dry run of the script."
)
parser.add_argument(
    '--n_uploads',
    default=1,
    type=int,
    help='Number of concurrent uploads.'
)
parser.add_argument(
    '--product',
    choices=['cpcir', 'gridsat'],
    required=True,
    help="Product to upload."
)
parser.add_argument(
    '--source',
    type=Path,
    required=True,
    help="Path to the local CCIC record"
)
parser.add_argument(
    '--variables',
    nargs='+',
    default=['inpainted', 'p_tiwp', 'tiwp', 'tiwp_ci'],
)

args = parser.parse_args()

if args.date_end is None:
    args.date_end = args.date_start

# Find files within the date range
files = sorted(
    [
        f
        for f in glob.glob(str(args.source / args.product / f'**/ccic_{args.product}_*.zarr'))
        if (args.date_start <= datetime_from_filename(f)) & (datetime_from_filename(f) <= args.date_end)
    ]
)

# TODO: Call each file with uploader using concurrency and a progressbar