"""
This script computes the monthly means from the existing CCIC data record.
"""

import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import gc
import logging
import sys
from pathlib import Path
import warnings

import ccic
from dateutil.relativedelta import relativedelta
import numpy as np
from tqdm import tqdm
import xarray as xr

def find_files(year: int, month: int, source: Path, product: str) -> list[Path]:
    """
    Find the files for year `year` and month `month` at a directory.

    Note: it is assumed the CCIC files are stored in the following directory
        structure: {source}/{product}/{year}/
    """
    path = source / product / str(year)
    files = path.glob(f'ccic_{product}_{year}{month:02d}*.*')
    return sorted(list(files))


def process_month(
        files: list[Path],
        product: str,
        status_bar: bool = True,
        precision: str = "single"
) -> xr.Dataset:
    """
    Compute the monthly means for the given month, stratified by the
    product temporal resolution (represented by the original variable
    name + `_stratified`) as well as monthly averages regardless of the
    timestamp (represented by the original variable name + `_aggregated`)
    """
    # Create a dataset to populate with means
    with warnings.catch_warnings():
        # If file is in zarr, xarray will yield these warnings until
        # using the zarr engine
        warnings.filterwarnings(
            "ignore",
            message="('netcdf4'|'scipy'|'h5netcdf') fails while guessing"
        )
        ds = xr.load_dataset(files[0])

    # Drop credibile interval variable
    if 'tiwp_ci' in ds:
        ds = ds.drop_vars('tiwp_ci')

    # Save the original attributes for later
    attrs = ds.attrs
    attrs_data = {}
    for name in list(ds.variables):
        attrs_data[name] = ds[name].attrs
    ds.attrs = {}

    # Set the time dimension
    if product == 'gridsat':
        time_deltas = [i * np.timedelta64(3, 'h') for i in range(8)]
    else:
        time_deltas = [i * np.timedelta64(30, 'm') for i in range(24 * 2)]

    # .astype('datetime64[M]').astype('datetime64[m]'):
    # hack to avoid dealing with days, i.e. date set to YYYYmmddT00:00
    time_offset = ds.time.values[0].astype('datetime64[M]').astype('datetime64[m]')
    time_values = np.array([time_offset + delta for delta in time_deltas])
    time_bounds = time_values[:-1] + 0.5 * (time_values[1:] - time_values[:-1])
    bnds_dtype = time_bounds.dtype

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=(
            "Converting non-nanosecond precision datetime "
            "values to nanosecond precision."
            )
        )
        # Floor day to first day of the month
        ds['time'] = ds['time'] - np.array(
            [np.timedelta64(d - 1, 'D') for d in ds.time.dt.day.values]
        )
        ds = ds.reindex({'time': time_values}, method=None, fill_value=0)

    # Initialize all variables to zero and create a count variable
    variables = set(ds.variables) - set(ds.coords)

    float_type = np.float32
    if precision == "double":
        float_type = np.float64

    for v in variables:
        # float instead of float32 to avoid any limitations accumulating
        # .copy to assign coordinates correctly
        ds[v] = ds[v].copy(
            data=np.zeros_like(ds[v]), deep=True
        ).astype(float_type)
        ds[f'{v}_count'] = ds[v].copy(
            data=np.zeros_like(ds[v]), deep=True
        ).astype(np.int16)

    # Accumulate values
    itr = files
    if status_bar:
        itr = tqdm(files, dynamic_ncols=True)

    for path in itr:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="('netcdf4'|'scipy'|'h5netcdf') fails while guessing"
            )
            ds_f = xr.load_dataset(path)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=("Converting non-nanosecond precision")
            )
            # Replace the timestamps day with first day of the month
            # and reindex to handle time dimension
            ds_f['time'] = ds_f['time'] - np.array(
                [np.timedelta64(d - 1, 'D') for d in ds_f.time.dt.day.values]
            )
        time_ind = np.digitize(
            ds_f.time.data.astype(bnds_dtype).astype(np.int64),
            time_bounds.astype(np.int64)
        )
        for v in variables:
            is_finite = np.isfinite(ds_f[v].data)
            ds[v][{"time": time_ind}] += np.where(is_finite, ds_f[v].data, 0)
            ds[f'{v}_count'][{"time": time_ind}] += is_finite.astype(int)

    # Divide by the total count
    for v in variables:
        non_zero_count = ds[f'{v}_count'].data > 0
        # .copy to set dimensions correctly
        ds[v] = ds[v].copy(
            data=np.divide(ds[v].data, ds[f'{v}_count'].data,
                           out=np.full_like(ds[v].data, np.nan),
                           where=non_zero_count),
            deep=True
        ).astype(np.float32)
        ds[f'{v}_count'] = ds[f'{v}_count'].astype(np.int16)

    # Update attributes
    ds.attrs = attrs
    ds.attrs["history"] = f"{datetime.now()}: Monthly means computation"
    ds.attrs["input_filename"] = [f.name for f in files]
    for v in variables:
        ds[v].attrs = attrs_data[v]
        ds[f'{v}_count'].attrs['units'] = '1'
        ds[f'{v}_count'].attrs['long_name'] = (
            "Count of '{:}' values used "
            "for the stratified monthly mean"
            ).format(ds[v].attrs['long_name'])
        ds[v].attrs['long_name'] = '{:}, stratified monthly mean'.format(
            ds[v].attrs['long_name']
        )
    for coord in ds.coords:
        ds[coord].attrs = attrs_data[coord]

    # Append `_stratified` the stratified variables
    ds = ds.rename({v: f'{v}_stratified' for v in variables})
    ds = ds.rename({f'{v}_count': f'{v}_stratified_count' for v in variables})

    # Compute monthly means irrespective of timestamp
    # Setting dtype='datetime64[M]' seems to not have effect
    ds['month'] = (('month',), [ds.time.data.min()])
    for v in variables:
        ds[v] = (
            np.divide(
                (ds[f'{v}_stratified'] * ds[f'{v}_stratified_count']).sum('time', skipna=True),
                ds[f'{v}_stratified_count'].sum('time', skipna=True),
                out=np.full_like(ds[f'{v}_stratified'].sum('time'), np.nan),
                where=(ds[f'{v}_stratified_count'].sum('time', skipna=True).data > 0)
            )
        ).expand_dims({'month': 1})

    # Change the name and data type of variable `time`
    ds = ds.rename({'time': 'hour_of_day'})
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=("Converting non-nanosecond precision")
        )
        ds['hour_of_day'] = (
            ds.hour_of_day - ds.hour_of_day.astype('datetime64[D]').astype(ds.hour_of_day.dtype)
        )

    # Set attributes for the full monthly mean data
    attrs_data['month'] = {'long_name': 'month', 'standard_name': 'month'}
    for v in variables:
        ds[v].attrs = attrs_data[v]
        ds[v].attrs['long_name'] = '{:}, monthly mean'.format(attrs_data[v]['long_name'])

    return ds


def calculate_mean(current_month: datetime, status_bar: bool,
                   precision: str, compress: bool=True) -> None:
    """
    Helper function encapsulating all processing for calculating means for
    a given month.

    Args:
        current_month: A numpy datetime.datetime object representing the month
            for which to calculate the averages.
        status_bar: Whether or not to display the progress for each month using
            a progress bar.
        precision: precision to use (single or double) for accumulating values
        compress: apply zlib maximum compression to the saved netCDF
    """
    year = current_month.year
    month = current_month.month
    logging.info(f"Finding {args.product} files for {year}-{month:02d}")
    files = find_files(year, month,
                        args.source, args.product)

    # Check that there is the expected number of files
    _, n_days = calendar.monthrange(year, month)
    n_expected_files = n_days * (8 if args.product == 'gridsat' else 24)
    n_observed_files = len(files)

    # This can fail for the first years of the data products
    # A quick fix is to manually check the files used and use --ignore_missing_files
    if (n_observed_files != n_expected_files):
        if args.ignore_missing_files:
            logging.warning(f"Using {n_observed_files}/{n_expected_files}"
                            f" retrievals to compute the means")
        else:
            raise ValueError(f"Expected {n_expected_files} retrievals "
                                f"but found {n_observed_files} retrievals")
    if len(files) == 0:
        logging.warning(f"No files to process for {year}-{month:02d}")
        return

    logging.info(f"Processing {year}-{month:02d}")
    ds = process_month(files, args.product, status_bar=status_bar, precision=precision)

    fname_dst = f'ccic_{args.product}_{year}{month:02d}_monthlymean.nc'
    f_dst = args.destination / fname_dst
    logging.info(f'Writing {f_dst}')
    ds.to_netcdf(
        f_dst,
        encoding={var: {'zlib': True, 'complevel': 9} for var in ds} if compress else None
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--product',
        choices=['gridsat', 'cpcir'],
        required=True,
        help="product to process"
    )
    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help="directory of the CCIC data record"
    )
    parser.add_argument(
        '--destination',
        type=Path,
        required=True,
        help="directory to save the monthly means"
    )
    parser.add_argument(
        '--month',
        required=True,
        help="month to process in the format YYYYmm"
    )
    parser.add_argument(
        '--month_end',
        nargs='?',
        default=None,
        help="process until this month in the format YYYYmm"
    )
    parser.add_argument(
        '--ignore_missing_files',
        action='store_true',
        help="ignore missing expected retrievals"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='verbose mode'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=1,
        help="The number of processes to use for parallel processing. If '1', files will be processes sequentially."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="double",
        help="Whether to use double or single precision for accumulating data."
    )
    parser.add_argument(
        '--uncompressed',
        action='store_true',
        help="Do not apply zlib compression to the final netCDF"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if (args.month_end is None) or args.month_end < args.month:
        args.month_end = args.month

    current_month = datetime.strptime(args.month, '%Y%m')
    month_end = datetime.strptime(args.month_end, '%Y%m')
    n_processes = args.n_processes

    precision = args.precision
    if not precision in ["single", "double"]:
        logging.error(
            "Precision should be one of ['single', 'double'] got '%s'.",
            precision
        )

    if n_processes == 1:
        while current_month <= month_end:
            calculate_mean(current_month, True, precision, not args.uncompressed)
            current_month += relativedelta(months=1)
        sys.exit(0)

    pool = ProcessPoolExecutor(max_workers=n_processes)
    tasks = []
    months = []
    while current_month <= month_end:
        tasks.append(pool.submit(calculate_mean, current_month, False, precision, not args.uncompressed))
        months.append(current_month)
        current_month += relativedelta(months=1)

    for month, task in zip(months, tasks):
        try:
            task.result()
            del task
        except Exception:
            logging.exception("The following error was encountered when processing %s.", month)
    sys.exit(0)

