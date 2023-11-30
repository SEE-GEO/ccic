"""
This script computes the monthly means from the existing CCIC data record.
"""

import argparse
import calendar
from datetime import datetime
import logging
from pathlib import Path
import re
import warnings

import ccic
import numpy as np
from tqdm import tqdm
import xarray as xr

def get_year_month(month_i: int, year_offset: int,
                   month_offset: int) -> tuple[int, int]:
    """Auxiliary function to that returns a (year, month) from an offset."""
    year_plus, month_plus = divmod((month_offset - 1) + month_i, 12)

    return (year_offset + year_plus, month_offset + month_plus)

def find_files(year: int, month: int, source: Path, product: str) -> list[Path]:
    """
    Find the files for year `year` and month `month`
    at directory `source` for product `product`.
    """
    files = source.rglob(f'ccic_{product}_{year}{month:02d}*.zarr')
    pattern = re.compile(r'ccic_gridsat_201001[0-3]{1}[0-9]{1}[0-2]{1}[0-9]{1}00.zarr')
    files = sorted([f for f in files if pattern.match(f.name)])
    return files

def process_month(files: list[Path], product: str) -> xr.Dataset:
    """
    Compute the monthly means for the given month.
    """
    # Create a dataset to populate with means
    ds = xr.open_zarr(files[0]).load()

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
    time_values = [time_offset + delta for delta in time_deltas]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=(
            "Converting non-nanosecond precision datetime "
            "values to nanosecond precision."
            )
        )
        ds['time'] = ds['time'].astype('datetime64[M]').astype('datetime64[m]')
        ds = ds.reindex({'time': time_values}, method=None, fill_value=0)

    # Initialize all variables to zero and create a count variable
    variables = set(ds.variables) - set(ds.coords)
    for v in variables:
        # float instead of float32 to avoid any limitations accumulating
        ds[v] = ds[v].copy(
            data=np.zeros_like(ds[v]), deep=True
        ).astype(float)
        ds[f'{v}_count'] = ds[v].copy(
            data=np.zeros_like(ds[v]), deep=True
        ).astype(int)

    # Accumulate values
    for f in tqdm(files, dynamic_ncols=True):
        ds_f = xr.open_zarr(f).load()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=(
                "Converting non-nanosecond precision"
                )
            )
            # Reindex to handle time dimension
            time_delta_from_month_start = ds_f['time'] - ds_f['time'].astype('datetime64[M]')
            days_from_month_start = time_delta_from_month_start.astype('timedelta64[D]')
            time_f = ds_f['time'] - days_from_month_start
            ds_f['time'] = time_f.astype('datetime64[m]')
            ds_f = ds_f.reindex({'time': time_values}, method=None, fill_value=np.nan)
        
        for v in variables:
            is_finite = np.isfinite(ds_f[v].data)
            ds[v] = ds[v] + ds[v].copy(
                data=np.where(is_finite, ds_f[v].data, 0), deep=True
            )
            ds[f'{v}_count'] = ds[f'{v}_count'] + ds[f'{v}_count'].copy(
                data=is_finite.astype(int), deep=True
            )

    # Divide by the total count
    for v in variables:
        non_zero_count = ds[f'{v}_count'].data > 0
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
        ds[f'{v}_count'].attrs['long_name'] = "Count of '{:}' values used \
            for the monthly mean".format(ds[v].attrs['long_name'])
        ds[f'{v}_count'].attrs['units'] = '1'
        ds[v].attrs['long_name'] = '{:}, monthly mean'.format(
            ds[v].attrs['long_name']
        )
    for coord in ds.coords:
        ds[coord].attrs = attrs_data[coord]

    return ds


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
        type=int,
        help="month to process in the format YYYYMM"
    )
    parser.add_argument(
        '--month_end',
        nargs='?',
        default=None,
        type=int,
        help="process until this month in the format YYYYMM"
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

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if (args.month_end is None) or args.month_end < args.month:
        args.month_end = args.month
    
    n_months_to_process = sum(divmod(args.month_end - args.month, 12)) + 1

    year_offset, month_offset = divmod(args.month, 100)

    for month_i in range(n_months_to_process):
        year, month = get_year_month(month_i, year_offset, month_offset)
        files = find_files(year, month, args.source, args.product)

        # Check that there is the expected number of files
        _, n_days = calendar.monthrange(year, month)
        n_expected_files = n_days * (8 if args.product == 'gridsat' else 24)
        n_observed_files = len(files)

        if (n_observed_files != n_expected_files):
            if args.ignore_missing_files:
                logging.warning(f"Using {n_observed_files}/{n_expected_files}"
                                f" retrievals to compute the means")
            else:
                raise ValueError(f"Expected {n_expected_files} retrievals "
                                 f"but found {n_observed_files} retrievals")

        logging.info(f"Processing {year}-{month:02d}")
        ds = process_month(files, args.product)

        fname_dst = f'ccic_{args.product}_{year}{month:02d}_monthlymean.nc'
        f_dst = args.destination / fname_dst
        logging.info(f'Writing {f_dst}')
        ds.to_netcdf(f_dst)