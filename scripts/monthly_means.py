"""
This script computes the monthly means from the existing CCIC data record.
"""

import argparse
import calendar
from pathlib import Path
import re

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
    files = [f for f in files if pattern.match(f.name)]
    return files

def process_month(files: list[Path]) -> xr.Dataset:
    """
    Compute the monthly means for the given month.
    """
    pass

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

    args = parser.parse_args()

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

        if (n_observed_files != n_expected_files) and (not args.ignore_missing_files):
            raise ValueError(f"Expected {n_expected_files} retrievals "
                             f"but found {n_observed_files} retrievals")
        
        ds = process_month(files)