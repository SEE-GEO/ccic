"""
This script computes the monthly means from the existing CCIC data record.

But we coarsen the grid and bin the data to keep PDFs.
"""

import argparse
import calendar
from contextlib import nullcontext
import datetime
from pathlib import Path

import ccic
import dask
from dask.distributed import Client
from flox.xarray import xarray_reduce
import numpy as np
import pandas as pd
from upath import UPath
import xarray as xr

from scripts.monthly_means import find_files

DATASET_LEVEL_ATTRS = {
    "description": (
        "Monthly aggregation of TIWP and cosine-latitude-weighted TIWP, binned by TIWP, latitude, and longitude. "
        "From these statistics, one can compute the (weighted) mean TIWP for each bin "
        "or aggregate over a set of bins. "
        "Concerning the bins, xarray does not support serializing pandas.IntervalIndex objects, "
        "so the left and right edges of the bins are stored as separate variables. "
        "The bin edges are left-inclusive and right-exclusive. "
        "Coordinates representing bins can be transformed to pandas.IntervalIndex objects using "
        "`ds.pipe(to_interval_index)`, where `to_interval_index` is, e.g., the function in "
        "ds.attrs['func=to_interval_index']."
    ),
    "func=to_interval_index": """
def to_interval_index(ds):
    for var in ["tiwp_bins", "latitude_bins", "longitude_bins"]:
        interval_index = pd.IntervalIndex.from_arrays(
            ds[f"{var}_left"].data,
            ds[f"{var}_right"].data,
            closed='left'
        )
        ds[var] = interval_index
        ds = ds.drop_vars([f"{var}_left", f"{var}_right"])
        ds = ds.set_coords(var)
    return ds
"""
}

VAR_LEVEL_ATTRS = {
    "tiwp_nansum": {
        "description": "Sum of non-NaN TIWP values in each bin. To compute the mean TIWP in each bin, divide this variable by `tiwp_count`.",
        "units": "kg m-2"
    },
    "tiwp_count": {
        "description": "Count of non-NaN TIWP values in each bin. To compute the mean TIWP in each bin, divide `tiwp_nansum` by this variable.",
        "units": "1"
    },
    "weighted_tiwp_nansum": {
        "description": "Sum of non-NaN cosine-latitude-weighted TIWP values in each bin. To compute the mean cosine-latitude-weighted TIWP in each bin, divide this variable by `weighted_tiwp_count`.",
        "units": "kg m-2"
    },
    "weighted_tiwp_count": {
        "description": "Sum of the cosine-latitude-weights for each non-NaN TIWP in each bin. To compute the mean cosine-latitude-weighted TIWP in each bin, divide `weighted_tiwp_nansum` by this variable.",
        "units": "1"
    }
}

def process_month(args: argparse.Namespace, year_month: datetime.datetime):
    print(f"Processing data for {year_month.strftime('%Y-%m')} with product {args.product}...")
    files = []
    for source in args.source:
        files.extend(find_files(year_month.year, year_month.month, source, args.product))

    # Remove files that are the same, but with different source paths
    # Sort S3Path objects to the end of the list so that any file is overwritten by S3 files if they exist
    files.sort(key=lambda p: 1 if 'S3Path' in str(type(p)) else 0)
    files = list({Path(f).name: f for f in files}.values())

    # Check that there is the expected number of files
    _, n_days = calendar.monthrange(year_month.year, year_month.month)
    n_expected_files = n_days * (8 if args.product == "gridsat" else 24)
    n_observed_files = len(files)
    if n_observed_files != n_expected_files:
        if not args.ignore_missing_files:
            raise ValueError(
                f"Expected {n_expected_files} files for {year_month.strftime('%Y-%m')}, "
                f"but found {n_observed_files}. Use --ignore_missing_files to ignore this error."
            )

    ds = xr.open_mfdataset(
        # Get mappers, as they are remote files
        [f.fs.get_mapper(str(f)) for f in files],
        combine='nested',
        concat_dim='time',
        parallel=True,
        engine='zarr'
    )

    # Rechunk to avoid too many small chunks
    ds = ds.chunk({'time': -1})

    # Drop variables that are not considered for now
    for var in ['p_tiwp', 'tiwp_ci', 'cloud_prob_2d', 'ci_bounds', 'inpainted']:
        if var in ds:
            ds = ds.drop_vars(var)

    # Clear attrs
    ds = ds.drop_attrs()

    # There are some problems with decimal hours in CPCIR
    # Clean up for stratified means
    ds['time'] = ds.time.dt.round('30min')

    # 'Floor' everything to the first day of the month for the monthly means
    ds['time'] = ds['time'] - (ds.time.dt.day - 1) * np.timedelta64(1, 'D')

    groupers = {
        'tiwp': {
            'values': pd.IntervalIndex.from_breaks(
                args.tiwp_bin_edges,
                closed='left'
            ),
            'isbin': True
        },
        'time': {
            'values': np.unique(ds.time.values),
            'isbin': False
        },
        'latitude': {
            'values': pd.IntervalIndex.from_breaks(
                np.arange(-70., 71., args.grid_resolution),
                closed='left'
            ),
            'isbin': True
        },
        'longitude': {
            'values': pd.IntervalIndex.from_breaks(
                np.arange(-180., 181., args.grid_resolution),
                closed='left'
            ),
            'isbin': True
        }
    }


    weights = np.cos(np.deg2rad(ds.latitude))
    weights = weights.broadcast_like(ds.tiwp).where(ds.tiwp.notnull())
    valid_weights = weights.where(ds.tiwp.notnull())

    with dask.config.set({"array.slicing.split_large_chunks": False}):
        # Using flox here as we're also binning by TIWP, which is a variable
        # Otherwise, we could use xarray's native groupers
        results = {
            key: xarray_reduce(
                values,
                *[ds[k] for k in groupers.keys()],
                func=key.split('_')[-1],
                expected_groups=tuple(v['values'] for v in groupers.values()),
                isbin=tuple(v['isbin'] for v in groupers.values()),
                # dim='time',
                engine='flox',
                method='map-reduce',
            ).rename(key)
            for key, values in zip(
                ['tiwp_nansum', 'tiwp_count', 'weighted_tiwp_nansum', 'weighted_tiwp_count'],
                [ds.tiwp, ds.tiwp, ds.tiwp * weights, valid_weights]
            )
        }

    # Merge results
    ds_combined = xr.merge(list(results.values()))

    # Consistency in data types
    ds_combined['tiwp_count'] = ds_combined['tiwp_count'].astype('int32')
    ds_combined['weighted_tiwp_count'] = ds_combined['weighted_tiwp_count'].astype('int32')
    
    # The bins are `object` dtypes, as they are numpy arrays of Interval objects.
    # Convert to pandas IntervalIndex for consistency
    for var in ['tiwp_bins', 'latitude_bins', 'longitude_bins']:
        ds_combined[var] = pd.IntervalIndex(ds_combined[var].data)
        ds_combined[f"{var}_left"] = (var, ds_combined.indexes[var].left)
        ds_combined[f'{var}_right'] = (var, ds_combined.indexes[var].right)
        ds_combined = ds_combined.set_coords([f"{var}_left", f"{var}_right"])
        ds_combined = ds_combined.drop_vars(var)

    # Add some info on what each variable represents, including the bins are defined
    ds_combined.attrs.update(DATASET_LEVEL_ATTRS)
    for var_name, attributes in VAR_LEVEL_ATTRS.items():
        ds_combined[var_name].attrs.update(attributes)

    # Save to netCDF
    # But since we may run this in a distributed environment and there can
    # be trouble with writing to the same file, we first load the results
    # into memory and then write to disk
    ds_combined = ds_combined.compute()
    ds_combined.to_netcdf(
        args.destination / f"ccic_{args.product}_{year_month.strftime('%Y%m')}_monthlymean_resolved.nc",
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        choices=["gridsat", "cpcir"],
        required=True,
        help="product to process",
    )
    parser.add_argument(
        "--source",
        nargs='+',
        required=True,
        help="directory of the CCIC data record",
    )
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="directory to save the monthly means",
    )
    parser.add_argument(
        "--month", required=True, help="month to process in the format YYYYmm"
    )
    parser.add_argument(
        "--month_end",
        nargs="?",
        default=None,
        help="process until this month in the format YYYYmm",
    )
    parser.add_argument(
        "--ignore_missing_files",
        action="store_true",
        help="ignore missing expected retrievals",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose mode"
    )
    parser.add_argument(
        '--grid_resolution',
        type=float,
        default=2.0,
    )
    parser.add_argument(
        '--tiwp_bin_edges',
        type=float,
        nargs='+',
        default=np.concatenate(
            [
                [0],
                np.geomspace(1e-3, 10, 12), # Same as np.logspace(-3, 1, 12)
                [np.inf]
            ]
        )
    )
    parser.add_argument(
        '--scheduler',
        help=(
            "Dask cluster (scheduler) to use. "
            "If not specified, will use the default scheduler."
        )
    )

    args = parser.parse_args()

    # Handle sources
    args.source = [UPath(source, anon=True) if 's3://' in source else UPath(source) for source in args.source]

    if (args.month_end is None) or args.month_end < args.month:
        args.month_end = args.month

    current_month = datetime.datetime.strptime(args.month, "%Y%m")
    month_end = datetime.datetime.strptime(args.month_end, "%Y%m")

    ctx = Client(args.scheduler) if args.scheduler else nullcontext()

    with ctx:
        for year_month in pd.date_range(current_month, month_end, freq="MS"):
            print(f"Processing {year_month.strftime('%Y-%m')}...")
            process_month(
                args,
                year_month
            )