"""
Prepare the netCDF files from https://doi.org/10.26023/72PW-V4FB-E0C
into a friendly netCDF format
"""

import argparse
import datetime
import os

import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('filepath',
    help='path to the netCDF file to read')
parser.add_argument('destination',
    help='path to save the friendly netCDF file')
args = parser.parse_args()

# Open dataset
ds = xr.open_dataset(args.filepath)

# Variables to keep
keep = set(
    [
        'latitude',
        'longitude',
        'altitude',
        'lniwc_error',
        'iwc_ret',
        'iwc_IWC_Z_T',
        'height_2D'
    ]
)

# Discard variables
ds = ds.drop(set(list(ds.keys())) - set(['latitude', 'longitude', 'altitude', 'lniwc_error', 'iwc_ret', 'iwc_IWC_Z_T', 'height_2D'])).drop_vars('range')

# Get UTC timestamps
timestamp_base = datetime.datetime.strptime(os.path.basename(args.filepath).split('_')[2], '%Y%m%d')
utc_time = np.array([timestamp_base + datetime.timedelta(hours=float(h)) for h in ds.time.values], dtype='datetime64')
ds_utc_time = xr.Dataset(data_vars={'utc_time': (['time'], utc_time)}, coords={'time': (['time'], ds.time.values)})

# Incorporate them into the dataset
ds = xr.merge((ds, ds_utc_time), combine_attrs="no_conflicts")

# Write to file
ds.to_netcdf(os.path.join(args.destination, 'friendly_{:}'.format(os.path.basename(args.filepath))))