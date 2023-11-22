"""
Parses the source CSV files from https://doi.org/10.26023/8V5Y-GB2E-CX07
into a friendly CSV format for analysing TWC

It requires the files from https://doi.org/10.26023/KJDH-MXGE-HK0V
"""

import argparse
import datetime
import io
import os
import tarfile

import numpy as np
import pandas as pd
import xmltodict

parser = argparse.ArgumentParser()
parser.add_argument('filepath',
    help='csv file to read')
parser.add_argument('navigationfolder',
    help='folder containing the tarballs with navigation data')
parser.add_argument('destination',
    help='where to place the friendly CSV')

args = parser.parse_args()

date = datetime.datetime.strptime(os.path.basename(args.filepath)[:6], '%y%m%d')
tarfilepath = os.path.join(args.navigationfolder, '{:}-MetNav.tar'.format(date.strftime('%Y%m%d')))

# Read the navigation data into a dataframe, extracting data on-the-fly
tar = tarfile.open(tarfilepath)
navigation_data = dict()
for member in [n for n in tar.getnames() if 'IWG1-10hz' in n]:
    if 'xml' in member:
        navigation_data['xml'] = tar.extractfile(member).read()
    else:
        navigation_data['csv'] = tar.extractfile(member).read()
tar.close()

# Get the column names
params = xmltodict.parse(navigation_data['xml'].decode('utf-8'))['config']['parameter']
column_names = ['Header'] + [params[i]['label'] for i in range(len(params))]

# Get the navigation dataframe
df_navigation = pd.read_csv(io.StringIO(navigation_data['csv'].decode("utf-8")), header=None, names=column_names)
df_navigation = df_navigation[['System Timestamp', 'Latitude (deg)', 'Longitude (deg)', 'GPS Altitude MSL (m)', 'GPS Altitude (m)', 'Pressure Altitude (ft)', 'RADAR Altitude (ft)']]
df_navigation['System Timestamp'] = df_navigation['System Timestamp'].apply(lambda x: np.datetime64(x))

# Keep only one record per second by truncating the decimal numbers
# and pick the closest timestamp to the 'pure' second
df_navigation['System Timestamp'] = df_navigation['System Timestamp'].apply(lambda x: x.replace(microsecond=0))
df_navigation = df_navigation.drop_duplicates(subset='System Timestamp', keep='first', ignore_index=True)

# Read the probe data
with open(args.filepath) as handle:
    lines = [line.rstrip() for line in handle]

# Data table starts from line 14, and headers in 12-13
assert lines[12] == 'Time,Paltft,PressMMS,SATMMS,TASMMS,Surapply,IKPTWC5s'
keys = lines[12].split(',')

# To store the data:
df = pd.DataFrame()
for i in range(14, len(lines)):
    row_split = lines[i].split(',')
    hours, minutes, seconds = map(int, row_split[0].split(':'))
    timestamp = date + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    row_data = {
        keys[0]: timestamp
    }
    for i, e in enumerate(map(float, row_split[1:len(keys)]), start=1):
        row_data[keys[i]] = e
    df = pd.concat((df, pd.DataFrame(row_data, index=[0])), ignore_index=True)

df = df[['Time', 'Paltft', 'IKPTWC5s']]

# Merge
df_combined = pd.merge(df_navigation.rename(columns={'System Timestamp': 'Time'}), df, on='Time')

FOOT_TO_METRE = 0.3048
df_combined['Pressure Altitude (m)'] = FOOT_TO_METRE * df_combined['Pressure Altitude (ft)']
df_combined['Paltm (metres)'] = FOOT_TO_METRE * df_combined['Paltft']
df_combined = df_combined.drop(columns=['Pressure Altitude (ft)', 'Paltft', 'GPS Altitude MSL (m)', 'RADAR Altitude (ft)']) # Seems like `GPS Altitude MSL (m)` can be all nan, and `RADAR Altitude (ft)` does not vary

# `Pressure Altitude (m)` and `Paltm (metres)` determined to be very similar checking the numbers for `180727-FLT01-IKP2-R1-V2.7.csv`
# Keep only one variable
df_combined = df_combined.drop(columns='Pressure Altitude (m)')

# Rename to keep information
df_combined = df_combined.rename(columns={
    'Time':'UTC',
    'IKPTWC5s': 'TWC_gm3',
    'Latitude (deg)': 'latitude',
    'Longitude (deg)': 'longitude',
    'Paltm (metres)': 'altitude_pressure_metres',
    'GPS Altitude (m)': 'altitude_gps_metres'})

# Write to file
df_combined.to_csv(os.path.join(args.destination, 'friendly_{:}'.format(os.path.basename(args.filepath))), index=False)