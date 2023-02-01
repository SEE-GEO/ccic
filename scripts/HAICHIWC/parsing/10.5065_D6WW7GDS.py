"""
Parses the source CSV files from https://doi.org/10.5065/D6WW7GDS
into a friendly CSV format for analysing TWC
"""

import argparse
import datetime
import re
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('filepath',
    help='the csv file to parse')
parser.add_argument('destination',
    help='the directory to place the parsed csv file')

args = parser.parse_args()



with open(args.filepath) as handle:
    lines = [line.rstrip() for line in handle]

# Data table starts from line 4
assert lines[4] == 'Stimech,Slat,Slong,SIAltm,SINShead,Spress,SSAT,STAT,SRHWVSS,STAS,Szaccg,Swdir,Swspd,XKBZR5s'
keys = lines[4].split(',')

# Time offset
pattern = r'f20-(.*)-[0-9]{6}-IKPV5.csv'
year, month, day = map(int, re.search(pattern, os.path.basename(args.filepath)).group(1).split('-'))
timestamp_start = datetime.datetime(year, month, day)

# To store the data:
df = pd.DataFrame()
for i in range(5, len(lines)):
    row_split = lines[i].split(',')
    hours, minutes, seconds = map(int, row_split[0].split(':'))
    timestamp = timestamp_start + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    row_data = {
        keys[0]: timestamp
    }
    for i, e in enumerate(map(float, row_split[1:len(keys)]), start=1):
        row_data[keys[i]] = e
    df = pd.concat((df, pd.DataFrame(row_data, index=[0])), ignore_index=True)

# Keep only some variables
df = df[['Stimech', 'Slat', 'Slong', 'SIAltm', 'XKBZR5s']]

# Remap columns
df = df.rename(columns={'Stimech': 'timestamp_UTC', 'Slat': 'latitude', 'Slong': 'longitude', 'SIAltm': 'altitude_metres', 'XKBZR5s': 'TWC_?'})

# To CSV
df.to_csv(os.path.join(args.destination, 'friendly_{:}'.format(os.path.basename(args.filepath))), index=False)