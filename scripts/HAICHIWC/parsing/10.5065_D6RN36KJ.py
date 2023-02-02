"""
Parses the source CSV files from https://doi.org/10.5065/D6RN36KJ
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

# Data table starts from line 5, and headers in 3-4
assert lines[3] == 'TIMECH2,PstatE,TATE,SAT2,BoeTAT,BoeingSAT,TASgood,XMach,LATE,LNGE,PALTFT,CDPcnc,PIPcnc,IKPTWCsur5s'
keys = lines[3].split(',')

# Time offset
pattern = r'(\d{8}).*'
date = re.search(pattern, os.path.basename(args.filepath)).group(1)
year, month, day = map(int, (date[:4], date[4:6], date[6:]))
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
df = df[['TIMECH2', 'LATE', 'LNGE', 'PALTFT', 'IKPTWCsur5s']]

# Remap columns
df = df.rename(columns={'TIMECH2': 'timestamp_UTC', 'LATE': 'latitude', 'LNGE': 'longitude', 'PALTFT': 'altitude_feet', 'IKPTWCsur5s': 'TWC_gm3'})

FOOT_TO_METRES = 0.3048
df['altitude_metres'] = FOOT_TO_METRES * df['altitude_feet']
df = df.drop(columns='altitude_feet')

# To CSV
df.to_csv(os.path.join(args.destination, 'friendly_{:}'.format(os.path.basename(args.filepath))), index=False)