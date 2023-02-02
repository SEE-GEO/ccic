"""
Parses the source CSV files from https://doi.org/10.26023/R3B8-Y0DS-0C13
into a friendly CSV format for analysing TWC.

Requires that the files from https://doi.org/10.5065/D6WW7GDS
are previously parsed with the script 10.5065_D6WW7GDS.py
"""

import argparse
import datetime
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('filepath',
    help='the CSV file to parse')
parser.add_argument('auxfile',
    help='the corresponding (parsed) CSV file obtained with '
         '10.5065_D6WW7GDS.py (DOI: 10.5065/D6WW7GDS)')
parser.add_argument('destination',
    help='where to place the parsed CSV')

args = parser.parse_args()

df_aux = pd.read_csv(args.auxfile)
df_in = pd.read_csv(args.filepath)

if 'TWC robust' in df_in.columns:
    df_in = df_in.rename(columns={'TWC robust': 'TWC_robust'})
elif 'Unnamed: 1' in df_in.columns:
    df_in = df_in.rename(columns={'Unnamed: 1': 'TWC_robust'})

df_in = df_in[['GMT','TWC_robust']]
df_in = df_in[df_in.isnull().any(axis=1) == False]

TIMESTAMP_BASE = datetime.datetime.strptime(
    os.path.basename(args.filepath).split('_')[3], '%Y%m%d'
)

def fun(row):
    """Convenience function"""
    hours, minutes, seconds = map(int, row.split(':'))
    return TIMESTAMP_BASE + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

df_in['GMT'] = df_in.GMT.apply(fun)
df_in = df_in.rename(columns={'GMT': 'UTC'})

df_in['UTC'] = df_in['UTC'].apply(str)
df_aux['UTC'] = df_aux['UTC'].apply(str)
df = pd.merge(df_in, df_aux, on='UTC')

df = df[['UTC', 'TWC_robust', 'latitude', 'longitude', 'altitude_pressure_metres']].rename(columns={'TWC_robust': 'TWC_?'})
df['UTC'] = df['UTC'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

assert df.index.size > 0

df.to_csv(os.path.join(args.destination, 'friendly_{:}'.format(os.path.basename(args.filepath))), index=False)