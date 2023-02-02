"""
Combine the differnet CSV files with TWC data from the
HAIC-HIWC campaigns. It requires that the data from the campaigns
has been prepared with the scripts 10.5065_D6RN36KJ.py, 10.5065_D6WW7GDS.py, etc.

Note: the resulting dataframe has not been cleaned on purpose.
    Values of -999.0 for TWC are invalid values
"""

import glob
import os

import pandas as pd

orders = {
    'amell60263': {
        'campaign': 'HAIC-HIWC',
        'instrument': 'IKP2',
        'DOI': 'https://doi.org/10.5065/D6WW7GDS',
        'notes': 'SAFIRE-provided Altitude from INS; hence likely pressure altitude.',
        'location': 'Darwin, Australia'
    },
    'amell32766': {
        'campaign': 'HAIC-HIWC',
        'instrument': 'ROBUST',
        'DOI': 'https://doi.org/10.26023/R3B8-Y0DS-0C13',
        'notes': 'SAFIRE-provided Altitude from INS; hence likely pressure altitude.',
        'location': 'Darwin, Australia'
    },
    'amell60877': {
        'campaign': 'HAIC-HIWC_2015',
        'instrument': 'IKP2',
        'DOI': 'https://doi.org/10.5065/D61N7ZV7',
        'notes': 'SAFIRE-provided Altitude from INS; hence likely pressure altitude.',
        'location': 'Cayenne, French Guiana'
    },
    'amell60679': {
        'campaign': 'HAIC-HIWC_2015',
        'instrument': 'ROBUST',
        'DOI': 'https://doi.org/10.26023/A3V5-TJDV-HY14',
        'notes': 'SAFIRE-provided Altitude from INS; hence likely pressure altitude.',
        'location': 'Cayenne, French Guiana'
    },
    'amell61210': {
        'campaign': 'HIWC-RADAR',
        'instrument': 'IKP2',
        'DOI': 'https://doi.org/10.5065/D6RN36KJ',
        'notes': '',
        'location': 'Fort Lauderdale, Florida'
    },
    'amell32226': {
        'campaign': 'HIWC-RADAR-2018',
        'instrument': 'IKP2',
        'DOI': 'https://doi.org/10.26023/8V5Y-GB2E-CX07',
        'notes': '',
        'location': 'Fort Lauderdale, Florida; Palmdale, California; Kona, Hawaii'
    },
}

# Path to the directory with 'friendly' CSVs,
# i.e. the CSVs prepared with the scripts in this directory
path_friendly_csv = '/mnt/data_copper2/ccic/flight_campaigns/HAICHIWC/orders/friendly'
#%%
# Create a dataframe for each order
for order_id, data in orders.items():
    order_dataframes = [pd.read_csv(f) for f in glob.glob(os.path.join(path_friendly_csv, order_id, '*csv'))]
    orders[order_id]['data'] = pd.concat(order_dataframes, ignore_index=True)

# %%
# Recover the dataframe and update the 'notes' value if TWC does not have a unit,
# which is indicated by having the column TWC_? instead of TWC_gm3
for order_id, data in orders.items():
    df = orders[order_id]['data']
    if 'TWC_?' in df.columns:
        orders[order_id]['notes'] = '{:} TWC unit missing, assuming g/m3'.format(orders[order_id]['notes'])
        orders[order_id]['data'] = orders[order_id]['data'].rename(columns={'TWC_?': 'TWC_gm3'})

#%%
# Create a dataframe for each order that contains as extra columns,
# (implying repeated information), the information `campaign`, `instrument`,
# `DOI`, `location`, and `notes`
for order_id, data in orders.items():
    df =  data['data']
    df['campaign'] = data['campaign']
    df['instrument'] = data['instrument']
    df['DOI'] = data['DOI']
    df['notes'] = data['notes']
    df['location'] = data['location']

    # Let's just use the pressure altitude, since it's present in all datasets
    if 'altitude_gps_metres' in df.columns:
        df = df.drop(columns='altitude_gps_metres')

    orders[order_id]['df'] = df

#%%
# Finally, combine all dataframes from all orders into a single dataframe
df_all_orders = pd.concat(
    [data['df'] for data in orders.values()],
    ignore_index=True
)

# and write to file
df_all_orders.to_csv('HAICHIWC_TWC.csv', index=False)