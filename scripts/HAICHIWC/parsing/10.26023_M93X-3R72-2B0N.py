"""
Parses a flight track KML file from the dataset https://doi.org/10.26023/M93X-3R72-2B0N

The parsing has been constructed by manually identifying the appropriate
elements in the KML file.
"""
import argparse
import datetime
import os
import re

import pandas as pd
import xmltodict

parser = argparse.ArgumentParser()
parser.add_argument('kmlfile',
    help="the kml file to parse")
parser.add_argument('destination',
    help="where to save the data as a CSV file")

args = parser.parse_args()

with open(args.kmlfile) as handle:
    xml_dict = xmltodict.parse(handle.read())

placemarks = xml_dict['kml']['Document']['Folder']['Placemark']
description_pattern = re.compile(r"<b> Position of GBTM </b> <br/>Location <br/>lon: (.*)<br/> lat: (.*)<br/>alt: (.*)<br/>Time <br/> (.*)")
coordinates = pd.DataFrame()
for placemark in placemarks:
    assert placemark['Style']['@id'] == 'position'
    assert placemark['styleUrl'] == '#position'
    assert placemark['Point']['altitudeMode'] == 'absolute'
    description_data = re.search(description_pattern, placemark['description']).groups()
    c1, c2, c3 = map(float, description_data[:3])
    c4 = datetime.datetime.strptime(description_data[-1], '%Y-%m-%d %H:%M:%S')
    # Coordinates: Lon, lat, altitude, UTC timestamp
    coordinates = pd.concat(
        (
            coordinates,
            pd.DataFrame(
                {
                    'longitude_deg': c1,
                    'latitude_deg': c2,
                    'altitude_?': c3,
                    'timestamp_utc': c4
                },
                index=[0]
            )
        ),
        ignore_index=True
    )

coordinates.to_csv(os.path.join(args.destination, os.path.basename(args.kmlfile).replace('.kml', '.csv')))