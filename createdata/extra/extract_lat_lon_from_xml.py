import requests
import numpy as np
import re
import pandas as pd
from os.path import isfile
import xml.etree.ElementTree as ET

input_xml = '../data/pingpongmap_3.xml'

def write_file(data, output_file):
	# Concat to old table
	data = data.sort_values(by=['id'])
	# Save complete table to file
	data.to_csv(output_file, header=True, index=False)
	print("Write to file")
	return data

def get_trainingdata():
	""" Get GPS positions of tables from pingpongmap.net """

	output_file = "../data/lat_long.csv"

	tree = ET.parse(input_xml)
	root = tree.getroot()

	# Extract latitude and longitude
	ids = []
	latitudes = []
	longitudes = []

	for child in root:
		ids.append(child.attrib['id'])
		if child.attrib['lat']:
			latitudes.append(float(child.attrib['lat']))
			longitudes.append(float(child.attrib['lng']))
		else:
			latitudes.append(999.)
			longitudes.append(999.)

	tables_df = pd.DataFrame(
			{"id": np.array(ids).astype(int), "latitude": latitudes,
			 "longitude": longitudes})
	write_file(tables_df, output_file)



get_trainingdata()

