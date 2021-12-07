import requests
import numpy as np
import re
import pandas as pd
from os.path import isfile

def write_file(old_data, new_data, output_file):
	# Concat to old table
	data = pd.concat([old_data, new_data])
	# Save complete table to file
	data.to_csv(output_file, header=True, index=False)
	print("Write to file")
	return data

def get_trainingdata(start_id=1, end_id=200):
	""" Scrape GPS positions of tables from pingpongmap.net """

	output_file = "lat_long.csv"
	if isfile(output_file):
		output_df = pd.read_csv(output_file)

		if start_id in output_df['id'].values and end_id in output_df['id'].values:
			return output_df
	else:
		output_df = pd.DataFrame({'id' : pd.Series(dtype=int),
						'latitude' : pd.Series(dtype=float),
						'longitude' : pd.Series(dtype=float)
					})
	

	# Website from which we extact geolocation of ping-poing tables
	base_website = "https://www.pingpongmap.net/?id="

	# Extact latitude and longitude
	ids = []
	latitudes = []
	longitudes = []

	counter_tables = 0  # initialize counter

	for index in range(start_id, end_id + 1):
		if index in output_df['id'].values:
			print(f"Skip {index}")
			continue

		url = base_website + str(index)
		r = requests.get(url)
		page_source = r.text

		latitudes_in_html = re.findall("newLat =(.*);", page_source)[1].strip()  # look in html
		longitudes_in_html = re.findall("newLng =(.*);", page_source)[1].strip()

		if not latitudes_in_html:
			print(f"{index}: No latitudes")
			continue 

		ids.append(index)  # save id
		latitudes.append(
			float(latitudes_in_html)
		)  # 1st element is a default Lat, 2nd is what we want
		longitudes.append(float(longitudes_in_html))
		print(f"Extracting Lat & Long for n. {counter_tables} table with id : {index}")
		counter_tables += 1  # update counter

		if counter_tables % 50 == 0:
			new_tables = pd.DataFrame({"id": np.array(ids).astype(int), "latitude": latitudes, "longitude": longitudes})
			output_df = write_file(output_df, new_tables, output_file)
			ids = []
			latitudes = []
			longitudes = []

	new_tables = pd.DataFrame({"id": np.array(ids).astype(int), "latitude": latitudes, "longitude": longitudes})
	write_file(output_df, new_tables, output_file)

start = 1010
end = 1040
get_trainingdata(start, end)
