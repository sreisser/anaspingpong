import requests
import numpy as np
import re


# Website from which we extact geolocation of ping-poing tables
base_website = "https://www.pingpongmap.net/?id="

# Extact latitude and longitude
ids = []
latitudes = []
longitudes = []

id = 1  # initialize id
counter_tables = 1  # initialize counter
n_tables_needed = 200  # for how many tables we need info

while True:
    url = base_website + str(id)
    r = requests.get(url)
    page_source = r.text

    latitudes_in_html = re.findall("newLat =(.*);", page_source)  # look in html
    longitudes_in_html = re.findall("newLng =(.*);", page_source)

    if "Berlin" in page_source:
        ids.append(id)  # save id
        latitudes.append(
            float(latitudes_in_html[1])
        )  # 1st element is a default Lat, 2nd is what we want
        longitudes.append(float(longitudes_in_html[1]))
        print(f"Extracting Lat & Long for n. {counter_tables} table with id : {id}")
        counter_tables += 1  # update counter

    if counter_tables == n_tables_needed:
        break
    id += 1

# Concatenate IDs Lat and Long
result = np.array(list(zip(ids, latitudes, longitudes)))

# Save results
np.save("lat_long.npy", result)
