import math
import pandas as pd
import os
import itertools
import random
import shutil
from PIL import Image


def long2tile(lon, zoom):
    return math.floor((lon + 180) / 360 * math.pow(2, zoom))


def lat2tile(lat, zoom):
    return math.floor(
        (
            1
            - math.log(
                math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)
            )
            / math.pi
        )
        / 2
        * math.pow(2, zoom)
    )


path_lat_long_reference = (
    r"/home/geomi/gm/projects/dsr/portfolio_project/publicpingpong/data"
)

path_negative_tiles = (
    r"/home/geomi/gm/projects/dsr/portfolio_project/trainingdata/negative/"
)

path_all_tiles = r"/home/geomi/gm/projects/dsr/portfolio_project/MapTilesDownloader/src/output/bing_test/20"

ZOOM = 20

list_of_xs = os.listdir(path_all_tiles)
list_of_xs = list(map(int, list_of_xs))  # convert to numbes

list_of_ys = []

for x in list_of_xs:
    list_of_ys_in_fold = os.listdir(os.path.join(path_all_tiles, str(x)))
    list_of_ys_in_fold = [
        int(f.split("_")[0].split(".")[0]) for f in list_of_ys_in_fold
    ]
    # list_of_ys_in_fold = list(map(int, list_of_ys_in_fold)) # convert to numbes
    list_of_ys.extend(list_of_ys_in_fold)

list_of_ys = list(set(list_of_ys))

min_x_BER = min(list_of_xs)
max_x_BER = max(list_of_xs)

min_y_BER = min(list_of_ys)
max_y_BER = max(list_of_ys)

print(min_x_BER, max_x_BER, min_y_BER, max_y_BER)

df_table = pd.read_csv(os.path.join(path_lat_long_reference, "lat_long.csv"))

# Transform latitude to tile_y & longitude to tile_X
df_table["tile_y"] = df_table.loc[:, "latitude"].apply(lambda y: lat2tile(y, ZOOM))
df_table["tile_x"] = df_table.loc[:, "longitude"].apply(lambda x: long2tile(x, ZOOM))

tiles_x_ref = df_table["tile_x"].values.astype(int).tolist()
tiles_y_ref = df_table["tile_y"].values.astype(int).tolist()

# Drop tables if latitude is 999 -> no existing table (code by S.)
df_table = df_table.query("latitude != 999")

n_neg_tables_randomly_selected = 800
n_neg_tables_around_tables = 700


# SELECT TILES RANDOMLY FROM BERLIN RANGE EXCLUDING TILES CORRESPONDING TO TABLES +-2

# Create all combinations of x-y pairs
x_all = list(range(min_x_BER, max_x_BER))
y_all = list(range(min_y_BER, max_y_BER))
xy_all = list(itertools.product(x_all, y_all))
print(len(x_all), len(y_all), len(xy_all))

# Select randomly from these combinations
# Check if file exists
# Check if file is similar to +-2 tiles around existing tables
n_neg_tables_found_randomly = 0
xy_all_shuffled = xy_all.copy()
random.shuffle(xy_all_shuffled)
xy_subset = xy_all_shuffled[
    : n_neg_tables_randomly_selected * 2
]  # get twice the wanted number because we might discard some
xy_reference = list(zip(tiles_x_ref, tiles_y_ref))
xy_selected = [
    xy for xy in xy_subset if xy not in xy_reference
]  # check they are not table tiles

# Loop through data and copy to folder with negative samples
for xy in xy_selected:
    path_orig = os.path.join(path_all_tiles, str(xy[0]), str(xy[1]) + ".jpeg")
    filename = f"{str(ZOOM)}_{str(xy[0])}_{str(xy[1])}.jpeg"
    path_dest = os.path.join(path_negative_tiles, filename)
    # check if file exists in original data
    if os.path.isfile(path_orig):
        # If file already in training data folder -> skip
        if os.path.isfile(path_dest):
            continue
        shutil.copy(path_orig, path_dest)
        n_neg_tables_found_randomly += 1

    if n_neg_tables_found_randomly == n_neg_tables_randomly_selected:
        break


# SELECT TILES AROUND TABLES -> -4 to -3

# go through reference tables and take tiles around the tables
xy_reference_shuffled = xy_reference.copy()
random.shuffle(xy_reference_shuffled)

n_neg_tables_found_around_tables = 0

for xyr in xy_reference_shuffled:
    # Define cluster of images in +-3
    xs_range3 = list(range(xyr[0] - 3, xyr[0] + 4))
    ys_range3 = list(range(xyr[1] - 3, xyr[1] + 4))
    xy_clust_range3 = list(itertools.product(xs_range3, ys_range3))

    # Define cluster of images in +-4
    xs_range4 = list(range(xyr[0] - 4, xyr[0] + 5))
    ys_range4 = list(range(xyr[1] - 4, xyr[1] + 5))
    xy_clust_range4 = list(itertools.product(xs_range4, ys_range4))

    # Get the difference, ie tiles on periphery of table
    xy_clust_around = list(set(xy_clust_range4) - set(xy_clust_range3))
    # print(len(xy_clust_range4), len(xy_clust_range3), len(xy_clust_around))

    # Save tiles on periphery of tables if the exist in our dataset (ie, table in Berlin)

    for xy in xy_clust_around:
        path_orig = os.path.join(path_all_tiles, str(xy[0]), str(xy[1]) + ".jpeg")
        filename = f"{str(ZOOM)}_{str(xy[0])}_{str(xy[1])}.jpeg"
        path_dest = os.path.join(path_negative_tiles, filename)

        # Check if file exists in original data
        if os.path.isfile(path_orig):
            # If file already in training data folder -> skip
            if os.path.isfile(path_dest):
                continue
            shutil.copy(path_orig, path_dest)
            n_neg_tables_found_around_tables += 1

        if n_neg_tables_found_around_tables == n_neg_tables_around_tables:
            break

    if n_neg_tables_found_around_tables == n_neg_tables_around_tables:
        break
