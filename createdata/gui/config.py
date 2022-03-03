import os

# Configuration parameters

PATH_BERLIN_TILES = (
    "../../data/data_berlin_tiles/MapTilesDownloader/src/output/bing_test/20"
)
PATH_REFERENCE_TABLES_LAT_LONG = "../../data/lat_long.csv"

ZOOM = 20
EXTEND_TILES = 2
SOURCE = "http://ecn.t0.tiles.virtualearth.net/tiles/a{quad}.jpeg?g=129&mkt=en&stl=H"
# SOURCE = "https://mt0.google.com/vt?lyrs=s&x={x}&s=&y={y}&z={z}"
PATH_NEGATIVE_TILES_TMP = (
    "../../data/data_train/classification/train_validation/negative_tmp"
)

if "google" in SOURCE:
    SOURCE_CODE = "google"
    PATH_POSITIVE_TILES = (
        "../../data/data_train/classification/train_validation/positive_google"
    )
    PATH_LAST_ID = "./create_positive_last_id_checked_google.txt"
else:
    SOURCE_CODE = "bing"
    PATH_POSITIVE_TILES = (
        "../../data/data_train/classification/train_validation/positive"
    )
    PATH_LAST_ID = "./create_positive_last_id_checked.txt"

FLAG_NEG_POS = 1  # 0 for selecting negative tiles, 1 for positive
if FLAG_NEG_POS == 0:
    PATH_SAVE_TILES = PATH_NEGATIVE_TILES_TMP
elif FLAG_NEG_POS == 1:
    PATH_SAVE_TILES = PATH_POSITIVE_TILES

if not os.path.isdir(PATH_SAVE_TILES):
    os.makedirs(PATH_SAVE_TILES)

IMAGE_SHAPE = (256, 256, 3)

SHIFT_TYPES = ["r", "b", "rb"]

# For each shift-type: define difference in the number of tile_x & tile_y from original
x_dif_from_orig = {"r": 0, "b": -1, "rb": -1}
y_dif_from_orig = {"r": -1, "b": 0, "rb": -1}

# Coordinates of TOIs relative to the corresponding shifted group of tiles

# Dimensions for shifted group of tiles are:
# r: 5x4, b: 4x5, rb: 4x4
# eg. "r": [2, 1] means -> from the 5x4 tiles after shifting right -> get tile in position [2,1]
coords_tois_onshifted = {
    "r": [[2, 1], [2, 2]],
    "b": [[1, 2], [2, 2]],
    "rb": [[1, 1], [1, 2], [2, 1], [2, 2]],
}

# We want to have 3x3 tiles to inspect, with original tile + 8 shifted around it.
# For each defined TOI above define corresponding position in the new 3x3 tiles group
# [1,1] will be filled by original central tile
coords_tois_onnew = {
    "r": [[1, 0], [1, 2]],
    "b": [[0, 1], [2, 1]],
    "rb": [[0, 0], [0, 2], [2, 0], [2, 2]],
}
