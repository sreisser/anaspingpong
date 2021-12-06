"""Generating negative samples 
"""

from os import listdir
from os.path import isfile, join
import shutil

# get min and max tile_x and tile_y in positive samples

path_positive_tiles = (
    r"/home/geomi/gm/projects/dsr/portfolio_project/trainingdata/positive_tiles/"
)
path_negative_tiles = (
    r"/home/geomi/gm/projects/dsr/portfolio_project/trainingdata/negative_tiles/"
)

path_all_tiles = r"/home/geomi/gm/projects/dsr/portfolio_project/MapTilesDownloader/src/output/bing_test"

sel_tiles_zoom = [
    int(f.split("_")[0])
    for f in listdir(path_positive_tiles)
    if isfile(join(path_positive_tiles, f))
]

sel_tiles_val_1 = [
    int(f.split("_")[1])
    for f in listdir(path_positive_tiles)
    if isfile(join(path_positive_tiles, f))
]

sel_tiles_val_2 = [
    int(f.split("_")[2].split(".")[0])
    for f in listdir(path_positive_tiles)
    if isfile(join(path_positive_tiles, f))
]

sel_tiles_v1v2 = [[v1, v2] for v1, v2 in zip(sel_tiles_val_1, sel_tiles_val_2)]


zoom = sel_tiles_zoom[0]  # same for all
min_val1 = min(sel_tiles_val_1)
max_val1 = max(sel_tiles_val_1)

min_val2 = min(sel_tiles_val_2)
max_val2 = max(sel_tiles_val_2)

# print(zoom, min_val1, max_val1, min_val2, max_val2)

count_n_neg_samples = 0

for val1 in range(min_val1, max_val1):
    for val2 in range(min_val2, max_val2):
        tile_v1v2 = [val1, val2]
        if tile_v1v2 not in sel_tiles_v1v2:
            # transfer from from database of all tiles to folder for negative_samples
            path_orig = join(path_all_tiles, str(zoom), str(val1), str(val2) + ".jpeg")
            filename = f"{str(zoom)}_{str(val1)}_{str(val2)}.jpeg"
            path_dest = join(path_negative_tiles, filename)
            # check if file exists in original data
            if isfile(path_orig):
                # If file already in training data folder -> skip
                if isfile(path_dest):
                    continue
                shutil.copy(path_orig, path_dest)
                count_n_neg_samples += 1
        if count_n_neg_samples == 200:
            break
    if count_n_neg_samples == 200:
        break

print("Total number of negative samples created: ", str(count_n_neg_samples))
