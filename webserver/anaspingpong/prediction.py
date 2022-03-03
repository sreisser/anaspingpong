from anaspingpong.utils import Utils
import os
import glob2 as glob
import numpy as np
from anaspingpong.predictor import Predictor
import cv2
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image

EXTEND_TILES = 1
ZOOM = 20
SOURCE = "http://ecn.t0.tiles.virtualearth.net/tiles/a{quad}.jpeg?g=129&mkt=en&stl=H"


# SOURCE = "https://mt0.google.com/vt?lyrs=s&x={x}&s=&y={y}&z={z}"


def get_tables(latitude, longitude):
    start = datetime.now()
    predictor = Predictor()
    longitudes = np.empty((0, 1))
    latitudes = np.empty((0, 1))
    # create folder to store all positive predicted
    dir_predicted = "data/predicted"
    Path(dir_predicted).mkdir(exist_ok=True)

    # download center tile +- EXTEND_TILES in all directions
    # if shift=True, additional tiles shifted by 0.5 will be created
    download_folder = download_tables(latitude, longitude, shift=True)

    # create life test data
    dataset = glob.glob(f'{download_folder}/*')

    # predict probabilities
    i_tile = 0
    for tile in dataset:
        i_tile += 1
        print(f'tile {i_tile}')
        tilename = os.path.basename(tile)
        img = cv2.imread(tile)
        label_pred = predictor.predictor(img)
        scores = label_pred['instances'].scores.numpy()
        boxes = label_pred['instances'].pred_boxes.tensor.numpy()

        if len(scores) > 0:
            shutil.copyfile(tile, os.path.join(dir_predicted, tilename))
        for i, score in enumerate(scores):
            image_height, _, _ = img.shape
            box_center = np.array([np.mean([boxes[i][0], boxes[i][2]]),
                                   np.mean([boxes[i][1], boxes[i][3]])]) \
                         / image_height
            tilename = tilename.split('.')[0]
            tile_split = tilename.split('_')
            x = float(tile_split[0]) + box_center[0]
            y = float(tile_split[1]) + box_center[1]
            print(tile_split)
            if len(tile_split) == 4 and tile_split[3] == 'rb':
                x += 0.5
                y += 0.5
            z = int(tile_split[2])
            long = Utils.tile2long(x, z)
            lat = Utils.tile2lat(y, z)
            dist = np.array([Utils.measure(lat, long, ilat, ilong)
                             for ilat, ilong in zip(latitudes, longitudes)])

            # allow only tables which are more than 2m from the ones already
            # in the list
            keys = np.where(dist < 2.)
            if len(keys[0]) == 0:
                longitudes = np.append(longitudes, long)
                latitudes = np.append(latitudes, lat)

    # remove download folder
    shutil.rmtree(download_folder)

    print(f'Detected {len(longitudes)} tables')
    time_delta = datetime.now() - start
    print(f'Time spent for prediction: {time_delta}')
    return longitudes, latitudes


def download_tables(latitude, longitude, shift=True):
    center_x = Utils.long2tile(longitude, ZOOM)
    center_y = Utils.lat2tile(latitude, ZOOM)
    z = ZOOM

    tempDirectory = os.path.join("temp", f'tmp{Utils.randomString()}')
    os.makedirs(tempDirectory)
    images_array = []
    for i in range(-EXTEND_TILES, EXTEND_TILES + 1):
        images_row = []
        for j in range(-EXTEND_TILES, EXTEND_TILES + 1):
            # take n tiles left and right, up and down of center tile
            x = center_x + i
            y = center_y + j
            tempFile = f"{x}_{y}_{ZOOM}" + ".jpeg"
            tempFilePath = os.path.join(tempDirectory, tempFile)
            result = Utils.downloadFile(SOURCE, tempFilePath, x, y, z)
            if shift:
                images_row.append(np.asarray(Image.open(tempFilePath)))
        if shift:
            images_array.append(images_row)
    print(f'Downloaded tiles to folder {tempDirectory}')

    if shift:
        # create shifted tiles
        images_array = np.array(images_array)
        array_shape = images_array.shape
        tile_width = array_shape[2]
        tile_height = array_shape[3]
        merged_width = array_shape[0] * tile_width
        merged_image_array = images_array.reshape(
            (array_shape[0], merged_width, tile_height, array_shape[4]))
        merged_image_array = np.concatenate(merged_image_array, axis=1)
        for i in range(EXTEND_TILES * 2):
            for j in range(EXTEND_TILES * 2):
                x1, x2 = (int(tile_width * (i + 0.5)), int(tile_width * (i + 1.5)))
                y1, y2 = (int(tile_height * (j + 0.5)), int(tile_height * (j + 1.5)))
                im = Image.fromarray(merged_image_array[x1:x2, y1:y2])
                x = center_x + j - EXTEND_TILES
                y = center_y + i - EXTEND_TILES
                tempFile = f"{x}_{y}_{ZOOM}_rb" + ".jpeg"
                tempFilePath = os.path.join(tempDirectory, tempFile)
                im.save(tempFilePath)

    return tempDirectory


