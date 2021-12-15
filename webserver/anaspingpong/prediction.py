from anaspingpong.utils import Utils
import os
import glob2 as glob
import numpy as np
from anaspingpong.predictor import Predictor
import cv2


EXTEND_TILES = 2
ZOOM = 20
SOURCE = "http://ecn.t0.tiles.virtualearth.net/tiles/a{quad}.jpeg?g=129&mkt=en&stl=H"
#SOURCE = "https://mt0.google.com/vt?lyrs=s&x={x}&s=&y={y}&z={z}"

#YAML = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'

def get_tables(latitude, longitude):
    predictor = Predictor()
    longitudes = np.empty((0, 1))
    latitudes = np.empty((0, 1))
  #  all_scores = np.empty((0, 1))
    # download center tile +- EXTEND_TILES in all directions
    download_folder = download_tables(latitude, longitude)

    # create life test data
    dataset = glob.glob(f'{download_folder}/*')

    # predict probabilities
    i_tile = 0
    for tile in dataset:
        print(f'Evaluating tile {i_tile} {tile}')
        i_tile += 1
        tilename = os.path.basename(tile)
        img = cv2.imread(tile)
        label_pred = predictor.predictor(img)
        scores = label_pred['instances'].scores.numpy()
        boxes = label_pred['instances'].pred_boxes.tensor.numpy()
        for i, score in enumerate(scores):
            print(score)
            image_height, _, _ = img.shape
            box_center = np.array([np.mean([boxes[i][0], boxes[i][2]]),
                          np.mean([boxes[i][1], boxes[i][3]])]) \
                         / image_height
            tile_split = tilename.split('_')
            x = float(tile_split[0]) + box_center[0]
            y = float(tile_split[1]) + box_center[1]
            z = int(tile_split[2].replace('.jpeg', ''))
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

    print(f'Detected {len(longitudes)} tables')
    return longitudes, latitudes


def download_tables(latitude, longitude):
    center_x = Utils.long2tile(longitude, ZOOM)
    center_y = Utils.lat2tile(latitude, ZOOM)
    z = ZOOM

    tempDirectory = os.path.join("temp", f'tmp{Utils.randomString()}')
    os.makedirs(tempDirectory)
    for i in range(-EXTEND_TILES, EXTEND_TILES+1):
        for j in range(-EXTEND_TILES, EXTEND_TILES+1):
            # take n tiles left and right, up and down of center tile
            x = center_x + i
            y = center_y + j
            tempFile = f"{x}_{y}_{ZOOM}" + ".jpeg"
            tempFilePath = os.path.join(tempDirectory, tempFile)
            result = Utils.downloadFile(SOURCE, tempFilePath, x, y, z)
    print(f'Downloaded tiles to folder {tempDirectory}')
    return tempDirectory
